"""Temporary eager orchestration for variable-``M`` general XD.

The dense kernels in :mod:`development.general_xd` operate on one fixed
observed dimension.  This module evaluates the deterministic mask groups made
by :mod:`development.general_validation`, restores inference rows, and merges
every informative group before performing one global M-step.  Group discovery
and the Python tuple loop are deliberately outside whole-operation JIT and
autodiff guarantees.

The grouped update never runs a group-local M-step.  It combines log component
masses, local component means, and local centered covariances with a weighted
Chan merge, applies the ridge once, and evaluates the candidate objective in a
second pass.  Any active fitting failure rolls the entire candidate back.
"""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from .general_validation import GroupedGeneralFitInputs, GroupedGeneralInputs
from .general_xd import (
    _component_log_weight_reductions,
    _exp_with_gradual_underflow,
    _has_nonzero_floating_magnitude,
    _log_positive_with_gradual_underflow,
    _multiply_by_log_scale_with_gradual_underflow,
    posterior_components_general,
)
from .identity_xd import EStep, Params


Array = jax.Array


class GroupedStepStatus(IntEnum):
    """Outcome of one host-orchestrated global grouped update."""

    SUCCESS = 0
    NUMERICAL_FAILURE = 1
    COMPONENT_COLLAPSED = 2


class GroupedFailureStage(IntEnum):
    """Stage at which a grouped candidate became unusable."""

    NONE = 0
    CURRENT_STATISTICS = 1
    M_STEP = 2
    CANDIDATE_OBJECTIVE = 3


class GroupedPosteriorResult(NamedTuple):
    """Restored dense posterior leaves and raw per-group inference status."""

    e_step: EStep
    group_numerical_failure: Array


class GroupedGeneralSufficientStatistics(NamedTuple):
    """Global weighted statistics merged across all informative groups.

    ``failed_pairs`` and ``group_numerical_failure`` describe fitting-active
    failures only: a raw factor failure on a zero-weight row remains visible in
    :class:`GroupedPosteriorResult` but has no statistical effect here.
    ``centered_covariance`` is normalized by component mass and does not include
    the covariance ridge.
    """

    mass: Array
    first_moment: Array
    second_moment: Array
    log_mass: Array
    component_mean: Array
    centered_covariance: Array
    weighted_log_likelihood: Array
    informative_weight: Array
    objective: Array
    numerical_failure: Array
    failed_pairs: Array
    group_numerical_failure: Array


class GroupedGeneralEMStepResult(NamedTuple):
    """One global grouped update with exact whole-state rollback metadata."""

    parameters: Params
    e_step: GroupedPosteriorResult
    statistics: GroupedGeneralSufficientStatistics
    objective: Array
    previous_objective: Array
    attempted_objective: Array
    attempted_objective_valid: Array
    status: Array
    failure_stage: Array
    collapsed: Array
    collapsed_components: Array
    numerical_failure: Array
    candidate_group_numerical_failure: Array
    candidate_failed_pairs: Array


class _LocalCenteredStatistics(NamedTuple):
    log_mass: Array
    component_mean: Array
    centered_covariance: Array
    objective_contribution: Array
    active_failed_pairs: Array
    numerical_failure: Array


def _stable_symmetrize(value: Array) -> Array:
    """Symmetrize without overflowing a finite near-limit matrix addition."""

    half = jax.lax.optimization_barrier(0.5 * value)
    return half + half.swapaxes(-1, -2)


def _positive_weight(value: Array) -> Array:
    """Return selected-dtype positive weights without flushing subnormals."""

    return (
        jnp.isfinite(value)
        & (~jnp.signbit(value))
        & _has_nonzero_floating_magnitude(value)
    )


def _restore_rows(
    grouped: GroupedGeneralInputs,
    values: Sequence[Array],
) -> Array:
    """Concatenate group-leading arrays and restore original input order."""

    concatenated = jnp.concatenate(tuple(values), axis=0)
    restoration = jnp.asarray(grouped.restoration_indices, dtype=jnp.int32)
    return concatenated[restoration]


def _posterior_group_leaves(
    grouped: GroupedGeneralInputs,
    *,
    parameters: Params,
    factor_jitter: Array | float,
) -> tuple[GroupedPosteriorResult, tuple[EStep, ...]]:
    """Evaluate every fixed-``M`` leaf and restore all row-leading fields."""

    leaves = tuple(
        posterior_components_general(
            parameters,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            factor_jitter=factor_jitter,
        )
        for group in grouped.groups
    )
    group_failure = jnp.stack(
        tuple(jnp.asarray(leaf.numerical_failure) for leaf in leaves)
    )
    restored = EStep(
        component_log_density=_restore_rows(
            grouped, tuple(leaf.component_log_density for leaf in leaves)
        ),
        component_log_joint=_restore_rows(
            grouped, tuple(leaf.component_log_joint for leaf in leaves)
        ),
        score_samples=_restore_rows(
            grouped, tuple(leaf.score_samples for leaf in leaves)
        ),
        responsibilities=_restore_rows(
            grouped, tuple(leaf.responsibilities for leaf in leaves)
        ),
        conditional_mean=_restore_rows(
            grouped, tuple(leaf.conditional_mean for leaf in leaves)
        ),
        conditional_covariance=_restore_rows(
            grouped, tuple(leaf.conditional_covariance for leaf in leaves)
        ),
        numerical_failure=jnp.any(group_failure),
        failed_pairs=_restore_rows(
            grouped, tuple(leaf.failed_pairs for leaf in leaves)
        ),
    )
    return GroupedPosteriorResult(restored, group_failure), leaves


def posterior_components_grouped(
    grouped: GroupedGeneralInputs,
    *,
    factor_jitter: Array | float = 0.0,
) -> GroupedPosteriorResult:
    """Evaluate and restore every deterministic fixed-``M`` mask group.

    The returned E-step preserves raw inference failure status.  In particular,
    it does not hide a failed row merely because that row later has zero fitting
    weight.
    """

    result, _ = _posterior_group_leaves(
        grouped,
        parameters=grouped.parameters,
        factor_jitter=factor_jitter,
    )
    return result


def _normalized_objective_contribution(
    e_step: EStep,
    sample_weight: Array,
    *,
    observed_dimension: int,
    log_informative_weight: Array,
) -> tuple[Array, Array, Array]:
    """Return a group's normalized objective and active failure diagnostics."""

    positive = _positive_weight(sample_weight)
    active_failed_pairs = e_step.failed_pairs & positive[:, None]
    if observed_dimension == 0:
        return (
            jnp.asarray(0.0, dtype=sample_weight.dtype),
            active_failed_pairs,
            jnp.asarray(False),
        )

    finite_score = jnp.isfinite(e_step.score_samples)
    usable_score = positive & finite_score
    log_weight = _log_positive_with_gradual_underflow(sample_weight)
    normalized_log_weight = jnp.where(
        positive,
        log_weight - log_informative_weight,
        -jnp.inf,
    )
    safe_score = jnp.where(usable_score, e_step.score_samples, 0.0)
    contribution = jnp.sum(
        _multiply_by_log_scale_with_gradual_underflow(
            safe_score, normalized_log_weight
        )
    )
    numerical_failure = (
        jnp.any(active_failed_pairs)
        | jnp.any(positive & (~finite_score))
        | (~jnp.isfinite(contribution))
    )
    # The dense leaf's finite score may normalize over only the successful
    # components.  If a positive-weight row has any failed pair, that partial
    # score is an internal fallback rather than the contracted grouped
    # objective.  Propagate a nonfinite value just as the convenience scoring
    # leaves do; zero-weight failures remain excluded above.
    reported_contribution = jnp.where(
        numerical_failure,
        jnp.asarray(jnp.nan, dtype=contribution.dtype),
        contribution,
    )
    return reported_contribution, active_failed_pairs, numerical_failure


def _local_centered_statistics(
    e_step: EStep,
    sample_weight: Array,
    *,
    observed_dimension: int,
    log_informative_weight: Array,
) -> _LocalCenteredStatistics:
    """Reduce one group without converting subnormal component mass first."""

    n_components = e_step.responsibilities.shape[-1]
    latent_dimension = e_step.conditional_mean.shape[-1]
    dtype = e_step.conditional_mean.dtype
    objective, active_failed_pairs, objective_failure = (
        _normalized_objective_contribution(
            e_step,
            sample_weight,
            observed_dimension=observed_dimension,
            log_informative_weight=log_informative_weight,
        )
    )
    if observed_dimension == 0:
        return _LocalCenteredStatistics(
            log_mass=jnp.full((n_components,), -jnp.inf, dtype=dtype),
            component_mean=jnp.zeros(
                (n_components, latent_dimension), dtype=dtype
            ),
            centered_covariance=jnp.zeros(
                (n_components, latent_dimension, latent_dimension), dtype=dtype
            ),
            objective_contribution=objective,
            active_failed_pairs=active_failed_pairs,
            numerical_failure=objective_failure,
        )

    log_mass, normalized_log_weight, _ = _component_log_weight_reductions(
        e_step, sample_weight
    )
    active_pair = jnp.isfinite(normalized_log_weight)
    conditional_mean = jnp.where(
        active_pair[..., None], e_step.conditional_mean, 0.0
    )
    component_mean = jnp.sum(
        _multiply_by_log_scale_with_gradual_underflow(
            conditional_mean, normalized_log_weight[..., None]
        ),
        axis=0,
    )
    centered = jnp.where(
        active_pair[..., None],
        conditional_mean - component_mean[None, :, :],
        0.0,
    )
    conditional_covariance = jnp.where(
        active_pair[..., None, None], e_step.conditional_covariance, 0.0
    )
    weighted_covariance = _multiply_by_log_scale_with_gradual_underflow(
        conditional_covariance, normalized_log_weight[..., None, None]
    )
    half_weighted_centered = _multiply_by_log_scale_with_gradual_underflow(
        centered, 0.5 * normalized_log_weight[..., None]
    )
    centered_covariance = jnp.sum(
        weighted_covariance
        + half_weighted_centered[..., :, None]
        * half_weighted_centered[..., None, :],
        axis=0,
    )
    centered_covariance = _stable_symmetrize(centered_covariance)

    has_mass = jnp.isfinite(log_mass)
    local_moments_finite = (
        jnp.all(jnp.isfinite(component_mean), axis=-1)
        & jnp.all(jnp.isfinite(centered_covariance), axis=(-2, -1))
    )
    numerical_failure = (
        objective_failure
        | jnp.any(jnp.isposinf(log_mass))
        | jnp.any(has_mass & (~local_moments_finite))
    )
    return _LocalCenteredStatistics(
        log_mass=log_mass,
        component_mean=jnp.where(
            has_mass[:, None], component_mean, jnp.zeros_like(component_mean)
        ),
        centered_covariance=jnp.where(
            has_mass[:, None, None],
            centered_covariance,
            jnp.zeros_like(centered_covariance),
        ),
        objective_contribution=objective,
        active_failed_pairs=active_failed_pairs,
        numerical_failure=numerical_failure,
    )


def _merge_centered_statistics(
    old_log_mass: Array,
    old_mean: Array,
    old_covariance: Array,
    local: _LocalCenteredStatistics,
) -> tuple[Array, Array, Array]:
    """Merge normalized centered moments using the weighted Chan identity."""

    new_log_mass = jnp.logaddexp(old_log_mass, local.log_mass)
    new_has_mass = jnp.isfinite(new_log_mass)
    safe_new_log_mass = jnp.where(new_has_mass, new_log_mass, 0.0)
    old_log_fraction = jnp.where(
        jnp.isfinite(old_log_mass),
        old_log_mass - safe_new_log_mass,
        -jnp.inf,
    )
    local_log_fraction = jnp.where(
        jnp.isfinite(local.log_mass),
        local.log_mass - safe_new_log_mass,
        -jnp.inf,
    )

    # Anchor the mean at the larger-mass summary and add only the smaller
    # fraction times the signed difference.  A fixed old anchor loses a tiny
    # group through FTZ when it comes second, while a tiny old group followed
    # by a dominant local group can round the local fraction to exactly one and
    # cancel the retained tiny contribution.  The log-scaled smaller term is
    # stable in both lexicographic group orders.
    old_is_anchor = old_log_mass >= local.log_mass
    anchor_mean = jnp.where(
        old_is_anchor[:, None], old_mean, local.component_mean
    )
    other_mean = jnp.where(
        old_is_anchor[:, None], local.component_mean, old_mean
    )
    smaller_log_fraction = jnp.where(
        old_is_anchor, local_log_fraction, old_log_fraction
    )
    mean_delta = other_mean - anchor_mean
    weighted_mean_delta = jax.lax.optimization_barrier(
        _multiply_by_log_scale_with_gradual_underflow(
            mean_delta, smaller_log_fraction[:, None]
        )
    )
    new_mean = anchor_mean + weighted_mean_delta

    # Form sqrt(old_fraction * local_fraction) * delta before the outer
    # product.  This avoids overflowing ``delta @ delta.T`` when a far-away
    # local mean carries a very small component-mass fraction but its final
    # Chan correction remains representable.
    delta = local.component_mean - old_mean
    cross_log_scale = 0.5 * (old_log_fraction + local_log_fraction)
    half_weighted_delta = _multiply_by_log_scale_with_gradual_underflow(
        delta, cross_log_scale[:, None]
    )
    cross = (
        half_weighted_delta[:, :, None]
        * half_weighted_delta[:, None, :]
    )
    new_covariance = (
        _multiply_by_log_scale_with_gradual_underflow(
            old_covariance, old_log_fraction[:, None, None]
        )
        + _multiply_by_log_scale_with_gradual_underflow(
            local.centered_covariance,
            local_log_fraction[:, None, None],
        )
        + cross
    )
    new_covariance = _stable_symmetrize(new_covariance)
    return (
        new_log_mass,
        jnp.where(new_has_mass[:, None], new_mean, jnp.zeros_like(new_mean)),
        jnp.where(
            new_has_mass[:, None, None],
            new_covariance,
            jnp.zeros_like(new_covariance),
        ),
    )


def _statistics_from_leaves(
    fit: GroupedGeneralFitInputs,
    leaves: Sequence[EStep],
) -> GroupedGeneralSufficientStatistics:
    """Build stable global statistics from already evaluated group leaves."""

    grouped = fit.grouped
    dtype = grouped.parameters.means.dtype
    n_components, latent_dimension = grouped.parameters.means.shape
    informative_weight = jnp.asarray(fit.informative_weight, dtype=dtype)
    log_informative_weight = _log_positive_with_gradual_underflow(
        informative_weight
    )
    global_log_mass = jnp.full((n_components,), -jnp.inf, dtype=dtype)
    global_mean = jnp.zeros((n_components, latent_dimension), dtype=dtype)
    global_covariance = jnp.zeros(
        (n_components, latent_dimension, latent_dimension), dtype=dtype
    )
    objective = jnp.asarray(0.0, dtype=dtype)
    active_failed_pairs: list[Array] = []
    group_failure: list[Array] = []

    for group, leaf in zip(grouped.groups, leaves, strict=True):
        local = _local_centered_statistics(
            leaf,
            group.sample_weight,
            observed_dimension=group.observations.shape[-1],
            log_informative_weight=log_informative_weight,
        )
        global_log_mass, global_mean, global_covariance = (
            _merge_centered_statistics(
                global_log_mass,
                global_mean,
                global_covariance,
                local,
            )
        )
        objective = objective + local.objective_contribution
        active_failed_pairs.append(local.active_failed_pairs)
        group_failure.append(local.numerical_failure)

    mass = _exp_with_gradual_underflow(global_log_mass)
    first_moment = _multiply_by_log_scale_with_gradual_underflow(
        global_mean, global_log_mass[:, None]
    )
    weighted_centered_covariance = (
        _multiply_by_log_scale_with_gradual_underflow(
            global_covariance, global_log_mass[:, None, None]
        )
    )
    half_weighted_mean = _multiply_by_log_scale_with_gradual_underflow(
        global_mean, 0.5 * global_log_mass[:, None]
    )
    second_moment = (
        weighted_centered_covariance
        + half_weighted_mean[:, :, None] * half_weighted_mean[:, None, :]
    )
    weighted_log_likelihood = _multiply_by_log_scale_with_gradual_underflow(
        objective, log_informative_weight
    )
    restored_failed_pairs = _restore_rows(grouped, tuple(active_failed_pairs))
    group_failure_array = jnp.stack(tuple(group_failure))
    reductions_are_finite = (
        jnp.all(jnp.isfinite(mass))
        & jnp.all(jnp.isfinite(first_moment))
        & jnp.all(jnp.isfinite(second_moment))
        & jnp.all(jnp.isfinite(global_mean))
        & jnp.all(jnp.isfinite(global_covariance))
        & jnp.isfinite(weighted_log_likelihood)
        & jnp.isfinite(informative_weight)
        & _has_nonzero_floating_magnitude(informative_weight)
        & jnp.isfinite(objective)
    )
    numerical_failure = jnp.any(group_failure_array) | (~reductions_are_finite)
    return GroupedGeneralSufficientStatistics(
        mass=mass,
        first_moment=first_moment,
        second_moment=second_moment,
        log_mass=global_log_mass,
        component_mean=global_mean,
        centered_covariance=global_covariance,
        weighted_log_likelihood=weighted_log_likelihood,
        informative_weight=informative_weight,
        objective=objective,
        numerical_failure=numerical_failure,
        failed_pairs=restored_failed_pairs,
        group_numerical_failure=group_failure_array,
    )


def _current_posterior_and_statistics(
    fit: GroupedGeneralFitInputs,
) -> tuple[GroupedPosteriorResult, GroupedGeneralSufficientStatistics]:
    posterior, leaves = _posterior_group_leaves(
        fit.grouped,
        parameters=fit.grouped.parameters,
        factor_jitter=fit.controls.factor_jitter,
    )
    return posterior, _statistics_from_leaves(fit, leaves)


def sufficient_statistics_grouped(
    fit: GroupedGeneralFitInputs,
) -> GroupedGeneralSufficientStatistics:
    """Return one stable global reduction without any group-local M-step."""

    _, statistics = _current_posterior_and_statistics(fit)
    return statistics


def _candidate_parameters(
    old: Params,
    statistics: GroupedGeneralSufficientStatistics,
    *,
    covariance_ridge: Array,
) -> tuple[Params, Array, Array]:
    """Form and validate one global M-step candidate."""

    dtype = old.means.dtype
    latent_dimension = old.means.shape[1]
    mass_is_valid = (
        jnp.isfinite(statistics.log_mass)
        & _has_nonzero_floating_magnitude(statistics.mass)
    )
    finite_log_mass = jnp.where(
        mass_is_valid, statistics.log_mass, -jnp.inf
    )
    log_total_mass = jsp.special.logsumexp(finite_log_mass)
    safe_log_total_mass = jnp.where(jnp.isfinite(log_total_mass), log_total_mass, 0.0)
    candidate_weight = _exp_with_gradual_underflow(
        jnp.where(
            mass_is_valid,
            statistics.log_mass - safe_log_total_mass,
            -jnp.inf,
        )
    )
    identity = jnp.eye(latent_dimension, dtype=dtype)
    candidate_covariance = _stable_symmetrize(
        statistics.centered_covariance + covariance_ridge * identity
    )
    factor = jax.lax.linalg.cholesky(
        candidate_covariance, symmetrize_input=False
    )
    parameter_is_valid = (
        jnp.isfinite(candidate_weight)
        & (~jnp.signbit(candidate_weight))
        & _has_nonzero_floating_magnitude(candidate_weight)
        & jnp.all(jnp.isfinite(statistics.component_mean), axis=-1)
        & jnp.all(jnp.isfinite(candidate_covariance), axis=(-2, -1))
        & jnp.all(jnp.isfinite(factor), axis=(-2, -1))
        & jnp.all(jnp.diagonal(factor, axis1=-2, axis2=-1) > 0.0, axis=-1)
    )
    collapsed_components = (~mass_is_valid) | (~parameter_is_valid)
    candidate = Params(
        weights=candidate_weight,
        means=statistics.component_mean,
        covariances=candidate_covariance,
    )
    return candidate, jnp.any(collapsed_components), collapsed_components


def _objective_for_parameters(
    fit: GroupedGeneralFitInputs,
    parameters: Params,
) -> tuple[Array, Array, Array, Array]:
    """Evaluate only the grouped objective and active failure diagnostics."""

    grouped = fit.grouped
    _, leaves = _posterior_group_leaves(
        grouped,
        parameters=parameters,
        factor_jitter=fit.controls.factor_jitter,
    )
    dtype = parameters.means.dtype
    log_informative_weight = _log_positive_with_gradual_underflow(
        jnp.asarray(fit.informative_weight, dtype=dtype)
    )
    objective = jnp.asarray(0.0, dtype=dtype)
    failed_pairs: list[Array] = []
    group_failure: list[Array] = []
    for group, leaf in zip(grouped.groups, leaves, strict=True):
        contribution, active_failure, numerical_failure = (
            _normalized_objective_contribution(
                leaf,
                group.sample_weight,
                observed_dimension=group.observations.shape[-1],
                log_informative_weight=log_informative_weight,
            )
        )
        objective = objective + contribution
        failed_pairs.append(active_failure)
        group_failure.append(numerical_failure)
    group_failure_array = jnp.stack(tuple(group_failure))
    restored_failed_pairs = _restore_rows(grouped, tuple(failed_pairs))
    objective_is_valid = (~jnp.any(group_failure_array)) & jnp.isfinite(objective)
    return (
        objective,
        objective_is_valid,
        group_failure_array,
        restored_failed_pairs,
    )


def _scalar_enum(value: IntEnum) -> Array:
    return jnp.asarray(int(value), dtype=jnp.int32)


def one_em_step_grouped(
    fit: GroupedGeneralFitInputs,
) -> GroupedGeneralEMStepResult:
    """Run one global grouped XD update with candidate-objective validation.

    The eager validation boundary has already established finite positive
    informative weight and valid scalar controls.  Current-statistic failure,
    global component collapse, or second-pass candidate failure returns the
    exact input parameters; no partial group update is exposed.
    """

    old = fit.grouped.parameters
    current_posterior, statistics = _current_posterior_and_statistics(fit)
    dtype = old.means.dtype
    n_samples = fit.grouped.n_samples
    n_components = old.weights.shape[0]
    zero_group_failure = jnp.zeros((len(fit.grouped.groups),), dtype=bool)
    zero_failed_pairs = jnp.zeros((n_samples, n_components), dtype=bool)
    zero_collapsed = jnp.zeros((n_components,), dtype=bool)
    invalid_attempt = jnp.asarray(jnp.nan, dtype=dtype)

    if bool(jax.device_get(statistics.numerical_failure)):
        return GroupedGeneralEMStepResult(
            parameters=old,
            e_step=current_posterior,
            statistics=statistics,
            objective=statistics.objective,
            previous_objective=statistics.objective,
            attempted_objective=invalid_attempt,
            attempted_objective_valid=jnp.asarray(False),
            status=_scalar_enum(GroupedStepStatus.NUMERICAL_FAILURE),
            failure_stage=_scalar_enum(GroupedFailureStage.CURRENT_STATISTICS),
            collapsed=jnp.asarray(False),
            collapsed_components=zero_collapsed,
            numerical_failure=jnp.asarray(True),
            candidate_group_numerical_failure=zero_group_failure,
            candidate_failed_pairs=zero_failed_pairs,
        )

    candidate, collapsed, collapsed_components = _candidate_parameters(
        old,
        statistics,
        covariance_ridge=fit.controls.covariance_ridge,
    )
    if bool(jax.device_get(collapsed)):
        return GroupedGeneralEMStepResult(
            parameters=old,
            e_step=current_posterior,
            statistics=statistics,
            objective=statistics.objective,
            previous_objective=statistics.objective,
            attempted_objective=invalid_attempt,
            attempted_objective_valid=jnp.asarray(False),
            status=_scalar_enum(GroupedStepStatus.COMPONENT_COLLAPSED),
            failure_stage=_scalar_enum(GroupedFailureStage.M_STEP),
            collapsed=collapsed,
            collapsed_components=collapsed_components,
            numerical_failure=jnp.asarray(False),
            candidate_group_numerical_failure=zero_group_failure,
            candidate_failed_pairs=zero_failed_pairs,
        )

    (
        attempted_objective,
        attempted_objective_valid,
        candidate_group_failure,
        candidate_failed_pairs,
    ) = _objective_for_parameters(fit, candidate)
    if not bool(jax.device_get(attempted_objective_valid)):
        return GroupedGeneralEMStepResult(
            parameters=old,
            e_step=current_posterior,
            statistics=statistics,
            objective=statistics.objective,
            previous_objective=statistics.objective,
            # A partially normalized internal score from the successful pairs
            # is not the candidate's contracted objective.  Keep the detailed
            # pair/group masks, but expose a nonfinite attempted objective so
            # it cannot be mistaken for a valid likelihood endpoint.
            attempted_objective=invalid_attempt,
            attempted_objective_valid=attempted_objective_valid,
            status=_scalar_enum(GroupedStepStatus.NUMERICAL_FAILURE),
            failure_stage=_scalar_enum(GroupedFailureStage.CANDIDATE_OBJECTIVE),
            collapsed=jnp.asarray(False),
            collapsed_components=zero_collapsed,
            numerical_failure=jnp.asarray(True),
            candidate_group_numerical_failure=candidate_group_failure,
            candidate_failed_pairs=candidate_failed_pairs,
        )

    return GroupedGeneralEMStepResult(
        parameters=candidate,
        e_step=current_posterior,
        statistics=statistics,
        objective=attempted_objective,
        previous_objective=statistics.objective,
        attempted_objective=attempted_objective,
        attempted_objective_valid=attempted_objective_valid,
        status=_scalar_enum(GroupedStepStatus.SUCCESS),
        failure_stage=_scalar_enum(GroupedFailureStage.NONE),
        collapsed=jnp.asarray(False),
        collapsed_components=zero_collapsed,
        numerical_failure=jnp.asarray(False),
        candidate_group_numerical_failure=candidate_group_failure,
        candidate_failed_pairs=candidate_failed_pairs,
    )


__all__ = [
    "GroupedFailureStage",
    "GroupedGeneralEMStepResult",
    "GroupedGeneralSufficientStatistics",
    "GroupedPosteriorResult",
    "GroupedStepStatus",
    "one_em_step_grouped",
    "posterior_components_grouped",
    "sufficient_statistics_grouped",
]
