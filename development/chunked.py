"""Temporary bounded-memory kernels for identity-projection XD.

This module is deliberately outside the future public package namespace.  It
implements one identity-XD EM update by scanning over statically sized chunks
and returning only component-level reductions.  Callers must pass canonical,
already-validated fitting arrays; ``chunk_size`` is a static compilation
parameter.

The covariance statistic is accumulated in centered form.  Each chunk first
forms a weighted mean and covariance numerator, then merges those moments with
the running state using the parallel (Chan) weighted-variance identity.  No raw
second-moment subtraction is used, and no ``N x K x D x D`` posterior array is
retained.  The original inputs are scan constants, and each iteration safely
gathers only ``chunk_size`` rows.  This avoids a separate globally padded input
copy while bounding posterior workspace by the chunk size.
"""

from __future__ import annotations

import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .identity_xd import (
    Params,
    _canonical_inputs,
    _real_scalar_control,
    posterior_components,
)


Array = jax.Array


class ChunkedSufficientStatistics(NamedTuple):
    """Reduced centered statistics accumulated across all actual rows.

    ``mass`` has shape ``(K,)``, ``means`` has shape ``(K,D)``, and
    ``centered_covariance_numerator`` has shape ``(K,D,D)``.  The last field is
    the weighted sum of conditional covariance plus squared displacement from
    ``means``; dividing it by ``mass`` gives the exact, unregularized M-step
    covariance up to chunk-order rounding.
    """

    mass: Array
    means: Array
    centered_covariance_numerator: Array


class ChunkedEMStepResult(NamedTuple):
    """One bounded-memory EM update and its device-resident diagnostics.

    On numerical failure or component collapse, ``parameters`` is the exact
    canonical input state.  The result intentionally contains no per-row or
    per-row/component posterior arrays.
    """

    parameters: Params
    statistics: ChunkedSufficientStatistics
    log_likelihood: Array
    objective: Array
    actual_count: Array
    padded_count: Array
    collapsed: Array
    collapsed_components: Array
    numerical_failure: Array


class _Accumulator(NamedTuple):
    """Scan carry containing only scalar and component-level reductions."""

    mass: Array
    means: Array
    centered_covariance_numerator: Array
    log_likelihood: Array
    numerical_failure: Array


def _positive_chunk_size(value: object) -> int:
    """Return a positive static index integer for padding and scan shapes."""

    value_dtype = getattr(value, "dtype", None)
    if isinstance(value, bool) or (
        value_dtype is not None
        and jnp.issubdtype(value_dtype, jnp.bool_)
    ):
        raise TypeError("chunk_size must be a positive integer, not bool")
    try:
        chunk_size = operator.index(value)
    except TypeError as error:
        raise TypeError("chunk_size must be a positive static integer") from error
    if chunk_size < 1:
        raise ValueError(
            f"chunk_size must be a positive integer; received {chunk_size}"
        )
    return int(chunk_size)


def _merge_centered_moments(
    accumulator: _Accumulator,
    chunk_mass: Array,
    chunk_means: Array,
    chunk_numerator: Array,
    chunk_log_likelihood: Array,
    chunk_numerical_failure: Array,
) -> _Accumulator:
    """Merge weighted centered moments without raw-second-moment subtraction."""

    combined_mass = accumulator.mass + chunk_mass
    safe_combined_mass = jnp.where(combined_mass > 0.0, combined_mass, 1.0)
    delta = chunk_means - accumulator.means
    chunk_fraction = chunk_mass / safe_combined_mass
    combined_means = accumulator.means + chunk_fraction[:, None] * delta

    # Algebraically this is ``mass_a * mass_b / combined_mass``. Forming the
    # product first can underflow even when the final cross weight is
    # representable. Multiplying the smaller mass by the larger mass's fraction
    # keeps the intermediate in range. The square-root form likewise avoids
    # forming ``delta**2`` before applying a potentially tiny cross weight.
    smaller_mass = jnp.minimum(accumulator.mass, chunk_mass)
    larger_mass = jnp.maximum(accumulator.mass, chunk_mass)
    cross_weight = smaller_mass * (larger_mass / safe_combined_mass)
    positive_cross_weight = cross_weight > 0.0
    safe_delta = jnp.where(positive_cross_weight[:, None], delta, 0.0)
    scaled_delta = (
        jnp.sqrt(jnp.where(positive_cross_weight, cross_weight, 0.0))[:, None]
        * safe_delta
    )
    cross_numerator = (
        scaled_delta[:, :, None] * scaled_delta[:, None, :]
    )
    combined_numerator = (
        accumulator.centered_covariance_numerator
        + chunk_numerator
        + cross_numerator
    )
    combined_log_likelihood = (
        accumulator.log_likelihood + chunk_log_likelihood
    )
    return _Accumulator(
        mass=combined_mass,
        means=combined_means,
        centered_covariance_numerator=combined_numerator,
        log_likelihood=combined_log_likelihood,
        numerical_failure=(
            accumulator.numerical_failure
            | chunk_numerical_failure
            | (~jnp.isfinite(chunk_log_likelihood))
            | (~jnp.isfinite(combined_log_likelihood))
        ),
    )


def chunked_em_step(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    chunk_size: int,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> ChunkedEMStepResult:
    """Run one exact identity-XD update with bounded posterior storage.

    ``chunk_size`` is required, keyword-only, positive, and static.  A JIT
    caller must close over it or mark it as a static argument.  Logical padding
    changes only the final chunk's validity mask and the reported
    ``padded_count``: masked slots contribute neither likelihood nor sufficient
    statistics, and no full padded input array is constructed.

    The largest posterior covariance array has shape ``(chunk_size,K,D,D)``.
    ``lax.scan`` returns no stacked per-chunk outputs, so only its reduced carry
    survives the scan.  Original ``N``-row inputs remain resident, and each scan
    iteration materializes ``chunk_size``-row observation and covariance gathers.
    """

    static_chunk_size = _positive_chunk_size(chunk_size)
    (
        canonical_params,
        x,
        noise,
        _,
        n_components,
        dimension,
    ) = _canonical_inputs(params, observations, measurement_covariances)
    if x.ndim != 2:
        raise ValueError(
            "chunked_em_step observations must have fitting shape (N, D); "
            f"received {x.shape}"
        )
    n_samples = x.shape[0]
    if n_samples < 1:
        raise ValueError(
            "chunked_em_step observations must have fitting shape (N, D) "
            "with N >= 1"
        )

    dtype = canonical_params.means.dtype
    ridge, ridge_is_valid = _real_scalar_control(
        covariance_ridge,
        dtype=dtype,
        name="covariance_ridge",
    )
    safe_ridge = jnp.where(ridge_is_valid, ridge, 0.0)

    n_chunks = (n_samples + static_chunk_size - 1) // static_chunk_size
    padded_count_value = n_chunks * static_chunk_size
    chunk_offsets = jnp.arange(static_chunk_size)
    chunk_indices = jnp.arange(n_chunks)

    initial_accumulator = _Accumulator(
        mass=jnp.zeros((n_components,), dtype=dtype),
        # Starting from the old component means keeps the first Chan delta
        # small for high-offset data.  With zero accumulated mass, the first
        # positive-mass chunk still replaces this anchor exactly algebraically.
        means=canonical_params.means,
        centered_covariance_numerator=jnp.zeros(
            (n_components, dimension, dimension), dtype=dtype
        ),
        log_likelihood=jnp.asarray(0.0, dtype=dtype),
        numerical_failure=~ridge_is_valid,
    )

    def scan_chunk(
        accumulator: _Accumulator,
        chunk_index: Array,
    ) -> tuple[_Accumulator, None]:
        row_indices = chunk_index * static_chunk_size + chunk_offsets
        chunk_is_valid = row_indices < n_samples
        # Every gather index must be in range because accelerator out-of-bounds
        # semantics are not a padding policy.  Invalid final-chunk slots reuse
        # the last real row only as a safe numerical placeholder; the masks
        # below remove them from likelihood, failure status, and statistics.
        safe_row_indices = jnp.minimum(row_indices, n_samples - 1)
        chunk_observations = x[safe_row_indices]
        chunk_covariances = noise[safe_row_indices]
        e_step = posterior_components(
            canonical_params,
            chunk_observations,
            chunk_covariances,
            factor_jitter=factor_jitter,
        )

        valid_pair = chunk_is_valid[:, None]
        masked_responsibilities = jnp.where(
            valid_pair, e_step.responsibilities, 0.0
        )
        # Replace padding leaves before products are formed.  Merely
        # multiplying padding by zero could turn an overflowed dummy posterior
        # into ``0 * inf == nan``.
        safe_conditional_mean = jnp.where(
            valid_pair[..., None],
            e_step.conditional_mean,
            canonical_params.means[None, :, :],
        )
        safe_conditional_covariance = jnp.where(
            valid_pair[..., None, None],
            e_step.conditional_covariance,
            0.0,
        )

        chunk_mass = jnp.sum(masked_responsibilities, axis=0)
        safe_chunk_mass = jnp.where(chunk_mass > 0.0, chunk_mass, 1.0)
        anchored_displacement = (
            safe_conditional_mean - canonical_params.means[None, :, :]
        )
        displacement_sum = jnp.sum(
            masked_responsibilities[..., None] * anchored_displacement,
            axis=0,
        )
        chunk_mean_displacement = (
            displacement_sum / safe_chunk_mass[:, None]
        )
        chunk_means = (
            canonical_params.means
            + chunk_mean_displacement
        )

        # Center in displacement coordinates instead of subtracting two large
        # absolute locations. Mask zero-weight pairs before scaling so padded
        # rows cannot square a huge displacement and then form ``0 * inf``.
        centered_displacement = (
            anchored_displacement - chunk_mean_displacement[None, :, :]
        )
        positive_responsibility = masked_responsibilities > 0.0
        safe_centered_displacement = jnp.where(
            positive_responsibility[..., None], centered_displacement, 0.0
        )
        weighted_centered_displacement = (
            jnp.sqrt(masked_responsibilities)[..., None]
            * safe_centered_displacement
        )
        centered_second_moment = (
            masked_responsibilities[..., None, None]
            * safe_conditional_covariance
            + weighted_centered_displacement[..., :, None]
            * weighted_centered_displacement[..., None, :]
        )
        chunk_numerator = jnp.sum(centered_second_moment, axis=0)
        chunk_log_likelihood = jnp.sum(
            jnp.where(chunk_is_valid, e_step.score_samples, 0.0)
        )
        chunk_numerical_failure = jnp.any(
            e_step.failed_pairs & valid_pair
        ) | (~jnp.isfinite(chunk_log_likelihood))
        next_accumulator = _merge_centered_moments(
            accumulator,
            chunk_mass,
            chunk_means,
            chunk_numerator,
            chunk_log_likelihood,
            chunk_numerical_failure,
        )
        return next_accumulator, None

    final_accumulator, _ = jax.lax.scan(
        scan_chunk,
        initial_accumulator,
        chunk_indices,
    )

    mass_is_valid = jnp.isfinite(final_accumulator.mass) & (
        final_accumulator.mass > 0.0
    )
    safe_mass = jnp.where(mass_is_valid, final_accumulator.mass, 1.0)
    finite_mass = jnp.where(mass_is_valid, final_accumulator.mass, 0.0)
    total_mass = jnp.sum(finite_mass)
    safe_total_mass = jnp.where(
        jnp.isfinite(total_mass) & (total_mass > 0.0), total_mass, 1.0
    )

    candidate_weights = finite_mass / safe_total_mass
    candidate_means = final_accumulator.means
    candidate_covariances = (
        final_accumulator.centered_covariance_numerator
        / safe_mass[:, None, None]
    )
    identity = jnp.eye(dimension, dtype=dtype)
    candidate_covariances = candidate_covariances + safe_ridge * identity
    candidate_covariances = 0.5 * (
        candidate_covariances
        + jnp.swapaxes(candidate_covariances, -1, -2)
    )

    candidate_factors = jax.lax.linalg.cholesky(
        candidate_covariances, symmetrize_input=False
    )
    finite_parameters = (
        jnp.isfinite(candidate_weights)
        & (candidate_weights > 0.0)
        & jnp.all(jnp.isfinite(candidate_means), axis=-1)
        & jnp.all(jnp.isfinite(candidate_covariances), axis=(-2, -1))
        & jnp.all(jnp.isfinite(candidate_factors), axis=(-2, -1))
        & jnp.all(
            jnp.diagonal(candidate_factors, axis1=-2, axis2=-1) > 0.0,
            axis=-1,
        )
    )
    numerical_failure = final_accumulator.numerical_failure
    proposed_collapsed_components = (~mass_is_valid) | (~finite_parameters)
    collapsed_components = jnp.where(
        numerical_failure,
        jnp.zeros_like(proposed_collapsed_components),
        proposed_collapsed_components,
    )
    collapsed = jnp.any(collapsed_components)
    rollback = numerical_failure | collapsed

    candidate_params = Params(
        candidate_weights, candidate_means, candidate_covariances
    )
    returned_params = Params(
        jnp.where(
            rollback, canonical_params.weights, candidate_params.weights
        ),
        jnp.where(rollback, canonical_params.means, candidate_params.means),
        jnp.where(
            rollback,
            canonical_params.covariances,
            candidate_params.covariances,
        ),
    )
    actual_count = jnp.asarray(n_samples, dtype=jnp.int32)
    padded_count = jnp.asarray(padded_count_value, dtype=jnp.int32)
    log_likelihood = final_accumulator.log_likelihood
    objective = log_likelihood / actual_count
    statistics = ChunkedSufficientStatistics(
        mass=final_accumulator.mass,
        means=final_accumulator.means,
        centered_covariance_numerator=(
            final_accumulator.centered_covariance_numerator
        ),
    )
    return ChunkedEMStepResult(
        parameters=returned_params,
        statistics=statistics,
        log_likelihood=log_likelihood,
        objective=objective,
        actual_count=actual_count,
        padded_count=padded_count,
        collapsed=collapsed,
        collapsed_components=collapsed_components,
        numerical_failure=numerical_failure,
    )


__all__ = [
    "ChunkedEMStepResult",
    "ChunkedSufficientStatistics",
    "chunked_em_step",
]
