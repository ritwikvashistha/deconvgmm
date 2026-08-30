"""Temporary host fit control for grouped general-projection XD.

The fixed-observed-dimension numerical leaves remain JAX compatible, but mask
grouping and the global variable-``M`` update are eager Python orchestration.
Consequently both controllers in this module are host loops.  There is
deliberately no compiled whole-group fitting entry point.

This remains development code, not a public package API.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .fit_control import (
    FitMode,
    FitStatus,
    _invalid_objective,
    _nonnegative_count,
    _sanitize_attempted_objective,
    classify_objective_change,
)
from .general_grouped import (
    GroupedFailureStage,
    GroupedStepStatus,
    _objective_for_parameters,
    one_em_step_grouped,
)
from .general_validation import GroupedGeneralFitInputs
from .identity_xd import Params
from .metadata import (
    GENERAL_CONTRACT_ID,
    GENERAL_CONTRACT_VERSION,
    GeneralResultMetadata,
    InitializationProvenance,
    current_general_result_metadata,
    user_supplied_initialization,
)
from .validation import validate_convergence_controls


Array = jax.Array


class GroupedGeneralFitResult(NamedTuple):
    """Host result shared by fixed-step and dynamically converged fitting.

    ``history`` contains accepted objectives only.  The group/pair diagnostics
    describe the terminating attempt (or the invalid current state when no
    attempt was possible), while ``parameters`` and ``objective`` always refer
    to the exact last accepted state.
    """

    parameters: Params
    initial_parameters: Params
    objective: Array
    objective_valid: Array
    history: Array
    n_iter: Array
    iteration_limit: Array
    converged: Array
    status: Array
    mode: Array
    attempted_iteration: Array
    attempted_objective: Array
    attempted_objective_valid: Array
    numerical_failure: Array
    collapsed: Array
    collapsed_components: Array
    failure_stage: Array
    group_numerical_failure: Array
    failed_pairs: Array
    informative_weight: Array
    factor_jitter: Array
    covariance_ridge: Array
    tol: Array | None
    decrease_tol: Array | None
    initialization: InitializationProvenance
    metadata: GeneralResultMetadata


def _status(value: FitStatus | FitMode) -> Array:
    return jnp.asarray(int(value), dtype=jnp.int32)


def _zero_diagnostics(
    fit: GroupedGeneralFitInputs,
) -> tuple[Array, Array, Array]:
    """Return empty component, group, and restored pair diagnostics."""

    n_components = fit.grouped.parameters.weights.shape[0]
    return (
        jnp.zeros((n_components,), dtype=bool),
        jnp.zeros((len(fit.grouped.groups),), dtype=bool),
        jnp.zeros(
            (fit.grouped.n_samples, n_components),
            dtype=bool,
        ),
    )


def _with_parameters(
    fit: GroupedGeneralFitInputs,
    parameters: Params,
) -> GroupedGeneralFitInputs:
    """Replace only the canonical mixture state for the next global step."""

    return replace(
        fit,
        grouped=replace(fit.grouped, parameters=parameters),
    )


def _failure_diagnostics(step: object) -> tuple[Array, Array]:
    """Select the masks belonging to the stage that terminated one step."""

    failure_stage = GroupedFailureStage(
        int(jax.device_get(step.failure_stage))
    )
    if failure_stage == GroupedFailureStage.CURRENT_STATISTICS:
        return (
            step.statistics.group_numerical_failure,
            step.statistics.failed_pairs,
        )
    return (
        step.candidate_group_numerical_failure,
        step.candidate_failed_pairs,
    )


def _initial_state(
    fit: GroupedGeneralFitInputs,
    *,
    mode: FitMode,
) -> tuple[
    Array,
    list[Array],
    FitStatus,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """Evaluate the initial score without requiring update statistics.

    A finite current objective is a valid accepted state even when the raw
    moment reductions required by a later M-step would overflow.  Those
    reductions are therefore deferred until an update is actually attempted.
    """

    (
        objective,
        objective_is_valid,
        group_numerical_failure,
        failed_pairs,
    ) = _objective_for_parameters(fit, fit.grouped.parameters)
    initial_failure = not bool(jax.device_get(objective_is_valid)) or not bool(
        jax.device_get(jnp.isfinite(objective))
    )
    collapsed_components, zero_group_failure, zero_failed_pairs = (
        _zero_diagnostics(fit)
    )
    if initial_failure:
        objective = _invalid_objective(
            dtype=fit.grouped.parameters.means.dtype
        )
        return (
            objective,
            [],
            FitStatus.NUMERICAL_FAILURE,
            _status(GroupedFailureStage.CURRENT_STATISTICS),
            jnp.asarray(True),
            group_numerical_failure,
            failed_pairs,
            collapsed_components,
            jnp.asarray(False),
        )
    initial_status = (
        FitStatus.FIXED_STEPS_COMPLETE
        if mode == FitMode.FIXED_STEPS
        else FitStatus.MAX_ITER
    )
    return (
        objective,
        [objective],
        initial_status,
        _status(GroupedFailureStage.NONE),
        jnp.asarray(False),
        zero_group_failure,
        zero_failed_pairs,
        collapsed_components,
        jnp.asarray(False),
    )


def _history_array(history: list[Array], *, dtype: jnp.dtype) -> Array:
    """Stack accepted objectives, preserving an explicit empty failure case."""

    if history:
        return jnp.stack(history)
    return jnp.empty((0,), dtype=dtype)


def _result(
    fit: GroupedGeneralFitInputs,
    *,
    parameters: Params,
    objective: Array,
    history: list[Array],
    n_iter: int,
    iteration_limit: int,
    converged: bool,
    status: FitStatus,
    mode: FitMode,
    attempted_iteration: int,
    attempted_objective: Array,
    attempted_objective_valid: bool | Array,
    numerical_failure: bool | Array,
    collapsed: bool | Array,
    collapsed_components: Array,
    failure_stage: Array,
    group_numerical_failure: Array,
    failed_pairs: Array,
    tol: Array | None,
    decrease_tol: Array | None,
) -> GroupedGeneralFitResult:
    """Build one coherent host result without changing numerical state."""

    dtype = parameters.means.dtype
    history_array = _history_array(history, dtype=dtype)
    objective_valid = bool(history) and len(history) == n_iter + 1
    attempted_objective, attempted_objective_valid = (
        _sanitize_attempted_objective(
            attempted_objective,
            attempted_objective_valid,
            dtype=dtype,
        )
    )
    return GroupedGeneralFitResult(
        parameters=parameters,
        initial_parameters=fit.grouped.parameters,
        objective=objective,
        objective_valid=jnp.asarray(objective_valid),
        history=history_array,
        n_iter=jnp.asarray(n_iter, dtype=jnp.int32),
        iteration_limit=jnp.asarray(iteration_limit, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        status=_status(status),
        mode=_status(mode),
        attempted_iteration=jnp.asarray(
            attempted_iteration, dtype=jnp.int32
        ),
        attempted_objective=attempted_objective,
        attempted_objective_valid=jnp.asarray(attempted_objective_valid),
        numerical_failure=jnp.asarray(numerical_failure),
        collapsed=jnp.asarray(collapsed),
        collapsed_components=collapsed_components,
        failure_stage=failure_stage,
        group_numerical_failure=group_numerical_failure,
        failed_pairs=failed_pairs,
        informative_weight=fit.informative_weight,
        factor_jitter=fit.controls.factor_jitter,
        covariance_ridge=fit.controls.covariance_ridge,
        tol=tol,
        decrease_tol=decrease_tol,
        initialization=user_supplied_initialization(),
        metadata=current_general_result_metadata(),
    )


def fit_fixed_steps_grouped(
    fit: GroupedGeneralFitInputs,
    *,
    n_steps: int,
) -> GroupedGeneralFitResult:
    """Run exactly ``n_steps`` eager global updates unless one fails.

    Finite objective decreases remain accepted in this mode.  A numerical
    failure or component collapse terminates the logical trajectory and returns
    the exact last accepted state.
    """

    step_count = _nonnegative_count(n_steps, name="n_steps")
    (
        initial_objective,
        history,
        status,
        failure_stage,
        numerical_failure,
        group_numerical_failure,
        failed_pairs,
        collapsed_components,
        collapsed,
    ) = _initial_state(fit, mode=FitMode.FIXED_STEPS)
    current_parameters = fit.grouped.parameters
    current_objective = initial_objective
    attempted_iteration = 0
    attempted_objective = current_objective
    attempted_objective_valid = False
    n_iter = 0

    if bool(jax.device_get(numerical_failure)):
        return _result(
            fit,
            parameters=current_parameters,
            objective=current_objective,
            history=history,
            n_iter=n_iter,
            iteration_limit=step_count,
            converged=False,
            status=status,
            mode=FitMode.FIXED_STEPS,
            attempted_iteration=attempted_iteration,
            attempted_objective=attempted_objective,
            attempted_objective_valid=attempted_objective_valid,
            numerical_failure=numerical_failure,
            collapsed=collapsed,
            collapsed_components=collapsed_components,
            failure_stage=failure_stage,
            group_numerical_failure=group_numerical_failure,
            failed_pairs=failed_pairs,
            tol=None,
            decrease_tol=None,
        )

    for attempted_iteration in range(1, step_count + 1):
        step = one_em_step_grouped(_with_parameters(fit, current_parameters))
        attempted_objective = step.attempted_objective
        attempted_objective_valid = step.attempted_objective_valid
        step_status = GroupedStepStatus(int(jax.device_get(step.status)))
        if step_status != GroupedStepStatus.SUCCESS:
            group_numerical_failure, failed_pairs = _failure_diagnostics(step)
            failure_stage = step.failure_stage
            collapsed_components = step.collapsed_components
            numerical_failure = step.numerical_failure
            collapsed = step.collapsed
            status = (
                FitStatus.NUMERICAL_FAILURE
                if step_status == GroupedStepStatus.NUMERICAL_FAILURE
                else FitStatus.COMPONENT_COLLAPSED
            )
            break

        current_parameters = step.parameters
        current_objective = step.objective
        history.append(current_objective)
        n_iter += 1
        status = FitStatus.FIXED_STEPS_COMPLETE
        failure_stage = step.failure_stage
        group_numerical_failure = step.candidate_group_numerical_failure
        failed_pairs = step.candidate_failed_pairs
    return _result(
        fit,
        parameters=current_parameters,
        objective=current_objective,
        history=history,
        n_iter=n_iter,
        iteration_limit=step_count,
        converged=False,
        status=status,
        mode=FitMode.FIXED_STEPS,
        attempted_iteration=attempted_iteration,
        attempted_objective=attempted_objective,
        attempted_objective_valid=attempted_objective_valid,
        numerical_failure=numerical_failure,
        collapsed=collapsed,
        collapsed_components=collapsed_components,
        failure_stage=failure_stage,
        group_numerical_failure=group_numerical_failure,
        failed_pairs=failed_pairs,
        tol=None,
        decrease_tol=None,
    )


def fit_converged_grouped(
    fit: GroupedGeneralFitInputs,
    *,
    max_iter: int = 100,
    tol: object = 1e-6,
    decrease_tol: object = 1e-10,
) -> GroupedGeneralFitResult:
    """Run host-controlled global updates until convergence or termination.

    Only accepted objectives enter ``history``.  A material finite decrease is
    recorded as an attempted objective but rolls back the complete grouped
    state without incrementing ``n_iter``.
    """

    iteration_limit = _nonnegative_count(max_iter, name="max_iter")
    dtype = fit.grouped.parameters.means.dtype
    convergence_controls = validate_convergence_controls(
        tol=tol,
        decrease_tol=decrease_tol,
        dtype=dtype,
    )
    tolerance = convergence_controls.tol
    decrease_tolerance = convergence_controls.decrease_tol
    (
        initial_objective,
        history,
        status,
        failure_stage,
        numerical_failure,
        group_numerical_failure,
        failed_pairs,
        collapsed_components,
        collapsed,
    ) = _initial_state(fit, mode=FitMode.CONVERGED)
    current_parameters = fit.grouped.parameters
    current_objective = initial_objective
    attempted_iteration = 0
    attempted_objective = current_objective
    attempted_objective_valid = False
    n_iter = 0
    converged = False

    if bool(jax.device_get(numerical_failure)):
        return _result(
            fit,
            parameters=current_parameters,
            objective=current_objective,
            history=history,
            n_iter=n_iter,
            iteration_limit=iteration_limit,
            converged=converged,
            status=status,
            mode=FitMode.CONVERGED,
            attempted_iteration=attempted_iteration,
            attempted_objective=attempted_objective,
            attempted_objective_valid=attempted_objective_valid,
            numerical_failure=numerical_failure,
            collapsed=collapsed,
            collapsed_components=collapsed_components,
            failure_stage=failure_stage,
            group_numerical_failure=group_numerical_failure,
            failed_pairs=failed_pairs,
            tol=tolerance,
            decrease_tol=decrease_tolerance,
        )

    for attempted_iteration in range(1, iteration_limit + 1):
        step = one_em_step_grouped(_with_parameters(fit, current_parameters))
        attempted_objective = step.attempted_objective
        attempted_objective_valid = step.attempted_objective_valid
        step_status = GroupedStepStatus(int(jax.device_get(step.status)))
        if step_status != GroupedStepStatus.SUCCESS:
            group_numerical_failure, failed_pairs = _failure_diagnostics(step)
            failure_stage = step.failure_stage
            collapsed_components = step.collapsed_components
            numerical_failure = step.numerical_failure
            collapsed = step.collapsed
            status = (
                FitStatus.NUMERICAL_FAILURE
                if step_status == GroupedStepStatus.NUMERICAL_FAILURE
                else FitStatus.COMPONENT_COLLAPSED
            )
            break

        decision = classify_objective_change(
            previous=current_objective,
            current=step.objective,
            tol=tolerance,
            decrease_tol=decrease_tolerance,
        )
        decision_status = FitStatus(int(jax.device_get(decision.status)))
        failure_stage = step.failure_stage
        group_numerical_failure = step.candidate_group_numerical_failure
        failed_pairs = step.candidate_failed_pairs
        if not bool(jax.device_get(decision.accept)):
            status = decision_status
            break

        current_parameters = step.parameters
        current_objective = step.objective
        history.append(current_objective)
        n_iter += 1
        status = decision_status
        if bool(jax.device_get(decision.converged)):
            converged = True
            break
    else:
        status = FitStatus.MAX_ITER

    return _result(
        fit,
        parameters=current_parameters,
        objective=current_objective,
        history=history,
        n_iter=n_iter,
        iteration_limit=iteration_limit,
        converged=converged,
        status=status,
        mode=FitMode.CONVERGED,
        attempted_iteration=attempted_iteration,
        attempted_objective=attempted_objective,
        attempted_objective_valid=attempted_objective_valid,
        numerical_failure=numerical_failure,
        collapsed=collapsed,
        collapsed_components=collapsed_components,
        failure_stage=failure_stage,
        group_numerical_failure=group_numerical_failure,
        failed_pairs=failed_pairs,
        tol=tolerance,
        decrease_tol=decrease_tolerance,
    )


__all__ = [
    "GENERAL_CONTRACT_ID",
    "GENERAL_CONTRACT_VERSION",
    "FitMode",
    "FitStatus",
    "GeneralResultMetadata",
    "GroupedGeneralFitResult",
    "current_general_result_metadata",
    "fit_converged_grouped",
    "fit_fixed_steps_grouped",
]
