"""Temporary fit-control semantics for the identity-projection XD kernel.

This module is not a public API. ``fit_converged`` and the host-facing
``fit_fixed_steps`` wrapper are intentionally host controlled.
``fit_fixed_steps_kernel`` uses JAX control flow and is JIT-compatible when
``n_steps`` is marked static by the caller.
"""

from __future__ import annotations

import operator
from enum import IntEnum
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .identity_xd import (
    Params,
    _real_scalar_control,
    em_step,
    posterior_components,
)
from .metadata import (
    InitializationProvenance,
    ResultMetadata,
    current_result_metadata,
    user_supplied_initialization,
)
from .validation import (
    PreparedControls,
    _ControlValueError,
    validate_convergence_controls,
    validate_controls,
)


Array = jax.Array


class FitStatus(IntEnum):
    """Integer fit states that can be carried through JAX control flow."""

    CONTINUE = 0
    CONVERGED = 1
    MAX_ITER = 2
    OBJECTIVE_DECREASED = 3
    NUMERICAL_FAILURE = 4
    COMPONENT_COLLAPSED = 5
    FIXED_STEPS_COMPLETE = 6


class FitMode(IntEnum):
    """Distinguish dynamically converged and fixed-step execution."""

    CONVERGED = 0
    FIXED_STEPS = 1


class ObjectiveChange(NamedTuple):
    """Classification of one candidate objective relative to its predecessor."""

    normalized_change: Array
    status: Array
    accept: Array
    converged: Array


class FitResult(NamedTuple):
    """Temporary common result schema for both fit-control modes."""

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
    factor_jitter: Array
    covariance_ridge: Array
    tol: Array | None
    decrease_tol: Array | None
    initialization: InitializationProvenance
    metadata: ResultMetadata


class FixedStepKernelResult(NamedTuple):
    """Static-shape result returned by the compiled fixed-step kernel."""

    parameters: Params
    objective: Array
    history_buffer: Array
    history_length: Array
    n_iter: Array
    converged: Array
    status: Array
    mode: Array
    attempted_iteration: Array
    attempted_objective: Array
    attempted_objective_valid: Array
    numerical_failure: Array
    collapsed: Array
    collapsed_components: Array


class _FixedCarry(NamedTuple):
    """Device state for the fixed-step scan."""

    parameters: Params
    objective: Array
    n_iter: Array
    active: Array
    status: Array
    attempted_iteration: Array
    attempted_objective: Array
    attempted_objective_valid: Array
    numerical_failure: Array
    collapsed: Array
    collapsed_components: Array


def _status(value: FitStatus | FitMode) -> Array:
    return jnp.asarray(int(value), dtype=jnp.int32)


def _validated_fit_controls(
    params: Params,
    *,
    factor_jitter: object,
    covariance_ridge: object,
) -> tuple[Array, Array, Array]:
    """Return safe scalar controls and a device-resident value-failure flag.

    Shape and dtype errors raise while tracing. Negative/nonfinite scalar values
    and nonzero values lost to selected-dtype underflow become an explicit
    numerical-failure status, including on zero-step paths.
    """

    dtype = jnp.asarray(params.means).dtype
    jitter, jitter_is_valid = _real_scalar_control(
        factor_jitter, dtype=dtype, name="factor_jitter"
    )
    ridge, ridge_is_valid = _real_scalar_control(
        covariance_ridge, dtype=dtype, name="covariance_ridge"
    )
    controls_are_valid = jitter_is_valid & ridge_is_valid
    safe_jitter = jnp.where(jitter_is_valid, jitter, 0.0)
    safe_ridge = jnp.where(ridge_is_valid, ridge, 0.0)
    return safe_jitter, safe_ridge, ~controls_are_valid


def _prepared_host_fit_controls(
    params: Params,
    *,
    factor_jitter: object,
    covariance_ridge: object,
) -> PreparedControls:
    """Prepare raw eager controls or encode value failure for kernel rollback.

    Type and shape errors remain exceptions. Value-domain errors use an exact
    selected-dtype negative sentinel so the existing device status path stops
    before iteration zero and returns the unchanged parameter state.
    """

    dtype = jnp.asarray(params.means).dtype
    try:
        return validate_controls(
            factor_jitter=factor_jitter,
            covariance_ridge=covariance_ridge,
            dtype=dtype,
        )
    except _ControlValueError:
        return PreparedControls(
            factor_jitter=jnp.asarray(-1.0, dtype=dtype),
            covariance_ridge=jnp.asarray(0.0, dtype=dtype),
        )


def _real_scalar_eager(
    value: object,
    *,
    name: str,
    finite_nonnegative: bool,
) -> Array:
    """Validate a host-control scalar without coercing bool/complex values."""

    try:
        original = jnp.asarray(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real numeric scalar") from error
    if original.ndim != 0:
        raise ValueError(
            f"{name} must be a rank-zero scalar; received shape {original.shape}"
        )
    is_real_numeric = jnp.issubdtype(
        original.dtype, jnp.integer
    ) or jnp.issubdtype(original.dtype, jnp.floating)
    if not is_real_numeric:
        raise TypeError(
            f"{name} must be a real numeric scalar; received dtype "
            f"{original.dtype}"
        )
    if finite_nonnegative and (
        not bool(jnp.isfinite(original)) or bool(original < 0.0)
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return original


def _nonnegative_count(value: object, *, name: str) -> int:
    """Return one nonnegative Python integer for static/host loop construction."""

    value_dtype = getattr(value, "dtype", None)
    dtype_is_boolean = False
    if value_dtype is not None:
        try:
            dtype_is_boolean = bool(jnp.issubdtype(value_dtype, jnp.bool_))
        except TypeError:
            dtype_is_boolean = False
    if isinstance(value, bool) or dtype_is_boolean:
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if count < 0:
        raise ValueError(f"{name} must be nonnegative; received {count}")
    return int(count)


def classify_objective_change(
    *,
    previous: object,
    current: object,
    tol: object,
    decrease_tol: object,
) -> ObjectiveChange:
    """Classify one normalized objective change under the contract thresholds.

    The diagnostic is the raw computed change. A narrowly capped accommodation
    applies only at nonzero thresholds where subtractive cancellation can move
    a conceptual equality by a few representational bits.
    """

    previous_value = _real_scalar_eager(
        previous, name="previous", finite_nonnegative=False
    )
    current_value = _real_scalar_eager(
        current, name="current", finite_nonnegative=False
    )
    tolerance = _real_scalar_eager(
        tol, name="tol", finite_nonnegative=True
    )
    decrease_tolerance = _real_scalar_eager(
        decrease_tol, name="decrease_tol", finite_nonnegative=True
    )

    dtype = jnp.result_type(previous_value, current_value, tolerance)
    if not jnp.issubdtype(dtype, jnp.floating):
        dtype = jnp.asarray(0.0).dtype
    previous_array = jnp.asarray(previous_value, dtype=dtype)
    current_array = jnp.asarray(current_value, dtype=dtype)
    tolerance = jnp.asarray(tolerance, dtype=dtype)
    decrease_tolerance = jnp.asarray(decrease_tolerance, dtype=dtype)

    denominator = jnp.maximum(1.0, jnp.abs(previous_array))
    normalized_change = (current_array - previous_array) / denominator
    cancellation_uncertainty = (
        8.0
        * jnp.finfo(dtype).eps
        * (jnp.abs(previous_array) + jnp.abs(current_array))
        / denominator
    )
    tolerance_slack = jnp.where(
        tolerance > 0.0,
        jnp.minimum(cancellation_uncertainty, tolerance * 1e-6),
        0.0,
    )
    decrease_slack = jnp.where(
        decrease_tolerance > 0.0,
        jnp.minimum(
            cancellation_uncertainty, decrease_tolerance * 1e-6
        ),
        0.0,
    )

    objectives_are_finite = jnp.isfinite(previous_array) & jnp.isfinite(
        current_array
    )
    material_decrease = normalized_change < (
        -decrease_tolerance - decrease_slack
    )
    converged = (
        objectives_are_finite
        & (~material_decrease)
        & (normalized_change <= tolerance + tolerance_slack)
    )
    accept = objectives_are_finite & (~material_decrease)
    status = jnp.where(
        ~objectives_are_finite,
        _status(FitStatus.NUMERICAL_FAILURE),
        jnp.where(
            material_decrease,
            _status(FitStatus.OBJECTIVE_DECREASED),
            jnp.where(
                converged,
                _status(FitStatus.CONVERGED),
                _status(FitStatus.CONTINUE),
            ),
        ),
    )
    return ObjectiveChange(normalized_change, status, accept, converged)


def _objective_and_failure(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array,
) -> tuple[Array, Array]:
    e_step = posterior_components(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    objective = jnp.mean(e_step.score_samples)
    failure = e_step.numerical_failure | (~jnp.isfinite(objective))
    return objective, failure


def mean_log_likelihood(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the mean observed-data score under the fixed jitter policy."""

    objective, _ = _objective_and_failure(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return objective


def _select_params(predicate: Array, candidate: Params, current: Params) -> Params:
    return Params(
        jnp.where(predicate, candidate.weights, current.weights),
        jnp.where(predicate, candidate.means, current.means),
        jnp.where(predicate, candidate.covariances, current.covariances),
    )


def _invalid_objective(*, dtype: jnp.dtype) -> Array:
    """Return the host-result sentinel for an invalid objective diagnostic."""

    return jnp.asarray(jnp.nan, dtype=dtype)


def _sanitize_attempted_objective(
    value: Array,
    valid: bool | Array,
    *,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Make host attempt validity exactly equivalent to scalar finiteness."""

    canonical = jnp.asarray(value, dtype=dtype)
    valid_array = jnp.asarray(valid) & jnp.isfinite(canonical)
    return (
        jnp.where(valid_array, canonical, _invalid_objective(dtype=dtype)),
        valid_array,
    )


def _host_history(history: list[Array], *, dtype: jnp.dtype) -> Array:
    """Stack accepted host objectives or return the invalid-initial empty form."""

    if history:
        return jnp.stack(history)
    return jnp.empty((0,), dtype=dtype)


def fit_converged(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    max_iter: int = 100,
    tol: object = 1e-6,
    decrease_tol: object = 1e-10,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> FitResult:
    """Run host-controlled EM until convergence, failure, or ``max_iter``.

    Only accepted candidates enter history. A rejected decrease or failed
    candidate returns the exact last accepted parameters and trimmed history.
    This operation is not claimed to be JIT- or autodiff-compatible.
    """

    iteration_limit = _nonnegative_count(max_iter, name="max_iter")
    dtype = jnp.asarray(params.means).dtype
    convergence_controls = validate_convergence_controls(
        tol=tol,
        decrease_tol=decrease_tol,
        dtype=dtype,
    )
    tolerance = convergence_controls.tol
    decrease_tolerance = convergence_controls.decrease_tol
    prepared_controls = _prepared_host_fit_controls(
        params,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    safe_jitter, safe_ridge, control_failure = _validated_fit_controls(
        params,
        factor_jitter=prepared_controls.factor_jitter,
        covariance_ridge=prepared_controls.covariance_ridge,
    )

    objective, objective_failure = _objective_and_failure(
        params,
        observations,
        measurement_covariances,
        factor_jitter=safe_jitter,
    )
    initial_failure = control_failure | objective_failure
    objective_valid = not bool(jax.device_get(objective_failure))
    initial_objective = jnp.where(
        objective_valid,
        objective,
        _invalid_objective(dtype=dtype),
    )
    history = [initial_objective] if objective_valid else []
    current_params = params
    current_objective = initial_objective
    n_iter = 0
    converged = False
    attempted_iteration = 0
    attempted_objective = _invalid_objective(dtype=dtype)
    attempted_objective_valid = False
    numerical_failure = bool(jax.device_get(initial_failure))
    collapsed = False
    collapsed_components = jnp.zeros_like(params.weights, dtype=jnp.bool_)
    status = (
        FitStatus.NUMERICAL_FAILURE
        if numerical_failure
        else FitStatus.MAX_ITER
    )

    if not numerical_failure:
        for attempted_iteration in range(1, iteration_limit + 1):
            update = em_step(
                current_params,
                observations,
                measurement_covariances,
                factor_jitter=safe_jitter,
                covariance_ridge=safe_ridge,
            )
            if bool(jax.device_get(update.numerical_failure)):
                numerical_failure = True
                status = FitStatus.NUMERICAL_FAILURE
                attempted_objective = _invalid_objective(dtype=dtype)
                attempted_objective_valid = False
                break
            if bool(jax.device_get(update.collapsed)):
                collapsed = True
                collapsed_components = update.collapsed_components
                status = FitStatus.COMPONENT_COLLAPSED
                attempted_objective = _invalid_objective(dtype=dtype)
                attempted_objective_valid = False
                break

            candidate_objective, candidate_failure = _objective_and_failure(
                update.parameters,
                observations,
                measurement_covariances,
                factor_jitter=safe_jitter,
            )
            candidate_is_finite = bool(
                jax.device_get(jnp.isfinite(candidate_objective))
            )
            attempted_objective = candidate_objective
            candidate_failed = bool(jax.device_get(candidate_failure))
            attempted_objective_valid = (
                candidate_is_finite and not candidate_failed
            )
            if candidate_failed:
                numerical_failure = True
                status = FitStatus.NUMERICAL_FAILURE
                break

            decision = classify_objective_change(
                previous=current_objective,
                current=candidate_objective,
                tol=tolerance,
                decrease_tol=decrease_tolerance,
            )
            decision_status = FitStatus(int(jax.device_get(decision.status)))
            if not bool(jax.device_get(decision.accept)):
                status = decision_status
                break

            current_params = update.parameters
            current_objective = candidate_objective
            history.append(current_objective)
            n_iter += 1
            status = decision_status
            if bool(jax.device_get(decision.converged)):
                converged = True
                break
        else:
            status = FitStatus.MAX_ITER

    attempted_objective, attempted_objective_valid = (
        _sanitize_attempted_objective(
            attempted_objective,
            attempted_objective_valid,
            dtype=dtype,
        )
    )
    return FitResult(
        parameters=current_params,
        initial_parameters=params,
        objective=current_objective,
        objective_valid=jnp.asarray(objective_valid),
        history=_host_history(history, dtype=dtype),
        n_iter=jnp.asarray(n_iter, dtype=jnp.int32),
        iteration_limit=jnp.asarray(iteration_limit, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        status=_status(status),
        mode=_status(FitMode.CONVERGED),
        attempted_iteration=jnp.asarray(attempted_iteration, dtype=jnp.int32),
        attempted_objective=attempted_objective,
        attempted_objective_valid=jnp.asarray(attempted_objective_valid),
        numerical_failure=jnp.asarray(numerical_failure),
        collapsed=jnp.asarray(collapsed),
        collapsed_components=collapsed_components,
        factor_jitter=safe_jitter,
        covariance_ridge=safe_ridge,
        tol=tolerance,
        decrease_tol=decrease_tolerance,
        initialization=user_supplied_initialization(),
        metadata=current_result_metadata(),
    )


def fit_fixed_steps_kernel(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    n_steps: int,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> FixedStepKernelResult:
    """Run a static-length scan and return a fixed-shape history buffer.

    ``n_steps`` determines result shapes and must therefore be static when this
    function is wrapped in ``jax.jit``. On a successful path, every slot is an
    accepted EM update. On failure, later slots repeat the last valid objective.
    """

    step_count = _nonnegative_count(n_steps, name="n_steps")
    safe_jitter, safe_ridge, control_failure = _validated_fit_controls(
        params,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    initial_objective, objective_failure = _objective_and_failure(
        params,
        observations,
        measurement_covariances,
        factor_jitter=safe_jitter,
    )
    initial_failure = control_failure | objective_failure
    initial_carry = _FixedCarry(
        parameters=params,
        objective=initial_objective,
        n_iter=jnp.asarray(0, dtype=jnp.int32),
        active=~initial_failure,
        status=jnp.where(
            initial_failure,
            _status(FitStatus.NUMERICAL_FAILURE),
            _status(FitStatus.CONTINUE),
        ),
        attempted_iteration=jnp.asarray(0, dtype=jnp.int32),
        attempted_objective=initial_objective,
        attempted_objective_valid=jnp.asarray(False),
        numerical_failure=initial_failure,
        collapsed=jnp.asarray(False),
        collapsed_components=jnp.zeros_like(
            params.weights, dtype=jnp.bool_
        ),
    )

    def scan_step(
        carry: _FixedCarry, iteration_index: Array
    ) -> tuple[_FixedCarry, Array]:
        update = em_step(
            carry.parameters,
            observations,
            measurement_covariances,
            factor_jitter=safe_jitter,
            covariance_ridge=safe_ridge,
        )
        candidate_objective, candidate_evaluation_failure = (
            _objective_and_failure(
                update.parameters,
                observations,
                measurement_covariances,
                factor_jitter=safe_jitter,
            )
        )
        step_numerical_failure = (
            update.numerical_failure | candidate_evaluation_failure
        )
        step_collapsed = update.collapsed & (~step_numerical_failure)
        step_failed = step_numerical_failure | step_collapsed
        accept = carry.active & (~step_failed)
        newly_numerical = carry.active & step_numerical_failure
        newly_collapsed = carry.active & step_collapsed

        next_parameters = _select_params(
            accept, update.parameters, carry.parameters
        )
        next_objective = jnp.where(
            accept, candidate_objective, carry.objective
        )
        attempted_now = iteration_index + jnp.asarray(1, dtype=jnp.int32)
        candidate_is_finite = jnp.isfinite(candidate_objective)
        next_carry = _FixedCarry(
            parameters=next_parameters,
            objective=next_objective,
            n_iter=carry.n_iter + accept.astype(jnp.int32),
            active=carry.active & (~step_failed),
            status=jnp.where(
                newly_numerical,
                _status(FitStatus.NUMERICAL_FAILURE),
                jnp.where(
                    newly_collapsed,
                    _status(FitStatus.COMPONENT_COLLAPSED),
                    carry.status,
                ),
            ),
            attempted_iteration=jnp.where(
                carry.active, attempted_now, carry.attempted_iteration
            ),
            attempted_objective=jnp.where(
                carry.active & candidate_is_finite,
                candidate_objective,
                carry.attempted_objective,
            ),
            attempted_objective_valid=jnp.where(
                carry.active,
                (~step_failed) & candidate_is_finite,
                carry.attempted_objective_valid,
            ),
            numerical_failure=carry.numerical_failure | newly_numerical,
            collapsed=carry.collapsed | newly_collapsed,
            collapsed_components=jnp.where(
                newly_collapsed,
                update.collapsed_components,
                carry.collapsed_components,
            ),
        )
        return next_carry, next_objective

    final_carry, history_tail = jax.lax.scan(
        scan_step,
        initial_carry,
        jnp.arange(step_count, dtype=jnp.int32),
    )
    history_buffer = jnp.concatenate(
        (initial_objective[None], history_tail), axis=0
    )
    final_status = jnp.where(
        final_carry.status == _status(FitStatus.CONTINUE),
        _status(FitStatus.FIXED_STEPS_COMPLETE),
        final_carry.status,
    )
    return FixedStepKernelResult(
        parameters=final_carry.parameters,
        objective=final_carry.objective,
        history_buffer=history_buffer,
        history_length=final_carry.n_iter + jnp.asarray(1, dtype=jnp.int32),
        n_iter=final_carry.n_iter,
        converged=jnp.asarray(False),
        status=final_status,
        mode=_status(FitMode.FIXED_STEPS),
        attempted_iteration=final_carry.attempted_iteration,
        attempted_objective=final_carry.attempted_objective,
        attempted_objective_valid=final_carry.attempted_objective_valid,
        numerical_failure=final_carry.numerical_failure,
        collapsed=final_carry.collapsed,
        collapsed_components=final_carry.collapsed_components,
    )


def fit_fixed_steps(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    n_steps: int,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> FitResult:
    """Return a host-facing fixed-step result with logically trimmed history.

    Raw scalar controls are prepared eagerly before entering the compiled scan,
    preserving Python values that JAX might otherwise canonicalize away.
    """

    step_count = _nonnegative_count(n_steps, name="n_steps")
    dtype = jnp.asarray(params.means).dtype
    prepared_controls = _prepared_host_fit_controls(
        params,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    safe_jitter, safe_ridge, _ = _validated_fit_controls(
        params,
        factor_jitter=prepared_controls.factor_jitter,
        covariance_ridge=prepared_controls.covariance_ridge,
    )
    _, initial_objective_failure = _objective_and_failure(
        params,
        observations,
        measurement_covariances,
        factor_jitter=safe_jitter,
    )
    objective_valid = not bool(jax.device_get(initial_objective_failure))
    kernel_result = fit_fixed_steps_kernel(
        params,
        observations,
        measurement_covariances,
        n_steps=step_count,
        factor_jitter=prepared_controls.factor_jitter,
        covariance_ridge=prepared_controls.covariance_ridge,
    )
    history_length = int(jax.device_get(kernel_result.history_length))
    objective = jnp.where(
        objective_valid,
        kernel_result.objective,
        _invalid_objective(dtype=dtype),
    )
    history = (
        kernel_result.history_buffer[:history_length]
        if objective_valid
        else jnp.empty((0,), dtype=dtype)
    )
    attempted_objective, attempted_objective_valid = (
        _sanitize_attempted_objective(
            kernel_result.attempted_objective,
            kernel_result.attempted_objective_valid,
            dtype=dtype,
        )
    )
    return FitResult(
        parameters=kernel_result.parameters,
        initial_parameters=params,
        objective=objective,
        objective_valid=jnp.asarray(objective_valid),
        history=history,
        n_iter=kernel_result.n_iter,
        iteration_limit=jnp.asarray(step_count, dtype=jnp.int32),
        converged=kernel_result.converged,
        status=kernel_result.status,
        mode=kernel_result.mode,
        attempted_iteration=kernel_result.attempted_iteration,
        attempted_objective=attempted_objective,
        attempted_objective_valid=attempted_objective_valid,
        numerical_failure=kernel_result.numerical_failure,
        collapsed=kernel_result.collapsed,
        collapsed_components=kernel_result.collapsed_components,
        factor_jitter=safe_jitter,
        covariance_ridge=safe_ridge,
        tol=None,
        decrease_tol=None,
        initialization=user_supplied_initialization(),
        metadata=current_result_metadata(),
    )


__all__ = [
    "FixedStepKernelResult",
    "FitMode",
    "FitResult",
    "FitStatus",
    "ObjectiveChange",
    "classify_objective_change",
    "fit_converged",
    "fit_fixed_steps",
    "fit_fixed_steps_kernel",
    "mean_log_likelihood",
]
