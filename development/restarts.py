"""Temporary host-only selection across ordered XD initializations.

This module implements companion contract ``xdgmm-jax.restart-selection``
``0.1.0-draft.1``.  Version 1 accepts only explicitly ordered user-supplied
parameter candidates.  Candidate construction and all four wrappers are eager,
sequential host operations; they carry no whole-call JIT or autodiff claim.

Restart wrapper results are deliberately outside the current serialization
contract.  The selected single-fit result is retained byte-for-byte and is not
relabelled with new initialization or contract metadata.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Callable, NamedTuple, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from .fit_control import (
    FitMode,
    FitResult,
    FitStatus,
    _nonnegative_count,
    _prepared_host_fit_controls,
    fit_converged,
    fit_fixed_steps,
)
from .general_fit_control import (
    GroupedGeneralFitResult,
    _with_parameters,
    fit_converged_grouped,
    fit_fixed_steps_grouped,
)
from .general_validation import (
    GroupedGeneralFitInputs,
    _canonical_parameters,
)
from .identity_xd import Params
from .validation import (
    PrecisionError,
    ValidationError,
    _computation_dtype,
    validate_convergence_controls,
)


Array = jax.Array

RESTART_CONTRACT_ID = "xdgmm-jax.restart-selection"
RESTART_CONTRACT_VERSION = "0.1.0-draft.1"
RESTART_SELECTION_RULE_ID = "xdgmm-jax.highest-eligible-objective"
RESTART_SELECTION_RULE_VERSION = "0.1.0-draft.1"
USER_SUPPLIED_INITIALIZATION_KIND = "user_supplied_ordered"


class RestartSelectionStatus(IntEnum):
    """Terminal status of the collection-level selection operation."""

    SELECTED_ELIGIBLE = 0
    ALL_INITIALIZATIONS_FAILED = 1


class RestartCandidates(NamedTuple):
    """Canonical ordered initial states with one leading restart axis.

    Instances returned by :func:`user_supplied_restart_candidates` are
    validated tokens.  Direct construction is not a supported boundary.
    """

    weights: Array
    means: Array
    covariances: Array
    initialization_kind: str


class RestartDiagnostics(NamedTuple):
    """Bounded terminal summary for every attempted initialization."""

    objective: Array
    objective_valid: Array
    n_iter: Array
    status: Array
    converged: Array
    numerical_failure: Array
    collapsed: Array
    eligible: Array


class RestartSelection(NamedTuple):
    """Deterministic collection-level selection and companion identity."""

    restart_count: Array
    selected_restart: Array
    success: Array
    status: Array
    contract_id: str
    contract_version: str
    rule_id: str
    rule_version: str


class IdentityRestartFitResult(NamedTuple):
    """Selected identity fit plus candidate custody and bounded summaries."""

    selected_result: FitResult
    candidates: RestartCandidates
    diagnostics: RestartDiagnostics
    selection: RestartSelection


class GroupedGeneralRestartFitResult(NamedTuple):
    """Selected grouped fit plus candidate custody and bounded summaries."""

    selected_result: GroupedGeneralFitResult
    candidates: RestartCandidates
    diagnostics: RestartDiagnostics
    selection: RestartSelection


def _candidate_error(index: int, error: Exception) -> Exception:
    """Retain validation exception class while adding restart location."""

    message = f"initial_parameters[{index}]: {error}"
    if isinstance(error, PrecisionError):
        return PrecisionError(message)
    if isinstance(error, ValidationError):
        return ValidationError(message)
    if isinstance(error, TypeError):
        return TypeError(message)
    return ValueError(message)


def _stored_candidate_error(index: int, error: Exception) -> Exception:
    """Retain validation class for a forged or corrupted stacked row."""

    message = f"candidates[{index}]: {error}"
    if isinstance(error, PrecisionError):
        return PrecisionError(message)
    if isinstance(error, ValidationError):
        return ValidationError(message)
    if isinstance(error, TypeError):
        return TypeError(message)
    return ValueError(message)


def user_supplied_restart_candidates(
    initial_parameters: Sequence[Params],
    *,
    dtype: object,
) -> RestartCandidates:
    """Validate and stack a nonempty ordered candidate collection.

    Floating source dtypes may differ because the selected computation dtype is
    explicit.  Every candidate is independently validated in that dtype before
    any stacking occurs, so a later invalid candidate cannot leave a partially
    usable collection.
    """

    if isinstance(initial_parameters, Params):
        raise ValidationError(
            "initial_parameters must be a nonempty sequence of Params, not "
            "one Params instance"
        )
    try:
        supplied = tuple(initial_parameters)
    except TypeError as error:
        raise TypeError(
            "initial_parameters must be a nonempty sequence of Params"
        ) from error
    if not supplied:
        raise ValidationError(
            "initial_parameters must contain at least one candidate"
        )

    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    canonical: list[Params] = []
    expected_shape: tuple[int, int] | None = None
    for index, parameters in enumerate(supplied):
        try:
            candidate, n_components, latent_dimension = _canonical_parameters(
                parameters,
                requested_numpy_dtype=requested_numpy_dtype,
                requested_jax_dtype=requested_jax_dtype,
            )
        except (TypeError, ValueError, ValidationError, PrecisionError) as error:
            raise _candidate_error(index, error) from error
        shape = (n_components, latent_dimension)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValidationError(
                f"initial_parameters[{index}] has (K, D)={shape}; expected "
                f"common (K, D)={expected_shape}"
            )
        for field, value in zip(
            ("weights", "means", "covariances"), candidate, strict=True
        ):
            if value.dtype != requested_jax_dtype:
                raise RuntimeError(
                    f"initial_parameters[{index}].{field} did not retain "
                    "the selected computation dtype"
                )
        canonical.append(candidate)

    return RestartCandidates(
        weights=jnp.stack([candidate.weights for candidate in canonical]),
        means=jnp.stack([candidate.means for candidate in canonical]),
        covariances=jnp.stack(
            [candidate.covariances for candidate in canonical]
        ),
        initialization_kind=USER_SUPPLIED_INITIALIZATION_KIND,
    )


def _validated_candidate_shape(
    candidates: RestartCandidates,
) -> tuple[int, int, int, jnp.dtype]:
    """Validate the structural token shared by all restart wrappers."""

    if not isinstance(candidates, RestartCandidates):
        raise TypeError(
            "candidates must be returned by "
            "user_supplied_restart_candidates"
        )
    if candidates.initialization_kind != USER_SUPPLIED_INITIALIZATION_KIND:
        raise ValidationError("candidates initialization kind is unsupported")

    weights = candidates.weights
    means = candidates.means
    covariances = candidates.covariances
    if weights.ndim != 2:
        raise ValidationError(
            f"candidate weights must have shape (R,K); received {weights.shape}"
        )
    if means.ndim != 3:
        raise ValidationError(
            "candidate means must have shape (R,K,D); received "
            f"{means.shape}"
        )
    if covariances.ndim != 4:
        raise ValidationError(
            "candidate covariances must have shape (R,K,D,D); received "
            f"{covariances.shape}"
        )
    restart_count, n_components = weights.shape
    if restart_count < 1 or n_components < 1:
        raise ValidationError("candidate dimensions R and K must be positive")
    if restart_count > np.iinfo(np.int32).max:
        raise ValidationError("candidate count exceeds the int32 result schema")
    if means.shape[0] != restart_count or means.shape[1] != n_components:
        raise ValidationError("candidate weights and means have incompatible R/K")
    latent_dimension = means.shape[2]
    if latent_dimension < 1:
        raise ValidationError("candidate latent dimension D must be positive")
    expected_covariances = (
        restart_count,
        n_components,
        latent_dimension,
        latent_dimension,
    )
    if covariances.shape != expected_covariances:
        raise ValidationError(
            "candidate covariances must have shape "
            f"{expected_covariances}; received {covariances.shape}"
        )
    dtype = means.dtype
    if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValidationError("candidate computation dtype must be float32 or float64")
    if weights.dtype != dtype or covariances.dtype != dtype:
        raise ValidationError("every candidate leaf must share one computation dtype")

    # ``RestartCandidates`` is a public NamedTuple and therefore forgeable.
    # Revalidate every stored row before candidate zero can run.  The equality
    # check deliberately rejects values that would require even a permitted
    # symmetry repair: wrappers consume only exact canonical constructor output.
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    for index in range(restart_count):
        stored = _candidate_at(candidates, index)
        try:
            canonical, canonical_k, canonical_d = _canonical_parameters(
                stored,
                requested_numpy_dtype=requested_numpy_dtype,
                requested_jax_dtype=requested_jax_dtype,
            )
        except (TypeError, ValueError, ValidationError, PrecisionError) as error:
            raise _stored_candidate_error(index, error) from error
        if (canonical_k, canonical_d) != (n_components, latent_dimension):
            raise RuntimeError(
                "restart candidate validation changed established K/D"
            )
        if not _params_bits_equal(canonical, stored):
            raise ValidationError(
                f"candidates[{index}] is not bit-exact canonical constructor "
                "output; hidden parameter repair is forbidden"
            )
    return restart_count, n_components, latent_dimension, dtype


def _candidate_at(candidates: RestartCandidates, index: int) -> Params:
    """Return one stored canonical row without conversion or repair."""

    return Params(
        weights=candidates.weights[index],
        means=candidates.means[index],
        covariances=candidates.covariances[index],
    )


def _host_scalar(value: object, *, field: str) -> np.ndarray:
    """Return one synchronized host scalar or raise an internal invariant."""

    host = np.asarray(jax.device_get(value))
    if host.shape != ():
        raise RuntimeError(f"restart result invariant: {field} must be scalar")
    return host


def _host_bool(value: object, *, field: str) -> bool:
    host = _host_scalar(value, field=field)
    if host.dtype != np.dtype(bool):
        raise RuntimeError(
            f"restart result invariant: {field} must have boolean dtype"
        )
    return bool(host)


def _host_int32(value: object, *, field: str) -> int:
    host = _host_scalar(value, field=field)
    if host.dtype != np.dtype(np.int32):
        raise RuntimeError(
            f"restart result invariant: {field} must have int32 dtype"
        )
    return int(host)


def _bits_equal(left: object, right: object) -> bool:
    """Compare array shape, dtype, and raw bits, including signed zero."""

    left_host = np.asarray(jax.device_get(left))
    right_host = np.asarray(jax.device_get(right))
    if left_host.shape != right_host.shape or left_host.dtype != right_host.dtype:
        return False
    return left_host.tobytes(order="C") == right_host.tobytes(order="C")


def _params_bits_equal(left: object, right: Params) -> bool:
    try:
        left_leaves = (left.weights, left.means, left.covariances)
    except AttributeError:
        return False
    return all(
        _bits_equal(actual, expected)
        for actual, expected in zip(left_leaves, right, strict=True)
    )


def _validate_result_summary(
    result: FitResult | GroupedGeneralFitResult,
    *,
    candidate: Params,
    expected_type: type[FitResult] | type[GroupedGeneralFitResult],
    expected_mode: FitMode,
    dtype: jnp.dtype,
) -> tuple[float, bool, int, int, bool, bool, bool]:
    """Validate selection-relevant frozen single-fit invariants."""

    if not isinstance(result, expected_type):
        raise RuntimeError(
            "restart result invariant: controller returned an unsupported "
            "result type"
        )
    if not _params_bits_equal(result.initial_parameters, candidate):
        raise RuntimeError(
            "restart result invariant: initial_parameters do not exactly "
            "match the stored candidate"
        )
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    try:
        canonical_result, result_k, result_d = _canonical_parameters(
            result.parameters,
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
        )
    except (TypeError, ValueError, ValidationError, PrecisionError) as error:
        raise RuntimeError(
            "restart result parameter invariant: controller returned invalid "
            f"parameters: {error}"
        ) from error
    expected_k, expected_d = candidate.means.shape
    if (result_k, result_d) != (expected_k, expected_d):
        raise RuntimeError(
            "restart result parameter invariant: controller changed K or D"
        )
    if not _params_bits_equal(canonical_result, result.parameters):
        raise RuntimeError(
            "restart result parameter invariant: controller parameters are "
            "not bit-exact canonical output"
        )

    objective_host = _host_scalar(result.objective, field="objective")
    if objective_host.dtype != np.dtype(dtype):
        raise RuntimeError(
            "restart result invariant: objective has the wrong computation dtype"
        )
    objective = float(objective_host)
    objective_valid = _host_bool(
        result.objective_valid, field="objective_valid"
    )
    n_iter = _host_int32(result.n_iter, field="n_iter")
    iteration_limit = _host_int32(
        result.iteration_limit, field="iteration_limit"
    )
    attempted_iteration = _host_int32(
        result.attempted_iteration, field="attempted_iteration"
    )
    if n_iter < 0 or iteration_limit < 0 or attempted_iteration < 0:
        raise RuntimeError(
            "restart result invariant: iteration fields must be nonnegative"
        )

    status_value = _host_int32(result.status, field="status")
    mode_value = _host_int32(result.mode, field="mode")
    try:
        status = FitStatus(status_value)
        mode = FitMode(mode_value)
    except ValueError as error:
        raise RuntimeError(
            "restart result invariant: unknown status or mode"
        ) from error
    if status == FitStatus.CONTINUE or mode != expected_mode:
        raise RuntimeError(
            "restart result invariant: terminal status/mode is inconsistent"
        )

    converged = _host_bool(result.converged, field="converged")
    numerical_failure = _host_bool(
        result.numerical_failure, field="numerical_failure"
    )
    collapsed = _host_bool(result.collapsed, field="collapsed")
    if converged != (status == FitStatus.CONVERGED):
        raise RuntimeError(
            "restart result invariant: converged is inconsistent with status"
        )
    if numerical_failure != (status == FitStatus.NUMERICAL_FAILURE):
        raise RuntimeError(
            "restart result invariant: numerical_failure is inconsistent "
            "with status"
        )
    if collapsed != (status == FitStatus.COMPONENT_COLLAPSED):
        raise RuntimeError(
            "restart result invariant: collapsed is inconsistent with status"
        )
    if status in (
        FitStatus.CONVERGED,
        FitStatus.MAX_ITER,
        FitStatus.OBJECTIVE_DECREASED,
    ) and mode != FitMode.CONVERGED:
        raise RuntimeError(
            "restart result invariant: converged-mode status has wrong mode"
        )
    if status == FitStatus.FIXED_STEPS_COMPLETE and mode != FitMode.FIXED_STEPS:
        raise RuntimeError(
            "restart result invariant: fixed-step completion has wrong mode"
        )
    if status in (FitStatus.MAX_ITER, FitStatus.FIXED_STEPS_COMPLETE) and (
        n_iter != iteration_limit
    ):
        raise RuntimeError(
            "restart result invariant: terminal iteration count disagrees "
            "with its limit"
        )
    if status == FitStatus.CONVERGED and n_iter == 0:
        raise RuntimeError(
            "restart result invariant: convergence requires an accepted update"
        )

    history = np.asarray(jax.device_get(result.history))
    if history.ndim != 1 or history.dtype != np.dtype(dtype):
        raise RuntimeError(
            "restart result invariant: history must be a computation-dtype vector"
        )
    history_valid = (
        history.shape == (n_iter + 1,)
        and bool(np.all(np.isfinite(history)))
        and _bits_equal(history[-1], objective_host)
    )
    if objective_valid != (np.isfinite(objective) and history_valid):
        raise RuntimeError(
            "restart result invariant: objective_valid disagrees with "
            "objective/history"
        )
    if not objective_valid:
        if status != FitStatus.NUMERICAL_FAILURE or n_iter != 0:
            raise RuntimeError(
                "restart result invariant: invalid initial objective must be "
                "a zero-iteration numerical failure"
            )
        if history.shape != (0,) or not np.isnan(objective):
            raise RuntimeError(
                "restart result invariant: invalid objective requires empty "
                "history and a NaN sentinel"
            )

    attempted_host = _host_scalar(
        result.attempted_objective, field="attempted_objective"
    )
    if attempted_host.dtype != np.dtype(dtype):
        raise RuntimeError(
            "restart result invariant: attempted objective has wrong dtype"
        )
    attempted_valid = _host_bool(
        result.attempted_objective_valid,
        field="attempted_objective_valid",
    )
    if attempted_valid != bool(np.isfinite(attempted_host)):
        raise RuntimeError(
            "restart result invariant: attempted objective validity disagrees "
            "with finiteness"
        )

    return (
        objective,
        objective_valid,
        n_iter,
        status_value,
        converged,
        numerical_failure,
        collapsed,
    )


SingleResult = TypeVar("SingleResult", FitResult, GroupedGeneralFitResult)
CollectionResult = TypeVar(
    "CollectionResult", IdentityRestartFitResult, GroupedGeneralRestartFitResult
)


def _collect_and_select(
    candidates: RestartCandidates,
    *,
    expected_type: type[FitResult] | type[GroupedGeneralFitResult],
    expected_mode: FitMode,
    run_one: Callable[[Params], SingleResult],
    build_result: Callable[
        [SingleResult, RestartCandidates, RestartDiagnostics, RestartSelection],
        CollectionResult,
    ],
) -> CollectionResult:
    """Run every candidate and apply the frozen deterministic selection rule."""

    restart_count, _, _, dtype = _validated_candidate_shape(candidates)
    results: list[SingleResult] = []
    objectives: list[Array] = []
    objective_validity: list[Array] = []
    iteration_counts: list[Array] = []
    statuses: list[Array] = []
    converged_values: list[Array] = []
    failure_values: list[Array] = []
    collapsed_values: list[Array] = []
    eligible_values: list[bool] = []
    host_objectives: list[float] = []
    host_validity: list[bool] = []

    for index in range(restart_count):
        candidate = _candidate_at(candidates, index)
        result = run_one(candidate)
        (
            objective,
            objective_valid,
            _,
            status_value,
            _,
            _,
            _,
        ) = _validate_result_summary(
            result,
            candidate=candidate,
            expected_type=expected_type,
            expected_mode=expected_mode,
            dtype=dtype,
        )
        status = FitStatus(status_value)
        if expected_mode == FitMode.CONVERGED:
            eligible = status in (FitStatus.CONVERGED, FitStatus.MAX_ITER)
        else:
            eligible = status == FitStatus.FIXED_STEPS_COMPLETE
        eligible = eligible and objective_valid and np.isfinite(objective)

        results.append(result)
        objectives.append(result.objective)
        objective_validity.append(result.objective_valid)
        iteration_counts.append(result.n_iter)
        statuses.append(result.status)
        converged_values.append(result.converged)
        failure_values.append(result.numerical_failure)
        collapsed_values.append(result.collapsed)
        eligible_values.append(bool(eligible))
        host_objectives.append(objective)
        host_validity.append(objective_valid and np.isfinite(objective))

    eligible_indices = [
        index for index, eligible in enumerate(eligible_values) if eligible
    ]
    if eligible_indices:
        selected_index = eligible_indices[0]
        for index in eligible_indices[1:]:
            if host_objectives[index] > host_objectives[selected_index]:
                selected_index = index
        selection_success = True
        selection_status = RestartSelectionStatus.SELECTED_ELIGIBLE
    else:
        valid_indices = [
            index for index, valid in enumerate(host_validity) if valid
        ]
        selected_index = valid_indices[0] if valid_indices else 0
        for index in valid_indices[1:]:
            if host_objectives[index] > host_objectives[selected_index]:
                selected_index = index
        selection_success = False
        selection_status = (
            RestartSelectionStatus.ALL_INITIALIZATIONS_FAILED
        )

    diagnostics = RestartDiagnostics(
        objective=jnp.stack(objectives),
        objective_valid=jnp.stack(objective_validity),
        n_iter=jnp.stack(iteration_counts),
        status=jnp.stack(statuses),
        converged=jnp.stack(converged_values),
        numerical_failure=jnp.stack(failure_values),
        collapsed=jnp.stack(collapsed_values),
        eligible=jnp.asarray(eligible_values, dtype=jnp.bool_),
    )
    selection = RestartSelection(
        restart_count=jnp.asarray(restart_count, dtype=jnp.int32),
        selected_restart=jnp.asarray(selected_index, dtype=jnp.int32),
        success=jnp.asarray(selection_success, dtype=jnp.bool_),
        status=jnp.asarray(int(selection_status), dtype=jnp.int32),
        contract_id=RESTART_CONTRACT_ID,
        contract_version=RESTART_CONTRACT_VERSION,
        rule_id=RESTART_SELECTION_RULE_ID,
        rule_version=RESTART_SELECTION_RULE_VERSION,
    )
    return build_result(
        results[selected_index], candidates, diagnostics, selection
    )


def fit_converged_restarts(
    candidates: RestartCandidates,
    observations: Array,
    measurement_covariances: Array,
    *,
    max_iter: int = 100,
    tol: object = 1e-6,
    decrease_tol: object = 1e-10,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> IdentityRestartFitResult:
    """Run every identity candidate with one common converged-fit policy."""

    _, _, _, dtype = _validated_candidate_shape(candidates)
    iteration_limit = _nonnegative_count(max_iter, name="max_iter")
    convergence_controls = validate_convergence_controls(
        tol=tol,
        decrease_tol=decrease_tol,
        dtype=dtype,
    )
    tolerance = convergence_controls.tol
    decrease_tolerance = convergence_controls.decrease_tol
    controls = _prepared_host_fit_controls(
        _candidate_at(candidates, 0),
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )

    def run_one(parameters: Params) -> FitResult:
        return fit_converged(
            parameters,
            observations,
            measurement_covariances,
            max_iter=iteration_limit,
            tol=tolerance,
            decrease_tol=decrease_tolerance,
            factor_jitter=controls.factor_jitter,
            covariance_ridge=controls.covariance_ridge,
        )

    return _collect_and_select(
        candidates,
        expected_type=FitResult,
        expected_mode=FitMode.CONVERGED,
        run_one=run_one,
        build_result=IdentityRestartFitResult,
    )


def fit_fixed_steps_restarts(
    candidates: RestartCandidates,
    observations: Array,
    measurement_covariances: Array,
    *,
    n_steps: int,
    factor_jitter: float | Array = 0.0,
    covariance_ridge: float | Array = 0.0,
) -> IdentityRestartFitResult:
    """Run every identity candidate for one common fixed update count."""

    _validated_candidate_shape(candidates)
    step_count = _nonnegative_count(n_steps, name="n_steps")
    controls = _prepared_host_fit_controls(
        _candidate_at(candidates, 0),
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )

    def run_one(parameters: Params) -> FitResult:
        return fit_fixed_steps(
            parameters,
            observations,
            measurement_covariances,
            n_steps=step_count,
            factor_jitter=controls.factor_jitter,
            covariance_ridge=controls.covariance_ridge,
        )

    return _collect_and_select(
        candidates,
        expected_type=FitResult,
        expected_mode=FitMode.FIXED_STEPS,
        run_one=run_one,
        build_result=IdentityRestartFitResult,
    )


def _validated_grouped_compatibility(
    fit: GroupedGeneralFitInputs,
    candidates: RestartCandidates,
) -> None:
    """Reject incompatible candidate geometry before candidate zero runs."""

    if not isinstance(fit, GroupedGeneralFitInputs):
        raise TypeError("fit must be a validated GroupedGeneralFitInputs")
    _, n_components, latent_dimension, dtype = _validated_candidate_shape(
        candidates
    )
    fit_parameters = fit.grouped.parameters
    expected_components, expected_dimension = fit_parameters.means.shape
    if (n_components, latent_dimension) != (
        expected_components,
        expected_dimension,
    ):
        raise ValidationError(
            "candidate (K, D) is incompatible with grouped fit parameters: "
            f"received {(n_components, latent_dimension)}, expected "
            f"{(expected_components, expected_dimension)}"
        )
    if dtype != fit_parameters.means.dtype:
        raise ValidationError(
            "candidate computation dtype is incompatible with grouped fit "
            f"dtype: received {dtype}, expected {fit_parameters.means.dtype}"
        )


def fit_converged_grouped_restarts(
    fit: GroupedGeneralFitInputs,
    *,
    candidates: RestartCandidates,
    max_iter: int = 100,
    tol: object = 1e-6,
    decrease_tol: object = 1e-10,
) -> GroupedGeneralRestartFitResult:
    """Run every candidate through the common grouped converged controller."""

    _validated_grouped_compatibility(fit, candidates)
    _, _, _, dtype = _validated_candidate_shape(candidates)
    iteration_limit = _nonnegative_count(max_iter, name="max_iter")
    convergence_controls = validate_convergence_controls(
        tol=tol,
        decrease_tol=decrease_tol,
        dtype=dtype,
    )
    tolerance = convergence_controls.tol
    decrease_tolerance = convergence_controls.decrease_tol

    def run_one(parameters: Params) -> GroupedGeneralFitResult:
        return fit_converged_grouped(
            _with_parameters(fit, parameters),
            max_iter=iteration_limit,
            tol=tolerance,
            decrease_tol=decrease_tolerance,
        )

    return _collect_and_select(
        candidates,
        expected_type=GroupedGeneralFitResult,
        expected_mode=FitMode.CONVERGED,
        run_one=run_one,
        build_result=GroupedGeneralRestartFitResult,
    )


def fit_fixed_steps_grouped_restarts(
    fit: GroupedGeneralFitInputs,
    *,
    candidates: RestartCandidates,
    n_steps: int,
) -> GroupedGeneralRestartFitResult:
    """Run every candidate through the common grouped fixed-step controller."""

    _validated_grouped_compatibility(fit, candidates)
    step_count = _nonnegative_count(n_steps, name="n_steps")

    def run_one(parameters: Params) -> GroupedGeneralFitResult:
        return fit_fixed_steps_grouped(
            _with_parameters(fit, parameters), n_steps=step_count
        )

    return _collect_and_select(
        candidates,
        expected_type=GroupedGeneralFitResult,
        expected_mode=FitMode.FIXED_STEPS,
        run_one=run_one,
        build_result=GroupedGeneralRestartFitResult,
    )


__all__ = [
    "RESTART_CONTRACT_ID",
    "RESTART_CONTRACT_VERSION",
    "RESTART_SELECTION_RULE_ID",
    "RESTART_SELECTION_RULE_VERSION",
    "GroupedGeneralRestartFitResult",
    "IdentityRestartFitResult",
    "RestartCandidates",
    "RestartDiagnostics",
    "RestartSelection",
    "RestartSelectionStatus",
    "fit_converged_grouped_restarts",
    "fit_converged_restarts",
    "fit_fixed_steps_grouped_restarts",
    "fit_fixed_steps_restarts",
    "user_supplied_restart_candidates",
]
