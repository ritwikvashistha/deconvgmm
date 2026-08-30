"""Red gates for the host fit state required by numerical serialization.

These tests deliberately stop before any writer or reader exists.  They freeze
only the temporary host-result information that Section 6 of the serialization
contract says a future serializer must be able to read without inference.
The compiled identity fixed-step result remains a separate static-buffer type.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
import development.fit_control as identity_control
import development.general_fit_control as general_control
import development.metadata as metadata_module
from development.fit_control import FitMode, FitStatus
from development.general_grouped import GroupedFailureStage, GroupedStepStatus
from development.general_validation import (
    PerItemFullNoise,
    PerItemProjection,
    group_masked_general_fit_inputs,
)
from development.identity_xd import Params


IDENTITY_HOST_RESULT_FIELDS = (
    "parameters",
    "initial_parameters",
    "objective",
    "objective_valid",
    "history",
    "n_iter",
    "iteration_limit",
    "converged",
    "status",
    "mode",
    "attempted_iteration",
    "attempted_objective",
    "attempted_objective_valid",
    "numerical_failure",
    "collapsed",
    "collapsed_components",
    "factor_jitter",
    "covariance_ridge",
    "tol",
    "decrease_tol",
    "initialization",
    "metadata",
)

GROUPED_HOST_RESULT_FIELDS = (
    "parameters",
    "initial_parameters",
    "objective",
    "objective_valid",
    "history",
    "n_iter",
    "iteration_limit",
    "converged",
    "status",
    "mode",
    "attempted_iteration",
    "attempted_objective",
    "attempted_objective_valid",
    "numerical_failure",
    "collapsed",
    "collapsed_components",
    "failure_stage",
    "group_numerical_failure",
    "failed_pairs",
    "informative_weight",
    "factor_jitter",
    "covariance_ridge",
    "tol",
    "decrease_tol",
    "initialization",
    "metadata",
)

FIXED_STEP_KERNEL_FIELDS = (
    "parameters",
    "objective",
    "history_buffer",
    "history_length",
    "n_iter",
    "converged",
    "status",
    "mode",
    "attempted_iteration",
    "attempted_objective",
    "attempted_objective_valid",
    "numerical_failure",
    "collapsed",
    "collapsed_components",
)

DTYPES = (
    pytest.param(jnp.float64, id="float64"),
    pytest.param(jnp.float32, id="float32"),
)

FACTOR_JITTER = 2.0**-12
COVARIANCE_RIDGE = 2.0**-10
CONVERGENCE_TOL = 2.0**10
DECREASE_TOL = 2.0**-20


def _ordinary_identity_problem(dtype) -> tuple[Params, jax.Array, jax.Array]:
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=dtype),
        means=jnp.asarray([[-0.0]], dtype=dtype),
        covariances=jnp.asarray([[[1.1]]], dtype=dtype),
    )
    observations = jnp.asarray([[-0.5], [0.1], [0.9]], dtype=dtype)
    noise = jnp.asarray([[[0.1]], [[0.2]], [[0.15]]], dtype=dtype)
    return parameters, observations, noise


def _ordinary_grouped_problem(dtype):
    parameters, observations, noise = _ordinary_identity_problem(dtype)
    numpy_dtype = np.dtype(dtype)
    fit = group_masked_general_fit_inputs(
        parameters,
        np.asarray(observations, dtype=numpy_dtype),
        np.ones((observations.shape[0], 1), dtype=bool),
        projection=PerItemProjection(
            np.ones((observations.shape[0], 1, 1), dtype=numpy_dtype)
        ),
        noise=PerItemFullNoise(np.asarray(noise, dtype=numpy_dtype)),
        sample_weight=np.ones(observations.shape[0], dtype=numpy_dtype),
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
        dtype=dtype,
    )
    return parameters, fit


def _collapse_identity_problem(dtype=jnp.float64):
    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray([[0.0], [1_000_000.0]], dtype=dtype),
        covariances=jnp.asarray([[[1.0]], [[1.0]]], dtype=dtype),
    )
    observations = jnp.asarray([[-0.2], [0.1], [0.4]], dtype=dtype)
    noise = jnp.full((3, 1, 1), 0.1, dtype=dtype)
    return parameters, observations, noise


def _collapse_grouped_problem(dtype=jnp.float64):
    parameters, observations, noise = _collapse_identity_problem(dtype)
    numpy_dtype = np.dtype(dtype)
    fit = group_masked_general_fit_inputs(
        parameters,
        np.asarray(observations, dtype=numpy_dtype),
        np.ones((observations.shape[0], 1), dtype=bool),
        projection=PerItemProjection(
            np.ones((observations.shape[0], 1, 1), dtype=numpy_dtype)
        ),
        noise=PerItemFullNoise(np.asarray(noise, dtype=numpy_dtype)),
        sample_weight=np.ones(observations.shape[0], dtype=numpy_dtype),
        dtype=dtype,
    )
    return parameters, fit


def _invalid_identity_initial_objective(dtype=jnp.float64):
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=dtype),
        means=jnp.asarray([[0.0]], dtype=dtype),
        covariances=jnp.asarray([[[1.0]]], dtype=dtype),
    )
    observations = jnp.asarray([[0.1]], dtype=dtype)
    # The raw numerical leaf is intentionally given a nonfactorable effective
    # covariance so its initial objective is invalid before any update.
    noise = jnp.asarray([[[-2.0]]], dtype=dtype)
    return parameters, observations, noise


def _invalid_grouped_initial_objective(dtype=jnp.float64):
    parameters = Params(
        weights=jnp.asarray([1.0], dtype=dtype),
        means=jnp.asarray([[0.0]], dtype=dtype),
        covariances=jnp.asarray([[[1.0]]], dtype=dtype),
    )
    numpy_dtype = np.dtype(dtype)
    # R=0 and S=0 are individually valid, but the positive-weight observed
    # covariance is singular in exact zero-jitter mode.
    fit = group_masked_general_fit_inputs(
        parameters,
        np.asarray([[0.1]], dtype=numpy_dtype),
        np.ones((1, 1), dtype=bool),
        projection=PerItemProjection(np.zeros((1, 1, 1), dtype=numpy_dtype)),
        noise=PerItemFullNoise(np.zeros((1, 1, 1), dtype=numpy_dtype)),
        sample_weight=np.ones((1,), dtype=numpy_dtype),
        dtype=dtype,
    )
    return parameters, fit


def _assert_array_bits_equal(actual, expected) -> None:
    actual_host = np.asarray(actual)
    expected_host = np.asarray(expected)
    assert actual_host.shape == expected_host.shape
    assert actual_host.dtype == expected_host.dtype
    assert actual_host.tobytes(order="C") == expected_host.tobytes(order="C")


def _assert_params_bits_equal(actual: Params, expected: Params) -> None:
    assert isinstance(actual, Params)
    for actual_field, expected_field in zip(actual, expected, strict=True):
        _assert_array_bits_equal(actual_field, expected_field)


def _assert_scalar_value(value, expected, *, dtype) -> None:
    value_host = np.asarray(value)
    expected_host = np.asarray(expected, dtype=np.dtype(dtype))
    assert value_host.shape == ()
    assert value_host.dtype == np.dtype(dtype)
    _assert_array_bits_equal(value_host, expected_host)


def _assert_host_state(
    result,
    *,
    fields,
    initial_parameters,
    dtype,
    iteration_limit,
    mode,
    objective_valid,
    factor_jitter,
    covariance_ridge,
    tol,
    decrease_tol,
) -> None:
    assert result._fields == fields
    _assert_params_bits_equal(result.initial_parameters, initial_parameters)
    assert bool(np.asarray(result.objective_valid)) is objective_valid
    assert int(np.asarray(result.iteration_limit)) == iteration_limit
    assert int(np.asarray(result.mode)) == int(mode)

    _assert_scalar_value(result.factor_jitter, factor_jitter, dtype=dtype)
    _assert_scalar_value(result.covariance_ridge, covariance_ridge, dtype=dtype)
    if mode == FitMode.FIXED_STEPS:
        assert result.tol is None
        assert result.decrease_tol is None
    else:
        assert tol is not None
        assert decrease_tol is not None
        _assert_scalar_value(result.tol, tol, dtype=dtype)
        _assert_scalar_value(result.decrease_tol, decrease_tol, dtype=dtype)

    assert isinstance(
        result.initialization, metadata_module.InitializationProvenance
    )
    assert result.initialization._fields == ("kind",)
    assert result.initialization.kind == "user_supplied"

    assert np.asarray(result.objective).shape == ()
    assert np.asarray(result.objective).dtype == np.dtype(dtype)
    assert np.asarray(result.attempted_objective).shape == ()
    assert np.asarray(result.attempted_objective).dtype == np.dtype(dtype)
    assert np.asarray(result.history).dtype == np.dtype(dtype)
    for parameter_field in result.parameters:
        assert np.asarray(parameter_field).dtype == np.dtype(dtype)

    history = np.asarray(result.history)
    n_iter = int(np.asarray(result.n_iter))
    returned_state_is_valid = bool(
        np.isfinite(float(np.asarray(result.objective)))
        and history.shape == (n_iter + 1,)
        and bool(np.all(np.isfinite(history)))
        and history[-1].tobytes() == np.asarray(result.objective).tobytes()
    )
    assert bool(np.asarray(result.objective_valid)) is returned_state_is_valid
    if objective_valid:
        assert returned_state_is_valid
    else:
        assert history.shape == (0,)
        assert n_iter == 0
        assert not np.isfinite(float(np.asarray(result.objective)))
        _assert_params_bits_equal(result.parameters, initial_parameters)

    attempted_objective_is_finite = np.isfinite(
        float(np.asarray(result.attempted_objective))
    )
    assert bool(np.asarray(result.attempted_objective_valid)) is bool(
        attempted_objective_is_finite
    )


def test_host_result_field_order_and_compiled_kernel_schema_are_separate():
    assert identity_control.FitResult._fields == IDENTITY_HOST_RESULT_FIELDS
    assert (
        general_control.GroupedGeneralFitResult._fields
        == GROUPED_HOST_RESULT_FIELDS
    )
    assert (
        identity_control.FixedStepKernelResult._fields
        == FIXED_STEP_KERNEL_FIELDS
    )
    assert "initial_parameters" not in FIXED_STEP_KERNEL_FIELDS
    assert "objective_valid" not in FIXED_STEP_KERNEL_FIELDS
    assert "iteration_limit" not in FIXED_STEP_KERNEL_FIELDS
    assert "tol" not in FIXED_STEP_KERNEL_FIELDS
    assert "decrease_tol" not in FIXED_STEP_KERNEL_FIELDS


def test_general_contract_metadata_and_initialization_type_are_centralized():
    required_metadata_exports = {
        "GENERAL_CONTRACT_ID",
        "GENERAL_CONTRACT_VERSION",
        "GeneralResultMetadata",
        "InitializationProvenance",
        "current_general_result_metadata",
    }
    assert required_metadata_exports <= set(metadata_module.__all__)
    assert (
        general_control.GENERAL_CONTRACT_ID
        == metadata_module.GENERAL_CONTRACT_ID
    )
    assert (
        general_control.GENERAL_CONTRACT_VERSION
        == metadata_module.GENERAL_CONTRACT_VERSION
    )
    assert (
        general_control.GeneralResultMetadata
        is metadata_module.GeneralResultMetadata
    )
    assert (
        general_control.current_general_result_metadata
        is metadata_module.current_general_result_metadata
    )
    assert (
        development.InitializationProvenance
        is metadata_module.InitializationProvenance
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", [FitMode.FIXED_STEPS, FitMode.CONVERGED])
@pytest.mark.parametrize("iteration_limit", [0, 1])
def test_identity_success_and_zero_step_results_retain_serializable_state(
    dtype, mode, iteration_limit
):
    parameters, observations, noise = _ordinary_identity_problem(dtype)
    if mode == FitMode.FIXED_STEPS:
        result = identity_control.fit_fixed_steps(
            parameters,
            observations,
            noise,
            n_steps=iteration_limit,
            factor_jitter=FACTOR_JITTER,
            covariance_ridge=COVARIANCE_RIDGE,
        )
        expected_tol = None
        expected_decrease_tol = None
        assert int(np.asarray(result.status)) == int(
            FitStatus.FIXED_STEPS_COMPLETE
        )
    else:
        result = identity_control.fit_converged(
            parameters,
            observations,
            noise,
            max_iter=iteration_limit,
            tol=CONVERGENCE_TOL,
            decrease_tol=DECREASE_TOL,
            factor_jitter=FACTOR_JITTER,
            covariance_ridge=COVARIANCE_RIDGE,
        )
        expected_tol = CONVERGENCE_TOL
        expected_decrease_tol = DECREASE_TOL
        expected_status = (
            FitStatus.MAX_ITER
            if iteration_limit == 0
            else FitStatus.CONVERGED
        )
        assert int(np.asarray(result.status)) == int(expected_status)

    _assert_host_state(
        result,
        fields=IDENTITY_HOST_RESULT_FIELDS,
        initial_parameters=parameters,
        dtype=dtype,
        iteration_limit=iteration_limit,
        mode=mode,
        objective_valid=True,
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
        tol=expected_tol,
        decrease_tol=expected_decrease_tol,
    )
    assert int(np.asarray(result.n_iter)) == iteration_limit
    if iteration_limit == 0:
        _assert_params_bits_equal(result.parameters, parameters)
    else:
        assert not all(
            np.asarray(actual).tobytes() == np.asarray(initial).tobytes()
            for actual, initial in zip(
                result.parameters, parameters, strict=True
            )
        )
        _assert_params_bits_equal(result.initial_parameters, parameters)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", [FitMode.FIXED_STEPS, FitMode.CONVERGED])
@pytest.mark.parametrize("iteration_limit", [0, 1])
def test_grouped_success_and_zero_step_results_retain_serializable_state(
    dtype, mode, iteration_limit
):
    user_parameters, fit = _ordinary_grouped_problem(dtype)
    if mode == FitMode.FIXED_STEPS:
        result = general_control.fit_fixed_steps_grouped(
            fit, n_steps=iteration_limit
        )
        expected_tol = None
        expected_decrease_tol = None
        assert int(np.asarray(result.status)) == int(
            FitStatus.FIXED_STEPS_COMPLETE
        )
    else:
        result = general_control.fit_converged_grouped(
            fit,
            max_iter=iteration_limit,
            tol=CONVERGENCE_TOL,
            decrease_tol=DECREASE_TOL,
        )
        expected_tol = CONVERGENCE_TOL
        expected_decrease_tol = DECREASE_TOL
        expected_status = (
            FitStatus.MAX_ITER
            if iteration_limit == 0
            else FitStatus.CONVERGED
        )
        assert int(np.asarray(result.status)) == int(expected_status)

    _assert_host_state(
        result,
        fields=GROUPED_HOST_RESULT_FIELDS,
        initial_parameters=user_parameters,
        dtype=dtype,
        iteration_limit=iteration_limit,
        mode=mode,
        objective_valid=True,
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
        tol=expected_tol,
        decrease_tol=expected_decrease_tol,
    )
    assert int(np.asarray(result.n_iter)) == iteration_limit
    if iteration_limit == 0:
        _assert_params_bits_equal(result.parameters, user_parameters)
    else:
        assert not all(
            np.asarray(actual).tobytes() == np.asarray(initial).tobytes()
            for actual, initial in zip(
                result.parameters, user_parameters, strict=True
            )
        )
        _assert_params_bits_equal(result.initial_parameters, user_parameters)


@pytest.mark.parametrize("family", ["identity", "grouped"])
def test_component_collapse_retains_initial_custody_and_exact_rollback(family):
    if family == "identity":
        parameters, observations, noise = _collapse_identity_problem()
        result = identity_control.fit_fixed_steps(
            parameters, observations, noise, n_steps=3
        )
        fields = IDENTITY_HOST_RESULT_FIELDS
    else:
        parameters, fit = _collapse_grouped_problem()
        result = general_control.fit_fixed_steps_grouped(fit, n_steps=3)
        fields = GROUPED_HOST_RESULT_FIELDS

    _assert_host_state(
        result,
        fields=fields,
        initial_parameters=parameters,
        dtype=jnp.float64,
        iteration_limit=3,
        mode=FitMode.FIXED_STEPS,
        objective_valid=True,
        factor_jitter=0.0,
        covariance_ridge=0.0,
        tol=None,
        decrease_tol=None,
    )
    assert int(np.asarray(result.status)) == int(FitStatus.COMPONENT_COLLAPSED)
    assert bool(np.asarray(result.collapsed))
    assert not bool(np.asarray(result.numerical_failure))
    assert np.any(np.asarray(result.collapsed_components))
    assert int(np.asarray(result.n_iter)) == 0
    _assert_params_bits_equal(result.parameters, parameters)


def test_identity_candidate_numerical_failure_keeps_last_valid_objective(
    monkeypatch,
):
    parameters, observations, noise = _ordinary_identity_problem(jnp.float64)
    candidate = Params(
        weights=parameters.weights,
        means=parameters.means + 1.0,
        covariances=parameters.covariances,
    )
    objective_results = iter(
        [
            (jnp.asarray(-10.0), jnp.asarray(False)),
            (jnp.asarray(jnp.nan), jnp.asarray(True)),
        ]
    )
    monkeypatch.setattr(
        identity_control,
        "_objective_and_failure",
        lambda *_args, **_kwargs: next(objective_results),
    )
    monkeypatch.setattr(
        identity_control,
        "em_step",
        lambda *_args, **_kwargs: SimpleNamespace(
            parameters=candidate,
            numerical_failure=jnp.asarray(False),
            collapsed=jnp.asarray(False),
            collapsed_components=jnp.asarray([False]),
        ),
    )

    result = identity_control.fit_converged(
        parameters,
        observations,
        noise,
        max_iter=2,
        tol=CONVERGENCE_TOL,
        decrease_tol=DECREASE_TOL,
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
    )

    _assert_host_state(
        result,
        fields=IDENTITY_HOST_RESULT_FIELDS,
        initial_parameters=parameters,
        dtype=jnp.float64,
        iteration_limit=2,
        mode=FitMode.CONVERGED,
        objective_valid=True,
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
        tol=CONVERGENCE_TOL,
        decrease_tol=DECREASE_TOL,
    )
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_params_bits_equal(result.parameters, parameters)
    _assert_array_bits_equal(result.history, np.asarray([-10.0]))


def test_grouped_candidate_numerical_failure_keeps_last_valid_objective(
    monkeypatch,
):
    parameters, fit = _ordinary_grouped_problem(jnp.float64)
    initial_objective = general_control._objective_for_parameters(
        fit, fit.grouped.parameters
    )[0]
    group_failure = np.zeros(len(fit.grouped.groups), dtype=bool)
    group_failure[0] = True
    failed_pairs = np.zeros(
        (fit.grouped.n_samples, parameters.weights.shape[0]), dtype=bool
    )
    failed_pairs[0, 0] = True
    failed_step = SimpleNamespace(
        parameters=parameters,
        objective=initial_objective,
        attempted_objective=jnp.asarray(jnp.nan, dtype=jnp.float64),
        attempted_objective_valid=jnp.asarray(False),
        status=jnp.asarray(
            int(GroupedStepStatus.NUMERICAL_FAILURE), dtype=jnp.int32
        ),
        failure_stage=jnp.asarray(
            int(GroupedFailureStage.CANDIDATE_OBJECTIVE), dtype=jnp.int32
        ),
        numerical_failure=jnp.asarray(True),
        collapsed=jnp.asarray(False),
        collapsed_components=jnp.zeros_like(parameters.weights, dtype=bool),
        candidate_group_numerical_failure=jnp.asarray(group_failure),
        candidate_failed_pairs=jnp.asarray(failed_pairs),
        statistics=SimpleNamespace(
            group_numerical_failure=jnp.zeros_like(
                jnp.asarray(group_failure)
            ),
            failed_pairs=jnp.zeros_like(jnp.asarray(failed_pairs)),
        ),
    )
    monkeypatch.setattr(
        general_control, "one_em_step_grouped", lambda _fit: failed_step
    )

    result = general_control.fit_converged_grouped(
        fit,
        max_iter=2,
        tol=CONVERGENCE_TOL,
        decrease_tol=DECREASE_TOL,
    )

    _assert_host_state(
        result,
        fields=GROUPED_HOST_RESULT_FIELDS,
        initial_parameters=parameters,
        dtype=jnp.float64,
        iteration_limit=2,
        mode=FitMode.CONVERGED,
        objective_valid=True,
        factor_jitter=FACTOR_JITTER,
        covariance_ridge=COVARIANCE_RIDGE,
        tol=CONVERGENCE_TOL,
        decrease_tol=DECREASE_TOL,
    )
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_params_bits_equal(result.parameters, parameters)
    _assert_array_bits_equal(result.history[0], initial_objective)
    np.testing.assert_array_equal(result.group_numerical_failure, group_failure)
    np.testing.assert_array_equal(result.failed_pairs, failed_pairs)


@pytest.mark.parametrize("mode", [FitMode.FIXED_STEPS, FitMode.CONVERGED])
def test_identity_invalid_initial_objective_has_empty_host_history(mode):
    parameters, observations, noise = _invalid_identity_initial_objective()
    if mode == FitMode.FIXED_STEPS:
        result = identity_control.fit_fixed_steps(
            parameters, observations, noise, n_steps=2
        )
        expected_tol = None
        expected_decrease_tol = None
    else:
        result = identity_control.fit_converged(
            parameters,
            observations,
            noise,
            max_iter=2,
            tol=CONVERGENCE_TOL,
            decrease_tol=DECREASE_TOL,
        )
        expected_tol = CONVERGENCE_TOL
        expected_decrease_tol = DECREASE_TOL

    _assert_host_state(
        result,
        fields=IDENTITY_HOST_RESULT_FIELDS,
        initial_parameters=parameters,
        dtype=jnp.float64,
        iteration_limit=2,
        mode=mode,
        objective_valid=False,
        factor_jitter=0.0,
        covariance_ridge=0.0,
        tol=expected_tol,
        decrease_tol=expected_decrease_tol,
    )
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not bool(np.asarray(result.attempted_objective_valid))


@pytest.mark.parametrize("mode", [FitMode.FIXED_STEPS, FitMode.CONVERGED])
def test_grouped_invalid_initial_objective_has_empty_host_history(mode):
    parameters, fit = _invalid_grouped_initial_objective()
    if mode == FitMode.FIXED_STEPS:
        result = general_control.fit_fixed_steps_grouped(fit, n_steps=2)
        expected_tol = None
        expected_decrease_tol = None
    else:
        result = general_control.fit_converged_grouped(
            fit,
            max_iter=2,
            tol=CONVERGENCE_TOL,
            decrease_tol=DECREASE_TOL,
        )
        expected_tol = CONVERGENCE_TOL
        expected_decrease_tol = DECREASE_TOL

    _assert_host_state(
        result,
        fields=GROUPED_HOST_RESULT_FIELDS,
        initial_parameters=parameters,
        dtype=jnp.float64,
        iteration_limit=2,
        mode=mode,
        objective_valid=False,
        factor_jitter=0.0,
        covariance_ridge=0.0,
        tol=expected_tol,
        decrease_tol=expected_decrease_tol,
    )
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not bool(np.asarray(result.attempted_objective_valid))


def test_compiled_fixed_step_buffer_is_unchanged_for_invalid_initial_objective():
    parameters, observations, noise = _invalid_identity_initial_objective()
    result = identity_control.fit_fixed_steps_kernel(
        parameters, observations, noise, n_steps=2
    )

    assert result._fields == FIXED_STEP_KERNEL_FIELDS
    assert np.asarray(result.history_buffer).shape == (3,)
    assert int(np.asarray(result.history_length)) == 1
    assert int(np.asarray(result.n_iter)) == 0
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert bool(np.asarray(result.numerical_failure))
    _assert_params_bits_equal(result.parameters, parameters)
