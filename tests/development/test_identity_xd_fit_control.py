"""Red tests for converged and fixed-step identity-XD fit control.

The numerical expectations in this file come from the independent, explicit
NumPy oracle in :mod:`tests.reference.identity_xd`.  These tests deliberately
target the temporary ``development.fit_control`` module so the fit semantics
can be settled before a public package namespace is chosen.
"""

from __future__ import annotations

from enum import IntEnum
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development.fit_control as fit_control_module
from development.fit_control import (
    FitMode,
    FitStatus,
    classify_objective_change,
    fit_converged,
    fit_fixed_steps,
    mean_log_likelihood,
)
from development.identity_xd import Params, em_step
from tests.reference.identity_xd import (
    identity_e_step as reference_e_step,
    identity_em_step as reference_em_step,
)


DTYPE_CASES = (
    pytest.param(jnp.float64, 5e-10, 5e-12, 5e-10, id="float64"),
    pytest.param(jnp.float32, 1e-4, 1e-5, 2e-4, id="float32"),
)

REQUIRED_RESULT_FIELDS = {
    "parameters",
    "objective",
    "history",
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
}

REQUIRED_KERNEL_RESULT_FIELDS = (REQUIRED_RESULT_FIELDS - {"history"}) | {
    "history_buffer",
    "history_length",
}


def _problem(dtype):
    """Return one deterministic, noncollapsed ``N=40, K=2, D=2`` fixture."""

    observations = jnp.asarray(
        [
            [-2.20, -0.90],
            [-2.00, -1.30],
            [-1.90, -0.60],
            [-1.80, -1.00],
            [-1.70, -0.30],
            [-1.60, -1.20],
            [-1.50, -0.70],
            [-1.40, -1.50],
            [-1.30, -0.40],
            [-1.20, -0.90],
            [-1.10, -0.20],
            [-1.00, -1.10],
            [-0.90, -0.50],
            [-0.80, -1.30],
            [-0.70, -0.10],
            [-0.60, -0.80],
            [-0.50, 0.00],
            [-0.40, -0.60],
            [-0.20, 0.10],
            [0.00, -0.30],
            [0.10, 0.40],
            [0.30, 0.80],
            [0.50, 0.20],
            [0.60, 1.00],
            [0.70, 0.50],
            [0.80, 1.30],
            [0.90, 0.70],
            [1.00, 1.50],
            [1.10, 0.30],
            [1.20, 1.10],
            [1.30, 0.60],
            [1.40, 1.40],
            [1.50, 0.80],
            [1.60, 1.70],
            [1.70, 0.40],
            [1.80, 1.20],
            [1.90, 0.90],
            [2.00, 1.60],
            [2.10, 0.70],
            [2.30, 1.30],
        ],
        dtype=dtype,
    )
    noise_patterns = jnp.asarray(
        [
            [[0.10, 0.02], [0.02, 0.14]],
            [[0.16, -0.03], [-0.03, 0.11]],
            [[0.08, 0.01], [0.01, 0.18]],
            [[0.13, 0.04], [0.04, 0.17]],
        ],
        dtype=dtype,
    )
    measurement_covariances = jnp.tile(noise_patterns, (10, 1, 1))
    parameters = Params(
        weights=jnp.asarray([0.62, 0.38], dtype=dtype),
        means=jnp.asarray([[-0.55, -0.15], [0.55, 0.35]], dtype=dtype),
        covariances=jnp.asarray(
            [
                [[1.55, 0.22], [0.22, 0.95]],
                [[1.20, -0.14], [-0.14, 1.35]],
            ],
            dtype=dtype,
        ),
    )
    return parameters, observations, measurement_covariances


def _collapse_problem(dtype=jnp.float64):
    """Return a valid fixture whose second component has exactly zero mass."""

    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray(
            [[0.0, 0.0], [1_000_000.0, 1_000_000.0]], dtype=dtype
        ),
        covariances=jnp.asarray([np.eye(2), np.eye(2)], dtype=dtype),
    )
    observations = jnp.asarray(
        [[-1.0, -0.5], [-0.3, 0.7], [0.4, -0.6], [1.1, 0.5]],
        dtype=dtype,
    )
    measurement_covariances = jnp.broadcast_to(
        jnp.eye(2, dtype=dtype) * jnp.asarray(0.1, dtype=dtype),
        (4, 2, 2),
    )
    return parameters, observations, measurement_covariances


def _tagged_parameters(tag: float) -> Params:
    """Return a one-component state whose mean identifies a test candidate."""

    return Params(
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        means=jnp.asarray([[tag]], dtype=jnp.float64),
        covariances=jnp.asarray([[[1.0]]], dtype=jnp.float64),
    )


def _reference_objective(params, observations, measurement_covariances) -> float:
    e_step = reference_e_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(measurement_covariances, dtype=np.float64),
        np.asarray(params.weights, dtype=np.float64),
        np.asarray(params.means, dtype=np.float64),
        np.asarray(params.covariances, dtype=np.float64),
    )
    return float(np.mean(e_step.score_samples))


def _reference_trajectory(initial, observations, measurement_covariances, n_steps):
    current = initial
    objectives = [_reference_objective(current, observations, measurement_covariances)]
    for _ in range(n_steps):
        next_parameters, _, _ = reference_em_step(
            np.asarray(observations, dtype=np.float64),
            np.asarray(measurement_covariances, dtype=np.float64),
            np.asarray(current.weights, dtype=np.float64),
            np.asarray(current.means, dtype=np.float64),
            np.asarray(current.covariances, dtype=np.float64),
        )
        current = Params(
            weights=next_parameters.weights,
            means=next_parameters.means,
            covariances=next_parameters.covariances,
        )
        objectives.append(
            _reference_objective(current, observations, measurement_covariances)
        )
    return current, np.asarray(objectives)


def _assert_params_close(actual, expected, *, rtol, atol) -> None:
    for actual_field, expected_field in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_field),
            np.asarray(expected_field),
            rtol=rtol,
            atol=atol,
        )


def _assert_params_exact(actual, expected) -> None:
    for actual_field, expected_field in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_field), np.asarray(expected_field)
        )


def _assert_result_schema(result) -> None:
    assert REQUIRED_RESULT_FIELDS.issubset(result._fields)


def _fixed_step_kernel_function():
    assert hasattr(fit_control_module, "FixedStepKernelResult")
    assert hasattr(fit_control_module, "fit_fixed_steps_kernel")
    return fit_control_module.fit_fixed_steps_kernel


def _assert_kernel_result_schema(result) -> None:
    assert REQUIRED_KERNEL_RESULT_FIELDS.issubset(result._fields)
    assert "history" not in result._fields


def _assert_collapsed_components(result, expected) -> None:
    np.testing.assert_array_equal(
        np.asarray(result.collapsed_components), np.asarray(expected)
    )


def _as_bool(value) -> bool:
    return bool(np.asarray(value))


def _as_int(value) -> int:
    return int(np.asarray(value))


def _assert_status(actual, expected: IntEnum) -> None:
    assert _as_int(actual) == int(expected)


def test_fit_control_enums_have_stable_named_states():
    """The temporary result uses integer states that survive JAX transforms."""

    assert issubclass(FitStatus, IntEnum)
    assert issubclass(FitMode, IntEnum)
    assert [member.name for member in FitStatus] == [
        "CONTINUE",
        "CONVERGED",
        "MAX_ITER",
        "OBJECTIVE_DECREASED",
        "NUMERICAL_FAILURE",
        "COMPONENT_COLLAPSED",
        "FIXED_STEPS_COMPLETE",
    ]
    assert [member.name for member in FitMode] == ["CONVERGED", "FIXED_STEPS"]


@pytest.mark.parametrize(
    "current,expected_change,expected_status,expected_accept,expected_converged",
    [
        pytest.param(
            -10.0 - 1.1e-9,
            -1.1e-10,
            "OBJECTIVE_DECREASED",
            False,
            False,
            id="material-decrease",
        ),
        pytest.param(
            -10.0 - 1.0e-9,
            -1.0e-10,
            "CONVERGED",
            True,
            True,
            id="decrease-boundary",
        ),
        pytest.param(
            -9.99999,
            1.0e-6,
            "CONVERGED",
            True,
            True,
            id="tolerance-boundary",
        ),
        pytest.param(
            -9.999989,
            1.1e-6,
            "CONTINUE",
            True,
            False,
            id="above-tolerance",
        ),
    ],
)
def test_xd_ip_conv_002_exact_objective_threshold_classification(
    current,
    expected_change,
    expected_status,
    expected_accept,
    expected_converged,
):
    """XD-IP-CONV-002: signed normalized changes use both thresholds."""

    result = classify_objective_change(
        previous=-10.0,
        current=current,
        tol=1e-6,
        decrease_tol=1e-10,
    )

    assert result._fields == (
        "normalized_change",
        "status",
        "accept",
        "converged",
    )
    raw_expected_change = (
        np.float64(current) - np.float64(-10.0)
    ) / np.float64(10.0)
    # The decimal values name the conceptual boundary; binary subtraction is
    # allowed its ordinary representation error, but the returned diagnostic
    # must be the raw computed change rather than a snapped display value.
    np.testing.assert_allclose(
        raw_expected_change, expected_change, rtol=2e-7, atol=1e-18
    )
    np.testing.assert_array_max_ulp(
        np.asarray(result.normalized_change), raw_expected_change, maxulp=2
    )
    _assert_status(result.status, FitStatus[expected_status])
    assert _as_bool(result.accept) is expected_accept
    assert _as_bool(result.converged) is expected_converged


@pytest.mark.parametrize(
    "direction,expected_status,expected_accept",
    [
        pytest.param(np.inf, FitStatus.CONTINUE, True, id="positive-nextafter"),
        pytest.param(
            -np.inf,
            FitStatus.OBJECTIVE_DECREASED,
            False,
            id="negative-nextafter",
        ),
    ],
)
def test_xd_ip_conv_002_zero_thresholds_have_no_general_comparison_slack(
    direction, expected_status, expected_accept
):
    previous = np.float64(-10.0)
    current = np.nextafter(previous, np.float64(direction))
    expected_change = (current - previous) / max(1.0, abs(previous))

    result = classify_objective_change(
        previous=previous,
        current=current,
        tol=0.0,
        decrease_tol=0.0,
    )

    np.testing.assert_array_max_ulp(
        np.asarray(result.normalized_change), expected_change, maxulp=2
    )
    _assert_status(result.status, expected_status)
    assert _as_bool(result.accept) is expected_accept
    assert not _as_bool(result.converged)


@pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
def test_xd_ip_conv_002_nonfinite_candidate_is_a_rejected_numerical_failure(
    current,
):
    result = classify_objective_change(
        previous=-10.0,
        current=current,
        tol=1e-6,
        decrease_tol=1e-10,
    )

    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    assert not _as_bool(result.accept)
    assert not _as_bool(result.converged)


@pytest.mark.parametrize(
    "tol,decrease_tol",
    [
        pytest.param(-1.0, 1e-10, id="negative-tol"),
        pytest.param(np.inf, 1e-10, id="infinite-tol"),
        pytest.param(np.nan, 1e-10, id="nan-tol"),
        pytest.param(1e-6, -1.0, id="negative-decrease-tol"),
        pytest.param(1e-6, np.inf, id="infinite-decrease-tol"),
        pytest.param(1e-6, np.nan, id="nan-decrease-tol"),
    ],
)
def test_xd_ip_conv_002_invalid_tolerances_fail_validation(tol, decrease_tol):
    with pytest.raises((TypeError, ValueError)):
        classify_objective_change(
            previous=-10.0,
            current=-9.0,
            tol=tol,
            decrease_tol=decrease_tol,
        )


@pytest.mark.parametrize("field", ["tol", "decrease_tol"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.float64(1e300), id="finite-source-overflow"),
        pytest.param(np.float64(1e-300), id="positive-source-underflow"),
    ],
)
def test_converged_tolerances_must_survive_selected_float32_conversion(
    monkeypatch, field, value
):
    parameters, observations, measurement_covariances = _problem(jnp.float32)
    calls = 0

    def must_not_step(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("invalid tolerance must fail before fitting")

    monkeypatch.setattr(fit_control_module, "em_step", must_not_step)
    arguments = {"tol": 1e-6, "decrease_tol": 1e-10, field: value}
    with pytest.raises(
        (ValueError, OverflowError),
        match=rf"{field}|float32|finite|zero|underflow",
    ):
        fit_converged(
            parameters,
            observations,
            measurement_covariances,
            max_iter=1,
            **arguments,
        )
    assert calls == 0


@pytest.mark.parametrize("dtype,_rtol,_atol,log_rtol", DTYPE_CASES)
def test_mean_log_likelihood_matches_independent_numpy_oracle(
    dtype, _rtol, _atol, log_rtol
):
    parameters, observations, measurement_covariances = _problem(dtype)
    expected = _reference_objective(
        parameters, observations, measurement_covariances
    )

    eager = mean_log_likelihood(parameters, observations, measurement_covariances)
    compiled = jax.jit(mean_log_likelihood)(
        parameters, observations, measurement_covariances
    )

    assert np.asarray(eager).shape == ()
    log_atol = 5e-10 if dtype == jnp.float64 else 2e-5
    np.testing.assert_allclose(
        np.asarray(eager), expected, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(compiled),
        np.asarray(eager),
        rtol=log_rtol,
        atol=log_atol,
    )


@pytest.mark.parametrize("dtype,_rtol,_atol,log_rtol", DTYPE_CASES)
def test_xd_ip_conv_003_zero_iterations_returns_exact_initialized_state(
    dtype, _rtol, _atol, log_rtol
):
    """XD-IP-CONV-003: max_iter=0 evaluates but never modifies theta(0)."""

    parameters, observations, measurement_covariances = _problem(dtype)
    expected_objective = _reference_objective(
        parameters, observations, measurement_covariances
    )

    result = fit_converged(
        parameters,
        observations,
        measurement_covariances,
        max_iter=0,
        tol=1e-6,
        decrease_tol=1e-10,
    )

    _assert_result_schema(result)
    _assert_params_exact(result.parameters, parameters)
    assert _as_int(result.n_iter) == 0
    assert not _as_bool(result.converged)
    _assert_status(result.status, FitStatus.MAX_ITER)
    _assert_status(result.mode, FitMode.CONVERGED)
    assert np.asarray(result.history).shape == (1,)
    log_atol = 5e-10 if dtype == jnp.float64 else 2e-5
    np.testing.assert_allclose(
        np.asarray(result.objective),
        expected_objective,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.history),
        [expected_objective],
        rtol=log_rtol,
        atol=log_atol,
    )
    assert _as_int(result.attempted_iteration) == 0
    assert not _as_bool(result.attempted_objective_valid)
    assert not _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False, False])


def test_xd_ip_conv_001_huge_tolerance_commits_only_the_first_update():
    """XD-IP-CONV-001: convergence stops mutation at the accepted candidate."""

    dtype = jnp.float64
    parameters, observations, measurement_covariances = _problem(dtype)
    direct_step = em_step(parameters, observations, measurement_covariances)
    assert not _as_bool(direct_step.numerical_failure)
    assert not _as_bool(direct_step.collapsed)
    expected_objective = _reference_objective(
        direct_step.parameters, observations, measurement_covariances
    )

    result = fit_converged(
        parameters,
        observations,
        measurement_covariances,
        max_iter=5,
        tol=1e6,
        decrease_tol=1e-10,
    )
    forced_five = fit_fixed_steps(
        parameters,
        observations,
        measurement_covariances,
        n_steps=5,
    )

    _assert_result_schema(result)
    assert _as_int(result.n_iter) == 1
    assert _as_bool(result.converged)
    _assert_status(result.status, FitStatus.CONVERGED)
    _assert_status(result.mode, FitMode.CONVERGED)
    assert np.asarray(result.history).shape == (2,)
    _assert_params_close(
        result.parameters, direct_step.parameters, rtol=5e-10, atol=5e-12
    )
    np.testing.assert_allclose(
        np.asarray(result.objective), expected_objective, rtol=5e-10, atol=5e-10
    )
    np.testing.assert_allclose(
        np.asarray(result.history)[-1], np.asarray(result.objective), rtol=0.0, atol=0.0
    )
    assert _as_int(result.attempted_iteration) == 1
    assert _as_bool(result.attempted_objective_valid)
    np.testing.assert_allclose(
        np.asarray(result.attempted_objective),
        np.asarray(result.objective),
        rtol=0.0,
        atol=0.0,
    )
    assert not _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False, False])

    assert _as_int(forced_five.n_iter) == 5
    assert not np.allclose(
        np.asarray(result.parameters.means),
        np.asarray(forced_five.parameters.means),
        rtol=1e-7,
        atol=1e-8,
    )


@pytest.mark.parametrize("dtype,rtol,atol,log_rtol", DTYPE_CASES)
@pytest.mark.parametrize("n_steps", [0, 1, 5])
def test_xd_ip_fixed_001_executes_exact_requested_updates(
    dtype, rtol, atol, log_rtol, n_steps
):
    """XD-IP-FIXED-001: result state, count, objective, and history agree."""

    parameters, observations, measurement_covariances = _problem(dtype)
    reference_parameters, reference_history = _reference_trajectory(
        parameters, observations, measurement_covariances, n_steps
    )

    result = fit_fixed_steps(
        parameters,
        observations,
        measurement_covariances,
        n_steps=n_steps,
    )

    _assert_result_schema(result)
    _assert_params_close(
        result.parameters, reference_parameters, rtol=rtol, atol=atol
    )
    assert _as_int(result.n_iter) == n_steps
    assert not _as_bool(result.converged)
    _assert_status(result.status, FitStatus.FIXED_STEPS_COMPLETE)
    _assert_status(result.mode, FitMode.FIXED_STEPS)
    assert np.asarray(result.history).shape == (n_steps + 1,)
    np.testing.assert_allclose(
        np.asarray(result.history),
        reference_history,
        rtol=log_rtol,
        atol=5e-10 if dtype == jnp.float64 else 2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.objective),
        reference_history[-1],
        rtol=log_rtol,
        atol=5e-10 if dtype == jnp.float64 else 2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(result.history)[-1], np.asarray(result.objective), rtol=0.0, atol=0.0
    )
    assert _as_int(result.attempted_iteration) == n_steps
    assert _as_bool(result.attempted_objective_valid) is (n_steps > 0)
    if n_steps > 0:
        np.testing.assert_allclose(
            np.asarray(result.attempted_objective),
            np.asarray(result.objective),
            rtol=0.0,
            atol=0.0,
        )
    assert not _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False, False])

    if n_steps == 0:
        _assert_params_exact(result.parameters, parameters)


@pytest.mark.parametrize("dtype,rtol,atol,log_rtol", DTYPE_CASES)
def test_xd_ip_fixed_001_is_jittable_when_n_steps_is_static(
    dtype, rtol, atol, log_rtol
):
    parameters, observations, measurement_covariances = _problem(dtype)
    kernel_function = _fixed_step_kernel_function()
    eager = kernel_function(
        parameters, observations, measurement_covariances, n_steps=5
    )
    compiled = jax.jit(kernel_function, static_argnames=("n_steps",))(
        parameters,
        observations,
        measurement_covariances,
        n_steps=5,
    )

    _assert_kernel_result_schema(eager)
    _assert_kernel_result_schema(compiled)
    _assert_params_close(compiled.parameters, eager.parameters, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        np.asarray(compiled.history_buffer),
        np.asarray(eager.history_buffer),
        rtol=log_rtol,
        atol=5e-10 if dtype == jnp.float64 else 2e-5,
    )
    np.testing.assert_allclose(
        np.asarray(compiled.objective),
        np.asarray(eager.objective),
        rtol=log_rtol,
        atol=5e-10 if dtype == jnp.float64 else 2e-5,
    )
    for field in (
        "n_iter",
        "converged",
        "status",
        "mode",
        "attempted_iteration",
        "attempted_objective_valid",
        "numerical_failure",
        "collapsed",
        "collapsed_components",
        "history_length",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(compiled, field)), np.asarray(getattr(eager, field))
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_collapse_001_fixed_kernel_freezes_buffer_and_reports_component(
    compiled,
):
    """A first-attempt collapse retains only the initialized state logically."""

    parameters, observations, measurement_covariances = _collapse_problem()
    kernel_function = _fixed_step_kernel_function()
    run_kernel = (
        jax.jit(kernel_function, static_argnames=("n_steps",))
        if compiled
        else kernel_function
    )
    result = run_kernel(
        parameters,
        observations,
        measurement_covariances,
        n_steps=4,
    )

    _assert_kernel_result_schema(result)
    _assert_params_exact(result.parameters, parameters)
    assert np.asarray(result.history_buffer).shape == (5,)
    assert _as_int(result.history_length) == 1
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.COMPONENT_COLLAPSED)
    _assert_status(result.mode, FitMode.FIXED_STEPS)
    assert not _as_bool(result.numerical_failure)
    assert _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False, True])
    assert _as_int(result.attempted_iteration) == 1
    assert not _as_bool(result.attempted_objective_valid)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_xd_ip_collapse_001_host_results_trim_history_and_roll_back(mode):
    parameters, observations, measurement_covariances = _collapse_problem()
    if mode == "fixed":
        result = fit_fixed_steps(
            parameters,
            observations,
            measurement_covariances,
            n_steps=4,
        )
        expected_mode = FitMode.FIXED_STEPS
    else:
        result = fit_converged(
            parameters,
            observations,
            measurement_covariances,
            max_iter=4,
            tol=0.0,
            decrease_tol=0.0,
        )
        expected_mode = FitMode.CONVERGED

    _assert_result_schema(result)
    _assert_params_exact(result.parameters, parameters)
    assert np.asarray(result.history).shape == (1,)
    np.testing.assert_array_equal(
        np.asarray(result.history)[0], np.asarray(result.objective)
    )
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.COMPONENT_COLLAPSED)
    _assert_status(result.mode, expected_mode)
    assert not _as_bool(result.numerical_failure)
    assert _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False, True])
    assert _as_int(result.attempted_iteration) == 1
    assert not _as_bool(result.attempted_objective_valid)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_ridge_001_negative_ridge_fails_fixed_kernel_at_zero_steps(
    compiled,
):
    parameters = _tagged_parameters(0.0)
    observations = jnp.asarray([[0.1]], dtype=jnp.float64)
    measurement_covariances = jnp.asarray([[[0.1]]], dtype=jnp.float64)
    kernel_function = _fixed_step_kernel_function()
    run_kernel = (
        jax.jit(kernel_function, static_argnames=("n_steps",))
        if compiled
        else kernel_function
    )
    result = run_kernel(
        parameters,
        observations,
        measurement_covariances,
        n_steps=0,
        covariance_ridge=-1e-3,
    )

    _assert_kernel_result_schema(result)
    _assert_params_exact(result.parameters, parameters)
    assert np.asarray(result.history_buffer).shape == (1,)
    assert _as_int(result.history_length) == 1
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    assert _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False])


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_xd_ip_ridge_001_negative_ridge_fails_host_fit_at_zero_steps(mode):
    parameters = _tagged_parameters(0.0)
    observations = jnp.asarray([[0.1]], dtype=jnp.float64)
    measurement_covariances = jnp.asarray([[[0.1]]], dtype=jnp.float64)
    if mode == "fixed":
        result = fit_fixed_steps(
            parameters,
            observations,
            measurement_covariances,
            n_steps=0,
            covariance_ridge=-1e-3,
        )
    else:
        result = fit_converged(
            parameters,
            observations,
            measurement_covariances,
            max_iter=0,
            covariance_ridge=-1e-3,
        )

    _assert_result_schema(result)
    _assert_params_exact(result.parameters, parameters)
    assert np.asarray(result.history).shape == (1,)
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    assert _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    _assert_collapsed_components(result, [False])


@pytest.mark.parametrize("mode", ["fixed", "converged"])
@pytest.mark.parametrize(
    "invalid_ridge",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.0 + 0.0j, id="complex"),
        pytest.param(jnp.asarray([0.0]), id="rank-one"),
    ],
)
def test_xd_ip_config_001_static_ridge_error_is_not_masked_by_bad_jitter(
    mode, invalid_ridge
):
    """Both host controls retain static validation precedence.

    A value-domain error in the first control must not turn a type/shape error
    in the second control into an ordinary numerical-failure result.
    """

    parameters, observations, measurement_covariances = _problem(jnp.float64)
    fit = fit_fixed_steps if mode == "fixed" else fit_converged
    count_argument = {"n_steps": 0} if mode == "fixed" else {"max_iter": 0}

    with pytest.raises((TypeError, ValueError)):
        fit(
            parameters,
            observations,
            measurement_covariances,
            factor_jitter=-1.0,
            covariance_ridge=invalid_ridge,
            **count_argument,
        )


def test_xd_ip_conv_004_candidate_failure_sanitizes_finite_attempt(monkeypatch):
    """A failed finite candidate becomes the host invalid-attempt sentinel."""

    theta0 = _tagged_parameters(0.0)
    theta1 = _tagged_parameters(1.0)
    objective_results = iter(
        [
            (jnp.asarray(-10.0), jnp.asarray(False)),
            (jnp.asarray(-9.0), jnp.asarray(True)),
        ]
    )

    monkeypatch.setattr(
        fit_control_module,
        "_objective_and_failure",
        lambda *_args, **_kwargs: next(objective_results),
    )
    monkeypatch.setattr(
        fit_control_module,
        "em_step",
        lambda *_args, **_kwargs: SimpleNamespace(
            parameters=theta1,
            numerical_failure=jnp.asarray(False),
            collapsed=jnp.asarray(False),
            collapsed_components=jnp.asarray([False]),
        ),
    )
    result = fit_converged(
        theta0,
        jnp.asarray([[0.0]]),
        jnp.asarray([[[0.0]]]),
        max_iter=1,
        tol=0.0,
        decrease_tol=0.0,
    )

    _assert_params_exact(result.parameters, theta0)
    np.testing.assert_array_equal(np.asarray(result.history), [-10.0])
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    assert _as_bool(result.numerical_failure)
    assert not _as_bool(result.attempted_objective_valid)
    assert not np.isfinite(float(np.asarray(result.attempted_objective)))
    _assert_collapsed_components(result, [False])


def test_xd_ip_conv_004_objective_decrease_harness_rolls_back_theta0(monkeypatch):
    """A tagged decreasing theta1 is attempted but never accepted."""

    theta0 = _tagged_parameters(0.0)
    theta1 = _tagged_parameters(1.0)
    objective_results = iter(
        [
            (jnp.asarray(-10.0), jnp.asarray(False)),
            (jnp.asarray(-11.0), jnp.asarray(False)),
        ]
    )
    monkeypatch.setattr(
        fit_control_module,
        "_objective_and_failure",
        lambda *_args, **_kwargs: next(objective_results),
    )
    monkeypatch.setattr(
        fit_control_module,
        "em_step",
        lambda *_args, **_kwargs: SimpleNamespace(
            parameters=theta1,
            numerical_failure=jnp.asarray(False),
            collapsed=jnp.asarray(False),
            collapsed_components=jnp.asarray([False]),
        ),
    )
    result = fit_converged(
        theta0,
        jnp.asarray([[0.0]]),
        jnp.asarray([[[0.0]]]),
        max_iter=1,
        tol=1e-6,
        decrease_tol=1e-10,
    )

    _assert_result_schema(result)
    _assert_params_exact(result.parameters, theta0)
    np.testing.assert_array_equal(np.asarray(result.history), [-10.0])
    assert _as_int(result.n_iter) == 0
    _assert_status(result.status, FitStatus.OBJECTIVE_DECREASED)
    assert not _as_bool(result.numerical_failure)
    assert not _as_bool(result.collapsed)
    assert _as_int(result.attempted_iteration) == 1
    assert _as_bool(result.attempted_objective_valid)
    np.testing.assert_array_equal(np.asarray(result.attempted_objective), -11.0)
    _assert_collapsed_components(result, [False])


@pytest.mark.parametrize(
    "invalid_count",
    [
        -1,
        1.5,
        pytest.param(True, id="python-boolean"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(jnp.asarray(True), id="jax-boolean-scalar"),
    ],
)
def test_xd_ip_conv_004_invalid_max_iter_fails_validation(invalid_count):
    parameters, observations, measurement_covariances = _problem(jnp.float64)

    with pytest.raises((TypeError, ValueError)):
        fit_converged(
            parameters,
            observations,
            measurement_covariances,
            max_iter=invalid_count,
            tol=1e-6,
            decrease_tol=1e-10,
        )


@pytest.mark.parametrize(
    "invalid_count",
    [
        -1,
        1.5,
        pytest.param(True, id="python-boolean"),
        pytest.param(np.bool_(True), id="numpy-boolean"),
        pytest.param(jnp.asarray(True), id="jax-boolean-scalar"),
    ],
)
def test_xd_ip_conv_004_invalid_n_steps_fails_validation(invalid_count):
    parameters, observations, measurement_covariances = _problem(jnp.float64)

    with pytest.raises((TypeError, ValueError)):
        fit_fixed_steps(
            parameters,
            observations,
            measurement_covariances,
            n_steps=invalid_count,
        )
