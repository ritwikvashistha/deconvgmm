"""Red tests for control values lost during selected-dtype conversion."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import (
    FitStatus,
    fit_converged,
    fit_fixed_steps,
    fit_fixed_steps_kernel,
)
from development.identity_xd import Params, em_step


CONTROL_CASES = (
    pytest.param("factor_jitter", -1e-50, id="negative-jitter"),
    pytest.param("covariance_ridge", -1e-50, id="negative-ridge"),
    pytest.param("factor_jitter", 1e-50, id="positive-jitter-underflow"),
    pytest.param("covariance_ridge", 1e-50, id="positive-ridge-underflow"),
)


def _float32_problem():
    parameters = Params(
        weights=jnp.asarray([0.45, 0.55], dtype=jnp.float32),
        means=jnp.asarray([[-0.5, 0.2], [0.8, -0.3]], dtype=jnp.float32),
        covariances=jnp.asarray(
            [
                [[0.7, 0.1], [0.1, 0.5]],
                [[0.9, -0.08], [-0.08, 0.6]],
            ],
            dtype=jnp.float32,
        ),
    )
    observations = jnp.asarray(
        [[-0.7, 0.3], [0.2, -0.4], [1.1, 0.5]], dtype=jnp.float32
    )
    noise = jnp.broadcast_to(
        jnp.eye(2, dtype=jnp.float32) * jnp.float32(0.1), (3, 2, 2)
    )
    return parameters, observations, noise


def _control(value: float):
    original = jnp.asarray(value, dtype=jnp.float64)
    assert float(np.asarray(original)) != 0.0
    assert float(np.asarray(original.astype(jnp.float32))) == 0.0
    return original


def _assert_exact_rollback(result, initial: Params) -> None:
    for actual, expected in zip(result.parameters, initial, strict=True):
        actual_array = np.asarray(actual)
        assert np.all(np.isfinite(actual_array))
        np.testing.assert_array_equal(actual_array, np.asarray(expected))


def _assert_unsuccessful(result, initial: Params) -> None:
    _assert_exact_rollback(result, initial)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))


@pytest.mark.parametrize(
    "dtype,control_value",
    (
        pytest.param(
            jnp.float32,
            np.nextafter(0.0, 1.0),
            id="positive-f64-subnormal-underflows-f32",
        ),
        pytest.param(
            jnp.float64,
            -np.nextafter(0.0, 1.0),
            id="negative-f64-subnormal-same-dtype",
        ),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_rank_zero_subnormal_control_survives_device_domain_checks(
    dtype, control_value, compiled
):
    parameters, observations, noise = _float32_problem()
    parameters = Params(
        *(jnp.asarray(field, dtype=dtype) for field in parameters)
    )
    observations = jnp.asarray(observations, dtype=dtype)
    noise = jnp.asarray(noise, dtype=dtype)
    control = jnp.asarray(control_value, dtype=jnp.float64)
    assert np.asarray(control) != 0.0
    run = jax.jit(em_step) if compiled else em_step

    result = run(
        parameters,
        observations,
        noise,
        factor_jitter=control,
    )

    _assert_unsuccessful(result, parameters)


@pytest.mark.parametrize("configuration_name,value", CONTROL_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_control_sign_and_underflow_are_checked_before_em_selected_dtype_cast(
    configuration_name, value, compiled
):
    parameters, observations, noise = _float32_problem()
    run = jax.jit(em_step) if compiled else em_step

    result = run(
        parameters,
        observations,
        noise,
        **{configuration_name: _control(value)},
    )

    _assert_unsuccessful(result, parameters)


@pytest.mark.parametrize("configuration_name,value", CONTROL_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_zero_step_fixed_kernel_reports_control_cast_underflow(
    configuration_name, value, compiled
):
    parameters, observations, noise = _float32_problem()
    kernel = (
        jax.jit(fit_fixed_steps_kernel, static_argnames=("n_steps",))
        if compiled
        else fit_fixed_steps_kernel
    )
    result = kernel(
        parameters,
        observations,
        noise,
        n_steps=0,
        **{configuration_name: _control(value)},
    )

    _assert_unsuccessful(result, parameters)
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert int(np.asarray(result.n_iter)) == 0
    assert int(np.asarray(result.history_length)) == 1
    assert np.asarray(result.history_buffer).shape == (1,)


@pytest.mark.parametrize("configuration_name,value", CONTROL_CASES)
@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_zero_step_host_fit_reports_control_cast_underflow(
    configuration_name, value, mode
):
    parameters, observations, noise = _float32_problem()
    if mode == "fixed":
        result = fit_fixed_steps(
            parameters,
            observations,
            noise,
            n_steps=0,
            **{configuration_name: _control(value)},
        )
    else:
        result = fit_converged(
            parameters,
            observations,
            noise,
            max_iter=0,
            **{configuration_name: _control(value)},
        )

    _assert_unsuccessful(result, parameters)
    assert int(np.asarray(result.status)) == int(FitStatus.NUMERICAL_FAILURE)
    assert int(np.asarray(result.n_iter)) == 0
    assert np.asarray(result.history).shape == (1,)
