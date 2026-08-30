"""Red tests for configuration and factorization-failure status semantics."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.identity_xd import Params, em_step, posterior_components


def _valid_problem(dtype=jnp.float64):
    parameters = Params(
        weights=jnp.asarray([0.45, 0.55], dtype=dtype),
        means=jnp.asarray([[-0.5, 0.2], [0.8, -0.3]], dtype=dtype),
        covariances=jnp.asarray(
            [
                [[0.7, 0.1], [0.1, 0.5]],
                [[0.9, -0.08], [-0.08, 0.6]],
            ],
            dtype=dtype,
        ),
    )
    observations = jnp.asarray([[-0.7, 0.3], [0.2, -0.4], [1.1, 0.5]], dtype=dtype)
    noise = jnp.broadcast_to(
        jnp.eye(2, dtype=dtype) * jnp.asarray(0.1, dtype=dtype), (3, 2, 2)
    )
    return parameters, observations, noise


def _assert_exact_finite_rollback(result, initial: Params) -> None:
    for returned, expected in zip(result.parameters, initial, strict=True):
        returned_array = np.asarray(returned)
        assert np.all(np.isfinite(returned_array))
        np.testing.assert_array_equal(returned_array, np.asarray(expected))


def test_xd_ip_config_001_vector_factor_jitter_is_rejected_statically():
    """Factorization jitter is one scalar, never a per-axis vector."""

    parameters, observations, noise = _valid_problem()
    vector_jitter = jnp.asarray([0.0, 1e-6])

    with pytest.raises(ValueError):
        posterior_components(
            parameters, observations, noise, factor_jitter=vector_jitter
        )
    with pytest.raises(ValueError):
        jax.jit(posterior_components)(
            parameters, observations, noise, factor_jitter=vector_jitter
        )


def test_xd_ip_config_001_vector_covariance_ridge_is_rejected_statically():
    """Covariance ridge is one scalar, never a per-axis vector."""

    parameters, observations, noise = _valid_problem()
    vector_ridge = jnp.asarray([0.0, 1e-6])

    with pytest.raises(ValueError):
        em_step(
            parameters, observations, noise, covariance_ridge=vector_ridge
        )
    with pytest.raises(ValueError):
        jax.jit(em_step)(
            parameters, observations, noise, covariance_ridge=vector_ridge
        )


@pytest.mark.parametrize("invalid_value", [-1e-3, np.nan, np.inf])
@pytest.mark.parametrize(
    "configuration_name", ["factor_jitter", "covariance_ridge"]
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_config_002_invalid_scalar_configuration_reports_failure(
    configuration_name, invalid_value, compiled
):
    """Invalid scalar configuration cannot masquerade as a successful update."""

    parameters, observations, noise = _valid_problem()
    run_em_step = jax.jit(em_step) if compiled else em_step
    result = run_em_step(
        parameters,
        observations,
        noise,
        **{configuration_name: invalid_value},
    )

    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    _assert_exact_finite_rollback(result, parameters)


@pytest.mark.parametrize(
    "invalid_control",
    [
        pytest.param(True, id="boolean"),
        pytest.param(1.0 + 2.0j, id="complex"),
        pytest.param("not-a-number", id="nonnumeric"),
    ],
)
@pytest.mark.parametrize(
    "configuration_name", ["factor_jitter", "covariance_ridge"]
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_config_003_invalid_control_dtype_is_rejected_before_casting(
    invalid_control, configuration_name, compiled
):
    """Boolean, complex, and nonnumeric controls are not coerced to float."""

    parameters, observations, noise = _valid_problem()
    operation = (
        posterior_components
        if configuration_name == "factor_jitter"
        else em_step
    )
    run_operation = jax.jit(operation) if compiled else operation

    with pytest.raises((TypeError, ValueError)):
        run_operation(
            parameters,
            observations,
            noise,
            **{configuration_name: invalid_control},
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_cov_001_factorization_failure_has_pair_status_and_rolls_back(
    compiled
):
    """A failed total covariance is a numerical failure, not a dead component."""

    dtype = jnp.float64
    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray([[0.0, 0.0], [0.8, -0.4]], dtype=dtype),
        covariances=jnp.asarray(
            [0.5 * np.eye(2), 2.0 * np.eye(2)], dtype=dtype
        ),
    )
    observations = jnp.asarray([[0.1, -0.2], [0.4, 0.3]], dtype=dtype)
    measurement_covariances = jnp.asarray(
        [0.1 * np.eye(2), -1.0 * np.eye(2)], dtype=dtype
    )
    expected_failed_pairs = np.array([[False, False], [True, False]])

    run_posterior = jax.jit(posterior_components) if compiled else posterior_components
    run_em_step = jax.jit(em_step) if compiled else em_step
    e_step = run_posterior(parameters, observations, measurement_covariances)
    result = run_em_step(parameters, observations, measurement_covariances)

    assert bool(np.asarray(e_step.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(e_step.failed_pairs), expected_failed_pairs
    )
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    np.testing.assert_array_equal(
        np.asarray(result.e_step.failed_pairs), expected_failed_pairs
    )
    _assert_exact_finite_rollback(result, parameters)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_xd_ip_cov_001_valid_float32_inputs_report_pair_factorization_failure(
    compiled,
):
    """A valid PSD/SPD input can still fail a rounded float32 factorization."""

    dtype = jnp.float32
    parameters = Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.zeros((2, 2), dtype=dtype),
        covariances=jnp.asarray(
            [np.diag([1e-8, 1e-8]), np.eye(2)], dtype=dtype
        ),
    )
    observations = jnp.asarray([[0.1, -0.2]], dtype=dtype)
    measurement_covariances = jnp.asarray(
        [[[1.0, 1.0], [1.0, 1.0]]], dtype=dtype
    )
    expected_failed_pairs = np.array([[True, False]])

    # The inputs satisfy their domains before V + S is rounded in float32.
    assert np.all(np.linalg.eigvalsh(np.asarray(measurement_covariances[0])) >= 0.0)
    np.linalg.cholesky(np.asarray(parameters.covariances))

    run_posterior = jax.jit(posterior_components) if compiled else posterior_components
    run_em_step = jax.jit(em_step) if compiled else em_step
    e_step = run_posterior(parameters, observations, measurement_covariances)
    result = run_em_step(parameters, observations, measurement_covariances)

    assert bool(np.asarray(e_step.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(e_step.failed_pairs), expected_failed_pairs
    )
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    np.testing.assert_array_equal(
        np.asarray(result.e_step.failed_pairs), expected_failed_pairs
    )
    _assert_exact_finite_rollback(result, parameters)
