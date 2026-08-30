"""Focused edge-case tests for the temporary identity-XD JAX kernel."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.identity_xd import (
    Params,
    em_step,
    marginalized_posterior,
    posterior_components,
)
from tests.reference.identity_xd import (
    identity_e_step as reference_e_step,
    identity_em_step as reference_em_step,
    marginalized_posterior as reference_marginalized_posterior,
)


DTYPE_CASES = (
    pytest.param(jnp.float64, 5e-10, 5e-12, id="float64"),
    pytest.param(jnp.float32, 1e-4, 1e-5, id="float32"),
)


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _assert_allclose(actual, expected, *, rtol: float, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )


def _well_conditioned_fixture(dtype):
    observations = jnp.asarray(
        [[-1.1, 0.2], [-0.4, -0.7], [0.3, 0.5], [0.9, -0.1], [1.5, 1.0]],
        dtype=dtype,
    )
    noise = jnp.asarray(
        [
            [[0.10, 0.02], [0.02, 0.15]],
            [[0.18, -0.03], [-0.03, 0.12]],
            [[0.08, 0.01], [0.01, 0.11]],
            [[0.14, 0.04], [0.04, 0.20]],
            [[0.12, -0.02], [-0.02, 0.09]],
        ],
        dtype=dtype,
    )
    parameters = _params(
        dtype,
        weights=[0.42, 0.58],
        means=[[-0.7, -0.1], [0.8, 0.5]],
        covariances=[
            [[0.75, 0.13], [0.13, 0.55]],
            [[0.65, -0.09], [-0.09, 0.85]],
        ],
    )
    return parameters, observations, noise


@pytest.mark.parametrize("dtype,_rtol,_atol", DTYPE_CASES)
def test_xd_ip_collapse_001_zero_mass_rolls_back_the_entire_parameter_state(
    dtype, _rtol, _atol
):
    """XD-IP-COLLAPSE-001: a dead component cannot leak partial updates."""

    observations = jnp.asarray(
        [
            [-1.0, -0.5],
            [-0.8, 0.4],
            [-0.3, -0.7],
            [0.0, 0.2],
            [0.2, 0.9],
            [0.6, -0.4],
            [0.9, 0.5],
            [1.2, -0.1],
        ],
        dtype=dtype,
    )
    noise = jnp.broadcast_to(
        jnp.eye(2, dtype=dtype) * jnp.asarray(0.1, dtype=dtype), (8, 2, 2)
    )
    parameters = _params(
        dtype,
        weights=[0.5, 0.5],
        means=[[0.0, 0.0], [1_000_000.0, 1_000_000.0]],
        covariances=[np.eye(2), np.eye(2)],
    )

    eager_result = em_step(parameters, observations, noise)
    compiled_result = jax.jit(em_step)(parameters, observations, noise)

    for result in (eager_result, compiled_result):
        assert float(np.asarray(result.statistics.mass[1])) == 0.0
        assert bool(np.asarray(result.collapsed))
        assert not bool(np.asarray(result.numerical_failure))
        np.testing.assert_array_equal(
            np.asarray(result.collapsed_components), np.array([False, True])
        )
        for returned, initial in zip(result.parameters, parameters, strict=True):
            returned_array = np.asarray(returned)
            assert np.all(np.isfinite(returned_array))
            np.testing.assert_array_equal(returned_array, np.asarray(initial))


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_xd_ip_jitter_001_factor_jitter_matches_effective_noise_oracle_only(
    dtype, rtol, atol
):
    """XD-IP-JITTER-001: jitter changes T/posterior, not stored V directly."""

    parameters, observations, noise = _well_conditioned_fixture(dtype)
    factor_jitter = 0.05
    reference_parameters, reference_estep, reference_statistics = reference_em_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
        factor_jitter=factor_jitter,
    )

    actual = em_step(
        parameters, observations, noise, factor_jitter=factor_jitter
    )
    assert not bool(np.asarray(actual.collapsed))

    for field in (
        "component_log_density",
        "component_log_joint",
        "score_samples",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
    ):
        _assert_allclose(
            getattr(actual.e_step, field),
            getattr(reference_estep, field),
            rtol=rtol,
            atol=atol,
        )
    for field in ("mass", "first_moment", "second_moment"):
        _assert_allclose(
            getattr(actual.statistics, field),
            getattr(reference_statistics, field),
            rtol=rtol,
            atol=atol,
        )
    for field in ("weights", "means", "covariances"):
        _assert_allclose(
            getattr(actual.parameters, field),
            getattr(reference_parameters, field),
            rtol=rtol,
            atol=atol,
        )

    directly_jittered_covariance = reference_parameters.covariances + (
        factor_jitter * np.eye(2)[None, :, :]
    )
    assert not np.allclose(
        np.asarray(actual.parameters.covariances),
        directly_jittered_covariance,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_xd_ip_ridge_001_covariance_ridge_affects_only_updated_covariances(
    dtype, rtol, atol
):
    """XD-IP-RIDGE-001: ridge is a stored V update, separate from jitter."""

    parameters, observations, noise = _well_conditioned_fixture(dtype)
    covariance_ridge = 0.02
    reference_parameters, _, _ = reference_em_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
        covariance_ridge=covariance_ridge,
    )
    exact_parameters, _, _ = reference_em_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
    )

    actual = em_step(
        parameters, observations, noise, covariance_ridge=covariance_ridge
    )
    assert not bool(np.asarray(actual.collapsed))

    _assert_allclose(
        actual.parameters.weights,
        exact_parameters.weights,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        actual.parameters.means, exact_parameters.means, rtol=rtol, atol=atol
    )
    _assert_allclose(
        actual.parameters.covariances,
        reference_parameters.covariances,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        actual.parameters.covariances,
        exact_parameters.covariances
        + covariance_ridge * np.eye(2)[None, :, :],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_xd_ip_shape_001_single_observation_matches_first_duplicated_batch_item(
    dtype, rtol, atol
):
    """XD-IP-SHAPE-001: unbatched inference retains no length-one batch axis."""

    parameters = _params(
        dtype,
        weights=[0.2, 0.35, 0.45],
        means=[[-1.0, 0.6], [0.4, -0.8], [1.5, 1.1]],
        covariances=[
            [[0.7, 0.12], [0.12, 0.5]],
            [[0.9, -0.16], [-0.16, 0.8]],
            [[0.6, 0.05], [0.05, 1.0]],
        ],
    )
    observation = jnp.asarray([0.2, 0.3], dtype=dtype)
    noise = jnp.asarray([[0.35, 0.07], [0.07, 0.25]], dtype=dtype)
    batch_observations = jnp.stack([observation, observation])
    batch_noise = jnp.stack([noise, noise])

    single = posterior_components(parameters, observation, noise)
    batch = posterior_components(parameters, batch_observations, batch_noise)
    expected_shapes = {
        "component_log_density": (3,),
        "component_log_joint": (3,),
        "score_samples": (),
        "responsibilities": (3,),
        "conditional_mean": (3, 2),
        "conditional_covariance": (3, 2, 2),
    }
    for field, expected_shape in expected_shapes.items():
        single_value = getattr(single, field)
        assert single_value.shape == expected_shape
        _assert_allclose(
            single_value, getattr(batch, field)[0], rtol=rtol, atol=atol
        )

    reference = reference_e_step(
        np.asarray(observation, dtype=np.float64)[None, :],
        np.asarray(noise, dtype=np.float64)[None, :, :],
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
    )
    for field in expected_shapes:
        _assert_allclose(
            getattr(single, field),
            getattr(reference, field)[0],
            rtol=rtol,
            atol=atol,
        )

    single_mean, single_covariance = marginalized_posterior(single)
    batch_mean, batch_covariance = marginalized_posterior(batch)
    reference_mean, reference_covariance = reference_marginalized_posterior(
        reference
    )
    assert single_mean.shape == (2,)
    assert single_covariance.shape == (2, 2)
    _assert_allclose(single_mean, batch_mean[0], rtol=rtol, atol=atol)
    _assert_allclose(single_covariance, batch_covariance[0], rtol=rtol, atol=atol)
    _assert_allclose(single_mean, reference_mean[0], rtol=rtol, atol=atol)
    _assert_allclose(
        single_covariance, reference_covariance[0], rtol=rtol, atol=atol
    )
