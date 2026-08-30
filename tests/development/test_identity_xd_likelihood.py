"""Ordinary likelihood and responsibility gates for the development kernel."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import mean_log_likelihood
from development.identity_xd import Params, posterior_components
from tests.fixtures.identity_likelihood import likelihood_fixture
from tests.reference.identity_xd import identity_e_step as reference_e_step


DTYPE_CASES = (
    pytest.param(jnp.float64, 5e-10, 5e-10, 5e-13, 1e-8, id="float64"),
    pytest.param(jnp.float32, 2e-4, 2e-5, 2e-5, 5e-3, id="float32"),
)


def _fixture(dtype):
    x, noise, weights, means, covariances = likelihood_fixture(np.float64)
    params = Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )
    return (
        params,
        jnp.asarray(x, dtype=dtype),
        jnp.asarray(noise, dtype=dtype),
        (x, noise, weights, means, covariances),
    )


@pytest.mark.parametrize(
    "dtype,log_rtol,log_atol,_row_atol,max_abs", DTYPE_CASES
)
def test_xd_ip_ll_001_likelihood_matches_independent_cholesky_oracle(
    dtype, log_rtol, log_atol, _row_atol, max_abs
):
    """XD-IP-LL-001: all ordinary observed-density reductions agree."""

    params, observations, noise, raw = _fixture(dtype)
    x64, noise64, weights64, means64, covariances64 = raw
    reference = reference_e_step(
        x64, noise64, weights64, means64, covariances64
    )
    actual = posterior_components(params, observations, noise)

    assert actual.component_log_density.shape == (11, 3)
    assert actual.component_log_joint.shape == (11, 3)
    assert actual.score_samples.shape == (11,)
    assert np.max(np.abs(reference.component_log_density)) < 1e3

    total_covariances = (
        covariances64[None, :, :, :] + noise64[:, None, :, :]
    )
    condition_numbers = np.linalg.cond(total_covariances)
    assert np.max(condition_numbers) <= 1e4

    for actual_values, expected_values in (
        (actual.component_log_density, reference.component_log_density),
        (actual.component_log_joint, reference.component_log_joint),
        (actual.score_samples, reference.score_samples),
    ):
        np.testing.assert_allclose(
            np.asarray(actual_values),
            expected_values,
            rtol=log_rtol,
            atol=log_atol,
        )
        assert np.max(
            np.abs(np.asarray(actual_values) - expected_values)
        ) <= max_abs

    expected_total = np.sum(reference.score_samples)
    expected_mean = np.mean(reference.score_samples)
    actual_total = jnp.sum(actual.score_samples)
    actual_mean = mean_log_likelihood(params, observations, noise)
    np.testing.assert_allclose(
        np.asarray(actual_total), expected_total, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(actual_mean), expected_mean, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(actual_total),
        np.asarray(jnp.sum(actual.score_samples)),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(actual_mean),
        np.asarray(jnp.mean(actual.score_samples)),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,row_atol,_max_abs",
    (
        pytest.param(jnp.float64, 5e-10, 5e-12, 5e-13, 1e-8, id="float64"),
        pytest.param(jnp.float32, 1e-4, 1e-5, 2e-5, 5e-3, id="float32"),
    ),
)
def test_xd_ip_resp_001_responsibilities_match_independent_log_softmax(
    dtype, ref_rtol, ref_atol, row_atol, _max_abs
):
    """XD-IP-RESP-001: ordinary responsibilities are normalized in log space."""

    params, observations, noise, raw = _fixture(dtype)
    reference = reference_e_step(*raw)
    actual = posterior_components(params, observations, noise)
    probabilities = np.asarray(actual.responsibilities)

    assert probabilities.shape == (11, 3)
    assert np.all(np.isfinite(probabilities))
    assert np.all(probabilities >= 0.0)
    np.testing.assert_allclose(
        probabilities,
        reference.responsibilities,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        probabilities.sum(axis=-1),
        np.ones(11),
        rtol=0.0,
        atol=row_atol,
    )

