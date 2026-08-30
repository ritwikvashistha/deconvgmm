"""Functional scoring and posterior gates for the temporary identity-XD API.

The expected values come from the literal likelihood fixture and the independent
NumPy oracle.  Eager validation is tested separately from the pure-JAX
operations: canonicalized arrays enter the functions in this module, while the
functions themselves remain suitable for JIT and vmap composition.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development import inference
from development.identity_xd import (
    Params,
    marginalized_posterior,
    posterior_components,
)
from development.validation import canonicalize_inference_inputs
from tests.fixtures.identity_likelihood import likelihood_fixture
from tests.reference.identity_xd import (
    identity_e_step as reference_e_step,
    marginalized_posterior as reference_marginalized_posterior,
)


DTYPE_CASES = (
    pytest.param(
        jnp.float64,
        5e-10,
        5e-12,
        5e-10,
        5e-10,
        id="float64",
    ),
    pytest.param(
        jnp.float32,
        1e-4,
        1e-5,
        2e-4,
        2e-5,
        id="float32",
    ),
)


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _literal_problem(dtype):
    observations, noise, weights, means, covariances = likelihood_fixture(
        np.float64
    )
    return (
        _params(dtype, weights, means, covariances),
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(noise, dtype=dtype),
        (observations, noise, weights, means, covariances),
    )


def _synchronized_numpy(value: jax.Array) -> np.ndarray:
    value.block_until_ready()
    return np.asarray(value)


def _assert_tree_allclose(actual, expected, *, rtol: float, atol: float) -> None:
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        np.testing.assert_allclose(
            _synchronized_numpy(actual_leaf),
            _synchronized_numpy(expected_leaf),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_ip_ll_001_functional_scores_match_oracle_and_exact_reductions(
    dtype,
    _ref_rtol,
    _ref_atol,
    log_rtol,
    log_atol,
):
    """XD-IP-LL-001: public-style score reductions have one definition."""

    params, observations, noise, raw = _literal_problem(dtype)
    reference = reference_e_step(*raw)

    per_observation = inference.score_samples(params, observations, noise)
    total = inference.log_likelihood(params, observations, noise)
    mean = inference.score(params, observations, noise)

    assert per_observation.shape == (11,)
    assert total.shape == ()
    assert mean.shape == ()
    assert per_observation.dtype == dtype
    assert total.dtype == dtype
    assert mean.dtype == dtype
    np.testing.assert_allclose(
        _synchronized_numpy(per_observation),
        reference.score_samples,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(total),
        np.sum(reference.score_samples),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(mean),
        np.mean(reference.score_samples),
        rtol=log_rtol,
        atol=log_atol,
    )

    # These are exact API identities, not merely tolerance-level equivalences.
    np.testing.assert_array_equal(
        _synchronized_numpy(total),
        _synchronized_numpy(jnp.sum(per_observation)),
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(mean),
        _synchronized_numpy(jnp.mean(per_observation)),
    )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_functional_scoring_preserves_single_and_arbitrary_batch_shapes(
    dtype,
    _ref_rtol,
    _ref_atol,
    log_rtol,
    log_atol,
):
    """A scalar item and a multi-axis batch retain their contract shapes."""

    params, observations, noise, raw = _literal_problem(dtype)
    raw_observations, raw_noise, weights, means, covariances = raw
    batch_observations = observations[:6].reshape(2, 3, 2)
    batch_noise = noise[:6].reshape(2, 3, 2, 2)
    reference = reference_e_step(
        raw_observations[:6],
        raw_noise[:6],
        weights,
        means,
        covariances,
    )

    batch_values = inference.score_samples(
        params, batch_observations, batch_noise
    )
    single_value = inference.score_samples(params, observations[0], noise[0])

    assert batch_values.shape == (2, 3)
    assert single_value.shape == ()
    np.testing.assert_allclose(
        _synchronized_numpy(batch_values),
        reference.score_samples.reshape(2, 3),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(single_value),
        reference.score_samples[0],
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(
            inference.log_likelihood(params, batch_observations, batch_noise)
        ),
        _synchronized_numpy(jnp.sum(batch_values)),
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(
            inference.score(params, batch_observations, batch_noise)
        ),
        _synchronized_numpy(jnp.mean(batch_values)),
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,_log_rtol,_log_atol", DTYPE_CASES
)
def test_xd_ip_post_002_functional_posterior_matches_marginalized_oracle(
    dtype,
    ref_rtol,
    ref_atol,
    _log_rtol,
    _log_atol,
):
    """Posterior means/covariances are component-marginalized moments."""

    params, observations, noise, raw = _literal_problem(dtype)
    raw_observations, raw_noise, weights, means, covariances = raw
    batch_observations = observations[:6].reshape(2, 3, 2)
    batch_noise = noise[:6].reshape(2, 3, 2, 2)

    reference_e = reference_e_step(
        raw_observations[:6],
        raw_noise[:6],
        weights,
        means,
        covariances,
    )
    expected_mean, expected_covariance = reference_marginalized_posterior(
        reference_e
    )
    expected_mean = expected_mean.reshape(2, 3, 2)
    expected_covariance = expected_covariance.reshape(2, 3, 2, 2)

    actual_mean, actual_covariance = inference.posterior(
        params, batch_observations, batch_noise
    )
    mean_only = inference.posterior_mean(
        params, batch_observations, batch_noise
    )
    component_result = posterior_components(
        params, batch_observations, batch_noise
    )
    component_mean, component_covariance = marginalized_posterior(
        component_result
    )

    assert actual_mean.shape == (2, 3, 2)
    assert actual_covariance.shape == (2, 3, 2, 2)
    assert mean_only.shape == (2, 3, 2)
    assert actual_mean.dtype == dtype
    assert actual_covariance.dtype == dtype
    assert mean_only.dtype == dtype
    np.testing.assert_allclose(
        _synchronized_numpy(actual_mean),
        expected_mean,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(actual_covariance),
        expected_covariance,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(actual_mean),
        _synchronized_numpy(mean_only),
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(actual_mean),
        _synchronized_numpy(component_mean),
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(actual_covariance),
        _synchronized_numpy(component_covariance),
    )

    single_mean, single_covariance = inference.posterior(
        params, observations[0], noise[0]
    )
    assert single_mean.shape == (2,)
    assert single_covariance.shape == (2, 2)
    np.testing.assert_allclose(
        _synchronized_numpy(single_mean),
        expected_mean[0, 0],
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(single_covariance),
        expected_covariance[0, 0],
        rtol=ref_rtol,
        atol=ref_atol,
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_ip_dtype_001_integer_observation_boundary_keeps_outputs_floating(
    dtype,
    ref_rtol,
    ref_atol,
    log_rtol,
    log_atol,
):
    """The eager boundary converts integer x before pure inference begins."""

    parameters = Params(
        weights=np.asarray([1.0], dtype=np.float64),
        means=np.asarray([[0.0, 0.0]], dtype=np.float64),
        covariances=np.asarray([np.eye(2)], dtype=np.float64),
    )
    integer_observation = np.asarray([0, 1], dtype=np.int32)
    noise = np.asarray([[0.5, 0.0], [0.0, 0.5]], dtype=np.float64)
    validated = canonicalize_inference_inputs(
        parameters, integer_observation, noise, dtype=dtype
    )
    reference = reference_e_step(
        integer_observation[None, :],
        noise[None, :, :],
        parameters.weights,
        parameters.means,
        parameters.covariances,
    )
    expected_mean, expected_covariance = reference_marginalized_posterior(
        reference
    )

    sample_score = inference.score_samples(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )
    total = inference.log_likelihood(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )
    mean_score = inference.score(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )
    posterior_mean, posterior_covariance = inference.posterior(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )
    mean_only = inference.posterior_mean(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )

    for value in (
        sample_score,
        total,
        mean_score,
        posterior_mean,
        posterior_covariance,
        mean_only,
    ):
        assert value.dtype == dtype
        assert np.issubdtype(_synchronized_numpy(value).dtype, np.floating)
    np.testing.assert_allclose(
        _synchronized_numpy(sample_score),
        reference.score_samples[0],
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(posterior_mean),
        expected_mean[0],
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(posterior_covariance),
        expected_covariance[0],
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(posterior_mean),
        [0.0, 2.0 / 3.0],
        rtol=ref_rtol,
        atol=ref_atol,
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_factor_jitter_is_forwarded_by_every_functional_inference_operation(
    dtype,
    ref_rtol,
    ref_atol,
    log_rtol,
    log_atol,
):
    """Scoring and posterior helpers share the kernel's effective model."""

    params, observations, noise, raw = _literal_problem(dtype)
    raw_observations, raw_noise, weights, means, covariances = raw
    factor_jitter = 0.125
    reference = reference_e_step(
        raw_observations,
        raw_noise,
        weights,
        means,
        covariances,
        factor_jitter=factor_jitter,
    )
    expected_mean, expected_covariance = reference_marginalized_posterior(
        reference
    )

    sample_scores = inference.score_samples(
        params, observations, noise, factor_jitter=factor_jitter
    )
    total = inference.log_likelihood(
        params, observations, noise, factor_jitter=factor_jitter
    )
    mean_score = inference.score(
        params, observations, noise, factor_jitter=factor_jitter
    )
    posterior_mean, posterior_covariance = inference.posterior(
        params, observations, noise, factor_jitter=factor_jitter
    )
    mean_only = inference.posterior_mean(
        params, observations, noise, factor_jitter=factor_jitter
    )

    np.testing.assert_allclose(
        _synchronized_numpy(sample_scores),
        reference.score_samples,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(total),
        np.sum(reference.score_samples),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(mean_score),
        np.mean(reference.score_samples),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(posterior_mean),
        expected_mean,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(posterior_covariance),
        expected_covariance,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(mean_only),
        _synchronized_numpy(posterior_mean),
    )

    without_jitter = inference.score_samples(params, observations, noise)
    assert np.any(
        _synchronized_numpy(sample_scores)
        != _synchronized_numpy(without_jitter)
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_functional_inference_is_eager_jit_equivalent_and_callback_free(
    dtype,
    ref_rtol,
    ref_atol,
    log_rtol,
    log_atol,
):
    """All five numerical-leaf helpers compile without hidden host work."""

    params, observations, noise, _ = _literal_problem(dtype)
    operations = (
        ("score_samples", log_rtol, log_atol),
        ("log_likelihood", log_rtol, log_atol),
        ("score", log_rtol, log_atol),
        ("posterior", ref_rtol, ref_atol),
        ("posterior_mean", ref_rtol, ref_atol),
    )

    for name, rtol, atol in operations:
        function = getattr(inference, name)
        jaxpr_text = str(jax.make_jaxpr(function)(params, observations, noise))
        assert "callback" not in jaxpr_text.lower()

        eager = function(params, observations, noise)
        compiled = jax.jit(function)(params, observations, noise)
        _assert_tree_allclose(compiled, eager, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_ip_vmap_001_functional_single_item_paths_match_native_batch(
    dtype,
    ref_rtol,
    ref_atol,
    log_rtol,
    log_atol,
):
    """XD-IP-VMAP-001: mapped scalar inference equals native batch work."""

    params, observations, noise, _ = _literal_problem(dtype)
    native_sample_scores = inference.score_samples(params, observations, noise)

    for name in ("score_samples", "log_likelihood", "score"):
        function = getattr(inference, name)
        mapped = jax.vmap(lambda x_i, s_i: function(params, x_i, s_i))(
            observations, noise
        )
        np.testing.assert_allclose(
            _synchronized_numpy(mapped),
            _synchronized_numpy(native_sample_scores),
            rtol=log_rtol,
            atol=log_atol,
        )

    native_mean, native_covariance = inference.posterior(
        params, observations, noise
    )
    mapped_mean, mapped_covariance = jax.vmap(
        lambda x_i, s_i: inference.posterior(params, x_i, s_i)
    )(observations, noise)
    mapped_mean_only = jax.vmap(
        lambda x_i, s_i: inference.posterior_mean(params, x_i, s_i)
    )(observations, noise)

    np.testing.assert_allclose(
        _synchronized_numpy(mapped_mean),
        _synchronized_numpy(native_mean),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(mapped_covariance),
        _synchronized_numpy(native_covariance),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        _synchronized_numpy(mapped_mean_only),
        _synchronized_numpy(native_mean),
        rtol=ref_rtol,
        atol=ref_atol,
    )


def test_functional_leaf_helpers_make_kernel_failure_observable():
    """Array-only conveniences must not turn a failed E-step into valid data."""

    params = _params(
        jnp.float64,
        [0.25, 0.75],
        [[0.0], [1.0]],
        [[[1.0]], [[1.0]]],
    )
    observation = jnp.asarray([0.2], dtype=jnp.float64)
    noise = jnp.asarray([[0.1]], dtype=jnp.float64)
    invalid_jitter = -1.0
    detailed = posterior_components(
        params, observation, noise, factor_jitter=invalid_jitter
    )

    assert bool(np.asarray(detailed.numerical_failure))
    assert np.all(np.asarray(detailed.failed_pairs))
    for value in (
        inference.predict_proba(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
        inference.score_samples(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
        inference.log_likelihood(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
        inference.score(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
        *inference.posterior(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
        inference.posterior_mean(
            params, observation, noise, factor_jitter=invalid_jitter
        ),
    ):
        assert np.all(np.isnan(_synchronized_numpy(value)))
    assert int(
        _synchronized_numpy(
            inference.predict(
                params, observation, noise, factor_jitter=invalid_jitter
            )
        )
    ) == -1
