"""Phase 2 prediction and sampling gates for the temporary identity-XD API."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.identity_xd import Params
from development.inference import (
    predict,
    predict_proba,
    sample_latent,
    sample_observed,
)
from tests.fixtures.identity_likelihood import likelihood_fixture


DTYPE_CASES = (
    pytest.param(jnp.float64, np.float64, 5e-10, 5e-12, id="float64"),
    pytest.param(jnp.float32, np.float32, 1e-4, 1e-5, id="float32"),
)


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _literal_likelihood_problem(dtype):
    observations, noise, weights, means, covariances = likelihood_fixture(
        np.float64
    )
    return (
        _params(dtype, weights, means, covariances),
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(noise, dtype=dtype),
    )


def _sampling_params(dtype) -> Params:
    return _params(
        dtype,
        [0.35, 0.65],
        [[-1.0, 0.5], [2.0, -0.75]],
        [
            [[1.0, 0.2], [0.2, 0.5]],
            [[0.6, -0.1], [-0.1, 1.2]],
        ],
    )


def _analytic_mixture_moments(
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(weights[:, None] * means, axis=0)
    offsets = means - mean
    covariance = np.sum(
        weights[:, None, None]
        * (covariances + offsets[:, :, None] * offsets[:, None, :]),
        axis=0,
    )
    return mean, covariance


def _synchronized_numpy(array: jax.Array) -> np.ndarray:
    array.block_until_ready()
    return np.asarray(array)


def _assert_empirical_moments(
    samples: np.ndarray,
    expected_mean: np.ndarray,
    expected_covariance: np.ndarray,
    *,
    dtype_atol: float,
    covariance_relative_bound: float,
) -> None:
    n_samples = samples.shape[0]
    empirical_mean = np.mean(samples, axis=0, dtype=np.float64)
    empirical_covariance = np.cov(
        np.asarray(samples, dtype=np.float64), rowvar=False, ddof=0
    )
    mean_bound = (
        6.0 * np.sqrt(np.diag(expected_covariance) / n_samples) + dtype_atol
    )
    assert np.all(np.abs(empirical_mean - expected_mean) <= mean_bound)

    relative_covariance_error = np.linalg.norm(
        empirical_covariance - expected_covariance, ord="fro"
    ) / np.linalg.norm(expected_covariance, ord="fro")
    assert relative_covariance_error <= covariance_relative_bound


@pytest.mark.parametrize("dtype,numpy_dtype,rtol,atol", DTYPE_CASES)
def test_xd_ip_vmap_001_single_prediction_matches_native_literal_batch(
    dtype,
    numpy_dtype,
    rtol,
    atol,
):
    """XD-IP-VMAP-001: single-item vmap equals the native N=11 batch path."""

    del numpy_dtype
    params, observations, noise = _literal_likelihood_problem(dtype)

    native_probabilities = predict_proba(params, observations, noise)
    mapped_probabilities = jax.vmap(
        lambda x_i, s_i: predict_proba(params, x_i, s_i)
    )(observations, noise)
    native_labels = predict(params, observations, noise)
    mapped_labels = jax.vmap(lambda x_i, s_i: predict(params, x_i, s_i))(
        observations, noise
    )

    assert native_probabilities.shape == (11, 3)
    assert mapped_probabilities.shape == (11, 3)
    assert native_labels.shape == (11,)
    assert mapped_labels.shape == (11,)
    np.testing.assert_allclose(
        _synchronized_numpy(mapped_probabilities),
        _synchronized_numpy(native_probabilities),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_array_equal(
        _synchronized_numpy(mapped_labels),
        _synchronized_numpy(native_labels),
    )


@pytest.mark.parametrize("dtype,numpy_dtype,rtol,atol", DTYPE_CASES)
def test_xd_ip_pred_001_exact_three_way_tie_uses_lowest_component(
    dtype,
    numpy_dtype,
    rtol,
    atol,
):
    """XD-IP-PRED-001: equal log-joints give equal q and argmax index zero."""

    del numpy_dtype, rtol
    params = _params(
        dtype,
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]],
        [
            [[0.7, 0.0], [0.0, 0.7]],
            [[0.7, 0.0], [0.0, 0.7]],
            [[0.7, 0.0], [0.0, 0.7]],
        ],
    )
    observation = jnp.asarray([0.0, 0.0], dtype=dtype)
    noise = jnp.asarray([[0.2, 0.0], [0.0, 0.2]], dtype=dtype)

    probabilities = predict_proba(params, observation, noise)
    label = predict(params, observation, noise)

    assert probabilities.shape == (3,)
    assert label.shape == ()
    np.testing.assert_allclose(
        _synchronized_numpy(probabilities),
        np.full(3, 1.0 / 3.0),
        rtol=0.0,
        atol=atol,
    )
    assert int(_synchronized_numpy(label)) == 0


@pytest.mark.parametrize("dtype,numpy_dtype,rtol,dtype_atol", DTYPE_CASES)
@pytest.mark.slow
def test_xd_ip_sample_001_latent_reproducibility_and_mixture_moments(
    dtype,
    numpy_dtype,
    rtol,
    dtype_atol,
):
    """XD-IP-SAMPLE-001: explicit-key draws meet analytic mixture moments."""

    del numpy_dtype, rtol
    params = _sampling_params(dtype)
    n_samples = 200_000
    key = jax.random.key(20260825)
    split_key = jax.random.split(key, 2)[1]

    draws = sample_latent(params, key, n_samples)
    reused = sample_latent(params, key, n_samples)
    independent = sample_latent(params, split_key, n_samples)
    samples = _synchronized_numpy(draws)
    reused_samples = _synchronized_numpy(reused)
    independent_samples = _synchronized_numpy(independent)

    assert samples.shape == (n_samples, 2)
    assert draws.dtype == dtype
    assert np.issubdtype(samples.dtype, np.floating)
    assert np.all(np.isfinite(samples))
    np.testing.assert_array_equal(samples, reused_samples)
    assert np.any(samples != independent_samples)

    expected_mean, expected_covariance = _analytic_mixture_moments(
        np.asarray([0.35, 0.65]),
        np.asarray([[-1.0, 0.5], [2.0, -0.75]]),
        np.asarray(
            [
                [[1.0, 0.2], [0.2, 0.5]],
                [[0.6, -0.1], [-0.1, 1.2]],
            ]
        ),
    )
    _assert_empirical_moments(
        samples,
        expected_mean,
        expected_covariance,
        dtype_atol=dtype_atol,
        covariance_relative_bound=0.02,
    )


OBSERVED_NOISE_CASES = (
    pytest.param([[0.0, 0.0], [0.0, 0.0]], id="zero"),
    pytest.param([[0.4, 0.15], [0.15, 0.3]], id="full"),
    pytest.param([[0.2, 0.2], [0.2, 0.2]], id="singular"),
)


@pytest.mark.parametrize("noise_matrix", OBSERVED_NOISE_CASES)
@pytest.mark.slow
def test_xd_ip_sample_002_observed_moments_include_psd_measurement_noise(
    noise_matrix,
):
    """XD-IP-SAMPLE-002: zero, full, and singular S have analytic moments."""

    dtype = jnp.float64
    params = _params(
        dtype,
        [1.0],
        [[0.2, -0.4]],
        [[[1.0, 0.25], [0.25, 0.7]]],
    )
    n_samples = 100_000
    one_noise = jnp.asarray(noise_matrix, dtype=dtype)
    noise = jnp.broadcast_to(one_noise, (n_samples, 2, 2))

    draws = sample_observed(params, jax.random.key(20260825), noise)
    samples = _synchronized_numpy(draws)

    assert samples.shape == (n_samples, 2)
    assert draws.dtype == dtype
    assert np.all(np.isfinite(samples))
    expected_covariance = np.asarray(
        [[1.0, 0.25], [0.25, 0.7]], dtype=np.float64
    ) + np.asarray(noise_matrix, dtype=np.float64)
    _assert_empirical_moments(
        samples,
        np.asarray([0.2, -0.4]),
        expected_covariance,
        dtype_atol=5e-12,
        covariance_relative_bound=0.025,
    )


def test_xd_ip_prng_001_random_functions_require_key_by_signature():
    """XD-IP-PRNG-001: neither random API may manufacture a default key."""

    params = _sampling_params(jnp.float64)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float64)

    with pytest.raises(TypeError):
        sample_latent(params, n=1)
    with pytest.raises(TypeError):
        sample_observed(params, S=noise)


def test_xd_ip_prng_001_observed_sampling_reuses_and_splits_keys_explicitly():
    """XD-IP-PRNG-001: reused observed keys repeat and split keys differ."""

    params = _sampling_params(jnp.float64)
    noise = jnp.broadcast_to(
        jnp.asarray([[0.1, 0.02], [0.02, 0.2]], dtype=jnp.float64),
        (64, 2, 2),
    )
    key = jax.random.key(20260825)
    split_key = jax.random.split(key, 2)[1]

    first = _synchronized_numpy(sample_observed(params, key, noise))
    repeated = _synchronized_numpy(sample_observed(params, key, noise))
    independent = _synchronized_numpy(sample_observed(params, split_key, noise))

    np.testing.assert_array_equal(first, repeated)
    assert np.any(first != independent)


@pytest.mark.parametrize(
    "invalid_key",
    [
        pytest.param(None, id="none"),
        pytest.param(1, id="integer"),
        pytest.param(jnp.zeros((3,), dtype=jnp.uint32), id="wrong-legacy-shape"),
        pytest.param(jnp.zeros((2, 2), dtype=jnp.uint32), id="batched-key"),
    ],
)
@pytest.mark.parametrize("n_samples", [0, 1], ids=["zero", "positive"])
def test_xd_ip_prng_001_rejects_invalid_explicit_keys_even_for_empty_draws(
    invalid_key, n_samples
):
    """An empty request does not bypass the explicit single-key contract."""

    params = _sampling_params(jnp.float64)
    noise = jnp.zeros((n_samples, 2, 2), dtype=jnp.float64)
    with pytest.raises((TypeError, ValueError)):
        sample_latent(params, invalid_key, n_samples)
    with pytest.raises((TypeError, ValueError)):
        sample_observed(params, invalid_key, noise)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_xd_ip_sample_001_and_002_zero_count_returns_typed_empty_arrays(dtype):
    """SAMPLE-001/002: zero draws are valid and preserve the feature axis."""

    params = _sampling_params(dtype)
    key = jax.random.key(20260825)

    latent = sample_latent(params, key, 0)
    observed = sample_observed(params, key, jnp.zeros((0, 2, 2), dtype=dtype))

    assert latent.shape == (0, 2)
    assert observed.shape == (0, 2)
    assert latent.dtype == dtype
    assert observed.dtype == dtype


@pytest.mark.parametrize(
    "invalid_n", [-1, 1.5, True, np.bool_(True), "2", jnp.asarray([1])]
)
def test_xd_ip_sample_001_rejects_invalid_sample_counts(invalid_n):
    """SAMPLE-001: n is a nonnegative scalar integer count, never inferred."""

    params = _sampling_params(jnp.float64)
    with pytest.raises((TypeError, ValueError), match="n|count|integer|nonnegative"):
        sample_latent(params, jax.random.key(20260825), invalid_n)


@pytest.mark.parametrize(
    "invalid_noise",
    [
        jnp.zeros((2, 2), dtype=jnp.float64),
        jnp.zeros((4, 2), dtype=jnp.float64),
        jnp.zeros((3, 2, 1), dtype=jnp.float64),
    ],
    ids=["unbatched-full", "diagonal-batch", "nonsquare-full"],
)
def test_xd_ip_sample_002_requires_exact_canonical_noise_shape(invalid_noise):
    """SAMPLE-002: observed n comes only from canonical ``(n,D,D)`` S."""

    params = _sampling_params(jnp.float64)
    with pytest.raises((TypeError, ValueError), match="S|noise|covariance|shape"):
        sample_observed(params, jax.random.key(20260825), invalid_noise)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_xd_ip_sample_002_materially_indefinite_noise_is_observable(dtype):
    """The pure sampler must not silently project an invalid row onto the PSD cone."""

    params = _sampling_params(dtype)
    noise = jnp.asarray(
        [
            [[0.2, 0.05], [0.05, 0.3]],
            [[1.0, 0.0], [0.0, -0.1]],
        ],
        dtype=dtype,
    )
    draws = sample_observed(params, jax.random.key(20260825), noise)
    actual = _synchronized_numpy(draws)

    assert np.all(np.isfinite(actual[0]))
    assert np.all(np.isnan(actual[1]))
