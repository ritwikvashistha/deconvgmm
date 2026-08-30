"""Red tests for the temporary general-XD inference convenience surface.

The fixed-observed-dimension numerical leaves are expected at these exact
signatures::

    score_samples_general(params, observations, projection_matrices,
                          measurement_covariances, *, factor_jitter=0.0)
    log_likelihood_general(params, observations, projection_matrices,
                           measurement_covariances, *, sample_weight=None,
                           factor_jitter=0.0)
    score_general(params, observations, projection_matrices,
                  measurement_covariances, *, sample_weight=None,
                  factor_jitter=0.0)
    predict_proba_general(params, observations, projection_matrices,
                          measurement_covariances, *, factor_jitter=0.0)
    predict_general(params, observations, projection_matrices,
                    measurement_covariances, *, factor_jitter=0.0)
    posterior_general(params, observations, projection_matrices,
                      measurement_covariances, *, factor_jitter=0.0)
    posterior_mean_general(params, observations, projection_matrices,
                           measurement_covariances, *, factor_jitter=0.0)

The grouped variants replace the four canonical fixed-``M`` positional inputs
with one :class:`GroupedGeneralInputs` positional input and retain only the
keyword-only ``factor_jitter`` control.  They use the already validated
``group.sample_weight`` arrays; no second weight argument is accepted.

These are numerical conveniences, not eager validation wrappers.  They expose
an :class:`~development.identity_xd.EStep` or
:class:`~development.general_grouped.GroupedPosteriorResult` failure through
NaN row sentinels and label ``-1`` instead of raising from traced code.
"""

from __future__ import annotations

import inspect

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
import development.general_inference as general_inference
from development.general_grouped import posterior_components_grouped
from development.general_validation import (
    PerItemFullNoise,
    PerItemProjection,
    PrecisionError,
    group_masked_general_inputs,
)
from development.general_xd import posterior_components_general
from development.identity_xd import Params
from tests.reference.general_xd import (
    general_e_step,
    general_grouped_objective,
    marginalized_posterior as reference_marginalized_posterior,
)


DTYPE_CASES = (
    pytest.param(
        jnp.float64,
        8e-10,
        8e-12,
        8e-10,
        8e-10,
        id="float64",
    ),
    pytest.param(
        jnp.float32,
        2e-4,
        2e-5,
        3e-4,
        3e-5,
        id="float32",
    ),
)


FIXED_NAMES = (
    "score_samples_general",
    "log_likelihood_general",
    "score_general",
    "predict_proba_general",
    "predict_general",
    "posterior_general",
    "posterior_mean_general",
)

GROUPED_NAMES = (
    "score_samples_grouped",
    "log_likelihood_grouped",
    "score_grouped",
    "predict_proba_grouped",
    "predict_grouped",
    "posterior_grouped",
    "posterior_mean_grouped",
)


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _ordinary_fixed_problem(dtype):
    params = _params(
        dtype,
        [0.2, 0.35, 0.45],
        [
            [-0.8, 0.25, 0.4, -0.15],
            [0.9, -0.5, 0.15, 0.7],
            [0.2, 0.8, -0.65, 0.35],
        ],
        [
            [
                [0.9, 0.12, -0.04, 0.03],
                [0.12, 0.7, 0.08, -0.02],
                [-0.04, 0.08, 0.6, 0.05],
                [0.03, -0.02, 0.05, 0.8],
            ],
            [
                [0.65, -0.06, 0.07, 0.02],
                [-0.06, 0.85, -0.03, 0.04],
                [0.07, -0.03, 0.75, 0.1],
                [0.02, 0.04, 0.1, 0.95],
            ],
            [
                [0.8, 0.04, 0.02, -0.08],
                [0.04, 0.75, 0.06, 0.03],
                [0.02, 0.06, 0.9, -0.05],
                [-0.08, 0.03, -0.05, 0.7],
            ],
        ],
    )
    observations = jnp.asarray(
        [
            [-1.1, 0.25],
            [0.4, -0.65],
            [1.2, 0.55],
            [-0.3, 0.95],
            [0.75, -0.1],
            [1.65, 0.35],
        ],
        dtype=dtype,
    )
    base_projection = np.asarray(
        [[1.0, 0.2, -0.15, 0.05], [-0.25, 0.75, 0.3, -0.4]],
        dtype=np.float64,
    )
    projection = jnp.asarray(
        np.stack(
            [
                base_projection
                + (sample - 2.5)
                * np.asarray(
                    [[0.01, -0.005, 0.003, 0.004],
                     [-0.004, 0.007, -0.002, 0.005]]
                )
                for sample in range(6)
            ]
        ),
        dtype=dtype,
    )
    noise = np.empty((6, 2, 2), dtype=np.float64)
    for sample in range(6):
        noise[sample] = np.asarray(
            [
                [0.18 + 0.015 * sample, 0.012 - 0.001 * sample],
                [0.012 - 0.001 * sample, 0.27 + 0.02 * sample],
            ]
        )
    return params, observations, projection, jnp.asarray(noise, dtype=dtype)


def _fixed_reference(params, observations, projection, noise):
    observed_dimension = observations.shape[-1]
    latent_dimension = params.means.shape[-1]
    flat_observations = np.asarray(observations).reshape(
        (-1, observed_dimension)
    )
    flat_projection = np.asarray(projection).reshape(
        (-1, observed_dimension, latent_dimension)
    )
    flat_noise = np.asarray(noise).reshape(
        (-1, observed_dimension, observed_dimension)
    )
    return general_e_step(
        flat_observations,
        flat_projection,
        flat_noise,
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
    )


def _synchronized_numpy(value):
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    return np.asarray(value)


def _assert_tree_allclose(actual, expected, *, rtol, atol):
    actual_leaves = jax.tree_util.tree_leaves(actual)
    expected_leaves = jax.tree_util.tree_leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        actual_array = _synchronized_numpy(actual_leaf)
        expected_array = _synchronized_numpy(expected_leaf)
        if np.issubdtype(actual_array.dtype, np.integer):
            np.testing.assert_array_equal(actual_array, expected_array)
        else:
            np.testing.assert_allclose(
                actual_array, expected_array, rtol=rtol, atol=atol
            )


def _prior_moments(params):
    weights = np.asarray(params.weights)
    means = np.asarray(params.means)
    covariances = np.asarray(params.covariances)
    mean = np.sum(weights[:, None] * means, axis=0)
    centered = means - mean
    covariance = np.sum(
        weights[:, None, None]
        * (covariances + centered[:, :, None] * centered[:, None, :]),
        axis=0,
    )
    return mean, 0.5 * (covariance + covariance.T)


def _assert_signature(function, positional_names, keyword_defaults):
    parameters = inspect.signature(function).parameters
    assert tuple(parameters) == tuple(positional_names) + tuple(keyword_defaults)
    for name in positional_names:
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[name].default is inspect.Parameter.empty
    for name, default in keyword_defaults.items():
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default == default


def test_general_inference_schema_signatures_and_exports():
    """The temporary surface is explicit and cannot infer grouped weights."""

    required = set(FIXED_NAMES + GROUPED_NAMES)
    assert required <= set(general_inference.__all__)
    for name in required:
        assert getattr(development, name) is getattr(general_inference, name)

    common = (
        "params",
        "observations",
        "projection_matrices",
        "measurement_covariances",
    )
    for name in (
        "score_samples_general",
        "predict_proba_general",
        "predict_general",
        "posterior_general",
        "posterior_mean_general",
    ):
        _assert_signature(
            getattr(general_inference, name),
            common,
            {"factor_jitter": 0.0},
        )
    for name in ("log_likelihood_general", "score_general"):
        _assert_signature(
            getattr(general_inference, name),
            common,
            {"sample_weight": None, "factor_jitter": 0.0},
        )
    for name in GROUPED_NAMES:
        _assert_signature(
            getattr(general_inference, name),
            ("grouped",),
            {"factor_jitter": 0.0},
        )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_shape_001_fixed_inference_matches_oracle_for_single_and_b23(
    dtype, ref_rtol, ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    batched_x = observations.reshape(2, 3, 2)
    batched_r = projection.reshape(2, 3, 2, 4)
    batched_s = noise.reshape(2, 3, 2, 2)
    reference = _fixed_reference(params, batched_x, batched_r, batched_s)
    expected_mean, expected_covariance = reference_marginalized_posterior(
        reference
    )

    actual_scores = general_inference.score_samples_general(
        params, batched_x, batched_r, batched_s
    )
    actual_probabilities = general_inference.predict_proba_general(
        params, batched_x, batched_r, batched_s
    )
    actual_labels = general_inference.predict_general(
        params, batched_x, batched_r, batched_s
    )
    actual_mean, actual_covariance = general_inference.posterior_general(
        params, batched_x, batched_r, batched_s
    )
    actual_mean_only = general_inference.posterior_mean_general(
        params, batched_x, batched_r, batched_s
    )

    assert actual_scores.shape == (2, 3)
    assert actual_probabilities.shape == (2, 3, 3)
    assert actual_labels.shape == (2, 3)
    assert actual_mean.shape == (2, 3, 4)
    assert actual_covariance.shape == (2, 3, 4, 4)
    assert actual_mean_only.shape == (2, 3, 4)
    for value in (
        actual_scores,
        actual_probabilities,
        actual_mean,
        actual_covariance,
        actual_mean_only,
    ):
        assert value.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(actual_scores),
        reference.score_samples.reshape(2, 3),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_probabilities),
        reference.responsibilities.reshape(2, 3, 3),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(
        np.asarray(actual_labels),
        np.argmax(reference.responsibilities, axis=-1).reshape(2, 3),
    )
    np.testing.assert_allclose(
        np.asarray(actual_mean),
        expected_mean.reshape(2, 3, 4),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual_covariance),
        expected_covariance.reshape(2, 3, 4, 4),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(np.asarray(actual_mean_only), np.asarray(actual_mean))

    single_reference = _fixed_reference(
        params, observations[0], projection[0], noise[0]
    )
    single_mean, single_covariance = reference_marginalized_posterior(
        single_reference
    )
    single_score = general_inference.score_samples_general(
        params, observations[0], projection[0], noise[0]
    )
    single_probability = general_inference.predict_proba_general(
        params, observations[0], projection[0], noise[0]
    )
    single_posterior = general_inference.posterior_general(
        params, observations[0], projection[0], noise[0]
    )
    assert single_score.shape == ()
    assert single_probability.shape == (3,)
    assert single_posterior[0].shape == (4,)
    assert single_posterior[1].shape == (4, 4)
    np.testing.assert_allclose(
        np.asarray(single_score),
        single_reference.score_samples[0],
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(single_probability),
        single_reference.responsibilities[0],
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(single_posterior[0]),
        single_mean[0],
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(single_posterior[1]),
        single_covariance[0],
        rtol=ref_rtol,
        atol=ref_atol,
    )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_weight_score_001_fixed_weighted_reductions_use_every_batch_axis(
    dtype, _ref_rtol, _ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    batched_x = observations.reshape(2, 3, 2)
    batched_r = projection.reshape(2, 3, 2, 4)
    batched_s = noise.reshape(2, 3, 2, 2)
    sample_weight = jnp.asarray(
        [[0.0, 0.25, 1.0], [2.0, 0.5, 3.0]], dtype=dtype
    )
    reference = _fixed_reference(params, batched_x, batched_r, batched_s)
    expected_scores = reference.score_samples.reshape(2, 3)
    expected_total = np.sum(np.asarray(sample_weight) * expected_scores)
    expected_mean = expected_total / np.sum(np.asarray(sample_weight))

    actual_scores = general_inference.score_samples_general(
        params, batched_x, batched_r, batched_s
    )
    unweighted_total = general_inference.log_likelihood_general(
        params, batched_x, batched_r, batched_s
    )
    unweighted_mean = general_inference.score_general(
        params, batched_x, batched_r, batched_s
    )
    weighted_total = general_inference.log_likelihood_general(
        params,
        batched_x,
        batched_r,
        batched_s,
        sample_weight=sample_weight,
    )
    weighted_mean = general_inference.score_general(
        params,
        batched_x,
        batched_r,
        batched_s,
        sample_weight=sample_weight,
    )

    assert unweighted_total.shape == ()
    assert unweighted_mean.shape == ()
    assert weighted_total.shape == ()
    assert weighted_mean.shape == ()
    np.testing.assert_allclose(
        np.asarray(actual_scores),
        expected_scores,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(unweighted_total),
        np.sum(expected_scores),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(unweighted_mean),
        np.mean(expected_scores),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(weighted_total),
        expected_total,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(weighted_mean),
        expected_mean,
        rtol=log_rtol,
        atol=log_atol,
    )

    scalar_weight = jnp.asarray(2.5, dtype=dtype)
    single_score = general_inference.score_samples_general(
        params, observations[0], projection[0], noise[0]
    )
    single_total = general_inference.log_likelihood_general(
        params,
        observations[0],
        projection[0],
        noise[0],
        sample_weight=scalar_weight,
    )
    single_mean = general_inference.score_general(
        params,
        observations[0],
        projection[0],
        noise[0],
        sample_weight=scalar_weight,
    )
    np.testing.assert_allclose(
        np.asarray(single_total),
        2.5 * np.asarray(single_score),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(single_mean),
        np.asarray(single_score),
        rtol=log_rtol,
        atol=log_atol,
    )


@pytest.mark.parametrize(
    "invalid_shape",
    ((), (1,), (2, 1), (6,), (2, 3, 1)),
    ids=("scalar", "singleton", "lower-rank", "flattened", "trailing-one"),
)
@pytest.mark.parametrize("name", ("log_likelihood_general", "score_general"))
def test_xd_gen_weight_score_001_rejects_every_inexact_weight_shape(
    name, invalid_shape
):
    params, observations, projection, noise = _ordinary_fixed_problem(
        jnp.float64
    )
    observations = observations.reshape(2, 3, 2)
    projection = projection.reshape(2, 3, 2, 4)
    noise = noise.reshape(2, 3, 2, 2)
    sample_weight = jnp.ones(invalid_shape, dtype=jnp.float64)

    with pytest.raises((TypeError, ValueError), match="sample_weight|shape"):
        getattr(general_inference, name)(
            params,
            observations,
            projection,
            noise,
            sample_weight=sample_weight,
        )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_weight_score_001_all_zero_fixed_weights_are_nonraising(
    dtype, _ref_rtol, _ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    sample_weight = jnp.zeros((6,), dtype=dtype)

    total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    normalized = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )

    np.testing.assert_allclose(
        np.asarray(total), np.asarray(0.0, dtype=np.dtype(dtype)),
        rtol=log_rtol, atol=log_atol
    )
    assert np.isnan(np.asarray(normalized))


@pytest.mark.parametrize(
    "dtype,case",
    (
        pytest.param(jnp.float32, "underflow", id="positive-f64-to-f32"),
        pytest.param(jnp.float64, "negative", id="negative-f64-same-dtype"),
    ),
)
def test_jax_subnormal_reduction_weight_domain_survives_device_comparisons(
    dtype, case
):
    """A JAX weight cannot disappear in conversion or sign checks."""

    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    minimum_subnormal = np.nextafter(0.0, 1.0)
    host_weight = np.zeros((6,), dtype=np.float64)
    if case == "underflow":
        host_weight[0] = minimum_subnormal
    else:
        host_weight[0] = -minimum_subnormal
        host_weight[1] = 1.0
    sample_weight = jnp.asarray(host_weight, dtype=jnp.float64)
    assert np.asarray(sample_weight)[0] != 0.0

    total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    normalized = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    jitted_total, jitted_normalized = jax.jit(
        lambda weight: (
            general_inference.log_likelihood_general(
                params,
                observations,
                projection,
                noise,
                sample_weight=weight,
            ),
            general_inference.score_general(
                params,
                observations,
                projection,
                noise,
                sample_weight=weight,
            ),
        )
    )(sample_weight)

    assert np.all(
        np.isnan(
            np.asarray(
                [total, normalized, jitted_total, jitted_normalized]
            )
        )
    )


def test_weighted_score_preserves_minimum_subnormal_common_scale_fixed_and_grouped():
    """A representable positive common scale cannot erase the denominator."""

    dtype = jnp.float64
    minimum_subnormal = np.nextafter(0.0, 1.0)
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray([[0.0], [1.0]], dtype=dtype)
    projection = jnp.asarray([[[1.0]], [[1.0]]], dtype=dtype)
    noise = jnp.asarray([[[1.0]], [[1.0]]], dtype=dtype)
    base_weight = jnp.asarray([1.0, 2.0], dtype=dtype)
    sample_weight = jnp.asarray(
        np.asarray([minimum_subnormal, 2.0 * minimum_subnormal]),
        dtype=dtype,
    )
    np.testing.assert_array_equal(
        np.asarray(sample_weight).view(np.uint64),
        np.asarray([1, 2], dtype=np.uint64),
    )
    expected = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=base_weight,
    )
    base_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=base_weight,
    )
    expected_total = np.asarray(base_total) * minimum_subnormal

    fixed = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    fixed_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    grouped = group_masked_general_inputs(
        params,
        np.asarray(observations),
        np.ones((2, 1), dtype=bool),
        projection=PerItemProjection(np.asarray(projection)),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.asarray(sample_weight),
        dtype=dtype,
    )
    grouped_score = general_inference.score_grouped(grouped)
    grouped_total = general_inference.log_likelihood_grouped(grouped)
    jitted_total = jax.jit(
        lambda value: general_inference.log_likelihood_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )
    )(observations)
    jitted_score = jax.jit(
        lambda value: general_inference.score_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )
    )(observations)

    assert expected_total != 0.0
    np.testing.assert_allclose(np.asarray(fixed), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(grouped_score), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(jitted_score), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(fixed_total), expected_total)
    np.testing.assert_array_equal(np.asarray(grouped_total), expected_total)
    np.testing.assert_array_equal(np.asarray(jitted_total), expected_total)


def test_weighted_reductions_scale_before_opposite_finite_scores_overflow():
    """Finite cancellation survives a large representable common weight."""

    dtype = jnp.float64
    target = 20.0
    log_two_pi = np.log(2.0 * np.pi)
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray(
        [[0.0], [np.sqrt(2.0 * target - log_two_pi)]], dtype=dtype
    )
    projection = jnp.asarray([[[0.0]], [[1.0]]], dtype=dtype)
    noise = jnp.asarray(
        [[[np.exp(-2.0 * target - log_two_pi)]], [[0.0]]], dtype=dtype
    )
    base_weight = jnp.ones((2,), dtype=dtype)
    common_scale = jnp.asarray(1e307, dtype=dtype)
    scaled_weight = base_weight * common_scale

    base_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=base_weight,
    )
    base_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=base_weight,
    )
    scaled_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=scaled_weight,
    )
    scaled_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=scaled_weight,
    )
    grouped = group_masked_general_inputs(
        params,
        np.asarray(observations),
        np.ones((2, 1), dtype=bool),
        projection=PerItemProjection(np.asarray(projection)),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.asarray(scaled_weight),
        dtype=dtype,
    )
    grouped_total = general_inference.log_likelihood_grouped(grouped)
    grouped_score = general_inference.score_grouped(grouped)

    expected_total = base_total * common_scale
    assert np.isfinite(np.asarray(expected_total))
    np.testing.assert_allclose(
        np.asarray(scaled_total), np.asarray(expected_total), rtol=8e-10
    )
    np.testing.assert_allclose(np.asarray(scaled_score), np.asarray(base_score))
    np.testing.assert_allclose(
        np.asarray(grouped_total), np.asarray(expected_total), rtol=8e-10
    )
    np.testing.assert_allclose(np.asarray(grouped_score), np.asarray(base_score))


def test_adjacent_large_weights_survive_overflowing_products_and_cancellation():
    """A finite residual retains weight low bits despite product overflow."""

    dtype = jnp.float64
    target = 3.0
    log_two_pi = np.log(2.0 * np.pi)
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray(
        [[0.0], [np.sqrt(2.0 * target - log_two_pi)]], dtype=dtype
    )
    projection = jnp.asarray([[[0.0]], [[1.0]]], dtype=dtype)
    noise = jnp.asarray(
        [[[np.exp(-2.0 * target - log_two_pi)]], [[0.0]]], dtype=dtype
    )
    first_weight = np.float64(8e307)
    sample_weight = jnp.asarray(
        [first_weight, np.nextafter(first_weight, np.inf)], dtype=dtype
    )
    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    score_values = np.asarray(scores)
    weight_values = np.asarray(sample_weight)
    common_weight_term = weight_values[0] * (
        score_values[0] + score_values[1]
    )
    adjacent_weight_term = (
        weight_values[1] - weight_values[0]
    ) * score_values[1]
    expected_total = common_weight_term + adjacent_weight_term
    expected_score = expected_total / np.sum(weight_values)

    fixed_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    fixed_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    grouped = group_masked_general_inputs(
        params,
        np.asarray(observations),
        np.ones((2, 1), dtype=bool),
        projection=PerItemProjection(np.asarray(projection)),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.asarray(sample_weight),
        dtype=dtype,
    )

    assert score_values[0] > 0.0 and score_values[1] < 0.0
    assert np.all(
        weight_values
        > np.finfo(np.float64).max / np.min(np.abs(score_values))
    )
    assert np.isfinite(common_weight_term)
    assert np.isfinite(adjacent_weight_term)
    assert np.isfinite(expected_total) and expected_total != 0.0
    np.testing.assert_allclose(np.asarray(fixed_total), expected_total)
    np.testing.assert_allclose(
        np.asarray(general_inference.log_likelihood_grouped(grouped)),
        expected_total,
    )
    np.testing.assert_allclose(np.asarray(fixed_score), expected_score)
    np.testing.assert_allclose(
        np.asarray(general_inference.score_grouped(grouped)), expected_score
    )

    def total_from_observations(value):
        return general_inference.log_likelihood_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    def score_from_observations(value):
        return general_inference.score_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    jitted_total = jax.jit(total_from_observations)(observations)
    jitted_score = jax.jit(score_from_observations)(observations)
    unweighted_gradient = jax.grad(
        lambda value: jnp.sum(
            general_inference.score_samples_general(
                params, value, projection, noise
            )
        )
    )(observations)
    expected_total_gradient = unweighted_gradient * sample_weight[:, None]
    expected_score_gradient = expected_total_gradient / np.sum(weight_values)
    total_gradient = jax.grad(total_from_observations)(observations)
    score_gradient = jax.jit(jax.grad(score_from_observations))(observations)

    np.testing.assert_allclose(np.asarray(jitted_total), expected_total)
    np.testing.assert_allclose(np.asarray(jitted_score), expected_score)
    assert np.all(np.isfinite(np.asarray(expected_total_gradient)))
    assert np.all(np.isfinite(np.asarray(expected_score_gradient)))
    np.testing.assert_allclose(
        np.asarray(total_gradient), np.asarray(expected_total_gradient)
    )
    np.testing.assert_allclose(
        np.asarray(score_gradient), np.asarray(expected_score_gradient)
    )


def test_unrecoverable_scale_separation_is_an_explicit_raw_failure():
    """An unrecoverable subnormal residual returns the failure sentinel."""

    dtype = jnp.float64
    target = 5.0
    log_two_pi = np.log(2.0 * np.pi)
    minimum_subnormal = np.nextafter(0.0, 1.0)
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray(
        [
            [0.0],
            [np.sqrt(2.0 * target - log_two_pi)],
            [np.sqrt(2.0 - log_two_pi)],
        ],
        dtype=dtype,
    )
    projection = jnp.asarray([[[0.0]], [[1.0]], [[1.0]]], dtype=dtype)
    noise = jnp.asarray(
        [
            [[np.exp(-2.0 * target - log_two_pi)]],
            [[0.0]],
            [[0.0]],
        ],
        dtype=dtype,
    )
    sample_weight = jnp.asarray(
        [4e307, 4e307, minimum_subnormal], dtype=dtype
    )
    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    grouped = group_masked_general_inputs(
        params,
        np.asarray(observations),
        np.ones((3, 1), dtype=bool),
        projection=PerItemProjection(np.asarray(projection)),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.asarray(sample_weight),
        dtype=dtype,
    )

    np.testing.assert_array_equal(
        np.asarray(scores), np.asarray([target, -target, -1.0])
    )
    assert 4e307 > np.finfo(np.float64).max / target
    fixed_total = np.asarray(
        general_inference.log_likelihood_general(
            params,
            observations,
            projection,
            noise,
            sample_weight=sample_weight,
        )
    )
    grouped_total = np.asarray(
        general_inference.log_likelihood_grouped(grouped)
    )
    assert np.isnan(fixed_total)
    assert np.isnan(grouped_total)


def test_nonrepresentable_raw_total_fails_but_finite_mean_is_preserved():
    """Raw overflow is explicit without erasing a representable mean."""

    dtype = jnp.float64
    target = 20.0
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray(
        [[np.sqrt(2.0 * target - np.log(2.0 * np.pi))]], dtype=dtype
    )
    projection = jnp.asarray([[[1.0]]], dtype=dtype)
    noise = jnp.zeros((1, 1, 1), dtype=dtype)
    sample_weight = jnp.asarray([1e307], dtype=dtype)
    expected_score = general_inference.score_samples_general(
        params, observations, projection, noise
    )[0]

    fixed_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    fixed_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    grouped = group_masked_general_inputs(
        params,
        np.asarray(observations),
        np.ones((1, 1), dtype=bool),
        projection=PerItemProjection(np.asarray(projection)),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.asarray(sample_weight),
        dtype=dtype,
    )
    grouped_total = general_inference.log_likelihood_grouped(grouped)
    grouped_score = general_inference.score_grouped(grouped)

    np.testing.assert_array_equal(
        np.isnan(np.asarray([fixed_total, grouped_total])),
        np.ones((2,), dtype=bool),
    )
    np.testing.assert_allclose(np.asarray(fixed_score), np.asarray(expected_score))
    np.testing.assert_allclose(
        np.asarray(grouped_score), np.asarray(expected_score)
    )


def test_nonrepresentable_weight_sum_is_an_explicit_fixed_failure():
    """Finite elements whose collection mass overflows cannot succeed."""

    dtype = jnp.float64
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.zeros((2, 1), dtype=dtype)
    projection = jnp.zeros((2, 1, 1), dtype=dtype)
    zero_score_variance = np.exp(-np.log(2.0 * np.pi))
    noise = jnp.full((2, 1, 1), zero_score_variance, dtype=dtype)
    half_maximum = np.finfo(np.float64).max / 2.0
    overflowing_element = np.nextafter(half_maximum, np.inf)
    sample_weight = jnp.full((2,), overflowing_element, dtype=dtype)

    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    fixed_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )
    fixed_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=sample_weight,
    )

    np.testing.assert_array_equal(np.asarray(scores), np.zeros((2,)))
    assert np.isnan(np.asarray(fixed_total))
    assert np.isnan(np.asarray(fixed_score))
    with pytest.raises(PrecisionError, match="informative total"):
        group_masked_general_inputs(
            params,
            np.asarray(observations),
            np.ones((2, 1), dtype=bool),
            projection=PerItemProjection(np.asarray(projection)),
            noise=PerItemFullNoise(np.asarray(noise)),
            sample_weight=np.asarray(sample_weight),
            dtype=dtype,
        )


def test_zero_valued_successful_score_has_finite_fixed_reduction_gradients():
    """The signed stable reducer stays differentiable through exact zero."""

    dtype = jnp.float64
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.zeros((1, 1), dtype=dtype)
    projection = jnp.zeros((1, 1, 1), dtype=dtype)
    zero_score_variance = np.exp(-np.log(2.0 * np.pi))
    noise = jnp.full((1, 1, 1), zero_score_variance, dtype=dtype)
    sample_weight = jnp.ones((1,), dtype=dtype)

    def total(value):
        return general_inference.log_likelihood_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    def normalized(value):
        return general_inference.score_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    total_gradient = jax.grad(total)(observations)
    normalized_gradient = jax.jit(jax.grad(normalized))(observations)

    np.testing.assert_array_equal(np.asarray(total(observations)), 0.0)
    np.testing.assert_array_equal(np.asarray(normalized(observations)), 0.0)
    np.testing.assert_array_equal(np.asarray(total_gradient), np.zeros((1, 1)))
    np.testing.assert_array_equal(
        np.asarray(normalized_gradient), np.zeros((1, 1))
    )


def test_exact_signed_cancellation_has_finite_fixed_reduction_gradients():
    """A zero sum of nonzero terms retains its ordinary derivatives."""

    dtype = jnp.float64
    target = 1.0
    log_two_pi = np.log(2.0 * np.pi)
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    observations = jnp.asarray(
        [[0.0], [np.sqrt(2.0 * target - log_two_pi)]], dtype=dtype
    )
    projection = jnp.asarray([[[0.0]], [[1.0]]], dtype=dtype)
    noise = jnp.asarray(
        [[[np.exp(-2.0 * target - log_two_pi)]], [[0.0]]], dtype=dtype
    )
    sample_weight = jnp.ones((2,), dtype=dtype)

    def total(value):
        return general_inference.log_likelihood_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    def normalized(value):
        return general_inference.score_general(
            params,
            value,
            projection,
            noise,
            sample_weight=sample_weight,
        )

    expected_total_gradient = jax.grad(
        lambda value: jnp.sum(
            general_inference.score_samples_general(
                params, value, projection, noise
            )
        )
    )(observations)
    expected_normalized_gradient = expected_total_gradient / 2.0
    total_gradient = jax.jit(jax.grad(total))(observations)
    normalized_gradient = jax.grad(normalized)(observations)

    np.testing.assert_array_equal(np.asarray(total(observations)), 0.0)
    np.testing.assert_array_equal(np.asarray(normalized(observations)), 0.0)
    assert np.all(np.isfinite(np.asarray(expected_total_gradient)))
    np.testing.assert_allclose(
        np.asarray(total_gradient), np.asarray(expected_total_gradient)
    )
    np.testing.assert_allclose(
        np.asarray(normalized_gradient),
        np.asarray(expected_normalized_gradient),
    )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,_log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_pred_001_dense_tie_uses_lowest_component_index(
    dtype, _ref_rtol, _ref_atol, _log_rtol, log_atol
):
    params = _params(
        dtype,
        [1.0 / 3.0] * 3,
        np.zeros((3, 3)),
        np.broadcast_to(
            np.asarray(
                [[0.8, 0.1, -0.04], [0.1, 0.7, 0.06], [-0.04, 0.06, 0.9]]
            ),
            (3, 3, 3),
        ),
    )
    observation = jnp.asarray([0.25, -0.4], dtype=dtype)
    projection = jnp.asarray(
        [[1.0, 0.3, -0.2], [0.1, -0.5, 0.8]], dtype=dtype
    )
    noise = jnp.asarray([[0.25, 0.04], [0.04, 0.35]], dtype=dtype)

    probabilities = general_inference.predict_proba_general(
        params, observation, projection, noise
    )
    label = general_inference.predict_general(
        params, observation, projection, noise
    )

    np.testing.assert_allclose(
        np.asarray(probabilities),
        np.full(3, 1.0 / 3.0),
        rtol=0.0,
        atol=log_atol,
    )
    assert int(np.asarray(label)) == 0


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,_log_rtol,_log_atol", DTYPE_CASES
)
def test_xd_gen_m0_001_fixed_m0_returns_exact_prior_inference(
    dtype, ref_rtol, ref_atol, _log_rtol, _log_atol
):
    params = _params(
        dtype,
        [0.2, 0.3, 0.5],
        [[-0.5, 0.2], [0.8, -0.3], [0.1, 1.0]],
        [
            [[0.7, 0.1], [0.1, 0.9]],
            [[0.6, -0.05], [-0.05, 0.8]],
            [[0.9, 0.02], [0.02, 0.5]],
        ],
    )
    batch_shape = (2, 3)
    observations = jnp.empty(batch_shape + (0,), dtype=dtype)
    projection = jnp.empty(batch_shape + (0, 2), dtype=dtype)
    noise = jnp.empty(batch_shape + (0, 0), dtype=dtype)
    expected_mean, expected_covariance = _prior_moments(params)

    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    probabilities = general_inference.predict_proba_general(
        params, observations, projection, noise
    )
    labels = general_inference.predict_general(
        params, observations, projection, noise
    )
    mean, covariance = general_inference.posterior_general(
        params, observations, projection, noise
    )
    mean_only = general_inference.posterior_mean_general(
        params, observations, projection, noise
    )
    total = general_inference.log_likelihood_general(
        params, observations, projection, noise
    )
    normalized = general_inference.score_general(
        params, observations, projection, noise
    )

    np.testing.assert_array_equal(np.asarray(scores), np.zeros(batch_shape))
    np.testing.assert_array_equal(
        np.asarray(probabilities),
        np.broadcast_to(np.asarray(params.weights), batch_shape + (3,)),
    )
    np.testing.assert_array_equal(
        np.asarray(labels), np.full(batch_shape, 2, dtype=np.int32)
    )
    np.testing.assert_allclose(
        np.asarray(mean),
        np.broadcast_to(expected_mean, batch_shape + (2,)),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(covariance),
        np.broadcast_to(expected_covariance, batch_shape + (2, 2)),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(np.asarray(mean_only), np.asarray(mean))
    assert float(np.asarray(total)) == 0.0
    assert float(np.asarray(normalized)) == 0.0


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.float32))
@pytest.mark.parametrize(
    "batch_shape", ((0,), (2, 0)), ids=("zero-leading", "zero-inner")
)
def test_zero_length_m_positive_fixed_inference_has_bounded_reduction_semantics(
    dtype, batch_shape
):
    params, _, _, _ = _ordinary_fixed_problem(dtype)
    observations = jnp.empty(batch_shape + (2,), dtype=dtype)
    projection = jnp.empty(batch_shape + (2, 4), dtype=dtype)
    noise = jnp.empty(batch_shape + (2, 2), dtype=dtype)

    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    probabilities = general_inference.predict_proba_general(
        params, observations, projection, noise
    )
    labels = general_inference.predict_general(
        params, observations, projection, noise
    )
    mean, covariance = general_inference.posterior_general(
        params, observations, projection, noise
    )
    mean_only = general_inference.posterior_mean_general(
        params, observations, projection, noise
    )
    total = general_inference.log_likelihood_general(
        params, observations, projection, noise
    )
    normalized = general_inference.score_general(
        params, observations, projection, noise
    )

    assert scores.shape == batch_shape
    assert probabilities.shape == batch_shape + (3,)
    assert labels.shape == batch_shape
    assert mean.shape == batch_shape + (4,)
    assert covariance.shape == batch_shape + (4, 4)
    assert mean_only.shape == batch_shape + (4,)
    assert scores.size == probabilities.size == labels.size == mean.size == 0
    assert float(np.asarray(total)) == 0.0
    assert np.isnan(np.asarray(normalized))


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.float32))
@pytest.mark.parametrize(
    "batch_shape", ((0,), (2, 0)), ids=("zero-leading", "zero-inner")
)
@pytest.mark.parametrize("observed_dimension", (0, 2), ids=("m0", "m2"))
def test_zero_length_fixed_invalid_global_jitter_poison_reductions_without_raising(
    dtype, batch_shape, observed_dimension
):
    params, _, _, _ = _ordinary_fixed_problem(dtype)
    observations = jnp.empty(batch_shape + (observed_dimension,), dtype=dtype)
    projection = jnp.empty(
        batch_shape + (observed_dimension, 4), dtype=dtype
    )
    noise = jnp.empty(
        batch_shape + (observed_dimension, observed_dimension), dtype=dtype
    )
    detailed = posterior_components_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )

    assert bool(np.asarray(detailed.numerical_failure))
    assert detailed.failed_pairs.size == 0
    scores = general_inference.score_samples_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )
    total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )
    normalized = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )
    probabilities = general_inference.predict_proba_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )
    labels = general_inference.predict_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )
    posterior = general_inference.posterior_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=-1.0,
    )

    assert scores.shape == batch_shape and scores.size == 0
    assert probabilities.shape == batch_shape + (3,)
    assert labels.shape == batch_shape
    assert posterior[0].shape == batch_shape + (4,)
    assert posterior[1].shape == batch_shape + (4, 4)
    assert np.isnan(np.asarray(total))
    assert np.isnan(np.asarray(normalized))


@pytest.mark.parametrize(
    "dtype,jitter_value",
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
def test_jax_subnormal_jitter_domain_survives_device_comparisons(
    dtype, jitter_value
):
    """A JAX subnormal cannot disappear in conversion or sign checks."""

    jitter = jnp.asarray(jitter_value, dtype=jnp.float64)
    assert np.asarray(jitter) != 0.0
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    detailed = posterior_components_general(
        params,
        observations,
        projection,
        noise,
        factor_jitter=jitter,
    )
    grouped, _ = _ordinary_grouped_problem(dtype)
    grouped_detailed = posterior_components_grouped(
        grouped, factor_jitter=jitter
    )
    jitted_total = jax.jit(
        lambda control: general_inference.log_likelihood_general(
            params,
            observations,
            projection,
            noise,
            factor_jitter=control,
        )
    )(jitter)

    assert bool(np.asarray(detailed.numerical_failure))
    assert np.all(np.asarray(detailed.failed_pairs))
    assert bool(np.asarray(grouped_detailed.e_step.numerical_failure))
    assert np.all(np.asarray(grouped_detailed.e_step.failed_pairs))
    assert np.all(
        np.isnan(
            np.asarray(
                general_inference.score_samples_general(
                    params,
                    observations,
                    projection,
                    noise,
                    factor_jitter=jitter,
                )
            )
        )
    )
    assert np.all(
        np.asarray(
            general_inference.predict_general(
                params,
                observations,
                projection,
                noise,
                factor_jitter=jitter,
            )
        )
        == -1
    )
    assert np.isnan(
        np.asarray(
            general_inference.log_likelihood_general(
                params,
                observations,
                projection,
                noise,
                factor_jitter=jitter,
            )
        )
    )
    assert np.isnan(np.asarray(jitted_total))
    assert np.all(
        np.isnan(
            np.asarray(
                general_inference.posterior_mean_grouped(
                    grouped, factor_jitter=jitter
                )
            )
        )
    )
    assert np.all(
        np.asarray(
            general_inference.predict_grouped(
                grouped, factor_jitter=jitter
            )
        )
        == -1
    )
    assert np.isnan(
        np.asarray(
            general_inference.score_grouped(
                grouped, factor_jitter=jitter
            )
        )
    )


def _fixed_failure_problem(dtype, sample_weight):
    params = _params(
        dtype,
        [0.4, 0.6],
        [[-0.2, 0.3], [0.7, -0.4]],
        [np.eye(2), np.asarray([[0.8, 0.05], [0.05, 1.1]])],
    )
    observations = jnp.asarray([[0.4], [5.0]], dtype=dtype)
    projection = jnp.asarray(
        [[[1.0, 0.0]], [[0.0, 0.0]]], dtype=dtype
    )
    noise = jnp.asarray([[[0.2]], [[0.0]]], dtype=dtype)
    return (
        params,
        observations,
        projection,
        noise,
        jnp.asarray(sample_weight, dtype=dtype),
    )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_infer_fail_001_fixed_sentinels_and_weight_participation(
    dtype, _ref_rtol, _ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise, zero_failed_weight = (
        _fixed_failure_problem(dtype, [2.0, 0.0])
    )
    detailed = posterior_components_general(
        params, observations, projection, noise
    )
    assert bool(np.asarray(detailed.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(detailed.failed_pairs), [[False, False], [True, True]]
    )

    scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    probabilities = general_inference.predict_proba_general(
        params, observations, projection, noise
    )
    labels = general_inference.predict_general(
        params, observations, projection, noise
    )
    mean, covariance = general_inference.posterior_general(
        params, observations, projection, noise
    )
    mean_only = general_inference.posterior_mean_general(
        params, observations, projection, noise
    )

    assert np.isfinite(np.asarray(scores[0]))
    assert np.isnan(np.asarray(scores[1]))
    assert np.all(np.isfinite(np.asarray(probabilities[0])))
    assert np.all(np.isnan(np.asarray(probabilities[1])))
    assert int(np.asarray(labels[1])) == -1
    assert np.all(np.isfinite(np.asarray(mean[0])))
    assert np.all(np.isnan(np.asarray(mean[1])))
    assert np.all(np.isfinite(np.asarray(covariance[0])))
    assert np.all(np.isnan(np.asarray(covariance[1])))
    np.testing.assert_array_equal(np.asarray(mean_only), np.asarray(mean))

    zero_weight_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=zero_failed_weight,
    )
    zero_weight_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=zero_failed_weight,
    )
    np.testing.assert_allclose(
        np.asarray(zero_weight_total),
        2.0 * np.asarray(scores[0]),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(zero_weight_score),
        np.asarray(scores[0]),
        rtol=log_rtol,
        atol=log_atol,
    )

    positive_failed_weight = jnp.asarray([2.0, 3.0], dtype=dtype)
    positive_total = general_inference.log_likelihood_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=positive_failed_weight,
    )
    positive_score = general_inference.score_general(
        params,
        observations,
        projection,
        noise,
        sample_weight=positive_failed_weight,
    )
    assert np.isnan(np.asarray(positive_total))
    assert np.isnan(np.asarray(positive_score))


def test_xd_gen_infer_fail_001_one_failed_pair_invalidates_entire_convenience_row():
    params = _params(
        jnp.float64,
        [0.4, 0.6],
        [[0.0], [1.0]],
        [[[1.0]], [[np.nan]]],
    )
    observation = jnp.asarray([0.25], dtype=jnp.float64)
    projection = jnp.asarray([[1.0]], dtype=jnp.float64)
    noise = jnp.asarray([[0.2]], dtype=jnp.float64)
    detailed = posterior_components_general(
        params, observation, projection, noise
    )

    np.testing.assert_array_equal(
        np.asarray(detailed.failed_pairs), [False, True]
    )
    assert np.isfinite(np.asarray(detailed.score_samples))
    assert np.all(np.isfinite(np.asarray(detailed.responsibilities)))
    assert np.isnan(
        np.asarray(
            general_inference.score_samples_general(
                params, observation, projection, noise
            )
        )
    )
    assert np.all(
        np.isnan(
            np.asarray(
                general_inference.predict_proba_general(
                    params, observation, projection, noise
                )
            )
        )
    )
    assert int(
        np.asarray(
            general_inference.predict_general(
                params, observation, projection, noise
            )
        )
    ) == -1
    posterior = general_inference.posterior_general(
        params, observation, projection, noise
    )
    assert np.all(np.isnan(np.asarray(posterior[0])))
    assert np.all(np.isnan(np.asarray(posterior[1])))


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_jit_001_fixed_conveniences_are_callback_free_and_jittable(
    dtype, ref_rtol, ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    sample_weight = jnp.asarray([0.5, 1.0, 2.0, 0.0, 1.5, 0.75], dtype=dtype)

    def all_operations(parameters, x, r, s, weight):
        return (
            general_inference.score_samples_general(parameters, x, r, s),
            general_inference.log_likelihood_general(
                parameters, x, r, s, sample_weight=weight
            ),
            general_inference.score_general(
                parameters, x, r, s, sample_weight=weight
            ),
            general_inference.predict_proba_general(parameters, x, r, s),
            general_inference.predict_general(parameters, x, r, s),
            *general_inference.posterior_general(parameters, x, r, s),
            general_inference.posterior_mean_general(parameters, x, r, s),
        )

    jaxpr_text = str(
        jax.make_jaxpr(all_operations)(
            params, observations, projection, noise, sample_weight
        )
    )
    assert "callback" not in jaxpr_text.lower()
    eager = all_operations(
        params, observations, projection, noise, sample_weight
    )
    compiled_function = jax.jit(all_operations)
    compiled = compiled_function(
        params, observations, projection, noise, sample_weight
    )
    _assert_tree_allclose(
        compiled,
        eager,
        rtol=max(ref_rtol, log_rtol),
        atol=max(ref_atol, log_atol),
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_xd_gen_vmap_001_fixed_single_item_paths_match_native_batch(
    dtype, ref_rtol, ref_atol, log_rtol, log_atol
):
    params, observations, projection, noise = _ordinary_fixed_problem(dtype)
    native_scores = general_inference.score_samples_general(
        params, observations, projection, noise
    )
    native_probabilities = general_inference.predict_proba_general(
        params, observations, projection, noise
    )
    native_labels = general_inference.predict_general(
        params, observations, projection, noise
    )
    native_mean, native_covariance = general_inference.posterior_general(
        params, observations, projection, noise
    )
    native_mean_only = general_inference.posterior_mean_general(
        params, observations, projection, noise
    )

    mapped = jax.vmap(
        lambda x_i, r_i, s_i: (
            general_inference.score_samples_general(params, x_i, r_i, s_i),
            general_inference.log_likelihood_general(params, x_i, r_i, s_i),
            general_inference.score_general(params, x_i, r_i, s_i),
            general_inference.predict_proba_general(params, x_i, r_i, s_i),
            general_inference.predict_general(params, x_i, r_i, s_i),
            *general_inference.posterior_general(params, x_i, r_i, s_i),
            general_inference.posterior_mean_general(params, x_i, r_i, s_i),
        )
    )(observations, projection, noise)

    for value in mapped[:3]:
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(native_scores),
            rtol=log_rtol,
            atol=log_atol,
        )
    np.testing.assert_allclose(
        np.asarray(mapped[3]),
        np.asarray(native_probabilities),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(np.asarray(mapped[4]), np.asarray(native_labels))
    np.testing.assert_allclose(
        np.asarray(mapped[5]),
        np.asarray(native_mean),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(mapped[6]),
        np.asarray(native_covariance),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(mapped[7]),
        np.asarray(native_mean_only),
        rtol=ref_rtol,
        atol=ref_atol,
    )


def _ordinary_grouped_problem(dtype):
    params, _, _, _ = _ordinary_fixed_problem(dtype)
    observations = np.asarray(
        [
            [-1.1, 0.2, 0.5],
            [0.4, -0.7, 0.1],
            [1.3, 0.5, -0.4],
            [-0.2, 0.9, 0.7],
            [0.8, -0.1, 0.35],
            [1.7, 0.4, -0.6],
            [0.1, -0.25, 0.8],
        ],
        dtype=np.dtype(dtype),
    )
    mask = np.asarray(
        [
            [True, False, False],
            [True, True, False],
            [False, False, False],
            [True, True, True],
            [True, False, True],
            [False, True, False],
            [False, False, False],
        ],
        dtype=bool,
    )
    base_projection = np.asarray(
        [
            [1.0, 0.2, -0.15, 0.05],
            [-0.25, 0.75, 0.3, -0.4],
            [0.35, -0.2, 0.6, 0.5],
        ],
        dtype=np.float64,
    )
    projection = np.stack(
        [base_projection + (row - 3.0) * 0.006 for row in range(7)]
    ).astype(np.dtype(dtype))
    noise = np.empty((7, 3, 3), dtype=np.dtype(dtype))
    for row in range(7):
        noise[row] = np.asarray(
            [
                [0.22 + 0.01 * row, 0.012, -0.006],
                [0.012, 0.31 + 0.015 * row, 0.008],
                [-0.006, 0.008, 0.27 + 0.02 * row],
            ],
            dtype=np.dtype(dtype),
        )
    sample_weight = np.asarray(
        [0.5, 1.0, 99.0, 2.0, 0.0, 0.75, 500.0],
        dtype=np.dtype(dtype),
    )
    grouped = group_masked_general_inputs(
        params,
        observations,
        mask,
        projection=PerItemProjection(projection),
        noise=PerItemFullNoise(noise),
        sample_weight=sample_weight,
        dtype=dtype,
    )
    return grouped, mask


def _grouped_reference(grouped):
    leaves = [
        general_e_step(
            np.asarray(group.observations),
            np.asarray(group.projection_matrices),
            np.asarray(group.measurement_covariances),
            np.asarray(grouped.parameters.weights),
            np.asarray(grouped.parameters.means),
            np.asarray(grouped.parameters.covariances),
        )
        for group in grouped.groups
    ]
    restoration = np.asarray(grouped.restoration_indices)

    def restore(values):
        return np.concatenate(values, axis=0)[restoration]

    score_samples = restore([leaf.score_samples for leaf in leaves])
    probabilities = restore([leaf.responsibilities for leaf in leaves])
    means_and_covariances = [
        reference_marginalized_posterior(leaf) for leaf in leaves
    ]
    posterior_mean = restore([value[0] for value in means_and_covariances])
    posterior_covariance = restore([value[1] for value in means_and_covariances])
    raw, informative_weight, objective = general_grouped_objective(
        leaves, [np.asarray(group.sample_weight) for group in grouped.groups]
    )
    return (
        leaves,
        score_samples,
        probabilities,
        posterior_mean,
        posterior_covariance,
        raw,
        informative_weight,
        objective,
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_grouped_conveniences_restore_rows_and_match_oracle(
    dtype, ref_rtol, ref_atol, log_rtol, log_atol
):
    grouped, mask = _ordinary_grouped_problem(dtype)
    (
        _leaves,
        expected_scores,
        expected_probabilities,
        expected_mean,
        expected_covariance,
        expected_total,
        expected_informative_weight,
        expected_score,
    ) = _grouped_reference(grouped)
    detailed = posterior_components_grouped(grouped)

    scores = general_inference.score_samples_grouped(grouped)
    probabilities = general_inference.predict_proba_grouped(grouped)
    labels = general_inference.predict_grouped(grouped)
    mean, covariance = general_inference.posterior_grouped(grouped)
    mean_only = general_inference.posterior_mean_grouped(grouped)
    total = general_inference.log_likelihood_grouped(grouped)
    normalized = general_inference.score_grouped(grouped)

    assert scores.shape == (7,)
    assert probabilities.shape == (7, 3)
    assert labels.shape == (7,)
    assert mean.shape == (7, 4)
    assert covariance.shape == (7, 4, 4)
    assert mean_only.shape == (7, 4)
    assert not bool(np.asarray(detailed.e_step.numerical_failure))
    assert not np.any(np.asarray(detailed.group_numerical_failure))
    np.testing.assert_allclose(
        np.asarray(scores), expected_scores, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(probabilities),
        expected_probabilities,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(
        np.asarray(labels), np.argmax(expected_probabilities, axis=-1)
    )
    np.testing.assert_allclose(
        np.asarray(mean), expected_mean, rtol=ref_rtol, atol=ref_atol
    )
    np.testing.assert_allclose(
        np.asarray(covariance),
        expected_covariance,
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(np.asarray(mean_only), np.asarray(mean))
    np.testing.assert_allclose(
        np.asarray(total), expected_total, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(normalized), expected_score, rtol=log_rtol, atol=log_atol
    )
    np.testing.assert_allclose(
        np.asarray(grouped.informative_weight),
        expected_informative_weight,
        rtol=0.0,
        atol=0.0,
    )
    # Fully missing weights 99 and 500 are excluded from the denominator.
    expected_weight_from_mask = np.sum(
        np.concatenate([np.asarray(group.sample_weight) for group in grouped.groups])[
            np.asarray(grouped.restoration_indices)
        ][np.any(mask, axis=1)]
    )
    np.testing.assert_allclose(
        expected_informative_weight, expected_weight_from_mask, rtol=0.0, atol=0.0
    )


def _grouped_failure_problem(dtype, sample_weight):
    params = _params(
        dtype,
        [0.4, 0.6],
        [[-0.2, 0.3], [0.7, -0.4]],
        [np.eye(2), np.asarray([[0.8, 0.05], [0.05, 1.1]])],
    )
    observations = np.asarray([[0.4, 3.0], [5.0, -2.0]], dtype=np.dtype(dtype))
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    projection = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.dtype(dtype),
    )
    noise = np.asarray(
        [
            [[0.2, 0.0], [0.0, 0.0]],
            [[0.2, 0.0], [0.0, 0.0]],
        ],
        dtype=np.dtype(dtype),
    )
    return group_masked_general_inputs(
        params,
        observations,
        mask,
        projection=PerItemProjection(projection),
        noise=PerItemFullNoise(noise),
        sample_weight=np.asarray(sample_weight, dtype=np.dtype(dtype)),
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "dtype,_ref_rtol,_ref_atol,log_rtol,log_atol", DTYPE_CASES
)
def test_grouped_failure_sentinels_restore_rows_and_obey_weight_participation(
    dtype, _ref_rtol, _ref_atol, log_rtol, log_atol
):
    zero_failed_weight = _grouped_failure_problem(dtype, [2.0, 0.0])
    detailed = posterior_components_grouped(zero_failed_weight)
    assert bool(np.asarray(detailed.e_step.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(detailed.e_step.failed_pairs),
        [[False, False], [True, True]],
    )
    # Lexicographic masks place the failed (False, True) group first.
    np.testing.assert_array_equal(
        np.asarray(detailed.group_numerical_failure), [True, False]
    )

    scores = general_inference.score_samples_grouped(zero_failed_weight)
    probabilities = general_inference.predict_proba_grouped(zero_failed_weight)
    labels = general_inference.predict_grouped(zero_failed_weight)
    mean, covariance = general_inference.posterior_grouped(zero_failed_weight)
    assert np.isfinite(np.asarray(scores[0]))
    assert np.isnan(np.asarray(scores[1]))
    assert np.all(np.isfinite(np.asarray(probabilities[0])))
    assert np.all(np.isnan(np.asarray(probabilities[1])))
    assert int(np.asarray(labels[1])) == -1
    assert np.all(np.isfinite(np.asarray(mean[0])))
    assert np.all(np.isnan(np.asarray(mean[1])))
    assert np.all(np.isfinite(np.asarray(covariance[0])))
    assert np.all(np.isnan(np.asarray(covariance[1])))

    total = general_inference.log_likelihood_grouped(zero_failed_weight)
    normalized = general_inference.score_grouped(zero_failed_weight)
    np.testing.assert_allclose(
        np.asarray(total),
        2.0 * np.asarray(scores[0]),
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(normalized),
        np.asarray(scores[0]),
        rtol=log_rtol,
        atol=log_atol,
    )

    positive_failed_weight = _grouped_failure_problem(dtype, [2.0, 3.0])
    assert np.isnan(
        np.asarray(
            general_inference.log_likelihood_grouped(positive_failed_weight)
        )
    )
    assert np.isnan(
        np.asarray(general_inference.score_grouped(positive_failed_weight))
    )


def _all_m0_grouped_problem(dtype):
    params = _params(
        dtype,
        [0.2, 0.3, 0.5],
        [[-0.5, 0.2], [0.8, -0.3], [0.1, 1.0]],
        [
            [[0.7, 0.1], [0.1, 0.9]],
            [[0.6, -0.05], [-0.05, 0.8]],
            [[0.9, 0.02], [0.02, 0.5]],
        ],
    )
    observations = np.asarray(
        [[0.1, -0.2], [0.4, 0.7], [-0.8, 0.3], [0.2, 1.1]],
        dtype=np.dtype(dtype),
    )
    mask = np.zeros((4, 2), dtype=bool)
    projection = np.broadcast_to(np.eye(2), (4, 2, 2)).astype(np.dtype(dtype))
    noise = np.broadcast_to(
        np.asarray([[0.2, 0.03], [0.03, 0.4]], dtype=np.dtype(dtype)),
        (4, 2, 2),
    ).copy()
    return group_masked_general_inputs(
        params,
        observations,
        mask,
        projection=PerItemProjection(projection),
        noise=PerItemFullNoise(noise),
        sample_weight=np.asarray([0.0, 2.0, 5.0, 0.0], dtype=np.dtype(dtype)),
        dtype=dtype,
    )


@pytest.mark.parametrize(
    "dtype,ref_rtol,ref_atol,_log_rtol,_log_atol", DTYPE_CASES
)
def test_xd_gen_m0_001_grouped_all_m0_total_and_score_are_exact_zero(
    dtype, ref_rtol, ref_atol, _log_rtol, _log_atol
):
    grouped = _all_m0_grouped_problem(dtype)
    expected_mean, expected_covariance = _prior_moments(grouped.parameters)
    detailed = posterior_components_grouped(grouped)

    scores = general_inference.score_samples_grouped(grouped)
    probabilities = general_inference.predict_proba_grouped(grouped)
    labels = general_inference.predict_grouped(grouped)
    mean, covariance = general_inference.posterior_grouped(grouped)
    mean_only = general_inference.posterior_mean_grouped(grouped)
    total = general_inference.log_likelihood_grouped(grouped)
    normalized = general_inference.score_grouped(grouped)

    assert not bool(np.asarray(detailed.e_step.numerical_failure))
    assert float(np.asarray(grouped.informative_weight)) == 0.0
    np.testing.assert_array_equal(np.asarray(scores), np.zeros(4))
    np.testing.assert_array_equal(
        np.asarray(probabilities),
        np.broadcast_to(np.asarray(grouped.parameters.weights), (4, 3)),
    )
    np.testing.assert_array_equal(np.asarray(labels), np.full(4, 2))
    np.testing.assert_allclose(
        np.asarray(mean),
        np.broadcast_to(expected_mean, (4, 2)),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_allclose(
        np.asarray(covariance),
        np.broadcast_to(expected_covariance, (4, 2, 2)),
        rtol=ref_rtol,
        atol=ref_atol,
    )
    np.testing.assert_array_equal(np.asarray(mean_only), np.asarray(mean))
    assert float(np.asarray(total)) == 0.0
    assert float(np.asarray(normalized)) == 0.0


@pytest.mark.parametrize("dtype", (jnp.float64, jnp.float32))
def test_grouped_all_m0_invalid_jitter_has_status_precedence_over_zero_convention(
    dtype,
):
    grouped = _all_m0_grouped_problem(dtype)
    detailed = posterior_components_grouped(grouped, factor_jitter=-1.0)
    assert bool(np.asarray(detailed.e_step.numerical_failure))
    assert np.all(np.asarray(detailed.e_step.failed_pairs))

    scores = general_inference.score_samples_grouped(
        grouped, factor_jitter=-1.0
    )
    probabilities = general_inference.predict_proba_grouped(
        grouped, factor_jitter=-1.0
    )
    labels = general_inference.predict_grouped(grouped, factor_jitter=-1.0)
    posterior = general_inference.posterior_grouped(
        grouped, factor_jitter=-1.0
    )
    total = general_inference.log_likelihood_grouped(
        grouped, factor_jitter=-1.0
    )
    normalized = general_inference.score_grouped(
        grouped, factor_jitter=-1.0
    )

    assert np.all(np.isnan(np.asarray(scores)))
    assert np.all(np.isnan(np.asarray(probabilities)))
    np.testing.assert_array_equal(np.asarray(labels), np.full(4, -1))
    assert np.all(np.isnan(np.asarray(posterior[0])))
    assert np.all(np.isnan(np.asarray(posterior[1])))
    assert np.isnan(np.asarray(total))
    assert np.isnan(np.asarray(normalized))
