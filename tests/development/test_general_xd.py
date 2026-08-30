"""Red contract tests for the temporary fixed-M general-projection XD core.

The intentionally absent ``development.general_xd`` module is expected to
provide these temporary functional signatures before this file can collect:

``posterior_components_general(params, x, R, S, *, factor_jitter=0)``
``sufficient_statistics_general(params, x, R, S, sample_weight, *, factor_jitter=0)``
``one_em_step_general(params, x, R, S, sample_weight, *, factor_jitter=0,
covariance_ridge=0)``

Canonical arrays use fixed observed dimension ``M``: ``x (N,M)``,
``R (N,M,D)``, ``S (N,M,M)``, with no implicit shared-array broadcasting.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development.general_xd as general_xd_module
from development import GeneralSufficientStatistics as ExportedGeneralStatistics
from development.general_xd import (
    one_em_step_general,
    posterior_components_general,
    sufficient_statistics_general,
)
from development.identity_xd import Params, posterior_components
from tests.reference.general_xd import (
    general_e_step,
    general_em_step,
    general_sufficient_statistics,
)


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _weighted_fixture(dtype=jnp.float64):
    weights = np.array([0.4, 0.6])
    means = np.array([[-0.8, 0.3], [1.2, -0.4]])
    covariances = np.array(
        [
            [[0.7, 0.1], [0.1, 0.5]],
            [[0.6, -0.08], [-0.08, 0.9]],
        ]
    )
    params = _params(dtype, weights, means, covariances)
    x = jnp.asarray([[-1.2], [0.4], [1.1], [2.0]], dtype=dtype)
    projection = jnp.asarray(
        [
            [[1.0, 0.25]],
            [[0.5, 1.0]],
            [[1.0, -0.5]],
            [[1.5, 0.2]],
        ],
        dtype=dtype,
    )
    noise = jnp.asarray([[[0.2]], [[0.1]], [[0.3]], [[0.15]]], dtype=dtype)
    return params, x, projection, noise


def _assert_exact_parameter_rollback(result, params):
    for actual, expected in zip(result.parameters, params, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    (
        pytest.param(jnp.float64, 5e-10, 5e-12, id="float64"),
        pytest.param(jnp.float32, 1e-4, 1e-5, id="float32"),
    ),
)
def test_general_scalar_projection_matches_analytic_oracle(dtype, rtol, atol):
    params = _params(dtype, [1.0], [[0.25]], [[[3.0]]])
    actual = posterior_components_general(
        params,
        jnp.asarray([[1.75]], dtype=dtype),
        jnp.asarray([[[2.0]]], dtype=dtype),
        jnp.asarray([[[0.5]]], dtype=dtype),
    )
    expected = general_e_step(
        [[1.75]], [[[2.0]]], [[[0.5]]], [1.0], [[0.25]], [[[3.0]]]
    )

    for field in (
        "component_log_density",
        "component_log_joint",
        "score_samples",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field)),
            getattr(expected, field),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    (
        pytest.param(jnp.float64, 5e-10, 5e-12, id="float64"),
        pytest.param(jnp.float32, 1e-4, 1e-5, id="float32"),
    ),
)
def test_general_identity_projection_matches_identity_kernel(dtype, rtol, atol):
    params = _params(
        dtype,
        [0.35, 0.65],
        [[-0.6, 0.2], [0.9, 0.5]],
        [
            [[0.8, 0.12], [0.12, 0.6]],
            [[0.5, -0.07], [-0.07, 0.9]],
        ],
    )
    x = jnp.asarray([[-0.8, 0.1], [0.5, -0.2], [1.2, 0.9]], dtype=dtype)
    noise = jnp.asarray(
        [
            [[0.2, 0.03], [0.03, 0.1]],
            [[0.15, -0.02], [-0.02, 0.25]],
            [[0.08, 0.01], [0.01, 0.18]],
        ],
        dtype=dtype,
    )
    projection = jnp.broadcast_to(jnp.eye(2, dtype=dtype), (3, 2, 2))

    actual = posterior_components_general(params, x, projection, noise)
    identity = posterior_components(params, x, noise)
    for field in (
        "component_log_density",
        "component_log_joint",
        "score_samples",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
        "failed_pairs",
        "numerical_failure",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(identity, field)),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "dtype",
    (
        pytest.param(jnp.float64, id="float64"),
        pytest.param(jnp.float32, id="float32"),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "batch_shape",
    (
        pytest.param((0,), id="zero-leading-axis"),
        pytest.param((2, 0), id="zero-inner-axis"),
    ),
)
@pytest.mark.parametrize(
    "observed_dimension",
    (
        pytest.param(2, id="m-positive"),
        pytest.param(0, id="m-zero"),
    ),
)
@pytest.mark.parametrize(
    "factor_jitter,expect_numerical_failure",
    (
        pytest.param(0.0, False, id="valid-jitter"),
        pytest.param(-1.0, True, id="invalid-jitter"),
    ),
)
def test_general_zero_sized_inference_batches_preserve_global_control_status(
    dtype,
    compiled,
    batch_shape,
    observed_dimension,
    factor_jitter,
    expect_numerical_failure,
):
    params = _params(
        dtype,
        [0.4, 0.6],
        [[-0.5, 0.2], [0.8, -0.3]],
        [np.eye(2), np.asarray([[0.8, 0.1], [0.1, 1.1]])],
    )
    x = jnp.empty(batch_shape + (observed_dimension,), dtype=dtype)
    projection = jnp.empty(
        batch_shape + (observed_dimension, 2), dtype=dtype
    )
    noise = jnp.empty(
        batch_shape + (observed_dimension, observed_dimension), dtype=dtype
    )
    run = (
        jax.jit(posterior_components_general)
        if compiled
        else posterior_components_general
    )

    posterior = run(
        params,
        x,
        projection,
        noise,
        factor_jitter=factor_jitter,
    )

    assert posterior.component_log_density.shape == batch_shape + (2,)
    assert posterior.component_log_joint.shape == batch_shape + (2,)
    assert posterior.score_samples.shape == batch_shape
    assert posterior.responsibilities.shape == batch_shape + (2,)
    assert posterior.conditional_mean.shape == batch_shape + (2, 2)
    assert posterior.conditional_covariance.shape == batch_shape + (2, 2, 2)
    assert posterior.failed_pairs.shape == batch_shape + (2,)
    assert posterior.failed_pairs.size == 0
    assert (
        bool(np.asarray(posterior.numerical_failure))
        is expect_numerical_failure
    )


@pytest.mark.parametrize(
    "x,projection,noise,weights,means,covariances",
    (
        pytest.param(
            [[3.0, -0.5]],
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
            [[[1.0, 0.0], [0.0, 0.25]]],
            [1.0],
            [[1.0, -2.0, 0.5]],
            [np.diag([4.0, 9.0, 1.0])],
            id="dimension-reducing-selector",
        ),
        pytest.param(
            [[3.0, 0.0]],
            [[[1.0, 1.0], [2.0, -1.0]]],
            [[[1.0, 0.0], [0.0, 2.0]]],
            [1.0],
            [[1.0, -1.0]],
            [[[2.0, 0.0], [0.0, 3.0]]],
            id="nonorthogonal",
        ),
    ),
)
def test_general_projection_fixtures_match_oracle(
    x, projection, noise, weights, means, covariances
):
    params = _params(jnp.float64, weights, means, covariances)
    actual = posterior_components_general(
        params,
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(projection, dtype=jnp.float64),
        jnp.asarray(noise, dtype=jnp.float64),
    )
    expected = general_e_step(
        x, projection, noise, weights, means, covariances
    )
    for field in (
        "component_log_density",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual, field)),
            getattr(expected, field),
            rtol=5e-10,
            atol=5e-12,
        )


def test_general_weight_scaling_and_zero_weight_row_match_oracle():
    params, x, projection, noise = _weighted_fixture()
    sample_weight = jnp.asarray([0.5, 2.0, 1.25, 0.0], dtype=jnp.float64)
    scale = 7.25

    actual_statistics = sufficient_statistics_general(
        params, x, projection, noise, sample_weight
    )
    scaled_statistics = sufficient_statistics_general(
        params, x, projection, noise, scale * sample_weight
    )
    expected_e_step = general_e_step(
        np.asarray(x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
    )
    expected_statistics = general_sufficient_statistics(
        expected_e_step, np.asarray(sample_weight)
    )
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual_statistics, field)),
            getattr(expected_statistics, field),
            rtol=5e-10,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(scaled_statistics, field)),
            scale * np.asarray(getattr(actual_statistics, field)),
            rtol=5e-10,
            atol=5e-12,
        )

    actual_step = one_em_step_general(
        params, x, projection, noise, sample_weight
    )
    scaled_step = one_em_step_general(
        params, x, projection, noise, scale * sample_weight
    )
    expected_parameters, _, _ = general_em_step(
        np.asarray(x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
        sample_weight=np.asarray(sample_weight),
    )
    assert not bool(np.asarray(actual_step.collapsed))
    assert not bool(np.asarray(actual_step.numerical_failure))
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual_step.parameters, field)),
            getattr(expected_parameters, field),
            rtol=5e-10,
            atol=5e-12,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(scaled_step.parameters, field)),
            np.asarray(getattr(actual_step.parameters, field)),
            rtol=5e-10,
            atol=5e-12,
        )

    truncated_step = one_em_step_general(
        params,
        x[:-1],
        projection[:-1],
        noise[:-1],
        sample_weight[:-1],
    )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(truncated_step.parameters, field)),
            np.asarray(getattr(actual_step.parameters, field)),
            rtol=5e-10,
            atol=5e-12,
        )


def test_general_zero_observed_dimension_returns_mixture_prior():
    weights = np.array([0.25, 0.75])
    means = np.array([[-1.0, 0.5], [2.0, -0.25]])
    covariances = np.array(
        [
            [[0.8, 0.1], [0.1, 0.6]],
            [[1.2, -0.2], [-0.2, 0.9]],
        ]
    )
    params = _params(jnp.float64, weights, means, covariances)
    x = jnp.empty((3, 0), dtype=jnp.float64)
    projection = jnp.empty((3, 0, 2), dtype=jnp.float64)
    noise = jnp.empty((3, 0, 0), dtype=jnp.float64)
    actual = posterior_components_general(params, x, projection, noise)

    np.testing.assert_array_equal(actual.component_log_density, np.zeros((3, 2)))
    np.testing.assert_array_equal(actual.score_samples, np.zeros(3))
    np.testing.assert_allclose(
        actual.responsibilities, np.broadcast_to(weights, (3, 2)), atol=0.0
    )
    np.testing.assert_array_equal(
        actual.conditional_mean, np.broadcast_to(means, (3, 2, 2))
    )
    np.testing.assert_array_equal(
        actual.conditional_covariance,
        np.broadcast_to(covariances, (3, 2, 2, 2)),
    )

    statistics = sufficient_statistics_general(
        params,
        x,
        projection,
        noise,
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64),
    )
    np.testing.assert_array_equal(statistics.mass, np.zeros(2))
    np.testing.assert_array_equal(statistics.first_moment, np.zeros((2, 2)))
    np.testing.assert_array_equal(statistics.second_moment, np.zeros((2, 2, 2)))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_float32_total_sample_mass_overflow_is_numerical_failure(compiled):
    params = _params(
        jnp.float32,
        [0.5, 0.5],
        [[0.0], [0.0]],
        [[[1.0]], [[1.0]]],
    )
    x = jnp.asarray([[0.0], [1.0]], dtype=jnp.float32)
    projection = jnp.ones((2, 1, 1), dtype=jnp.float32)
    noise = jnp.ones((2, 1, 1), dtype=jnp.float32)
    sample_weight = jnp.asarray([3e38, 3e38], dtype=jnp.float32)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)

    _assert_exact_parameter_rollback(result, params)
    assert np.all(np.isfinite(np.asarray(result.statistics.mass)))
    assert bool(np.asarray(result.numerical_failure))
    assert bool(np.asarray(result.statistics.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(result.collapsed_components, np.zeros(2, dtype=bool))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_sufficient_statistics_report_positive_weight_factor_failure(compiled):
    params = _params(jnp.float64, [1.0], [[0.0, 0.0]], [np.eye(2)])
    x = jnp.asarray([[0.0]], dtype=jnp.float64)
    projection = jnp.zeros((1, 1, 2), dtype=jnp.float64)
    noise = jnp.zeros((1, 1, 1), dtype=jnp.float64)
    sample_weight = jnp.ones((1,), dtype=jnp.float64)
    run = (
        jax.jit(sufficient_statistics_general)
        if compiled
        else sufficient_statistics_general
    )

    statistics = run(params, x, projection, noise, sample_weight)

    statistics_type = getattr(
        general_xd_module, "GeneralSufficientStatistics", ()
    )
    assert isinstance(statistics, statistics_type)
    assert statistics_type is ExportedGeneralStatistics
    assert statistics._fields == (
        "mass",
        "first_moment",
        "second_moment",
        "numerical_failure",
        "failed_pairs",
    )
    assert bool(np.asarray(statistics.numerical_failure))
    np.testing.assert_array_equal(statistics.failed_pairs, [[True]])


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "sample_weight",
    (
        pytest.param([-0.25, 1.0], id="negative"),
        pytest.param([np.nan, 1.0], id="nonfinite"),
    ),
)
def test_general_sufficient_statistics_report_invalid_weights(
    compiled, sample_weight
):
    params, x, projection, noise = _weighted_fixture()
    supplied = jnp.asarray(sample_weight + [1.0, 1.0], dtype=jnp.float64)
    run = (
        jax.jit(sufficient_statistics_general)
        if compiled
        else sufficient_statistics_general
    )

    statistics = run(params, x, projection, noise, supplied)

    assert bool(np.asarray(statistics.numerical_failure))
    np.testing.assert_array_equal(
        statistics.failed_pairs, np.zeros((4, 2), dtype=bool)
    )


@pytest.mark.parametrize(
    "dtype,scale,atol",
    (
        pytest.param(jnp.float64, 1e24, 8e-12, id="float64"),
        pytest.param(jnp.float32, 1e30, 0.0, id="float32"),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_square_zero_noise_has_exact_zero_covariance_at_huge_scale(
    dtype, scale, atol, compiled
):
    params = _params(
        dtype,
        [1.0],
        [[0.2, -0.3]],
        [[[scale, 0.1 * scale], [0.1 * scale, 0.8 * scale]]],
    )
    x = jnp.asarray([[1.0, -2.0]], dtype=dtype)
    projection = jnp.asarray(
        [[[1.25, 0.2], [-0.3, 0.9]]], dtype=dtype
    )
    noise = jnp.zeros((1, 2, 2), dtype=dtype)
    run = (
        jax.jit(posterior_components_general)
        if compiled
        else posterior_components_general
    )

    posterior = run(params, x, projection, noise)

    assert not bool(np.asarray(posterior.numerical_failure))
    np.testing.assert_allclose(
        np.asarray(posterior.conditional_covariance),
        np.zeros((1, 1, 2, 2)),
        rtol=0.0,
        atol=atol,
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "dtype,covariance_scale,rtol",
    (
        pytest.param(jnp.float64, 1e308, 2e-15, id="float64"),
        pytest.param(jnp.float32, 2e38, 2e-6, id="float32"),
    ),
)
def test_general_symmetrization_does_not_overflow_finite_covariance(
    compiled, dtype, covariance_scale, rtol
):
    params = _params(dtype, [1.0], [[0.0]], [[[covariance_scale]]])
    x = jnp.zeros((1, 1), dtype=dtype)
    projection = jnp.zeros((1, 1, 1), dtype=dtype)
    noise = jnp.ones((1, 1, 1), dtype=dtype)
    sample_weight = jnp.ones((1,), dtype=dtype)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)

    assert not bool(np.asarray(result.e_step.numerical_failure))
    assert not bool(np.asarray(result.statistics.numerical_failure))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_allclose(
        np.asarray(result.e_step.conditional_covariance),
        covariance_scale,
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.statistics.second_moment),
        covariance_scale,
        rtol=rtol,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        covariance_scale,
        rtol=rtol,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "dtype,stored_weights",
    (
        pytest.param(
            jnp.float64,
            [0.2, 0.3, 0.5000000000002],
            id="float64",
        ),
        pytest.param(jnp.float32, [0.2, 0.3, 0.50001], id="float32"),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_all_pair_failure_copies_stored_weights_exactly(
    dtype, stored_weights, compiled
):
    params = _params(
        dtype,
        stored_weights,
        np.zeros((3, 2)),
        np.broadcast_to(np.eye(2), (3, 2, 2)),
    )
    x = jnp.zeros((1, 1), dtype=dtype)
    projection = jnp.zeros((1, 1, 2), dtype=dtype)
    noise = jnp.zeros((1, 1, 1), dtype=dtype)
    run = (
        jax.jit(posterior_components_general)
        if compiled
        else posterior_components_general
    )

    posterior = run(params, x, projection, noise)

    assert bool(np.asarray(posterior.numerical_failure))
    np.testing.assert_array_equal(posterior.failed_pairs, np.ones((1, 3), dtype=bool))
    np.testing.assert_array_equal(posterior.responsibilities[0], params.weights)


def test_general_numpy_weight_source_is_checked_before_x64_disabled_conversion():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import numpy as np
import jax
import jax.numpy as jnp
from development.general_xd import one_em_step_general, sufficient_statistics_general
from development.identity_xd import Params

assert not jax.config.x64_enabled
params = Params(
    jnp.asarray([1.0], dtype=jnp.float32),
    jnp.asarray([[0.0]], dtype=jnp.float32),
    jnp.asarray([[[1.0]]], dtype=jnp.float32),
)
x = jnp.asarray([[0.0], [1.0]], dtype=jnp.float32)
projection = jnp.ones((2, 1, 1), dtype=jnp.float32)
noise = jnp.ones((2, 1, 1), dtype=jnp.float32)

for tiny in (-1e-50, 1e-50):
    source_weight = np.asarray([tiny, 1.0], dtype=np.float64)
    statistics = sufficient_statistics_general(
        params, x, projection, noise, source_weight
    )
    assert bool(np.asarray(statistics.numerical_failure))
    result = one_em_step_general(params, x, projection, noise, source_weight)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    for actual, expected in zip(result.parameters, params):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_general_eager_control_sources_are_checked_before_x64_disabled_conversion():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import numpy as np
import jax
import jax.numpy as jnp
from development.general_xd import one_em_step_general
from development.identity_xd import Params

assert not jax.config.x64_enabled
params = Params(
    jnp.asarray([1.0], dtype=jnp.float32),
    jnp.asarray([[0.0]], dtype=jnp.float32),
    jnp.asarray([[[1.0]]], dtype=jnp.float32),
)
x = jnp.asarray([[0.0], [1.0]], dtype=jnp.float32)
projection = jnp.ones((2, 1, 1), dtype=jnp.float32)
noise = jnp.ones((2, 1, 1), dtype=jnp.float32)
sample_weight = jnp.ones((2,), dtype=jnp.float32)

for control in ("factor_jitter", "covariance_ridge"):
    for tiny in (-1e-50, 1e-50):
        for source_value in (tiny, np.asarray(tiny, dtype=np.float64)):
            result = one_em_step_general(
                params,
                x,
                projection,
                noise,
                sample_weight,
                **{control: source_value},
            )
            assert bool(np.asarray(result.numerical_failure))
            assert not bool(np.asarray(result.collapsed))
            assert not np.any(np.asarray(result.collapsed_components))
            for actual, expected in zip(result.parameters, params):
                np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_extreme_common_weight_scale_preserves_minor_component(compiled):
    n_samples = 1000
    minor_mean = np.sqrt(2440.0)
    params = _params(
        jnp.float64,
        [0.5, 0.5],
        [[0.0], [minor_mean]],
        [[[1.0]], [[1.0]]],
    )
    x = jnp.zeros((n_samples, 1), dtype=jnp.float64)
    projection = jnp.ones((n_samples, 1, 1), dtype=jnp.float64)
    noise = jnp.ones((n_samples, 1, 1), dtype=jnp.float64)
    unit_weight = jnp.ones((n_samples,), dtype=jnp.float64)
    scale = np.ldexp(1.0, -200)
    scaled_weight = jnp.full((n_samples,), scale, dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    unit_result = run(params, x, projection, noise, unit_weight)
    scaled_result = run(params, x, projection, noise, scaled_weight)
    expected_parameters, expected_e_step, expected_statistics = general_em_step(
        np.asarray(x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
        sample_weight=np.ones(n_samples, dtype=np.float64),
    )

    assert expected_e_step.responsibilities[0, 1] == pytest.approx(
        1.2032781734919707e-265, rel=5e-13
    )
    assert not bool(np.asarray(unit_result.numerical_failure))
    assert not bool(np.asarray(unit_result.collapsed))
    assert not bool(np.asarray(scaled_result.numerical_failure))
    assert not bool(np.asarray(scaled_result.collapsed))
    assert float(np.asarray(scaled_result.statistics.mass[1])) > 0.0

    for field in ("mass", "first_moment", "second_moment"):
        unit_actual = np.asarray(getattr(unit_result.statistics, field))
        scaled_actual = np.asarray(getattr(scaled_result.statistics, field))
        oracle_actual = np.asarray(getattr(expected_statistics, field))
        np.testing.assert_allclose(
            unit_actual, oracle_actual, rtol=8e-10, atol=8e-12
        )
        np.testing.assert_allclose(
            scaled_actual,
            unit_actual * scale,
            rtol=8e-10,
            atol=4.0 * np.nextafter(0.0, 1.0),
        )
        np.testing.assert_allclose(
            scaled_actual,
            oracle_actual * scale,
            rtol=8e-10,
            atol=4.0 * np.nextafter(0.0, 1.0),
        )

    for field in ("weights", "means", "covariances"):
        unit_actual = np.asarray(getattr(unit_result.parameters, field))
        scaled_actual = np.asarray(getattr(scaled_result.parameters, field))
        oracle_actual = np.asarray(getattr(expected_parameters, field))
        np.testing.assert_allclose(
            unit_actual, oracle_actual, rtol=8e-10, atol=8e-12
        )
        np.testing.assert_allclose(
            scaled_actual, unit_actual, rtol=8e-10, atol=8e-12
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_min_subnormal_weights_have_representable_aggregate(compiled):
    params = _params(jnp.float64, [1.0], [[0.0]], [[[1.0]]])
    x = jnp.zeros((2, 1), dtype=jnp.float64)
    projection = jnp.ones((2, 1, 1), dtype=jnp.float64)
    noise = jnp.ones((2, 1, 1), dtype=jnp.float64)
    min_subnormal = np.nextafter(0.0, 1.0)
    tiny_weight = jnp.asarray(
        [min_subnormal, min_subnormal], dtype=jnp.float64
    )
    unit_weight = jnp.ones((2,), dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    tiny_result = run(params, x, projection, noise, tiny_weight)
    unit_result = run(params, x, projection, noise, unit_weight)

    assert not bool(np.asarray(tiny_result.numerical_failure))
    assert not bool(np.asarray(tiny_result.collapsed))
    assert not np.any(np.asarray(tiny_result.collapsed_components))
    np.testing.assert_array_equal(
        np.asarray(tiny_result.statistics.mass),
        np.asarray([2.0 * min_subnormal]),
    )
    np.testing.assert_array_equal(
        np.asarray(tiny_result.statistics.second_moment),
        np.asarray([[[min_subnormal]]]),
    )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(tiny_result.parameters, field)),
            np.asarray(getattr(unit_result.parameters, field)),
            rtol=8e-10,
            atol=8e-12,
        )


@pytest.mark.parametrize(
    "dtype,weight_values",
    (
        pytest.param(
            jnp.float32,
            [np.nextafter(0.0, 1.0), 1.0],
            id="positive-f64-subnormal-underflows-f32",
        ),
        pytest.param(
            jnp.float64,
            [-np.nextafter(0.0, 1.0), 1.0],
            id="negative-f64-subnormal-same-dtype",
        ),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_jax_subnormal_weight_domain_survives_device_checks(
    dtype, weight_values, compiled
):
    params = _params(dtype, [1.0], [[0.0]], [[[1.0]]])
    x = jnp.zeros((2, 1), dtype=dtype)
    projection = jnp.ones((2, 1, 1), dtype=dtype)
    noise = jnp.ones((2, 1, 1), dtype=dtype)
    sample_weight = jnp.asarray(weight_values, dtype=jnp.float64)
    assert np.asarray(sample_weight)[0] != 0.0
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)

    _assert_exact_parameter_rollback(result, params)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_subnormal_normalized_weight_preserves_huge_moments(compiled):
    params = _params(jnp.float64, [1.0], [[0.0]], [[[1e308]]])
    x = jnp.asarray([[0.0], [1e308]], dtype=jnp.float64)
    projection = jnp.ones((2, 1, 1), dtype=jnp.float64)
    noise = jnp.ones((2, 1, 1), dtype=jnp.float64)
    min_subnormal = np.nextafter(0.0, 1.0)
    sample_weight = jnp.asarray([1.0, min_subnormal], dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)
    expected_first = min_subnormal * 1e308
    expected_second = expected_first * 1e308

    assert not np.any(np.asarray(result.e_step.failed_pairs))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    np.testing.assert_array_equal(
        np.asarray(result.e_step.responsibilities), np.ones((2, 1))
    )
    np.testing.assert_allclose(
        np.asarray(result.statistics.mass), np.asarray([1.0]), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        np.asarray(result.statistics.first_moment),
        np.asarray([[expected_first]]),
        rtol=3e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.statistics.second_moment),
        np.asarray([[[expected_second]]]),
        rtol=3e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.means),
        np.asarray([[expected_first]]),
        rtol=3e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        np.asarray([[[expected_second]]]),
        rtol=3e-13,
        atol=0.0,
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_heterogeneous_weight_scale_preserves_minor_component(compiled):
    params = _params(
        jnp.float64,
        [0.5, 0.5],
        [[-0.02], [0.02]],
        [[[5e-7]], [[5e-7]]],
    )
    x = jnp.asarray([[-0.02], [0.02]], dtype=jnp.float64)
    projection = jnp.ones((2, 1, 1), dtype=jnp.float64)
    noise = jnp.full((2, 1, 1), 5e-7, dtype=jnp.float64)
    sample_weight = jnp.asarray([1e308, 1.0], dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)
    expected_parameters, expected_e_step, expected_statistics = general_em_step(
        np.asarray(x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
        sample_weight=np.asarray(sample_weight),
    )

    np.testing.assert_array_equal(
        np.asarray(result.e_step.responsibilities),
        expected_e_step.responsibilities,
    )
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert not np.any(np.asarray(result.collapsed_components))
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.statistics, field)),
            np.asarray(getattr(expected_statistics, field)),
            rtol=2e-14,
            atol=0.0,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.parameters, field)),
            np.asarray(getattr(expected_parameters, field)),
            rtol=8e-10,
            atol=4.0 * np.nextafter(0.0, 1.0),
        )
    assert np.asarray(result.parameters.weights[1]) == pytest.approx(
        1e-308, rel=8e-10, abs=4.0 * np.nextafter(0.0, 1.0)
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_log_effective_weight_recovers_underflowed_responsibility(compiled):
    log_odds = 710.0
    distance = np.sqrt(2.0 * log_odds * 1e-6)
    params = _params(
        jnp.float64,
        [0.5, 0.5],
        [[0.0], [distance]],
        [[[5e-7]], [[5e-7]]],
    )
    x = jnp.zeros((1, 1), dtype=jnp.float64)
    projection = jnp.ones((1, 1, 1), dtype=jnp.float64)
    noise = jnp.full((1, 1, 1), 5e-7, dtype=jnp.float64)
    sample_weight = jnp.asarray([1e308], dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)
    expected_parameters, expected_e_step, expected_statistics = general_em_step(
        np.asarray(x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(params.weights),
        np.asarray(params.means),
        np.asarray(params.covariances),
        sample_weight=np.asarray(sample_weight),
    )

    assert expected_e_step.responsibilities[0, 1] > 0.0
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert float(np.asarray(result.e_step.responsibilities[0, 1])) > 0.0
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.statistics, field)),
            np.asarray(getattr(expected_statistics, field)),
            rtol=3e-12,
            atol=4.0 * np.nextafter(0.0, 1.0),
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.parameters, field)),
            np.asarray(getattr(expected_parameters, field)),
            rtol=3e-12,
            atol=4.0 * np.nextafter(0.0, 1.0),
        )

    probe_x = jnp.asarray([[distance]], dtype=jnp.float64)
    posterior = posterior_components_general(
        result.parameters, probe_x, projection, noise
    )
    expected_posterior = general_e_step(
        np.asarray(probe_x),
        np.asarray(projection),
        np.asarray(noise),
        np.asarray(expected_parameters.weights),
        np.asarray(expected_parameters.means),
        np.asarray(expected_parameters.covariances),
    )
    assert expected_posterior.responsibilities[0, 1] > 0.0
    np.testing.assert_allclose(
        np.asarray(posterior.responsibilities),
        expected_posterior.responsibilities,
        rtol=3e-12,
        atol=4.0 * np.nextafter(0.0, 1.0),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_zero_weight_singular_row_has_no_fit_effect(compiled):
    params = _params(jnp.float64, [1.0], [[0.0, 0.0]], [np.eye(2)])
    x = jnp.asarray([[0.4], [9.0]], dtype=jnp.float64)
    projection = jnp.asarray([[[1.0, 0.0]], [[0.0, 0.0]]], dtype=jnp.float64)
    noise = jnp.asarray([[[0.5]], [[0.0]]], dtype=jnp.float64)
    sample_weight = jnp.asarray([1.0, 0.0], dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)
    truncated = run(
        params,
        x[:1],
        projection[:1],
        noise[:1],
        sample_weight[:1],
    )

    assert bool(np.asarray(result.e_step.failed_pairs[1, 0]))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.statistics.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        result.statistics.failed_pairs, np.zeros((2, 1), dtype=bool)
    )
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.statistics, field)),
            np.asarray(getattr(truncated.statistics, field)),
            rtol=5e-10,
            atol=5e-12,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(result.parameters, field)),
            np.asarray(getattr(truncated.parameters, field)),
            rtol=5e-10,
            atol=5e-12,
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_all_zero_weights_use_no_informative_collapse_path(compiled):
    params, x, projection, noise = _weighted_fixture()
    sample_weight = jnp.zeros((4,), dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)

    _assert_exact_parameter_rollback(result, params)
    assert not bool(np.asarray(result.statistics.numerical_failure))
    assert not bool(np.asarray(result.numerical_failure))
    assert bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        result.collapsed_components, np.ones(2, dtype=bool)
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "factor_jitter,expect_numerical_failure",
    (
        pytest.param(0.0, False, id="valid-controls"),
        pytest.param(-1.0, True, id="invalid-jitter"),
    ),
)
def test_general_zero_observed_dimension_step_preserves_whole_state_semantics(
    factor_jitter, expect_numerical_failure, compiled
):
    params = _params(
        jnp.float64,
        [0.4, 0.6],
        [[-0.5, 0.2], [0.8, -0.3]],
        [np.eye(2), np.asarray([[0.8, 0.1], [0.1, 1.1]])],
    )
    x = jnp.empty((3, 0), dtype=jnp.float64)
    projection = jnp.empty((3, 0, 2), dtype=jnp.float64)
    noise = jnp.empty((3, 0, 0), dtype=jnp.float64)
    sample_weight = jnp.asarray([1.0, 0.0, 2.0], dtype=jnp.float64)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(
        params,
        x,
        projection,
        noise,
        sample_weight,
        factor_jitter=factor_jitter,
    )

    _assert_exact_parameter_rollback(result, params)
    assert bool(np.asarray(result.numerical_failure)) is expect_numerical_failure
    assert (
        bool(np.asarray(result.statistics.numerical_failure))
        is expect_numerical_failure
    )
    assert bool(np.asarray(result.collapsed)) is (not expect_numerical_failure)
    np.testing.assert_array_equal(
        result.collapsed_components,
        np.full(2, not expect_numerical_failure, dtype=bool),
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_zero_weight_huge_finite_posterior_has_no_arithmetic_effect(compiled):
    params = _params(jnp.float32, [1.0], [[0.0]], [[[1e20]]])
    x = jnp.asarray([[0.0], [2e19]], dtype=jnp.float32)
    projection = jnp.ones((2, 1, 1), dtype=jnp.float32)
    noise = jnp.ones((2, 1, 1), dtype=jnp.float32)
    sample_weight = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    run = jax.jit(one_em_step_general) if compiled else one_em_step_general

    result = run(params, x, projection, noise, sample_weight)
    truncated = run(
        params,
        x[:1],
        projection[:1],
        noise[:1],
        sample_weight[:1],
    )

    assert np.all(np.isfinite(np.asarray(result.e_step.conditional_mean)))
    assert np.all(np.isfinite(np.asarray(result.e_step.score_samples)))
    assert not bool(np.asarray(result.statistics.numerical_failure))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_array_equal(
            np.asarray(getattr(result.statistics, field)),
            np.asarray(getattr(truncated.statistics, field)),
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_array_equal(
            np.asarray(getattr(result.parameters, field)),
            np.asarray(getattr(truncated.parameters, field)),
        )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_identity_valued_projection_gradient_uses_generic_gain(compiled):
    params = _params(
        jnp.float64,
        [1.0],
        [[0.2, -0.4]],
        [[[1.2, 0.15], [0.15, 0.8]]],
    )
    x = jnp.asarray([[1.1, -0.3]], dtype=jnp.float64)
    noise = jnp.zeros((1, 2, 2), dtype=jnp.float64)
    coefficient = jnp.asarray([0.7, -1.2], dtype=jnp.float64)

    def objective(projection):
        posterior = posterior_components_general(
            params, x, projection[None, ...], noise
        )
        return jnp.vdot(coefficient, posterior.conditional_mean[0, 0])

    gradient = jax.grad(objective)
    if compiled:
        gradient = jax.jit(gradient)
    projection = jnp.eye(2, dtype=jnp.float64)
    actual = np.asarray(gradient(projection))
    step = 1e-5
    expected = np.empty((2, 2), dtype=np.float64)
    for row in range(2):
        for column in range(2):
            perturbation = np.zeros((2, 2), dtype=np.float64)
            perturbation[row, column] = step
            expected[row, column] = (
                float(objective(projection + perturbation))
                - float(objective(projection - perturbation))
            ) / (2.0 * step)

    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    "dtype,atol",
    (
        pytest.param(jnp.float64, 2e-12, id="float64"),
        pytest.param(jnp.float32, 2e-5, id="float32"),
    ),
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_general_dimension_reducing_zero_noise_has_zero_projected_covariance(
    dtype, atol, compiled
):
    params = _params(
        dtype,
        [1.0],
        [[0.2, -0.3, 0.7]],
        [[[2.0, 0.2, -0.1], [0.2, 1.5, 0.3], [-0.1, 0.3, 0.9]]],
    )
    projection = jnp.asarray(
        [[[1.25, 0.2, -0.1], [-0.3, 0.9, 0.4]]], dtype=dtype
    )
    x = jnp.asarray([[1.0, -2.0]], dtype=dtype)
    noise = jnp.zeros((1, 2, 2), dtype=dtype)
    run = (
        jax.jit(posterior_components_general)
        if compiled
        else posterior_components_general
    )

    posterior = run(params, x, projection, noise)
    projected_covariance = (
        projection[0]
        @ posterior.conditional_covariance[0, 0]
        @ projection[0].T
    )

    assert not bool(np.asarray(posterior.numerical_failure))
    np.testing.assert_allclose(
        np.asarray(projected_covariance), np.zeros((2, 2)), rtol=0.0, atol=atol
    )
