"""Contract tests for the temporary, non-public identity-XD JAX kernel.

Expected values come from ``tests.reference.identity_xd``, an intentionally
loop-oriented NumPy implementation that does not import the JAX kernel.  The
test names retain the IDs from ``docs/capability-matrix.md`` so these checks can
move with the implementation once the public package namespace is selected.
"""

from __future__ import annotations

import jax

# The contract requires an actual float64 reference execution.  Enable it before
# constructing any arrays instead of accepting JAX's otherwise silent downcast.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.identity_xd import (
    EMStepResult,
    EStep,
    Params,
    SufficientStatistics,
    diagonal_covariance,
    em_step,
    full_covariance,
    isotropic_covariance,
    marginalized_posterior,
    posterior_components,
    sufficient_statistics,
)
from tests.reference.identity_xd import (
    identity_e_step as reference_e_step,
    identity_em_step as reference_em_step,
    marginalized_posterior as reference_marginalized_posterior,
)


DTYPE_CASES = (
    pytest.param(jnp.float64, 5e-10, 5e-12, 5e-13, id="float64"),
    pytest.param(jnp.float32, 1e-4, 1e-5, 2e-5, id="float32"),
)


def _as_params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _assert_allclose(actual, expected, *, rtol: float, atol: float) -> None:
    np.testing.assert_allclose(
        np.asarray(actual), np.asarray(expected), rtol=rtol, atol=atol
    )


def _assert_covariance_invariants(covariances, dtype, *, model: bool = False) -> None:
    values = np.asarray(covariances, dtype=np.float64)
    transposed = np.swapaxes(values, -1, -2)
    scale = np.maximum(1.0, np.linalg.norm(values, ord=2, axis=(-2, -1)))
    symmetry_residual = (
        np.linalg.norm(values - transposed, ord=np.inf, axis=(-2, -1)) / scale
    )

    if dtype == jnp.float64:
        assert np.all(symmetry_residual <= 2e-13)
    else:
        assert np.all(symmetry_residual <= 2e-6)

    if model:
        factors = np.linalg.cholesky(values)
        assert np.all(np.isfinite(factors))
        return

    minimum_eigenvalue = np.linalg.eigvalsh(0.5 * (values + transposed))[..., 0]
    if dtype == jnp.float64:
        assert np.all(minimum_eigenvalue >= -2e-11 * scale)
    else:
        assert np.all(minimum_eigenvalue >= -5e-5 * scale)


def _ordinary_fixture(dtype):
    observations = jnp.asarray(
        [
            [-1.5, 0.2],
            [-0.9, -0.4],
            [-0.2, 0.7],
            [0.4, -0.5],
            [1.1, 0.8],
            [1.8, 0.4],
            [0.7, 1.5],
        ],
        dtype=dtype,
    )
    measurement_covariances = jnp.asarray(
        [
            [[0.12, 0.02], [0.02, 0.08]],
            [[0.09, -0.01], [-0.01, 0.16]],
            [[0.20, 0.04], [0.04, 0.14]],
            [[0.07, 0.00], [0.00, 0.11]],
            [[0.15, -0.03], [-0.03, 0.18]],
            [[0.10, 0.01], [0.01, 0.06]],
            [[0.18, 0.05], [0.05, 0.22]],
        ],
        dtype=dtype,
    )
    parameters = _as_params(
        dtype,
        weights=[0.25, 0.45, 0.30],
        means=[[-1.0, 0.1], [0.3, -0.2], [1.2, 1.0]],
        covariances=[
            [[0.8, 0.12], [0.12, 0.6]],
            [[0.7, -0.08], [-0.08, 0.9]],
            [[0.5, 0.09], [0.09, 0.7]],
        ],
    )
    return parameters, observations, measurement_covariances


@pytest.mark.parametrize("dtype,rtol,atol,row_sum_atol", DTYPE_CASES)
def test_xd_ip_resp_002_far_tail_responsibilities_stay_finite_and_normalized(
    dtype, rtol, atol, row_sum_atol
):
    """XD-IP-RESP-002: normalization remains valid after density underflow."""

    dimension = 32
    observations = np.stack(
        [
            np.full(dimension, 1_000.0),
            np.full(dimension, -1_000.0),
            np.zeros(dimension),
            np.tile(np.array([1_000.0, -1_000.0]), dimension // 2),
        ]
    )
    noise_diagonals = np.stack(
        [np.linspace(0.01 + 0.01 * index, 0.20, dimension) for index in range(4)]
    )
    measurement_covariances = np.eye(dimension)[None, :, :] * noise_diagonals[
        :, :, None
    ]
    weights = np.array([0.2, 0.3, 0.5])
    means = np.stack(
        [
            np.full(dimension, -2.0),
            np.zeros(dimension),
            np.full(dimension, 2.0),
        ]
    )
    covariances = np.stack(
        [
            0.5 * np.eye(dimension),
            np.eye(dimension),
            2.0 * np.eye(dimension),
        ]
    )
    reference = reference_e_step(
        observations, measurement_covariances, weights, means, covariances
    )
    assert np.any(np.exp(reference.component_log_density) == 0.0)

    parameters = _as_params(dtype, weights, means, covariances)
    actual = posterior_components(
        parameters,
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(measurement_covariances, dtype=dtype),
    )

    assert isinstance(actual, EStep)
    assert np.all(np.isfinite(np.asarray(actual.component_log_density)))
    assert np.all(np.isfinite(np.asarray(actual.responsibilities)))
    assert np.all(np.asarray(actual.responsibilities) >= 0.0)
    assert not np.any(np.all(np.asarray(actual.responsibilities) == 0.0, axis=1))
    _assert_allclose(
        actual.responsibilities,
        reference.responsibilities,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        jnp.sum(actual.responsibilities, axis=-1),
        np.ones(observations.shape[0]),
        rtol=0.0,
        atol=row_sum_atol,
    )


@pytest.mark.parametrize("dtype,rtol,atol,_row_sum_atol", DTYPE_CASES)
def test_xd_ip_post_001_one_component_conditional_posterior_matches_oracle(
    dtype, rtol, atol, _row_sum_atol
):
    """XD-IP-POST-001: the analytic one-component posterior is recovered."""

    observations = np.array([[1.1, -0.5]])
    measurement_covariances = np.array([[[0.5, 0.1], [0.1, 0.3]]])
    weights = np.array([1.0])
    means = np.array([[-0.2, 0.7]])
    covariances = np.array([[[2.0, 0.4], [0.4, 1.0]]])
    reference = reference_e_step(
        observations, measurement_covariances, weights, means, covariances
    )
    reference_mean, reference_covariance = reference_marginalized_posterior(
        reference
    )

    parameters = _as_params(dtype, weights, means, covariances)
    actual = posterior_components(
        parameters,
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(measurement_covariances, dtype=dtype),
    )
    actual_mean, actual_covariance = marginalized_posterior(actual)

    _assert_allclose(actual.responsibilities, [[1.0]], rtol=0.0, atol=atol)
    _assert_allclose(
        actual.conditional_mean, reference.conditional_mean, rtol=rtol, atol=atol
    )
    _assert_allclose(
        actual.conditional_covariance,
        reference.conditional_covariance,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(actual_mean, reference_mean, rtol=rtol, atol=atol)
    _assert_allclose(
        actual_covariance, reference_covariance, rtol=rtol, atol=atol
    )
    if dtype == jnp.float64:
        latent_covariance = covariances[0]
        total_covariance = latent_covariance + measurement_covariances[0]
        gain = np.linalg.solve(total_covariance, latent_covariance.T).T
        expected_mean_from_gain = means[0] + gain @ (
            observations[0] - means[0]
        )
        subtractive_covariance = latent_covariance - gain @ latent_covariance
        subtractive_covariance = 0.5 * (
            subtractive_covariance + subtractive_covariance.T
        )

        _assert_allclose(
            actual.conditional_mean[0, 0],
            expected_mean_from_gain,
            rtol=rtol,
            atol=atol,
        )
        _assert_allclose(
            subtractive_covariance,
            reference.conditional_covariance[0, 0],
            rtol=rtol,
            atol=atol,
        )
        _assert_allclose(
            actual.conditional_covariance[0, 0],
            subtractive_covariance,
            rtol=rtol,
            atol=atol,
        )
    _assert_covariance_invariants(actual.conditional_covariance, dtype)
    _assert_covariance_invariants(actual_covariance, dtype)


@pytest.mark.parametrize("dtype,rtol,atol,_row_sum_atol", DTYPE_CASES)
def test_xd_ip_post_002_marginalized_covariance_includes_between_component_term(
    dtype, rtol, atol, _row_sum_atol
):
    """XD-IP-POST-002: total covariance includes mixture-mean dispersion."""

    observations = np.array([[0.2, 0.3]])
    measurement_covariances = np.array([[[0.35, 0.07], [0.07, 0.25]]])
    weights = np.array([0.20, 0.35, 0.45])
    means = np.array([[-1.0, 0.6], [0.4, -0.8], [1.5, 1.1]])
    covariances = np.array(
        [
            [[0.7, 0.12], [0.12, 0.5]],
            [[0.9, -0.16], [-0.16, 0.8]],
            [[0.6, 0.05], [0.05, 1.0]],
        ]
    )
    reference = reference_e_step(
        observations, measurement_covariances, weights, means, covariances
    )
    reference_mean, reference_covariance = reference_marginalized_posterior(
        reference
    )

    parameters = _as_params(dtype, weights, means, covariances)
    actual_e_step = posterior_components(
        parameters,
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(measurement_covariances, dtype=dtype),
    )
    actual_mean, actual_covariance = marginalized_posterior(actual_e_step)

    _assert_allclose(actual_mean, reference_mean, rtol=rtol, atol=atol)
    _assert_allclose(
        actual_covariance, reference_covariance, rtol=rtol, atol=atol
    )
    within_component = np.einsum(
        "nk,nkde->nde",
        np.asarray(actual_e_step.responsibilities),
        np.asarray(actual_e_step.conditional_covariance),
    )
    between_component = np.asarray(actual_covariance) - within_component
    assert np.trace(between_component[0]) > 10.0 * atol
    assert not any(
        np.allclose(np.asarray(actual_mean[0]), component_mean, rtol=rtol, atol=atol)
        for component_mean in np.asarray(actual_e_step.conditional_mean[0])
    )
    _assert_covariance_invariants(actual_covariance, dtype)


@pytest.mark.parametrize("dtype,rtol,atol,row_sum_atol", DTYPE_CASES)
def test_xd_ip_em_001_one_exact_em_step_matches_independent_numpy_oracle(
    dtype, rtol, atol, row_sum_atol
):
    """XD-IP-EM-001: E-step, statistics, and centered M-step all agree."""

    parameters, observations, measurement_covariances = _ordinary_fixture(dtype)
    reference_parameters, reference_estep, reference_statistics = reference_em_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(measurement_covariances, dtype=np.float64),
        np.asarray(parameters.weights, dtype=np.float64),
        np.asarray(parameters.means, dtype=np.float64),
        np.asarray(parameters.covariances, dtype=np.float64),
    )

    actual = em_step(parameters, observations, measurement_covariances)
    assert isinstance(actual, EMStepResult)
    assert isinstance(actual.e_step, EStep)
    assert isinstance(actual.statistics, SufficientStatistics)
    independently_reduced = sufficient_statistics(actual.e_step)

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
        _assert_allclose(
            getattr(independently_reduced, field),
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

    _assert_allclose(
        jnp.sum(actual.parameters.weights), 1.0, rtol=0.0, atol=row_sum_atol
    )
    assert not bool(np.asarray(actual.collapsed))
    assert not np.any(np.asarray(actual.collapsed_components))
    _assert_covariance_invariants(
        actual.parameters.covariances, dtype, model=True
    )


@pytest.mark.parametrize("dtype,rtol,atol,row_sum_atol", DTYPE_CASES)
def test_xd_ip_zero_001_zero_noise_recovers_ordinary_gmm_and_exact_latent_state(
    dtype, rtol, atol, row_sum_atol
):
    """XD-IP-ZERO-001: S=0 reduces to an ordinary full-covariance GMM."""

    observations = np.array(
        [[-1.2, 0.4], [0.0, -0.3], [0.8, 1.1], [1.7, -0.2]]
    )
    measurement_covariances = np.zeros((4, 2, 2))
    weights = np.array([0.3, 0.7])
    means = np.array([[-0.8, 0.2], [1.1, 0.5]])
    covariances = np.array(
        [[[0.9, 0.18], [0.18, 0.6]], [[0.7, -0.12], [-0.12, 1.0]]]
    )
    reference = reference_e_step(
        observations, measurement_covariances, weights, means, covariances
    )
    parameters = _as_params(dtype, weights, means, covariances)
    actual = posterior_components(
        parameters,
        jnp.asarray(observations, dtype=dtype),
        jnp.asarray(measurement_covariances, dtype=dtype),
    )

    _assert_allclose(
        actual.component_log_density,
        reference.component_log_density,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        actual.responsibilities, reference.responsibilities, rtol=rtol, atol=atol
    )
    _assert_allclose(
        jnp.sum(actual.responsibilities, axis=-1),
        np.ones(observations.shape[0]),
        rtol=0.0,
        atol=row_sum_atol,
    )
    expected_conditional_means = np.broadcast_to(
        observations[:, None, :], (observations.shape[0], weights.size, 2)
    )
    _assert_allclose(
        actual.conditional_mean,
        expected_conditional_means,
        rtol=rtol,
        atol=atol,
    )
    _assert_allclose(
        actual.conditional_covariance,
        np.zeros((observations.shape[0], weights.size, 2, 2)),
        rtol=0.0,
        atol=atol,
    )


@pytest.mark.parametrize("dtype,_rtol,_atol,_row_sum_atol", DTYPE_CASES)
def test_xd_ip_noise_001_covariance_adapters_are_explicit_and_unambiguous(
    dtype, _rtol, _atol, _row_sum_atol
):
    """XD-IP-NOISE-001: variance adapters populate only intended entries."""

    isotropic_variances = jnp.asarray([0.0, 0.1, 0.5, 2.0], dtype=dtype)
    actual_isotropic = isotropic_covariance(isotropic_variances, dimension=3)
    expected_isotropic = np.asarray(isotropic_variances)[:, None, None] * np.eye(
        3, dtype=np.asarray(isotropic_variances).dtype
    )[None, :, :]
    np.testing.assert_array_equal(np.asarray(actual_isotropic), expected_isotropic)
    off_diagonal = np.asarray(actual_isotropic).copy()
    off_diagonal[:, np.arange(3), np.arange(3)] = 0.0
    np.testing.assert_array_equal(off_diagonal, 0.0)

    diagonal_variances = jnp.asarray(
        [[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]], dtype=dtype
    )
    actual_diagonal = diagonal_covariance(diagonal_variances)
    expected_diagonal = np.stack(
        [np.diag(row) for row in np.asarray(diagonal_variances)]
    )
    np.testing.assert_array_equal(np.asarray(actual_diagonal), expected_diagonal)

    correlated = jnp.asarray(
        [
            [[0.4, 0.1, -0.02], [0.1, 0.5, 0.03], [-0.02, 0.03, 0.3]],
            [[0.8, -0.05, 0.01], [-0.05, 0.6, 0.08], [0.01, 0.08, 0.7]],
        ],
        dtype=dtype,
    )
    actual_full = full_covariance(correlated)
    np.testing.assert_array_equal(np.asarray(actual_full), np.asarray(correlated))

    single_isotropic = isotropic_covariance(jnp.asarray(0.25, dtype=dtype), 3)
    np.testing.assert_array_equal(
        np.asarray(single_isotropic),
        np.eye(3, dtype=np.asarray(single_isotropic).dtype) * 0.25,
    )

    with pytest.raises(ValueError):
        isotropic_covariance(jnp.ones((4, 1), dtype=dtype), dimension=3)
    with pytest.raises(ValueError):
        isotropic_covariance(jnp.asarray([0.1, -0.2], dtype=dtype), dimension=3)
    with pytest.raises(ValueError):
        diagonal_covariance(jnp.asarray([[0.1, -0.2, 0.3]], dtype=dtype))
    with pytest.raises(ValueError):
        full_covariance(jnp.ones((2, 3), dtype=dtype))


@pytest.mark.parametrize("dtype,rtol,atol,_row_sum_atol", DTYPE_CASES)
def test_xd_ip_jit_001_eager_and_jitted_posterior_and_em_step_agree(
    dtype, rtol, atol, _row_sum_atol
):
    """XD-IP-JIT-001: canonical posterior and one-step paths compile cleanly."""

    parameters, observations, measurement_covariances = _ordinary_fixture(dtype)

    eager_posterior = posterior_components(
        parameters, observations, measurement_covariances
    )
    compiled_posterior = jax.jit(posterior_components)(
        parameters, observations, measurement_covariances
    )
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager_posterior),
        jax.tree_util.tree_leaves(compiled_posterior),
        strict=True,
    ):
        _assert_allclose(compiled_leaf, eager_leaf, rtol=rtol, atol=atol)

    eager_update = em_step(parameters, observations, measurement_covariances)
    compiled_update = jax.jit(em_step)(
        parameters, observations, measurement_covariances
    )
    for eager_leaf, compiled_leaf in zip(
        jax.tree_util.tree_leaves(eager_update),
        jax.tree_util.tree_leaves(compiled_update),
        strict=True,
    ):
        _assert_allclose(compiled_leaf, eager_leaf, rtol=rtol, atol=atol)
