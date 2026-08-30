"""JAX-contract evidence for the masked (MAR) convMMD path (contract §16).

Covers the per-group fixed-``M`` leaf's jit callback-freedom / retrace invariance,
float32/float64 correctness, autodiff-vs-finite-difference gradients through the
grouped loss, the projected denoiser under vmap, explicit-PRNG semantics for the
masked Monte-Carlo loss and its statistical convergence, ``M=0`` inertness and the
prior-mean fallback, honest failure/status (a degenerate covariance surfaces as a
visible NaN), a valid grouped fit, and the eager validation rejections (non-boolean
/ mismatched mask, NaN including masked positions, per-item isotropic/diagonal
masked noise). Mask grouping and validation are host-only and outside the JIT row.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic_projected,
    denoise_projected,
    posterior_components_projected,
    to_canonical,
)
from development.convmmd_fit import ConvMMDFitStatus
from development.convmmd_grouped import (
    convmmd_loss_analytic_masked,
    fit_masked_analytic,
    group_masked_fit_inputs,
    group_masked_inputs,
    grouped_analytic_loss,
    grouped_denoise,
    grouped_mc_loss,
    grouped_posterior_components,
)
from development.general_validation import (
    NoInformativeWeightError,
    PerItemDiagonalNoise,
    PerItemFullNoise,
    PerItemIsotropicNoise,
    ValidationError,
)


# --------------------------------------------------------------------------- #
# problem builders
# --------------------------------------------------------------------------- #

_MASK = np.array(
    [
        [True, True, True],    # M=3 (fully observed)
        [True, False, True],   # M=2
        [False, True, True],   # M=2
        [True, True, False],   # M=2
        [True, False, False],  # M=1
        [False, False, False], # M=0
    ]
)


def _params(dtype=jnp.float64) -> ConvMMDParams:
    return ConvMMDParams(
        weights=jnp.asarray([0.4, 0.6], dtype=dtype),
        means=jnp.asarray([[-0.5, -0.1, 0.2], [0.8, 0.6, -0.3]], dtype=dtype),
        covariances=jnp.asarray(
            [
                [[0.74, 0.11, 0.02], [0.11, 0.63, -0.05], [0.02, -0.05, 0.58]],
                [[0.68, -0.07, 0.03], [-0.07, 0.82, 0.04], [0.03, 0.04, 0.71]],
            ],
            dtype=dtype,
        ),
    )


def _data(dtype=jnp.float64):
    observations = jnp.asarray(
        [
            [-0.9, 0.2, 0.1],
            [-0.1, -0.6, 0.4],
            [0.7, 0.4, -0.2],
            [1.3, 1.0, 0.3],
            [0.2, -0.2, 0.5],
            [0.0, 0.3, -0.4],
        ],
        dtype=dtype,
    )
    noise = jnp.stack(
        [
            (0.08 + 0.02 * i) * jnp.eye(3, dtype=dtype)
            + 0.01 * jnp.asarray(
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype
            )
            for i in range(6)
        ]
    )
    bandwidths = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
    return observations, noise, bandwidths


def _grouped(dtype=jnp.float64, mask=_MASK):
    params = _params(dtype)
    observations, noise, _ = _data(dtype)
    return group_masked_inputs(
        params,
        observations,
        jnp.asarray(mask),
        noise=PerItemFullNoise(noise),
        dtype=dtype,
    )


def _unconstrained(dtype=jnp.float64) -> ConvMMDUnconstrained:
    return ConvMMDUnconstrained(
        alphas=jnp.asarray([0.1, -0.2], dtype=dtype),
        means=jnp.asarray([[-0.5, -0.1, 0.2], [0.8, 0.6, -0.3]], dtype=dtype),
        unconstrained_L=jnp.asarray(
            [
                [[0.3, 0.0, 0.0], [0.2, 0.1, 0.0], [0.05, -0.1, 0.2]],
                [[0.25, 0.0, 0.0], [-0.1, 0.2, 0.0], [0.1, 0.05, 0.15]],
            ],
            dtype=dtype,
        ),
    )


def _block(value):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)


def _first_positive_group(grouped):
    for group in grouped.groups:
        if group.observations.shape[-1] > 0:
            return group
    raise AssertionError("no informative group")


# --------------------------------------------------------------------------- #
# CMMD-MISS-JIT-001: leaf callback-free + retrace invariant; grouped loss jit
# --------------------------------------------------------------------------- #


def test_projected_leaf_is_callback_free_and_does_not_retrace():
    params = _params()
    group = _first_positive_group(_grouped())
    _, _, bandwidths = _data()

    def analytic(p, x, proj, s):
        return convmmd_loss_analytic_projected(p, x, proj, s, bandwidths)

    def denoiser(p, x, proj, s):
        return denoise_projected(p, x, proj, s)

    for operation in (analytic, denoiser):
        text = str(
            jax.make_jaxpr(operation)(
                params,
                group.observations,
                group.projection_matrices,
                group.measurement_covariances,
            )
        )
        assert "callback" not in text.lower()

        trace_count = 0

        def counted(p, x, proj, s):
            nonlocal trace_count
            trace_count += 1
            return operation(p, x, proj, s)

        compiled = jax.jit(counted)
        _block(
            compiled(
                params,
                group.observations,
                group.projection_matrices,
                group.measurement_covariances,
            )
        )
        _block(
            compiled(
                params,
                group.observations + 0.01,
                group.projection_matrices,
                group.measurement_covariances,
            )
        )
        assert trace_count == 1


def test_grouped_analytic_loss_jits_over_fixed_structure():
    grouped = _grouped()
    _, _, bandwidths = _data()

    trace_count = 0

    def counted(p):
        nonlocal trace_count
        trace_count += 1
        return grouped_analytic_loss(p, grouped, bandwidths)

    compiled = jax.jit(counted)
    first = compiled(_params())
    _block(first)
    perturbed = _params()
    perturbed = ConvMMDParams(
        perturbed.weights, perturbed.means + 0.05, perturbed.covariances
    )
    _block(compiled(perturbed))
    assert trace_count == 1
    assert jnp.isfinite(first)


# --------------------------------------------------------------------------- #
# CMMD-MISS-DTYPE-001
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_masked_dtype_is_preserved(dtype):
    params = _params(dtype)
    grouped = _grouped(dtype)
    _, _, bandwidths = _data(dtype)
    loss = grouped_analytic_loss(params, grouped, bandwidths)
    posterior_mean = grouped_denoise(params, grouped)
    mc = grouped_mc_loss(params, grouped, bandwidths, jax.random.PRNGKey(0), 64)
    assert loss.dtype == dtype
    assert posterior_mean.dtype == dtype
    assert mc.dtype == dtype
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(posterior_mean))


# --------------------------------------------------------------------------- #
# CMMD-MISS-GRAD-001
# --------------------------------------------------------------------------- #


def _central_difference(
    function: Callable[[jax.Array], jax.Array], value: jax.Array, *, step: float
) -> np.ndarray:
    base = np.asarray(value, dtype=np.float64)
    gradient = np.empty_like(base)
    for index in np.ndindex(base.shape):
        positive = base.copy()
        negative = base.copy()
        positive[index] += step
        negative[index] -= step
        upper = float(np.asarray(function(jnp.asarray(positive))))
        lower = float(np.asarray(function(jnp.asarray(negative))))
        gradient[index] = (upper - lower) / (2.0 * step)
    return gradient


def test_masked_analytic_loss_gradient_matches_central_difference():
    grouped = _grouped()
    _, _, bandwidths = _data()
    base = _unconstrained()

    def loss_from_means(means):
        candidate = ConvMMDUnconstrained(base.alphas, means, base.unconstrained_L)
        return grouped_analytic_loss(to_canonical(candidate), grouped, bandwidths)

    def loss_from_alphas(alphas):
        candidate = ConvMMDUnconstrained(alphas, base.means, base.unconstrained_L)
        return grouped_analytic_loss(to_canonical(candidate), grouped, bandwidths)

    for function, argument in (
        (loss_from_means, base.means),
        (loss_from_alphas, base.alphas),
    ):
        automatic = np.asarray(jax.grad(function)(argument))
        numeric = _central_difference(function, argument, step=1e-5)
        assert np.all(np.isfinite(automatic))
        np.testing.assert_allclose(automatic, numeric, rtol=2e-5, atol=2e-6)


# --------------------------------------------------------------------------- #
# CMMD-MISS-VMAP-001
# --------------------------------------------------------------------------- #


def test_projected_denoiser_vmaps_over_group_rows():
    params = _params()
    group = _first_positive_group(_grouped())
    batched = denoise_projected(
        params,
        group.observations,
        group.projection_matrices,
        group.measurement_covariances,
    )

    def single(x, proj, s):
        return denoise_projected(
            params, x[None, :], proj[None, :, :], s[None, :, :]
        )[0]

    mapped = jax.vmap(single)(
        group.observations,
        group.projection_matrices,
        group.measurement_covariances,
    )
    np.testing.assert_allclose(
        np.asarray(mapped), np.asarray(batched), rtol=1e-12, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# CMMD-MISS-PRNG-001 and CMMD-MISS-MC-001
# --------------------------------------------------------------------------- #


def test_masked_mc_prng_semantics():
    params = _params()
    grouped = _grouped()
    _, _, bandwidths = _data()
    key = jax.random.PRNGKey(0)

    reuse_a = grouped_mc_loss(params, grouped, bandwidths, key, 128)
    reuse_b = grouped_mc_loss(params, grouped, bandwidths, key, 128)
    assert float(reuse_a) == float(reuse_b)  # identical key -> identical draw

    left, right = jax.random.split(key)
    split_a = grouped_mc_loss(params, grouped, bandwidths, left, 128)
    split_b = grouped_mc_loss(params, grouped, bandwidths, right, 128)
    assert float(split_a) != float(split_b)  # split keys -> independent draws

    with pytest.raises(TypeError):
        grouped_mc_loss(params, grouped, bandwidths)  # missing key


def test_masked_mc_converges_to_masked_analytic():
    params = _params()
    grouped = _grouped()
    _, _, bandwidths = _data()
    analytic = float(grouped_analytic_loss(params, grouped, bandwidths))

    def error_at(num_samples: int) -> tuple[float, float]:
        keys = jax.random.split(jax.random.PRNGKey(11), 8)
        estimates = [
            float(grouped_mc_loss(params, grouped, bandwidths, k, num_samples))
            for k in keys
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(256)
    fine_error, fine_se = error_at(8192)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 1e-9


# --------------------------------------------------------------------------- #
# CMMD-M0-001 (JAX side)
# --------------------------------------------------------------------------- #


def test_m0_rows_are_inert_and_return_prior_mean():
    params = _params()
    observations, noise, bandwidths = _data()

    informative = _MASK.any(axis=1)
    grouped_full = group_masked_inputs(
        params, observations, jnp.asarray(_MASK), noise=PerItemFullNoise(noise),
        dtype=jnp.float64,
    )
    grouped_dropped = group_masked_inputs(
        params,
        observations[informative],
        jnp.asarray(_MASK[informative]),
        noise=PerItemFullNoise(noise[informative]),
        dtype=jnp.float64,
    )
    full_loss = grouped_analytic_loss(params, grouped_full, bandwidths)
    dropped_loss = grouped_analytic_loss(params, grouped_dropped, bandwidths)
    np.testing.assert_allclose(
        float(full_loss), float(dropped_loss), rtol=1e-13, atol=1e-13
    )

    # Denoiser returns the prior mean / prior weights for every M=0 row.
    components = grouped_posterior_components(params, grouped_full)
    posterior_mean = grouped_denoise(params, grouped_full)
    prior_mean = jnp.sum(params.weights[:, None] * params.means, axis=0)
    for row in np.flatnonzero(~informative):
        np.testing.assert_allclose(
            np.asarray(posterior_mean[row]), np.asarray(prior_mean),
            rtol=1e-13, atol=1e-13,
        )
        np.testing.assert_allclose(
            np.asarray(components.responsibilities[row]),
            np.asarray(params.weights),
            rtol=1e-13, atol=1e-13,
        )


def test_all_m0_loss_is_zero_and_fit_raises():
    params = _params()
    observations, noise, bandwidths = _data()
    all_zero = jnp.zeros(observations.shape, dtype=bool)
    grouped = group_masked_inputs(
        params, observations, all_zero, noise=PerItemFullNoise(noise),
        dtype=jnp.float64,
    )
    loss = grouped_analytic_loss(params, grouped, bandwidths)
    assert float(loss) == 0.0

    with pytest.raises(NoInformativeWeightError):
        group_masked_fit_inputs(
            params, observations, all_zero, noise=PerItemFullNoise(noise),
            dtype=jnp.float64,
        )


def test_zero_informative_weight_loss_is_zero_and_matches_oracle():
    """Informative rows that all carry zero sample weight yield loss exactly 0
    (not 0/0 = NaN), consistently in the analytic and MC paths, under jit, and in
    agreement with the independent oracle."""

    from tests.reference import convmmd as oracle

    params = _params()
    observations, noise, bandwidths = _data()
    # Zero weight on every observed row; a nonzero weight only on the M=0 row.
    sample_weight = jnp.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 5.0])
    grouped = group_masked_inputs(
        params, observations, jnp.asarray(_MASK),
        noise=PerItemFullNoise(noise), sample_weight=sample_weight,
        dtype=jnp.float64,
    )
    loss = grouped_analytic_loss(params, grouped, bandwidths)
    jitted = jax.jit(lambda p: grouped_analytic_loss(p, grouped, bandwidths))(params)
    mc = grouped_mc_loss(params, grouped, bandwidths, jax.random.PRNGKey(0), 32)
    assert float(loss) == 0.0
    assert float(jitted) == 0.0
    assert float(mc) == 0.0

    oracle_loss = oracle.convmmd_loss_masked(
        np.asarray(params.weights), np.asarray(params.means),
        np.asarray(params.covariances), np.asarray(observations),
        np.asarray(_MASK), np.asarray(noise), np.asarray(bandwidths),
        sample_weight=np.asarray(sample_weight),
    ).loss
    assert oracle_loss == 0.0

    # The gradient is finite (no NaN leaking from the guarded zero branch).
    grad = jax.grad(lambda p: grouped_analytic_loss(p, grouped, bandwidths))(params)
    assert bool(jnp.all(jnp.isfinite(grad.means)))
    assert bool(jnp.all(jnp.isfinite(grad.covariances)))


# --------------------------------------------------------------------------- #
# CMMD-MISS-STATUS-001
# --------------------------------------------------------------------------- #


def test_degenerate_covariance_surfaces_as_visible_nan():
    params = _params()
    grouped = _grouped()
    _, _, bandwidths = _data()
    broken = ConvMMDParams(
        weights=params.weights,
        means=params.means,
        covariances=params.covariances.at[0].set(-10.0 * jnp.eye(3)),
    )
    loss = grouped_analytic_loss(broken, grouped, bandwidths)
    assert jnp.isnan(loss)


# --------------------------------------------------------------------------- #
# CMMD-MISS-FIT-001
# --------------------------------------------------------------------------- #


def test_valid_masked_fit_reduces_loss_with_spd_covariances():
    params = _params()
    observations, noise, bandwidths = _data()
    grouped = group_masked_fit_inputs(
        params, observations, jnp.asarray(_MASK), noise=PerItemFullNoise(noise),
        dtype=jnp.float64,
    )
    result = fit_masked_analytic(_unconstrained(), grouped, bandwidths, n_steps=150)
    assert not bool(result.numerical_failure)
    assert int(result.status) in (
        int(ConvMMDFitStatus.CONVERGED),
        int(ConvMMDFitStatus.MAX_ITER),
    )
    assert float(result.loss) <= float(result.history[0])
    eigenvalues = jnp.linalg.eigvalsh(result.parameters.covariances)
    assert bool(jnp.all(eigenvalues > 0.0))
    assert result.metadata.contract_id == "xdgmm-jax.convmmd"

    # The reported analytic loss recomputes exactly at the returned parameters.
    recomputed = grouped_analytic_loss(result.parameters, grouped, bandwidths)
    np.testing.assert_allclose(
        float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# CMMD-MISS-VAL-001
# --------------------------------------------------------------------------- #


def test_masked_validation_rejections():
    params = _params()
    observations, noise, _ = _data()
    mask = jnp.asarray(_MASK)

    def group(obs=observations, msk=mask, noise_spec=None):
        return group_masked_inputs(
            params, obs, msk,
            noise=noise_spec if noise_spec is not None else PerItemFullNoise(noise),
            dtype=jnp.float64,
        )

    # Non-boolean mask.
    with pytest.raises(ValidationError):
        group(msk=jnp.asarray(_MASK, dtype=jnp.float64))
    # Mismatched mask shape.
    with pytest.raises(ValidationError):
        group(msk=jnp.asarray(_MASK[:3]))
    # NaN in an observation, including at a masked position (row 5 is all-missing).
    with pytest.raises(ValidationError):
        group(obs=observations.at[5, 0].set(jnp.nan))
    with pytest.raises(ValidationError):
        group(obs=observations.at[0, 0].set(jnp.nan))
    # NaN in the measurement covariances.
    with pytest.raises(ValidationError):
        group(noise_spec=PerItemFullNoise(noise.at[0, 0, 0].set(jnp.nan)))
    # Per-item isotropic/diagonal masked noise is excluded.
    with pytest.raises(ValidationError):
        group(noise_spec=PerItemDiagonalNoise(jnp.ones((6, 3))))
    with pytest.raises(ValidationError):
        group(noise_spec=PerItemIsotropicNoise(jnp.ones((6,))))


def test_large_noise_is_not_treated_as_missing():
    params = _params()
    observations, noise, bandwidths = _data()
    # A large but finite covariance is ordinary noise, not missingness: a fully
    # observed row with huge noise stays in the M=D group and the loss is finite.
    loud_noise = noise.at[0].set(1.0e6 * jnp.eye(3))
    full_mask = jnp.ones(observations.shape, dtype=bool)
    grouped = group_masked_inputs(
        params, observations, full_mask, noise=PerItemFullNoise(loud_noise),
        dtype=jnp.float64,
    )
    assert len(grouped.groups) == 1
    assert grouped.groups[0].observations.shape[-1] == observations.shape[1]
    loss = grouped_analytic_loss(params, grouped, bandwidths)
    assert jnp.isfinite(loss)
