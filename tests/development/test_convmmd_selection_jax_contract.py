"""JAX-contract evidence for the known-selection convMMD path (contract §17).

Covers the Gaussian-window parameter transform and the per-group fixed-``M`` selected
leaves: jit callback-freedom / retrace invariance, float32/float64 correctness,
autodiff-vs-finite-difference gradients through the transform and grouped selected
analytic loss, the selection-aware denoiser under vmap, explicit-PRNG semantics for
the SNIS Monte-Carlo loss and its statistical convergence to the §17.5 analytic value,
the ESS diagnostic and its collapse under a selective window, and a valid selected
analytic fit. The SNIS loss is held only to a statistical (convergence) gate; no
machine-epsilon parity is claimed for a general ``Omega``.

Rows: ``CMMD-SEL-JIT-001``, ``CMMD-SEL-GRAD-001``, ``CMMD-SEL-PRNG-001``,
``CMMD-SEL-DTYPE-001``, ``CMMD-SEL-VMAP-001``, ``CMMD-SEL-FIT-001``,
``CMMD-SEL-MC-001`` (JAX), ``CMMD-SEL-ESS-001`` (JAX).
"""

from __future__ import annotations

from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import ConvMMDParams, ConvMMDUnconstrained, to_canonical
from development.convmmd_fit import ConvMMDFitStatus
from development.convmmd_grouped import (
    group_masked_fit_inputs,
    group_masked_inputs,
    grouped_denoise,
)
from development.convmmd_selection import (
    CallableSelection,
    GaussianSelection,
    convmmd_loss_snis_projected,
    effective_volume_gaussian,
    fit_selected_analytic,
    fit_selected_mc,
    grouped_analytic_selected_loss,
    grouped_denoise_selected,
    grouped_snis_denoise,
    grouped_snis_loss,
    select_gaussian_params,
    snis_effective_sample_size,
)
from development.general_validation import PerItemFullNoise


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
            + 0.01 * jnp.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=dtype)
            for i in range(6)
        ]
    )
    bandwidths = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
    return observations, noise, bandwidths


def _selection(dtype=jnp.float64) -> GaussianSelection:
    return GaussianSelection(
        location=jnp.asarray([0.2, -0.1, 0.15], dtype=dtype),
        precision=jnp.asarray(
            [[0.50, 0.08, 0.00], [0.08, 0.45, 0.03], [0.00, 0.03, 0.40]], dtype=dtype
        ),
    )


def _grouped(dtype=jnp.float64, mask=_MASK):
    params = _params(dtype)
    observations, noise, _ = _data(dtype)
    return group_masked_inputs(
        params, observations, jnp.asarray(mask),
        noise=PerItemFullNoise(noise), dtype=dtype,
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


# --- a small fully-observed homoscedastic instance for the SNIS gates ----------


def _snis_instance(dtype=jnp.float64):
    params = ConvMMDParams(
        weights=jnp.asarray([0.4, 0.6], dtype=dtype),
        means=jnp.asarray([[-0.7, 0.3], [0.8, -0.4]], dtype=dtype),
        covariances=jnp.asarray(
            [[[0.5, 0.1], [0.1, 0.4]], [[0.6, -0.15], [-0.15, 0.5]]], dtype=dtype
        ),
    )
    observations = jnp.asarray(
        [[-0.6, 0.2], [0.1, -0.3], [0.9, -0.5], [-0.2, 0.6], [0.4, 0.1]], dtype=dtype
    )
    noise = jnp.broadcast_to(
        jnp.asarray([[0.12, 0.03], [0.03, 0.10]], dtype=dtype), (5, 2, 2)
    )
    bandwidths = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
    selection = GaussianSelection(
        location=jnp.asarray([0.1, -0.1], dtype=dtype),
        precision=jnp.asarray([[0.6, 0.1], [0.1, 0.5]], dtype=dtype),
    )
    grouped = group_masked_inputs(
        params, observations, jnp.ones(observations.shape, dtype=bool),
        noise=PerItemFullNoise(noise), dtype=dtype,
    )
    return params, grouped, bandwidths, selection


def _snis_unconstrained(dtype=jnp.float64) -> ConvMMDUnconstrained:
    return ConvMMDUnconstrained(
        alphas=jnp.asarray([0.0, 0.1], dtype=dtype),
        means=jnp.asarray([[-0.7, 0.3], [0.8, -0.4]], dtype=dtype),
        unconstrained_L=jnp.asarray(
            [[[0.3, 0.0], [0.1, 0.2]], [[0.25, 0.0], [-0.1, 0.2]]], dtype=dtype
        ),
    )


# --------------------------------------------------------------------------- #
# CMMD-SEL-JIT-001
# --------------------------------------------------------------------------- #


def test_transform_and_selected_leaves_are_callback_free_and_do_not_retrace():
    params = _params()
    grouped = _grouped()
    _, _, bandwidths = _data()
    selection = _selection()

    # (a) the parameter transform
    transform_jaxpr = str(jax.make_jaxpr(select_gaussian_params)(params, selection))
    assert "callback" not in transform_jaxpr.lower()

    # (b) the grouped selected analytic loss
    def selected_loss(p):
        return grouped_analytic_selected_loss(p, grouped, bandwidths, selection)

    loss_jaxpr = str(jax.make_jaxpr(selected_loss)(params))
    assert "callback" not in loss_jaxpr.lower()

    trace_count = 0

    def counted(p):
        nonlocal trace_count
        trace_count += 1
        return selected_loss(p)

    compiled = jax.jit(counted)
    first = compiled(params)
    _block(first)
    perturbed = ConvMMDParams(params.weights, params.means + 0.05, params.covariances)
    _block(compiled(perturbed))
    assert trace_count == 1
    assert jnp.isfinite(first)

    # (c) the SNIS leaf (omega / num_samples closed over -> array-only jit)
    sp, sg, sb, ssel = _snis_instance()
    group = _first_positive_group(sg)
    from development.convmmd_selection import gaussian_selection_omega
    omega = gaussian_selection_omega(ssel)
    key = jax.random.PRNGKey(0)

    snis_count = 0

    def counted_snis(p, x, proj, s):
        nonlocal snis_count
        snis_count += 1
        return convmmd_loss_snis_projected(p, x, proj, s, sb, key, 64, omega)

    snis_jaxpr = str(
        jax.make_jaxpr(lambda p, x, proj, s: convmmd_loss_snis_projected(
            p, x, proj, s, sb, key, 64, omega
        ))(sp, group.observations, group.projection_matrices,
           group.measurement_covariances)
    )
    assert "callback" not in snis_jaxpr.lower()
    compiled_snis = jax.jit(counted_snis)
    _block(compiled_snis(sp, group.observations, group.projection_matrices,
                         group.measurement_covariances))
    _block(compiled_snis(sp, group.observations + 0.01, group.projection_matrices,
                         group.measurement_covariances))
    assert snis_count == 1


# --------------------------------------------------------------------------- #
# CMMD-SEL-DTYPE-001
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_selected_dtype_is_preserved(dtype):
    params = _params(dtype)
    grouped = _grouped(dtype)
    _, _, bandwidths = _data(dtype)
    selection = _selection(dtype)

    selected = select_gaussian_params(params, selection)
    loss = grouped_analytic_selected_loss(params, grouped, bandwidths, selection)
    posterior_mean = grouped_denoise_selected(params, grouped, selection)
    z_theta = effective_volume_gaussian(params, selection)
    assert selected.weights.dtype == dtype
    assert selected.covariances.dtype == dtype
    assert loss.dtype == dtype
    assert posterior_mean.dtype == dtype
    assert z_theta.dtype == dtype
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(posterior_mean))

    sp, sg, sb, ssel = _snis_instance(dtype)
    snis = grouped_snis_loss(sp, sg, sb, jax.random.PRNGKey(0), 64, ssel)
    assert snis.dtype == dtype
    assert jnp.isfinite(snis)


# --------------------------------------------------------------------------- #
# CMMD-SEL-GRAD-001
# --------------------------------------------------------------------------- #


def _central_difference(function, value, *, step):
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


def test_selected_analytic_loss_gradient_matches_central_difference():
    grouped = _grouped()
    _, _, bandwidths = _data()
    selection = _selection()
    base = _unconstrained()

    def loss_from_means(means):
        candidate = ConvMMDUnconstrained(base.alphas, means, base.unconstrained_L)
        return grouped_analytic_selected_loss(
            to_canonical(candidate), grouped, bandwidths, selection
        )

    def loss_from_alphas(alphas):
        candidate = ConvMMDUnconstrained(alphas, base.means, base.unconstrained_L)
        return grouped_analytic_selected_loss(
            to_canonical(candidate), grouped, bandwidths, selection
        )

    for function, argument in (
        (loss_from_means, base.means),
        (loss_from_alphas, base.alphas),
    ):
        automatic = np.asarray(jax.grad(function)(argument))
        numeric = _central_difference(function, argument, step=1e-5)
        assert np.all(np.isfinite(automatic))
        np.testing.assert_allclose(automatic, numeric, rtol=3e-5, atol=3e-6)


# --------------------------------------------------------------------------- #
# CMMD-SEL-VMAP-001
# --------------------------------------------------------------------------- #


def test_selected_denoiser_vmaps_over_group_rows():
    params = _params()
    grouped = _grouped()
    selection = _selection()
    selected_params = select_gaussian_params(params, selection)
    group = _first_positive_group(grouped)
    from development.convmmd import denoise_projected

    batched = denoise_projected(
        selected_params, group.observations, group.projection_matrices,
        group.measurement_covariances,
    )

    def single(x, proj, s):
        return denoise_projected(
            selected_params, x[None, :], proj[None, :, :], s[None, :, :]
        )[0]

    mapped = jax.vmap(single)(
        group.observations, group.projection_matrices, group.measurement_covariances
    )
    np.testing.assert_allclose(
        np.asarray(mapped), np.asarray(batched), rtol=1e-12, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# CMMD-SEL-PRNG-001 and CMMD-SEL-MC-001 (JAX)
# --------------------------------------------------------------------------- #


def test_snis_prng_semantics():
    params, grouped, bandwidths, selection = _snis_instance()
    key = jax.random.PRNGKey(0)

    reuse_a = grouped_snis_loss(params, grouped, bandwidths, key, 96, selection)
    reuse_b = grouped_snis_loss(params, grouped, bandwidths, key, 96, selection)
    assert float(reuse_a) == float(reuse_b)  # identical key -> identical draw

    left, right = jax.random.split(key)
    split_a = grouped_snis_loss(params, grouped, bandwidths, left, 96, selection)
    split_b = grouped_snis_loss(params, grouped, bandwidths, right, 96, selection)
    assert float(split_a) != float(split_b)  # split keys -> independent draws

    with pytest.raises(TypeError):
        grouped_snis_loss(params, grouped, bandwidths)  # missing key


def test_snis_converges_to_selected_analytic():
    """CMMD-SEL-MC-001 (JAX): the SNIS loss converges to the §17.5 analytic value.
    Statistical only (SNIS is biased at finite M): the spread shrinks with M and the
    fine estimate agrees within noise."""

    params, grouped, bandwidths, selection = _snis_instance()
    analytic = float(
        grouped_analytic_selected_loss(params, grouped, bandwidths, selection)
    )

    def error_at(num_samples):
        keys = jax.random.split(jax.random.PRNGKey(11), 6)
        estimates = [
            float(grouped_snis_loss(params, grouped, bandwidths, k, num_samples, selection))
            for k in keys
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(64)
    fine_error, fine_se = error_at(512)
    assert fine_se < coarse_se               # spread shrinks with M
    assert fine_error < 4.0 * fine_se + 5e-3  # fine estimate agrees within noise


# --------------------------------------------------------------------------- #
# CMMD-SEL-ESS-001 (JAX)
# --------------------------------------------------------------------------- #


def test_snis_ess_diagnostic_range_and_collapse():
    params, grouped, bandwidths, selection = _snis_instance()
    n_components = int(params.weights.shape[0])
    num_samples = 4000

    ess, z_hat = snis_effective_sample_size(
        params, selection, jax.random.PRNGKey(7), num_samples
    )
    assert jnp.isfinite(ess) and jnp.isfinite(z_hat)
    assert 0.0 < float(ess) <= n_components * num_samples
    assert 0.0 < float(z_hat) <= 1.0 + 1e-6

    # A far more selective window collapses the ESS.
    selective = GaussianSelection(selection.location, selection.precision * 30.0)
    ess_selective, _ = snis_effective_sample_size(
        params, selective, jax.random.PRNGKey(7), num_samples
    )
    assert float(ess_selective) < float(ess)


def test_snis_denoiser_converges_to_selected_analytic():
    """CMMD-SEL-DENOISE-003 (JAX): the SNIS denoiser converges to the §17.5 analytic
    selected denoiser (statistical: spread shrinks, fine estimate within noise)."""

    params, grouped, _bandwidths, selection = _snis_instance()
    analytic = np.asarray(grouped_denoise_selected(params, grouped, selection))

    def error_at(num_samples):
        estimates = np.stack([
            np.asarray(grouped_snis_denoise(params, grouped, jax.random.PRNGKey(s),
                                            num_samples, selection))
            for s in range(6)
        ])
        mean = estimates.mean(axis=0)
        row_error = float(np.max(np.linalg.norm(mean - analytic, axis=1)))
        row_se = float(np.max(np.linalg.norm(estimates.std(axis=0), axis=1) / np.sqrt(6)))
        return row_error, row_se

    _, coarse_se = error_at(500)
    fine_error, fine_se = error_at(8000)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 5e-3


def test_snis_denoiser_prng_and_dtype():
    params, grouped, _bandwidths, selection = _snis_instance()
    key = jax.random.PRNGKey(0)
    reuse_a = grouped_snis_denoise(params, grouped, key, 256, selection)
    reuse_b = grouped_snis_denoise(params, grouped, key, 256, selection)
    np.testing.assert_array_equal(np.asarray(reuse_a), np.asarray(reuse_b))
    left, right = jax.random.split(key)
    split_a = grouped_snis_denoise(params, grouped, left, 256, selection)
    split_b = grouped_snis_denoise(params, grouped, right, 256, selection)
    assert not np.array_equal(np.asarray(split_a), np.asarray(split_b))

    params32, grouped32, _b32, selection32 = _snis_instance(jnp.float32)
    mean32 = grouped_snis_denoise(params32, grouped32, key, 128, selection32)
    assert mean32.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(mean32))


def test_snis_denoiser_m0_row_is_selected_prior_mean():
    """An all-M=0 group denoises to the selected prior mean (transformed-prior mean)."""

    params, _grouped, _bandwidths, selection = _snis_instance()
    observations = jnp.zeros((4, 2), dtype=jnp.float64)
    mask = jnp.zeros((4, 2), dtype=bool)  # every row M=0
    noise = jnp.broadcast_to(jnp.asarray([[0.12, 0.03], [0.03, 0.10]]), (4, 2, 2))
    grouped = group_masked_inputs(
        params, observations, mask, noise=PerItemFullNoise(noise), dtype=jnp.float64
    )
    selected = select_gaussian_params(params, selection)
    prior_mean = jnp.sum(selected.weights[:, None] * selected.means, axis=0)

    estimates = np.stack([
        np.asarray(grouped_snis_denoise(params, grouped, jax.random.PRNGKey(s), 8000, selection))
        for s in range(6)
    ])
    mean_row0 = estimates.mean(axis=0)[0]
    assert np.linalg.norm(mean_row0 - np.asarray(prior_mean)) < 3e-2


def test_snis_gaussian_and_callable_specs_agree_at_same_key():
    """A GaussianSelection and a CallableSelection wrapping the same omega give the
    identical SNIS value at the same key (the spec type is a routing detail)."""

    params, grouped, bandwidths, selection = _snis_instance()
    from development.convmmd_selection import gaussian_selection_omega

    callable_selection = CallableSelection(gaussian_selection_omega(selection))
    key = jax.random.PRNGKey(3)
    gaussian_value = grouped_snis_loss(params, grouped, bandwidths, key, 128, selection)
    callable_value = grouped_snis_loss(
        params, grouped, bandwidths, key, 128, callable_selection
    )
    assert float(gaussian_value) == float(callable_value)


# --------------------------------------------------------------------------- #
# CMMD-SEL-FIT-001
# --------------------------------------------------------------------------- #


def test_valid_selected_analytic_fit_reduces_loss_with_spd_covariances():
    params = _params()
    observations, noise, bandwidths = _data()
    selection = _selection()
    grouped = group_masked_fit_inputs(
        params, observations, jnp.asarray(_MASK), noise=PerItemFullNoise(noise),
        dtype=jnp.float64,
    )
    result = fit_selected_analytic(
        _unconstrained(), grouped, bandwidths, selection, n_steps=150
    )
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
    recomputed = grouped_analytic_selected_loss(
        result.parameters, grouped, bandwidths, selection
    )
    np.testing.assert_allclose(
        float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12
    )


def test_selected_mc_fit_runs_and_returns_spd():
    """The stochastic SNIS selected fit completes without numerical failure and
    returns SPD covariances (fixed-step; not held to loss reduction or a spurious
    CONVERGED, mirroring the base MC fit)."""

    params, grouped, bandwidths, selection = _snis_instance()
    result = fit_selected_mc(
        _snis_unconstrained(), grouped, bandwidths, selection,
        jax.random.PRNGKey(0), num_samples=96, n_steps=25,
    )
    assert not bool(result.numerical_failure)
    eigenvalues = jnp.linalg.eigvalsh(result.parameters.covariances)
    assert bool(jnp.all(eigenvalues > 0.0))
    assert jnp.isfinite(result.loss)
    assert result.metadata.contract_id == "xdgmm-jax.convmmd"
