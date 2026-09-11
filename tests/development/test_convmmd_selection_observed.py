"""Omega(x-tilde) selected convMMD parity and contract (contract §17.12).

Selection on the OBSERVED value under the homoscedastic, fully-observed restriction:
the Gaussian window acts on the noise-convolved model, so the transform is the §17.5
identity on the inflated covariances ``Sigma+S``. Binds the pure-JAX observed-value
path (transform, analytic loss, effective volume, general-Omega SNIS) to the
independent NumPy oracle at float64 near machine epsilon and a declared float32
profile; checks the ``Psi^{-1}=0`` reduction to the base loss, the denoiser reducing
to the base MAR posterior, the homoscedastic guard, jit/grad, and a valid fit.

Rows: ``CMMD-SEL-OBS-XFORM-001``, ``CMMD-SEL-OBS-GAUSS-001``, ``CMMD-SEL-OBS-REDUCE-001``,
``CMMD-SEL-OBS-DENOISE-001``, ``CMMD-SEL-OBS-MC-001``, ``CMMD-SEL-OBS-VAL-001``,
``CMMD-SEL-OBS-JIT-001``, ``CMMD-SEL-OBS-FIT-001``.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    denoise,
    to_canonical,
)
from development.convmmd_fit import ConvMMDFitStatus
from development.convmmd_selection import (
    CallableSelection,
    GaussianSelection,
    convmmd_denoise_selected_observed,
    convmmd_loss_analytic_selected_observed,
    convmmd_loss_snis_selected_observed,
    effective_volume_gaussian_observed,
    fit_selected_observed_analytic,
    gaussian_selection_omega,
    select_gaussian_params_observed,
)
from tests.reference import convmmd as oracle


F64_RTOL, F64_ATOL = 5e-8, 5e-10
F32_RTOL, F32_ATOL = 2e-4, 2e-5
DTYPE_CASES = [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)]

DIMENSION, N_COMPONENTS, N_SAMPLES = 3, 2, 24


def _instance():
    rng = np.random.Generator(np.random.PCG64(20260902))

    def spd(scale):
        raw = rng.standard_normal((DIMENSION, DIMENSION))
        cov = raw @ raw.T
        cov *= scale / np.trace(cov)
        return cov + 0.4 * np.eye(DIMENSION)

    weights = np.array([0.4, 0.6])
    means = rng.standard_normal((N_COMPONENTS, DIMENSION))
    covariances = np.stack([spd(1.0 + 0.3 * k) for k in range(N_COMPONENTS)])
    noise = spd(0.3)  # homoscedastic
    observations = rng.standard_normal((N_SAMPLES, DIMENSION))
    bandwidths = np.array([0.5, 1.0, 2.0])
    location = rng.standard_normal(DIMENSION)
    precision = spd(0.6)
    return weights, means, covariances, noise, observations, bandwidths, location, precision


def _params(weights, means, covariances, dtype):
    return ConvMMDParams(
        jnp.asarray(weights, dtype), jnp.asarray(means, dtype),
        jnp.asarray(covariances, dtype),
    )


def _selection(location, precision, dtype):
    return GaussianSelection(jnp.asarray(location, dtype), jnp.asarray(precision, dtype))


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_observed_transform_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-OBS-XFORM-001."""

    w, m, c, s, _x, _bw, a, p = _instance()
    transform = oracle.gaussian_window_transform_observed(w, m, c, s, a, p)
    selected = select_gaussian_params_observed(
        _params(w, m, c, dtype), _selection(a, p, dtype), jnp.asarray(s, dtype)
    )
    assert selected.covariances.dtype == dtype
    np.testing.assert_allclose(np.asarray(selected.weights), transform.weights, rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(selected.means), transform.means, rtol=rtol, atol=atol)
    np.testing.assert_allclose(np.asarray(selected.covariances), transform.covariances, rtol=rtol, atol=atol)
    for cov in np.asarray(selected.covariances, np.float64):
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_observed_analytic_loss_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-OBS-GAUSS-001."""

    w, m, c, s, x, bw, a, p = _instance()
    loss = convmmd_loss_analytic_selected_observed(
        _params(w, m, c, dtype), jnp.asarray(x, dtype),
        noise=jnp.asarray(s, dtype), bandwidths=jnp.asarray(bw, dtype),
        selection=_selection(a, p, dtype),
    )
    oracle_loss = oracle.convmmd_loss_selected_observed(w, m, c, x, s, bw, a, p).loss
    assert loss.dtype == dtype
    np.testing.assert_allclose(np.asarray(loss), oracle_loss, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_observed_precision_zero_reduces_to_base_loss(dtype, rtol, atol):
    """CMMD-SEL-OBS-REDUCE-001: Psi^{-1}=0 == the base §4 loss with noise S."""

    w, m, c, s, x, bw, a, _p = _instance()
    params = _params(w, m, c, dtype)
    no_selection = GaussianSelection(
        jnp.asarray(a, dtype), jnp.zeros((DIMENSION, DIMENSION), dtype)
    )
    selected = convmmd_loss_analytic_selected_observed(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(s, dtype),
        bandwidths=jnp.asarray(bw, dtype), selection=no_selection,
    )
    noise_full = jnp.broadcast_to(jnp.asarray(s, dtype), (N_SAMPLES, DIMENSION, DIMENSION))
    base = convmmd_loss_analytic(params, jnp.asarray(x, dtype), noise_full, jnp.asarray(bw, dtype))
    np.testing.assert_allclose(np.asarray(selected), np.asarray(base), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_observed_denoiser_is_base_mar_posterior(dtype, rtol, atol):
    """CMMD-SEL-OBS-DENOISE-001: Omega(x-tilde) cancels -> the base MAR posterior."""

    w, m, c, s, x, _bw, _a, _p = _instance()
    params = _params(w, m, c, dtype)
    observed_denoise = convmmd_denoise_selected_observed(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(s, dtype)
    )
    noise_full = jnp.broadcast_to(jnp.asarray(s, dtype), (N_SAMPLES, DIMENSION, DIMENSION))
    base = denoise(params, jnp.asarray(x, dtype), noise_full)
    assert observed_denoise.dtype == dtype
    np.testing.assert_allclose(np.asarray(observed_denoise), np.asarray(base), rtol=rtol, atol=atol)
    oracle_denoise = oracle.denoise(w, m, c, x, np.broadcast_to(s, (N_SAMPLES, DIMENSION, DIMENSION))).posterior_mean
    np.testing.assert_allclose(np.asarray(observed_denoise), oracle_denoise, rtol=rtol, atol=atol)


def test_observed_effective_volume_in_range_and_reduces():
    w, m, c, s, _x, _bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    z = float(effective_volume_gaussian_observed(params, _selection(a, p, jnp.float64), jnp.asarray(s)))
    assert 0.0 < z <= 1.0 + 1e-12
    no_selection = GaussianSelection(jnp.asarray(a), jnp.zeros((DIMENSION, DIMENSION)))
    z_one = float(effective_volume_gaussian_observed(params, no_selection, jnp.asarray(s)))
    np.testing.assert_allclose(z_one, 1.0, rtol=0, atol=1e-9)


def test_observed_snis_converges_to_analytic():
    """CMMD-SEL-OBS-MC-001: SNIS Omega(x-tilde) converges to the §17.12 analytic value
    (statistical: spread shrinks, fine estimate within noise)."""

    w, m, c, s, x, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)
    analytic = float(convmmd_loss_analytic_selected_observed(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw), selection=selection))

    def error_at(num_samples):
        estimates = [
            float(convmmd_loss_snis_selected_observed(
                params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
                selection=selection, key=jax.random.PRNGKey(seed), num_samples=num_samples))
            for seed in range(6)
        ]
        return abs(np.mean(estimates) - analytic), float(np.std(estimates) / np.sqrt(6))

    _, coarse_se = error_at(64)
    fine_error, fine_se = error_at(512)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 5e-3

    # Gaussian-spec and callable-spec agree at the same key (spec type is routing).
    callable_selection = CallableSelection(gaussian_selection_omega(selection))
    key = jax.random.PRNGKey(3)
    v_g = convmmd_loss_snis_selected_observed(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=selection, key=key, num_samples=128)
    v_c = convmmd_loss_snis_selected_observed(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=callable_selection, key=key, num_samples=128)
    assert float(v_g) == float(v_c)


def test_observed_homoscedastic_guard_rejects_heteroscedastic():
    """CMMD-SEL-OBS-VAL-001: genuinely heteroscedastic noise is rejected (§17.11/§17.12)."""

    w, m, c, s, x, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    hetero = np.stack([
        (0.2 + 0.05 * i) * np.eye(DIMENSION) + 0.01 * np.ones((DIMENSION, DIMENSION))
        for i in range(N_SAMPLES)
    ])
    with pytest.raises(ValueError):
        convmmd_loss_analytic_selected_observed(
            params, jnp.asarray(x), noise=jnp.asarray(hetero),
            bandwidths=jnp.asarray(bw), selection=_selection(a, p, jnp.float64))
    # An (N,D,D) stack with identical rows IS accepted (homoscedastic in stacked form).
    homo_stack = np.broadcast_to(s, (N_SAMPLES, DIMENSION, DIMENSION)).copy()
    loss = convmmd_loss_analytic_selected_observed(
        params, jnp.asarray(x), noise=jnp.asarray(homo_stack),
        bandwidths=jnp.asarray(bw), selection=_selection(a, p, jnp.float64))
    assert jnp.isfinite(loss)


def test_observed_loss_jits_and_grads():
    """CMMD-SEL-OBS-JIT-001: the observed analytic loss is callback-free, jits without
    retrace, and is differentiable through the observed-space transform."""

    w, m, c, s, x, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)

    def loss(pp):
        return convmmd_loss_analytic_selected_observed(
            pp, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw), selection=selection)

    assert "callback" not in str(jax.make_jaxpr(loss)(params)).lower()
    trace_count = 0

    def counted(pp):
        nonlocal trace_count
        trace_count += 1
        return loss(pp)

    compiled = jax.jit(counted)
    compiled(params).block_until_ready()
    perturbed = ConvMMDParams(params.weights, params.means + 0.05, params.covariances)
    compiled(perturbed).block_until_ready()
    assert trace_count == 1

    grad = jax.grad(lambda mu: loss(ConvMMDParams(params.weights, mu, params.covariances)))(params.means)
    assert bool(jnp.all(jnp.isfinite(grad))) and float(jnp.linalg.norm(grad)) > 0


def test_observed_analytic_fit_reduces_loss_with_spd_covariances():
    """CMMD-SEL-OBS-FIT-001."""

    w, m, c, s, x, bw, a, p = _instance()
    initial = ConvMMDUnconstrained(
        alphas=jnp.asarray(np.log(w + 1e-8)),
        means=jnp.asarray(m),
        unconstrained_L=jnp.asarray(np.stack([0.3 * np.eye(DIMENSION) for _ in range(N_COMPONENTS)])),
    )
    result = fit_selected_observed_analytic(
        initial, jnp.asarray(x), jnp.asarray(s), jnp.asarray(bw), _selection(a, p, jnp.float64),
        n_steps=150,
    )
    assert not bool(result.numerical_failure)
    assert int(result.status) in (int(ConvMMDFitStatus.CONVERGED), int(ConvMMDFitStatus.MAX_ITER))
    assert float(result.loss) <= float(result.history[0])
    assert bool(jnp.all(jnp.linalg.eigvalsh(result.parameters.covariances) > 0.0))
    recomputed = convmmd_loss_analytic_selected_observed(
        result.parameters, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=_selection(a, p, jnp.float64))
    np.testing.assert_allclose(float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12)
