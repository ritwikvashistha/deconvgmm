"""Heteroscedastic Omega(x-tilde) selected convMMD parity and contract (contract §17.13).

Selection on the OBSERVED value with **per-object** measurement noise ``S_i``, fully
observed. Each detected object's ``S_i`` is KNOWN, so the §17.5 identity applies to the
inflated covariances ``Sigma_k + S_i`` **per object**: the selected observed density
conditional on ``S_i`` is the exact Gaussian mixture ``(pi''_i, nu''_i, B''_i)`` and the
loss is the sum of per-object one-sample discrepancies. This binds the pure-JAX
heteroscedastic path (per-object transform, analytic loss, general-Omega per-object SNIS,
denoiser) to the independent NumPy oracle at float64 near machine epsilon and a declared
float32 profile; checks the ``all-S_i-equal`` reduction to §17.12, the ``Psi^{-1}=0``
reduction to the base §4 loss with per-item noise, the base-MAR denoiser, the
**no-noise-model** invariant, jit/grad/vmap, and a valid fit.

Rows: ``CMMD-SEL-HET-XFORM-001``, ``CMMD-SEL-HET-GAUSS-001``, ``CMMD-SEL-HET-REDUCE-001``,
``CMMD-SEL-HET-DENOISE-001``, ``CMMD-SEL-HET-MC-001``, ``CMMD-SEL-HET-NOMODEL-001``,
``CMMD-SEL-HET-JIT-001``, ``CMMD-SEL-HET-FIT-001``.
"""

from __future__ import annotations

import inspect

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
)
from development.convmmd_fit import ConvMMDFitStatus
from development.convmmd_selection import (
    CallableSelection,
    GaussianSelection,
    convmmd_denoise_selected_observed_hetero,
    convmmd_loss_analytic_selected_observed,
    convmmd_loss_analytic_selected_observed_hetero,
    convmmd_loss_snis_selected_observed_hetero,
    fit_selected_observed_hetero_analytic,
    gaussian_selection_omega,
    select_gaussian_params_observed_hetero,
    snis_diagnostics_observed_hetero,
)
from tests.reference import convmmd as oracle


F64_RTOL, F64_ATOL = 5e-8, 5e-10
F32_RTOL, F32_ATOL = 2e-4, 2e-5
DTYPE_CASES = [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)]

DIMENSION, N_COMPONENTS, N_SAMPLES = 3, 2, 16


def _instance():
    """A genuinely heteroscedastic instance: every ``S_i`` distinct."""

    rng = np.random.Generator(np.random.PCG64(20260831))

    def spd(scale):
        raw = rng.standard_normal((DIMENSION, DIMENSION))
        cov = raw @ raw.T
        cov *= scale / np.trace(cov)
        return cov + 0.4 * np.eye(DIMENSION)

    weights = np.array([0.4, 0.6])
    means = rng.standard_normal((N_COMPONENTS, DIMENSION))
    covariances = np.stack([spd(1.0 + 0.3 * k) for k in range(N_COMPONENTS)])
    # per-object noise: each row a distinct SPD matrix (genuine heteroscedasticity)
    noise = np.stack([spd(0.2 + 0.5 * rng.random()) for _ in range(N_SAMPLES)])
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
def test_hetero_transform_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-HET-XFORM-001: the per-object transform equals the oracle for
    ARBITRARY per-object S_i and yields SPD B''_{k,i}."""

    w, m, c, s, _x, _bw, a, p = _instance()
    weights_j, means_j, covs_j = select_gaussian_params_observed_hetero(
        _params(w, m, c, dtype), _selection(a, p, dtype), jnp.asarray(s, dtype)
    )
    assert covs_j.dtype == dtype
    weights_j = np.asarray(weights_j)
    means_j = np.asarray(means_j)
    covs_j = np.asarray(covs_j)
    for i in range(N_SAMPLES):
        transform = oracle.gaussian_window_transform(w, m, c + s[i][None, :, :], a, p)
        np.testing.assert_allclose(weights_j[i], transform.weights, rtol=rtol, atol=atol)
        np.testing.assert_allclose(means_j[i], transform.means, rtol=rtol, atol=atol)
        np.testing.assert_allclose(covs_j[i], transform.covariances, rtol=rtol, atol=atol)
    for cov in covs_j.reshape(-1, DIMENSION, DIMENSION).astype(np.float64):
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_hetero_analytic_loss_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-HET-GAUSS-001."""

    w, m, c, s, x, bw, a, p = _instance()
    loss = convmmd_loss_analytic_selected_observed_hetero(
        _params(w, m, c, dtype), jnp.asarray(x, dtype),
        noise=jnp.asarray(s, dtype), bandwidths=jnp.asarray(bw, dtype),
        selection=_selection(a, p, dtype),
    )
    oracle_loss = oracle.convmmd_loss_selected_observed_hetero(w, m, c, x, s, bw, a, p).loss
    assert loss.dtype == dtype
    np.testing.assert_allclose(np.asarray(loss), oracle_loss, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_hetero_reduces_to_homoscedastic_and_base(dtype, rtol, atol):
    """CMMD-SEL-HET-REDUCE-001: all-S_i-equal == §17.12; Psi^{-1}=0 == base §4 loss."""

    w, m, c, s, x, bw, a, p = _instance()
    params = _params(w, m, c, dtype)

    # (a) all S_i equal to a single S -> the §17.12 homoscedastic analytic value.
    single = s[0]
    stacked = np.broadcast_to(single, (N_SAMPLES, DIMENSION, DIMENSION)).copy()
    het = convmmd_loss_analytic_selected_observed_hetero(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(stacked, dtype),
        bandwidths=jnp.asarray(bw, dtype), selection=_selection(a, p, dtype))
    homo = convmmd_loss_analytic_selected_observed(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(single, dtype),
        bandwidths=jnp.asarray(bw, dtype), selection=_selection(a, p, dtype))
    np.testing.assert_allclose(np.asarray(het), np.asarray(homo), rtol=rtol, atol=atol)

    # (b) Psi^{-1}=0 (no selection) -> the base §4 loss with per-item noise S_i.
    no_selection = GaussianSelection(jnp.asarray(a, dtype), jnp.zeros((DIMENSION, DIMENSION), dtype))
    het_zero = convmmd_loss_analytic_selected_observed_hetero(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(s, dtype),
        bandwidths=jnp.asarray(bw, dtype), selection=no_selection)
    base = convmmd_loss_analytic(params, jnp.asarray(x, dtype), jnp.asarray(s, dtype), jnp.asarray(bw, dtype))
    np.testing.assert_allclose(np.asarray(het_zero), np.asarray(base), rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_hetero_denoiser_is_base_mar_posterior(dtype, rtol, atol):
    """CMMD-SEL-HET-DENOISE-001: Omega(x-tilde) cancels -> base MAR posterior at each
    object's own S_i; genuinely heteroscedastic S_i is accepted."""

    w, m, c, s, x, _bw, _a, _p = _instance()
    params = _params(w, m, c, dtype)
    hetero_denoise = convmmd_denoise_selected_observed_hetero(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(s, dtype))
    base = denoise(params, jnp.asarray(x, dtype), jnp.asarray(s, dtype))
    assert hetero_denoise.dtype == dtype
    np.testing.assert_allclose(np.asarray(hetero_denoise), np.asarray(base), rtol=rtol, atol=atol)
    oracle_denoise = oracle.denoise(w, m, c, x, s).posterior_mean
    np.testing.assert_allclose(np.asarray(hetero_denoise), oracle_denoise, rtol=rtol, atol=atol)


def test_hetero_snis_converges_to_analytic_with_ess():
    """CMMD-SEL-HET-MC-001: the per-object SNIS (known S_i, no noise model) converges to
    the per-object analytic value (statistical: spread shrinks, within noise); per-object
    ESS is a positive diagnostic."""

    w, m, c, s, x, bw, a, p = _instance()
    # Use a small subset to keep the per-object (K,K,M,M) self tensor cheap.
    x, s = x[:4], s[:4]
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)
    analytic = float(convmmd_loss_analytic_selected_observed_hetero(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw), selection=selection))

    def error_at(num_samples):
        estimates = [
            float(convmmd_loss_snis_selected_observed_hetero(
                params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
                selection=selection, key=jax.random.PRNGKey(seed), num_samples=num_samples))
            for seed in range(6)
        ]
        return abs(np.mean(estimates) - analytic), float(np.std(estimates) / np.sqrt(6))

    _, coarse_se = error_at(64)
    fine_error, fine_se = error_at(384)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 5e-3

    # Gaussian-spec and callable-spec agree at the same key (spec type is routing only).
    callable_selection = CallableSelection(gaussian_selection_omega(selection))
    key = jax.random.PRNGKey(3)
    v_g = convmmd_loss_snis_selected_observed_hetero(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=selection, key=key, num_samples=128)
    v_c = convmmd_loss_snis_selected_observed_hetero(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=callable_selection, key=key, num_samples=128)
    assert float(v_g) == float(v_c)

    ess, z_hat = snis_diagnostics_observed_hetero(
        params, jnp.asarray(s), jax.random.PRNGKey(0), 128, selection)
    ess = np.asarray(ess)
    z_hat = np.asarray(z_hat)
    assert ess.shape == (x.shape[0],) and np.all(ess > 0.0) and np.all(ess <= N_COMPONENTS * 128 + 1e-6)
    assert np.all(z_hat > 0.0) and np.all(z_hat <= 1.0 + 1e-9)


def test_hetero_requires_no_noise_model():
    """CMMD-SEL-HET-NOMODEL-001: the heteroscedastic loss and denoiser consume only the
    detected objects' own S_i and expose no noise-model / covar_callback parameter (the
    pygmmis-contrast invariant of §17.13)."""

    forbidden = {"noise_model", "covar_callback", "covar_field", "noise_field", "noise_law"}
    for fn in (
        convmmd_loss_analytic_selected_observed_hetero,
        convmmd_loss_snis_selected_observed_hetero,
        convmmd_denoise_selected_observed_hetero,
        select_gaussian_params_observed_hetero,
        fit_selected_observed_hetero_analytic,
    ):
        names = set(inspect.signature(fn).parameters)
        assert not (names & forbidden), f"{fn.__name__} exposes a noise-model parameter"
        assert "noise" in names

    # Functionally: genuinely heteroscedastic S_i alone yields a finite exact result.
    w, m, c, s, x, bw, a, p = _instance()
    assert len({tuple(np.round(si.ravel(), 6)) for si in s}) == N_SAMPLES  # all distinct
    params = _params(w, m, c, jnp.float64)
    loss = convmmd_loss_analytic_selected_observed_hetero(
        params, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=_selection(a, p, jnp.float64))
    assert jnp.isfinite(loss)


def test_hetero_loss_jits_grads_and_vmaps():
    """CMMD-SEL-HET-JIT-001: the per-object analytic leaf is callback-free, jits without
    retrace on same-shape inputs, is differentiable through the per-object transform, and
    the SNIS leaf reads only the passed key."""

    w, m, c, s, x, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)

    def loss(pp):
        return convmmd_loss_analytic_selected_observed_hetero(
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

    # SNIS reads only the passed key: same key -> identical, split key -> differs.
    xs, ss = x[:3], s[:3]
    key = jax.random.PRNGKey(7)
    a1 = float(convmmd_loss_snis_selected_observed_hetero(
        params, jnp.asarray(xs), noise=jnp.asarray(ss), bandwidths=jnp.asarray(bw),
        selection=selection, key=key, num_samples=96))
    a2 = float(convmmd_loss_snis_selected_observed_hetero(
        params, jnp.asarray(xs), noise=jnp.asarray(ss), bandwidths=jnp.asarray(bw),
        selection=selection, key=key, num_samples=96))
    b = float(convmmd_loss_snis_selected_observed_hetero(
        params, jnp.asarray(xs), noise=jnp.asarray(ss), bandwidths=jnp.asarray(bw),
        selection=selection, key=jax.random.split(key)[0], num_samples=96))
    assert a1 == a2 and a1 != b


def test_hetero_analytic_fit_reduces_loss_with_spd_covariances():
    """CMMD-SEL-HET-FIT-001."""

    w, m, c, s, x, bw, a, p = _instance()
    initial = ConvMMDUnconstrained(
        alphas=jnp.asarray(np.log(w + 1e-8)),
        means=jnp.asarray(m),
        unconstrained_L=jnp.asarray(np.stack([0.3 * np.eye(DIMENSION) for _ in range(N_COMPONENTS)])),
    )
    result = fit_selected_observed_hetero_analytic(
        initial, jnp.asarray(x), jnp.asarray(s), jnp.asarray(bw), _selection(a, p, jnp.float64),
        n_steps=150,
    )
    assert not bool(result.numerical_failure)
    assert int(result.status) in (int(ConvMMDFitStatus.CONVERGED), int(ConvMMDFitStatus.MAX_ITER))
    assert float(result.loss) <= float(result.history[0])
    assert bool(jnp.all(jnp.linalg.eigvalsh(result.parameters.covariances) > 0.0))
    recomputed = convmmd_loss_analytic_selected_observed_hetero(
        result.parameters, jnp.asarray(x), noise=jnp.asarray(s), bandwidths=jnp.asarray(bw),
        selection=_selection(a, p, jnp.float64))
    np.testing.assert_allclose(float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12)
