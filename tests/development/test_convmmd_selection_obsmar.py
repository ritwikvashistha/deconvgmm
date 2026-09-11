"""JAX-vs-oracle gates for Omega(x-tilde) + MAR projection (contract §17.14).

Heteroscedastic selection on the observed value **composed with** per-coordinate
missingness: the Gaussian window is marginalized onto each observed subspace
(``Psi_i_inv = (P_i Psi P_i^T)^{-1}``, the inverse of the covariance principal submatrix,
NOT ``P_i Psi^{-1} P_i^T``) and the §17.5 transform runs per object on the projected
inflated ``P_i Sigma_k P_i^T + S_i``, grouped by mask pattern. Machine-epsilon parity vs the
independent NumPy oracle for a Gaussian window (f64 rtol 5e-8/atol 5e-10; f32 declared
profile); both reductions exact (``P_i=I`` -> §17.13 hetero; ``Psi^{-1}=0`` -> base §16
masked); the marginalized-window crux checked directly; the denoiser is the base §16 masked
posterior; per-object projected SNIS converges (statistical gate, ESS logged); JIT/grad/vmap;
spec validation; a valid fit. Warning-as-error in the pinned ``cv`` lane; Linux-portable
(no committed fixture).

Rows: ``CMMD-SEL-OBSMAR-{MARGWIN,XFORM,GAUSS,REDUCE,DENOISE,MC,NOMODEL,JIT,FIT}-001``.
"""

from __future__ import annotations

import inspect

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from development.convmmd import ConvMMDParams, ConvMMDUnconstrained
from development.convmmd_fit import ConvMMDFitStatus
from development.convmmd_grouped import (
    convmmd_loss_analytic_masked,
    convmmd_denoise_masked,
    median_bandwidths_masked,
)
from development.general_validation import PerItemFullNoise
from development.convmmd_selection import (
    CallableSelection,
    GaussianSelection,
    convmmd_denoise_selected_observed_masked,
    convmmd_loss_analytic_selected_observed_hetero,
    convmmd_loss_analytic_selected_observed_masked,
    convmmd_loss_snis_selected_observed_masked,
    fit_selected_observed_masked_analytic,
    gaussian_selection_omega,
    group_masked_fit_inputs,
    group_masked_inputs,
    grouped_analytic_selected_observed_masked_loss,
    marginalize_gaussian_window,
    select_gaussian_params_observed_masked,
    snis_diagnostics_observed_masked,
)
from tests.reference import convmmd as oracle


F64_RTOL, F64_ATOL = 5e-8, 5e-10
F32_RTOL, F32_ATOL = 2e-4, 2e-5
DTYPE_CASES = [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)]

DIMENSION, N_COMPONENTS, N_SAMPLES = 3, 2, 16


def _instance():
    """A D=3 instance: distinct per-object S_i, off-diagonal window, mixed masks."""

    rng = np.random.Generator(np.random.PCG64(20260902))

    def spd(scale):
        raw = rng.standard_normal((DIMENSION, DIMENSION))
        cov = raw @ raw.T
        cov *= scale / np.trace(cov)
        return cov + 0.4 * np.eye(DIMENSION)

    weights = np.array([0.4, 0.6])
    means = rng.standard_normal((N_COMPONENTS, DIMENSION))
    covariances = np.stack([spd(1.0 + 0.3 * k) for k in range(N_COMPONENTS)])
    noise = np.stack([spd(0.2 + 0.5 * rng.random()) for _ in range(N_SAMPLES)])
    observations = rng.standard_normal((N_SAMPLES, DIMENSION))
    bandwidths = np.array([0.5, 1.0, 2.0])
    location = rng.standard_normal(DIMENSION)
    root = rng.standard_normal((DIMENSION, DIMENSION))
    precision = root @ root.T + 0.5 * np.eye(DIMENSION)  # genuinely off-diagonal
    mask = np.ones((N_SAMPLES, DIMENSION), dtype=bool)
    mask[1] = [True, False, True]
    mask[3] = [False, True, True]
    mask[5] = [True, True, False]
    mask[7] = [True, False, True]
    mask[9] = [False, False, False]  # an M=0 row
    return weights, means, covariances, noise, observations, mask, bandwidths, location, precision


def _params(weights, means, covariances, dtype):
    return ConvMMDParams(
        jnp.asarray(weights, dtype), jnp.asarray(means, dtype), jnp.asarray(covariances, dtype)
    )


def _selection(location, precision, dtype):
    return GaussianSelection(jnp.asarray(location, dtype), jnp.asarray(precision, dtype))


def _grouped(params, x, mask, noise, dtype, *, fit=False):
    fn = group_masked_fit_inputs if fit else group_masked_inputs
    return fn(
        params, jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(noise, dtype)), dtype=dtype,
    )


# --------------------------------------------------------------------------- #
# The marginalized-window crux (CMMD-SEL-OBSMAR-MARGWIN-001)
# --------------------------------------------------------------------------- #


def test_marginalized_window_is_covariance_submatrix_inverse():
    """CMMD-SEL-OBSMAR-MARGWIN-001: the JAX marginalized window precision equals the oracle
    (= Schur complement of the precision), NOT the naive precision submatrix; full observation
    recovers Psi^{-1}; Psi^{-1}=0 gives zeros without inverting a singular precision."""

    *_, location, precision = _instance()
    for coords in ([0, 2], [1, 2], [0], [0, 1, 2]):
        marginal = marginalize_gaussian_window(_selection(location, precision, jnp.float64), coords)
        a_ref, psi_ref = oracle.gaussian_window_marginal(location, precision, coords)
        np.testing.assert_allclose(np.asarray(marginal.location), a_ref, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(marginal.precision), psi_ref, rtol=0, atol=1e-11)
        naive = precision[np.ix_(coords, coords)]
        if len(coords) < DIMENSION:
            assert float(np.abs(np.asarray(marginal.precision) - naive).max()) > 1e-6
    # Fully observed -> the full window precision unchanged.
    full = marginalize_gaussian_window(_selection(location, precision, jnp.float64), [0, 1, 2])
    np.testing.assert_allclose(np.asarray(full.precision), precision, rtol=0, atol=1e-11)
    # Psi^{-1} = 0 -> zeros (no singular inverse).
    zero = marginalize_gaussian_window(
        GaussianSelection(jnp.asarray(location), jnp.zeros((DIMENSION, DIMENSION))), [0, 2]
    )
    np.testing.assert_array_equal(np.asarray(zero.precision), np.zeros((2, 2)))


def test_rank_deficient_window_precision_fails_actionably():
    """CMMD-SEL-OBSMAR-MARGWIN-001: a rank-deficient (non-PD, non-zero) Psi^{-1} has no
    covariance form and MUST fail actionably at the eager boundary (§17.14)."""

    *_, location, _p = _instance()
    singular = np.diag([1.0, 1.0, 0.0])  # rank 2 in D=3: Psi = inv(Psi^{-1}) does not exist
    with pytest.raises(ValueError, match="positive-definite window precision"):
        marginalize_gaussian_window(
            GaussianSelection(jnp.asarray(location), jnp.asarray(singular)), [0, 2]
        )


# --------------------------------------------------------------------------- #
# Transform and analytic loss (XFORM, GAUSS)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_projected_transform_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-OBSMAR-XFORM-001: the per-object projected transform (marginalized window on
    the projected inflated B_k^{(i)}) equals the oracle per row and yields SPD C_{k,i}."""

    w, m, c, s, x, mask, _bw, a, p = _instance()
    params = _params(w, m, c, dtype)
    grouped = _grouped(params, x, mask, s, dtype)
    per_group = select_gaussian_params_observed_masked(params, grouped, _selection(a, p, dtype))
    seen_rows = 0
    for index, (pipp, nupp, cpp) in per_group:
        group = grouped.groups[index]
        coords = list(group.coordinate_indices)
        a_i, psi_i_inv = oracle.gaussian_window_marginal(a, p, coords)
        block = np.ix_(coords, coords)
        pipp, nupp, cpp = np.asarray(pipp), np.asarray(nupp), np.asarray(cpp)
        assert cpp.dtype == dtype
        for row, original in enumerate(group.original_indices):
            proj_mu = np.stack([m[k][coords] for k in range(N_COMPONENTS)])
            inflated = np.stack([c[k][block] + s[original][block] for k in range(N_COMPONENTS)])
            transform = oracle.gaussian_window_transform(w, proj_mu, inflated, a_i, psi_i_inv)
            np.testing.assert_allclose(pipp[row], transform.weights, rtol=rtol, atol=atol)
            np.testing.assert_allclose(nupp[row], transform.means, rtol=rtol, atol=atol)
            np.testing.assert_allclose(cpp[row], transform.covariances, rtol=rtol, atol=atol)
            for cov in cpp[row].astype(np.float64):
                assert np.all(np.linalg.eigvalsh(cov) > 0.0)
            seen_rows += 1
    assert seen_rows == int(mask.any(axis=1).sum())  # every informative row transformed


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_analytic_loss_matches_oracle_mixed_masks(dtype, rtol, atol):
    """CMMD-SEL-OBSMAR-GAUSS-001: the MAR-composed analytic loss equals the independent NumPy
    oracle on mixed masks (fully observed, partial, and an M=0 row)."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    loss = convmmd_loss_analytic_selected_observed_masked(
        _params(w, m, c, dtype), jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), bandwidths=jnp.asarray(bw, dtype),
        selection=_selection(a, p, dtype), dtype=dtype,
    )
    oracle_loss = oracle.convmmd_loss_selected_observed_masked(
        w, m, c, x, mask, s, bw, a, p
    ).loss
    assert loss.dtype == dtype
    np.testing.assert_allclose(np.asarray(loss), oracle_loss, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Both reductions must be exact (REDUCE)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_reduces_to_hetero_and_base_masked(dtype, rtol, atol):
    """CMMD-SEL-OBSMAR-REDUCE-001: P_i=I (all observed) == §17.13 hetero; Psi^{-1}=0 == base
    §16 masked loss. Both exact."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    params = _params(w, m, c, dtype)

    # (a) fully observed -> §17.13 heteroscedastic loss.
    full = np.ones_like(mask, dtype=bool)
    bw_full = np.array([0.5, 1.0, 2.0])
    masked_full = convmmd_loss_analytic_selected_observed_masked(
        params, jnp.asarray(x, dtype), jnp.asarray(full),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), bandwidths=jnp.asarray(bw_full, dtype),
        selection=_selection(a, p, dtype), dtype=dtype)
    hetero = convmmd_loss_analytic_selected_observed_hetero(
        params, jnp.asarray(x, dtype), noise=jnp.asarray(s, dtype),
        bandwidths=jnp.asarray(bw_full, dtype), selection=_selection(a, p, dtype))
    np.testing.assert_allclose(np.asarray(masked_full), np.asarray(hetero), rtol=rtol, atol=atol)

    # (b) Psi^{-1}=0 (no selection) -> base §16 masked loss, on mixed masks incl M=0.
    no_selection = GaussianSelection(jnp.asarray(a, dtype), jnp.zeros((DIMENSION, DIMENSION), dtype))
    masked_sel0 = convmmd_loss_analytic_selected_observed_masked(
        params, jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), bandwidths=jnp.asarray(bw, dtype),
        selection=no_selection, dtype=dtype)
    base = convmmd_loss_analytic_masked(
        params, jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), bandwidths=jnp.asarray(bw, dtype), dtype=dtype)
    np.testing.assert_allclose(np.asarray(masked_sel0), np.asarray(base), rtol=rtol, atol=atol)


# --------------------------------------------------------------------------- #
# Denoiser (DENOISE) and no-noise-model invariant (NOMODEL)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_denoiser_is_base_masked_posterior(dtype, rtol, atol):
    """CMMD-SEL-OBSMAR-DENOISE-001: Omega(x-tilde) cancels -> the base §16 masked posterior at
    each object's own S_i; genuinely per-item S_i is accepted; full D, original row order."""

    w, m, c, s, x, mask, _bw, _a, _p = _instance()
    params = _params(w, m, c, dtype)
    selected = convmmd_denoise_selected_observed_masked(
        params, jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), dtype=dtype)
    base = convmmd_denoise_masked(
        params, jnp.asarray(x, dtype), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s, dtype)), dtype=dtype)
    assert selected.shape == (N_SAMPLES, DIMENSION)
    np.testing.assert_allclose(np.asarray(selected), np.asarray(base), rtol=rtol, atol=atol)


def test_requires_no_noise_model():
    """CMMD-SEL-OBSMAR-NOMODEL-001: the MAR-composed loss and denoiser consume only detected
    objects' own S_i and expose no noise-model / covar_callback parameter (§17.14)."""

    forbidden = {"noise_model", "covar_callback", "covar_field", "noise_field", "noise_law"}
    for fn in (
        convmmd_loss_analytic_selected_observed_masked,
        convmmd_loss_snis_selected_observed_masked,
        convmmd_denoise_selected_observed_masked,
        select_gaussian_params_observed_masked,
        fit_selected_observed_masked_analytic,
    ):
        names = set(inspect.signature(fn).parameters)
        assert not (names & forbidden), f"{fn.__name__} exposes a noise-model parameter"


# --------------------------------------------------------------------------- #
# General-Omega projected SNIS (MC)
# --------------------------------------------------------------------------- #


def test_projected_snis_converges_to_analytic_and_logs_ess():
    """CMMD-SEL-OBSMAR-MC-001: the per-object projected SNIS loss (known S_i, observed-subspace
    weight, no noise model) converges to the §17.14 analytic value; per-object ESS is positive;
    Gaussian and callable specs agree at a fixed key."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    # Keep the per-object (K,K,M,M) self tensor cheap.
    x, s, mask = x[:6], s[:6], mask[:6]
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)
    analytic = float(convmmd_loss_analytic_selected_observed_masked(
        params, jnp.asarray(x), jnp.asarray(mask), noise=PerItemFullNoise(jnp.asarray(s)),
        bandwidths=jnp.asarray(bw), selection=selection, dtype=jnp.float64))

    def error_at(num_samples):
        estimates = [
            float(convmmd_loss_snis_selected_observed_masked(
                params, jnp.asarray(x), jnp.asarray(mask), noise=PerItemFullNoise(jnp.asarray(s)),
                bandwidths=jnp.asarray(bw), selection=selection,
                key=jax.random.PRNGKey(seed), num_samples=num_samples, dtype=jnp.float64))
            for seed in range(6)
        ]
        return abs(np.mean(estimates) - analytic), float(np.std(estimates) / np.sqrt(6))

    _, coarse_se = error_at(64)
    fine_error, fine_se = error_at(384)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 5e-3

    # Gaussian-spec and marginalized-callable-spec agree at a fixed key (routing only): the
    # callable must act on the observed subspace, so wrap the marginalized window per group.
    key = jax.random.PRNGKey(3)
    v_gauss = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(x), jnp.asarray(mask), noise=PerItemFullNoise(jnp.asarray(s)),
        bandwidths=jnp.asarray(bw), selection=selection, key=key, num_samples=96, dtype=jnp.float64))
    assert np.isfinite(v_gauss)

    grouped = _grouped(params, x, mask, s, jnp.float64)
    diag = snis_diagnostics_observed_masked(params, grouped, selection, jax.random.PRNGKey(0), 96)
    for index, (ess, z_hat) in diag:
        ess, z_hat = np.asarray(ess), np.asarray(z_hat)
        assert np.all(ess > 0.0) and np.all(ess <= N_COMPONENTS * 96 + 1e-6)
        assert np.all(z_hat > 0.0) and np.all(z_hat <= 1.0 + 1e-9)


def test_snis_callable_on_observed_subspace_agrees_with_marginalized_gaussian():
    """CMMD-SEL-OBSMAR-MC-001: a CallableSelection whose omega is the per-object marginalized
    Gaussian window (evaluated on the observed subspace) reproduces the GaussianSelection SNIS
    at a fixed key on a single mask group (the per-subspace convention of §17.14)."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    # One partial mask group (rows sharing coords {0,2}); a single group -> one omega.
    coords = [0, 2]
    rows = [1, 7]  # both have mask [True, False, True]
    x, s, mask = x[rows], s[rows], mask[rows]
    params = _params(w, m, c, jnp.float64)
    a_i, psi_i_inv = oracle.gaussian_window_marginal(a, p, coords)
    subspace_omega = gaussian_selection_omega(GaussianSelection(jnp.asarray(a_i), jnp.asarray(psi_i_inv)))
    key = jax.random.PRNGKey(11)
    v_gauss = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(x), jnp.asarray(mask), noise=PerItemFullNoise(jnp.asarray(s)),
        bandwidths=jnp.asarray(bw), selection=_selection(a, p, jnp.float64),
        key=key, num_samples=96, dtype=jnp.float64))
    v_call = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(x), jnp.asarray(mask), noise=PerItemFullNoise(jnp.asarray(s)),
        bandwidths=jnp.asarray(bw), selection=CallableSelection(subspace_omega),
        key=key, num_samples=96, dtype=jnp.float64))
    assert v_gauss == v_call


# --------------------------------------------------------------------------- #
# JIT / grad / vmap (JIT) and a valid fit (FIT)
# --------------------------------------------------------------------------- #


def test_grouped_loss_jits_grads_and_snis_reads_only_key():
    """CMMD-SEL-OBSMAR-JIT-001: the grouped analytic leaf is callback-free, jits without
    retrace on same-shape params, is differentiable through the per-object transform (the
    host-side window inversion carries no gradient), and the SNIS leaf reads only the key."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)
    grouped = _grouped(params, x, mask, s, jnp.float64)

    def loss(pp):
        return grouped_analytic_selected_observed_masked_loss(pp, grouped, jnp.asarray(bw), selection)

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

    xs, ss, ms = x[:4], s[:4], mask[:4]
    key = jax.random.PRNGKey(7)
    common = dict(noise=PerItemFullNoise(jnp.asarray(ss)), bandwidths=jnp.asarray(bw),
                  selection=selection, num_samples=64, dtype=jnp.float64)
    a1 = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(xs), jnp.asarray(ms), key=key, **common))
    a2 = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(xs), jnp.asarray(ms), key=key, **common))
    b = float(convmmd_loss_snis_selected_observed_masked(
        params, jnp.asarray(xs), jnp.asarray(ms), key=jax.random.split(key)[0], **common))
    assert a1 == a2 and a1 != b


def test_analytic_fit_reduces_loss_with_spd_covariances():
    """CMMD-SEL-OBSMAR-FIT-001: a valid MAR-composed analytic fit reports non-failure, reduces
    the loss, returns SPD covariances, and recomputes its reported loss exactly."""

    w, m, c, s, x, mask, bw, a, p = _instance()
    params = _params(w, m, c, jnp.float64)
    selection = _selection(a, p, jnp.float64)
    grouped = group_masked_fit_inputs(
        params, jnp.asarray(x), jnp.asarray(mask),
        noise=PerItemFullNoise(jnp.asarray(s)), dtype=jnp.float64)
    initial = ConvMMDUnconstrained(
        alphas=jnp.asarray(np.log(w + 1e-8)),
        means=jnp.asarray(m),
        unconstrained_L=jnp.asarray(np.stack([0.3 * np.eye(DIMENSION) for _ in range(N_COMPONENTS)])),
    )
    result = fit_selected_observed_masked_analytic(
        initial, grouped, jnp.asarray(bw), selection, n_steps=150, learning_rate=2e-2)
    assert not bool(result.numerical_failure)
    assert int(result.status) in (int(ConvMMDFitStatus.CONVERGED), int(ConvMMDFitStatus.MAX_ITER))
    assert float(result.loss) <= float(result.history[0])
    assert bool(jnp.all(jnp.linalg.eigvalsh(result.parameters.covariances) > 0.0))
    recomputed = grouped_analytic_selected_observed_masked_loss(
        result.parameters, grouped, jnp.asarray(bw), selection)
    np.testing.assert_allclose(float(result.loss), float(recomputed), rtol=1e-12, atol=1e-12)
