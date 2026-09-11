"""Independent-oracle self-consistency for Omega(x-tilde) + MAR projection (contract §17.14).

Checks the NumPy MAR-composed selection oracle against (a) the **marginalized-window crux**
-- ``Psi_i_inv = (P_i Psi P_i^T)^{-1}`` equals the Schur complement of the window precision
and differs from the naive precision submatrix ``P_i Psi^{-1} P_i^T`` (an independent route
to the same Gaussian marginal); (b) the *defining* per-object identity on the observed
subspace ``Omega_i(x) [sum_k pi_k N(x; P_i mu_k, B_k^{(i)})] == Z_i sum_k pi_{k,i}
N(x; nu_{k,i}, C_{k,i})``; (c) the reduction to the shipped §17.13 heteroscedastic oracle
when fully observed; (d) the reduction to the shipped §16 masked oracle at ``Psi^{-1}=0``;
and (e) the per-object projected SNIS reference (known ``S_i``, observed-subspace weight, no
noise model) converging to the analytic value. This is the oracle-side gate; the
JAX-vs-oracle machine-epsilon parity lives in
``tests/development/test_convmmd_selection_obsmar.py``. Live and Linux-portable (no committed
fixture); all gates run warning-as-error in the pinned lane.

Rows (reference side): ``CMMD-SEL-OBSMAR-MARGWIN-001``, ``CMMD-SEL-OBSMAR-XFORM-001``,
``CMMD-SEL-OBSMAR-GAUSS-001``, ``CMMD-SEL-OBSMAR-REDUCE-001``, ``CMMD-SEL-OBSMAR-MC-001``.
"""

from __future__ import annotations

import numpy as np

from tests.reference import convmmd as oracle


def _log_gaussian(x, mean, cov) -> float:
    dimension = x.shape[0]
    factor = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(factor, x - mean)
    return float(
        -0.5 * (dimension * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(factor)).sum()
                + whitened @ whitened)
    )


def _instance():
    """A deterministic D=3 instance with a genuinely off-diagonal window and mixed masks."""

    rng = np.random.Generator(np.random.PCG64(20260901))
    dimension, components, samples = 3, 2, 6

    def spd(scale):
        raw = rng.standard_normal((dimension, dimension))
        cov = raw @ raw.T
        cov *= scale / np.trace(cov)
        return cov + 0.4 * np.eye(dimension)

    weights = oracle.softmax(rng.normal(size=components))
    means = rng.standard_normal((components, dimension))
    covariances = np.stack([spd(1.0 + 0.3 * k) for k in range(components)])
    noise = np.stack([spd(0.2 + 0.5 * rng.random()) for _ in range(samples)])  # distinct S_i
    observations = rng.standard_normal((samples, dimension))
    bandwidths = np.array([0.7, 1.5])
    location = np.array([0.1, -0.2, 0.3])
    # Genuinely off-diagonal window precision (so the marginal != precision submatrix).
    root = rng.standard_normal((dimension, dimension))
    precision = root @ root.T + 0.5 * np.eye(dimension)
    # Mixed masks: fully observed, two partial patterns, and one M=0 row.
    mask = np.ones((samples, dimension), dtype=bool)
    mask[1] = [True, False, True]
    mask[2] = [False, True, True]
    mask[4] = [True, True, False]
    mask[5] = [False, False, False]
    return weights, means, covariances, observations, mask, noise, bandwidths, location, precision


def _gaussian_omega(location, precision):
    def omega(z):
        centered = z - location
        return np.exp(-0.5 * np.einsum("...d,de,...e->...", centered, precision, centered))

    return omega


def test_obsmar_marginal_window_is_schur_complement_not_precision_submatrix():
    """CMMD-SEL-OBSMAR-MARGWIN-001 (reference): the marginalized observed-subspace precision
    equals the Schur complement of the window precision (an independent route to the Gaussian
    Lebesgue marginal) and differs from the naive precision submatrix ``P_i Psi^{-1} P_i^T``;
    the ``Psi^{-1}=0`` branch gives zeros."""

    *_, location, precision = _instance()
    dimension = location.shape[0]
    for observed in ([0, 2], [1, 2], [0], [0, 1, 2]):
        unobs = [d for d in range(dimension) if d not in observed]
        a_i, psi_i_inv = oracle.gaussian_window_marginal(location, precision, observed)
        # Independent route: marginal precision = Schur complement of the precision blocks
        #   Lambda_CC - Lambda_CU Lambda_UU^{-1} Lambda_UC   (= (Psi[C,C])^{-1}).
        lam_cc = precision[np.ix_(observed, observed)]
        if unobs:
            lam_cu = precision[np.ix_(observed, unobs)]
            lam_uu = precision[np.ix_(unobs, unobs)]
            schur = lam_cc - lam_cu @ np.linalg.inv(lam_uu) @ lam_cu.T
        else:
            schur = lam_cc
        np.testing.assert_allclose(psi_i_inv, schur, rtol=0, atol=1e-11)
        np.testing.assert_allclose(a_i, location[observed], rtol=0, atol=0)
        # And it is NOT the naive precision submatrix whenever a coordinate is dropped.
        naive = precision[np.ix_(observed, observed)]
        if unobs:
            assert float(np.abs(psi_i_inv - naive).max()) > 1e-6
    # Psi^{-1} = 0 (no selection): marginal precision is zero, no singular inverse.
    _, zero_prec = oracle.gaussian_window_marginal(
        location, np.zeros((dimension, dimension)), [0, 2]
    )
    np.testing.assert_array_equal(zero_prec, np.zeros((2, 2)))


def test_obsmar_transform_defining_identity_on_observed_subspace():
    """CMMD-SEL-OBSMAR-XFORM-001 (reference): per detected object the marginalized-window
    transform on the projected inflated ``B_k^{(i)}`` reproduces the selected observed density
    ``Omega_i(x) p_tilde(x | P_i, S_i) / Z_i`` on the observed subspace R^{M_i}."""

    weights, means, covariances, _obs, mask, noise, _bw, a, precision = _instance()
    n_components = means.shape[0]
    rng = np.random.Generator(np.random.PCG64(3))
    for i in range(noise.shape[0]):
        coords = np.flatnonzero(mask[i])
        if coords.size == 0:
            continue
        block = np.ix_(coords, coords)
        a_i, psi_i_inv = oracle.gaussian_window_marginal(a, precision, coords)
        omega_i = _gaussian_omega(a_i, psi_i_inv)
        proj_mu = np.stack([means[k][coords] for k in range(n_components)])
        inflated = np.stack(
            [covariances[k][block] + noise[i][block] for k in range(n_components)]
        )  # B_k^{(i)}
        transform = oracle.gaussian_window_transform(weights, proj_mu, inflated, a_i, psi_i_inv)
        for _ in range(8):
            x = rng.standard_normal(coords.size) * 1.5 + a_i
            p_tilde = sum(
                weights[k] * np.exp(_log_gaussian(x, proj_mu[k], inflated[k]))
                for k in range(n_components)
            )
            selected = sum(
                transform.weights[k]
                * np.exp(_log_gaussian(x, transform.means[k], transform.covariances[k]))
                for k in range(n_components)
            )
            lhs = float(omega_i(x)) * p_tilde
            rhs = transform.effective_volume * selected
            assert abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)), (i, lhs, rhs)


def test_obsmar_full_observation_reduces_to_hetero_oracle():
    """CMMD-SEL-OBSMAR-REDUCE-001 (reference): with every mask fully observed, the MAR-composed
    oracle equals the shipped §17.13 heteroscedastic oracle exactly (P_i = I)."""

    weights, means, covariances, observations, _mask, noise, bandwidths, a, precision = _instance()
    full_mask = np.ones_like(_mask, dtype=bool)
    masked = oracle.convmmd_loss_selected_observed_masked(
        weights, means, covariances, observations, full_mask, noise, bandwidths, a, precision
    ).loss
    hetero = oracle.convmmd_loss_selected_observed_hetero(
        weights, means, covariances, observations, noise, bandwidths, a, precision
    ).loss
    np.testing.assert_allclose(masked, hetero, rtol=0, atol=1e-12)


def test_obsmar_no_selection_reduces_to_masked_oracle():
    """CMMD-SEL-OBSMAR-REDUCE-001 (reference): at ``Psi^{-1}=0`` the MAR-composed oracle equals
    the shipped §16 masked (MAR) oracle exactly, on mixed masks including an M=0 row."""

    weights, means, covariances, observations, mask, noise, bandwidths, a, _p = _instance()
    dimension = means.shape[1]
    zero = np.zeros((dimension, dimension))
    # A zero window with a finite location: still no selection (Psi_i_inv = 0 everywhere).
    masked_sel = oracle.convmmd_loss_selected_observed_masked(
        weights, means, covariances, observations, mask, noise, bandwidths, a, zero
    ).loss
    base_masked = oracle.convmmd_loss_masked(
        weights, means, covariances, observations, mask, noise, bandwidths
    ).loss
    np.testing.assert_allclose(masked_sel, base_masked, rtol=0, atol=1e-12)


def test_obsmar_mc_reference_converges_to_analytic_gaussian():
    """CMMD-SEL-OBSMAR-MC-001 (reference): the per-object projected SNIS reference (known S_i,
    marginalized-window weight on the observed subspace, no noise model) converges to the
    §17.14 analytic value; spread shrinks and the fine estimate agrees within noise."""

    weights, means, covariances, observations, mask, noise, bandwidths, a, precision = _instance()
    analytic = oracle.convmmd_loss_selected_observed_masked(
        weights, means, covariances, observations, mask, noise, bandwidths, a, precision
    ).loss

    def error_at(num_samples):
        rng = np.random.Generator(np.random.PCG64(2026))
        estimates = [
            oracle.monte_carlo_loss_selected_observed_masked(
                weights, means, covariances, observations, mask, noise, bandwidths,
                a, precision, rng, num_samples,
            )
            for _ in range(6)
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(96)
    fine_error, fine_se = error_at(768)
    assert fine_se < coarse_se                # spread shrinks with M
    assert fine_error < 4.0 * fine_se + 5e-3  # fine estimate agrees within noise
