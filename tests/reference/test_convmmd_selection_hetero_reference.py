"""Independent-oracle self-consistency for heteroscedastic Omega(x-tilde) (contract §17.13).

Checks the NumPy heteroscedastic oracle against (a) the *defining* per-object identity
``Omega(x) [sum_k pi_k N(x; mu_k, Sigma_k+S_i)] == Z_{theta,i} sum_k pi''_{k,i}
N(x; nu''_{k,i}, B''_{k,i})`` for random ``x`` and an arbitrary per-object ``S_i``, (b) the
reduction to the shipped **homoscedastic** oracle when every ``S_i`` is equal, and (c) the
per-object SNIS reference (known ``S_i``, no noise model) converging to the analytic value.
This is the oracle-side gate; the JAX-vs-oracle machine-epsilon parity lives in
``tests/development/test_convmmd_selection_hetero.py``. Live and Linux-portable (no
committed fixture); all gates run warning-as-error in the pinned lane.

Rows (reference side): ``CMMD-SEL-HET-XFORM-001``, ``CMMD-SEL-HET-GAUSS-001``,
``CMMD-SEL-HET-REDUCE-001``, ``CMMD-SEL-HET-MC-001``.
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
    """A deterministic, genuinely heteroscedastic fully-observed instance."""

    rng = np.random.Generator(np.random.PCG64(20260831))
    dimension, components, samples = 2, 2, 4

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
    location = np.array([0.1, -0.2])
    precision = np.array([[0.6, 0.1], [0.1, 0.5]])
    return weights, means, covariances, observations, noise, bandwidths, location, precision


def _gaussian_omega(location, precision):
    def omega(z):
        centered = z - location
        return np.exp(-0.5 * np.einsum("...d,de,...e->...", centered, precision, centered))

    return omega


def test_hetero_transform_defining_identity():
    """CMMD-SEL-HET-XFORM-001 (reference): per object, the §17.5 transform on the inflated
    Sigma_k+S_i reproduces the selected observed density Omega(x) p_tilde(x|S_i)/Z_i."""

    weights, means, covariances, _obs, noise, _bw, a, precision = _instance()
    n_components, dimension = means.shape
    rng = np.random.Generator(np.random.PCG64(7))
    omega = _gaussian_omega(a, precision)
    for i in range(noise.shape[0]):
        inflated = covariances + noise[i][None, :, :]  # B_k^{(i)} = Sigma_k + S_i
        transform = oracle.gaussian_window_transform(weights, means, inflated, a, precision)
        for _ in range(8):
            x = rng.standard_normal(dimension) * 1.5 + a
            p_tilde = sum(
                weights[k] * np.exp(_log_gaussian(x, means[k], inflated[k]))
                for k in range(n_components)
            )
            selected_mixture = sum(
                transform.weights[k]
                * np.exp(_log_gaussian(x, transform.means[k], transform.covariances[k]))
                for k in range(n_components)
            )
            lhs = float(omega(x)) * p_tilde
            rhs = transform.effective_volume * selected_mixture
            assert abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)), (i, lhs, rhs)


def test_hetero_all_equal_reduces_to_homoscedastic_oracle():
    """CMMD-SEL-HET-REDUCE-001 / GAUSS-001 (reference): with every S_i equal, the
    per-object heteroscedastic oracle equals the shipped homoscedastic §17.12 oracle."""

    weights, means, covariances, observations, noise, bandwidths, a, precision = _instance()
    single = noise[0]
    stacked = np.broadcast_to(single, noise.shape).copy()
    hetero = oracle.convmmd_loss_selected_observed_hetero(
        weights, means, covariances, observations, stacked, bandwidths, a, precision).loss
    homo = oracle.convmmd_loss_selected_observed(
        weights, means, covariances, observations, single, bandwidths, a, precision).loss
    np.testing.assert_allclose(hetero, homo, rtol=0, atol=1e-12)


def test_hetero_mc_reference_converges_to_analytic_gaussian():
    """CMMD-SEL-HET-MC-001 (reference): for a Gaussian Omega the per-object SNIS reference
    (known S_i, no noise model) converges to the per-object analytic value; the spread
    shrinks and the fine estimate agrees within noise. Statistical gate only."""

    weights, means, covariances, observations, noise, bandwidths, a, precision = _instance()
    analytic = oracle.convmmd_loss_selected_observed_hetero(
        weights, means, covariances, observations, noise, bandwidths, a, precision).loss
    omega = _gaussian_omega(a, precision)

    def error_at(num_samples):
        rng = np.random.Generator(np.random.PCG64(2026))
        estimates = [
            oracle.monte_carlo_loss_selected_observed_hetero(
                weights, means, covariances, observations, noise, bandwidths,
                omega, rng, num_samples,
            )
            for _ in range(6)
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(96)
    fine_error, fine_se = error_at(768)
    assert fine_se < coarse_se               # spread shrinks with M
    assert fine_error < 4.0 * fine_se + 5e-3  # fine estimate agrees within noise


def test_hetero_mc_reference_nongaussian_finite_and_trends():
    """CMMD-SEL-HET-MC-001 (reference, non-Gaussian): for a smooth logistic Omega the
    per-object SNIS reference is finite and its spread shrinks with M; no closed form."""

    weights, means, covariances, observations, noise, bandwidths, _a, _p = _instance()
    direction = np.array([1.0, -0.5])

    def omega(z):
        return 1.0 / (1.0 + np.exp(-(z @ direction)))  # smooth half-space in (0, 1)

    def spread_at(num_samples):
        rng = np.random.Generator(np.random.PCG64(11))
        values = np.array([
            oracle.monte_carlo_loss_selected_observed_hetero(
                weights, means, covariances, observations, noise, bandwidths,
                omega, rng, num_samples,
            )
            for _ in range(6)
        ])
        assert np.all(np.isfinite(values))
        return float(values.std() / np.sqrt(len(values)))

    coarse = spread_at(96)
    fine = spread_at(768)
    assert fine < coarse  # spread shrinks with M (convergence trend)
