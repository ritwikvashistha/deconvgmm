"""Independent-oracle self-consistency for known-selection convMMD (contract §17).

Pins the stored ``convmmd_selection_001`` fixture (archive + per-payload SHA-256)
and checks the NumPy selection oracle against the *defining* §17.5 identity — for
random latent points ``Omega(z) q_theta(z) == Z_theta * sum_k pi'_k N(z; mu'_k,
Sigma'_k)`` — plus the transform invariants (SPD, simplex, covariance shrinkage,
effective volume in ``(0, 1]``) and the near-no-selection reduction. This is the
oracle-side gate; the JAX-vs-oracle machine-epsilon parity lives in
``tests/development/test_convmmd_selection_parity.py``. All gates run
warning-as-error in the pinned lane.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference import convmmd as oracle


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_selection_001.npz"
METADATA = FIXTURE_DIR / "convmmd_selection_001.metadata.json"
ARCHIVE_SHA256 = "0a5205c15e720f9b86c31bb8f2b34d561a777d2310f81c81f25c94c808b0a27c"


def _load_det_npz():
    spec = importlib.util.spec_from_file_location(
        "_sel_det_npz",
        Path(__file__).resolve().parents[2] / "scripts" / "deterministic_npz.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> dict[str, np.ndarray]:
    with np.load(ARCHIVE) as data:
        return {name: data[name] for name in data.files}


def _log_gaussian(x, mean, cov) -> float:
    dimension = x.shape[0]
    factor = np.linalg.cholesky(cov)
    whitened = np.linalg.solve(factor, x - mean)
    return float(
        -0.5 * (dimension * np.log(2.0 * np.pi) + 2.0 * np.log(np.diag(factor)).sum()
                + whitened @ whitened)
    )


def test_fixture_digest_is_pinned():
    assert ARCHIVE.is_file()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256, (
        f"selection fixture digest changed to {digest}; regenerate the pin "
        "deliberately and update ARCHIVE_SHA256 in the same change"
    )


def test_metadata_matches_payloads_and_version():
    metadata = json.loads(METADATA.read_text())
    assert metadata["archive_sha256"] == ARCHIVE_SHA256
    assert metadata["contract_id"] == "xdgmm-jax.convmmd"
    assert metadata["contract_version"] == "0.3.0-draft.1"
    assert metadata["selection_convention"] == "on_z"
    fixture = _fixture()
    deterministic_npz = _load_det_npz()
    for name, expected in metadata["payload_sha256"].items():
        payload = deterministic_npz.npy_bytes(fixture[name])
        assert hashlib.sha256(payload).hexdigest() == expected, name


def test_transform_matches_defining_identity():
    """Omega(z) q(z) == Z_theta * sum_k pi'_k N(z; mu'_k, Sigma'_k) for random z."""

    fixture = _fixture()
    weights = fixture["eval_weights"]
    means = fixture["eval_means"]
    covariances = fixture["eval_covariances"]
    a = fixture["selection_location"]
    precision = fixture["selection_precision"]

    transform = oracle.gaussian_window_transform(
        weights, means, covariances, a, precision
    )
    # The stored transform equals the freshly computed one.
    np.testing.assert_array_equal(transform.weights, fixture["oracle_selected_weights"])
    np.testing.assert_array_equal(transform.means, fixture["oracle_selected_means"])
    np.testing.assert_array_equal(
        transform.covariances, fixture["oracle_selected_covariances"]
    )

    rng = np.random.Generator(np.random.PCG64(2026))
    n_components, dimension = means.shape
    for _ in range(16):
        z = rng.standard_normal(dimension) * 1.5 + a
        omega = float(np.exp(-0.5 * (z - a) @ precision @ (z - a)))
        q = sum(
            weights[k] * np.exp(_log_gaussian(z, means[k], covariances[k]))
            for k in range(n_components)
        )
        selected_mixture = sum(
            transform.weights[k]
            * np.exp(_log_gaussian(z, transform.means[k], transform.covariances[k]))
            for k in range(n_components)
        )
        lhs = omega * q
        rhs = transform.effective_volume * selected_mixture
        assert abs(lhs - rhs) <= 1e-11 * max(1.0, abs(lhs)), (lhs, rhs)


def test_transform_invariants():
    fixture = _fixture()
    transform = oracle.gaussian_window_transform(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["selection_location"],
        fixture["selection_precision"],
    )
    # pi' on the simplex.
    assert transform.weights.shape == fixture["eval_weights"].shape
    np.testing.assert_allclose(transform.weights.sum(), 1.0, rtol=0, atol=1e-13)
    assert np.all(transform.weights > 0.0)
    # Sigma'_k SPD and shrunk relative to Sigma_k (selection adds precision).
    for k in range(transform.covariances.shape[0]):
        eig = np.linalg.eigvalsh(transform.covariances[k])
        assert np.all(eig > 0.0)
        shrink = np.linalg.eigvalsh(
            fixture["eval_covariances"][k] - transform.covariances[k]
        )
        assert np.all(shrink >= -1e-12)
    # Effective volume in (0, 1] for Omega <= 1; stored value matches.
    assert 0.0 < transform.effective_volume <= 1.0 + 1e-12
    np.testing.assert_allclose(
        transform.effective_volume, float(fixture["oracle_effective_volume"]),
        rtol=0, atol=1e-13,
    )
    assert float(fixture["oracle_effective_volume"]) < 1.0  # genuinely selective


def test_near_no_selection_reduces_to_masked():
    """A vanishing precision (Omega -> 1) reduces the selected oracle to the masked
    oracle and drives Z_theta -> 1."""

    fixture = _fixture()
    weights = fixture["eval_weights"]
    means = fixture["eval_means"]
    covariances = fixture["eval_covariances"]
    tiny = 1e-9 * np.eye(means.shape[1])

    transform = oracle.gaussian_window_transform(
        weights, means, covariances, fixture["selection_location"], tiny
    )
    np.testing.assert_allclose(transform.weights, weights, rtol=0, atol=1e-7)
    np.testing.assert_allclose(transform.means, means, rtol=0, atol=1e-6)
    np.testing.assert_allclose(transform.covariances, covariances, rtol=0, atol=1e-6)
    np.testing.assert_allclose(transform.effective_volume, 1.0, rtol=0, atol=1e-7)

    selected = oracle.convmmd_loss_selected(
        weights, means, covariances,
        fixture["observations"], fixture["observed_mask"],
        fixture["measurement_covariances"], fixture["bandwidths"],
        fixture["selection_location"], tiny,
    ).loss
    masked = oracle.convmmd_loss_masked(
        weights, means, covariances,
        fixture["observations"], fixture["observed_mask"],
        fixture["measurement_covariances"], fixture["bandwidths"],
    ).loss
    np.testing.assert_allclose(selected, masked, rtol=0, atol=1e-7)


def test_selected_denoise_rows_sum_to_one_and_shapes():
    fixture = _fixture()
    denoised = oracle.denoise_selected(
        fixture["eval_weights"], fixture["eval_means"], fixture["eval_covariances"],
        fixture["observations"], fixture["observed_mask"],
        fixture["measurement_covariances"],
        fixture["selection_location"], fixture["selection_precision"],
    )
    assert denoised.posterior_mean.shape == fixture["observations"].shape
    np.testing.assert_allclose(
        denoised.responsibilities.sum(axis=1), 1.0, rtol=0, atol=1e-13
    )
    np.testing.assert_array_equal(
        denoised.responsibilities, fixture["oracle_selected_responsibilities"]
    )
    np.testing.assert_array_equal(
        denoised.posterior_mean, fixture["oracle_selected_posterior_mean"]
    )


# ---------------------------------------------------------------------------
# SNIS Monte-Carlo reference (§17.4): general Omega, statistical gate only
# ---------------------------------------------------------------------------


def _small_selection_instance():
    """A tiny fully-observed homoscedastic instance for the SNIS reference gate."""

    rng = np.random.Generator(np.random.PCG64(31))
    dimension, components, samples = 2, 2, 4
    means = rng.normal(size=(components, dimension))
    raw = rng.normal(size=(components, dimension, dimension))
    covariances = np.stack([m @ m.T + 0.5 * np.eye(dimension) for m in raw])
    weights = oracle.softmax(rng.normal(size=components))
    observations = rng.normal(size=(samples, dimension))
    noise_block = np.array([[0.12, 0.03], [0.03, 0.10]])
    noise = np.broadcast_to(noise_block, (samples, dimension, dimension)).copy()
    observed_mask = np.ones((samples, dimension), dtype=bool)
    bandwidths = np.array([0.7, 1.5])
    location = np.array([0.1, -0.2])
    precision = np.array([[0.6, 0.1], [0.1, 0.5]])
    return (
        weights, means, covariances, observations, observed_mask, noise,
        bandwidths, location, precision,
    )


def _gaussian_omega(location, precision):
    def omega(z):
        centered = z - location
        return np.exp(-0.5 * np.einsum("...d,de,...e->...", centered, precision, centered))

    return omega


def test_snis_reference_converges_to_analytic_gaussian():
    """CMMD-SEL-MC-001 (reference): for a Gaussian Omega the SNIS reference converges
    to the §17.5 analytic value; the spread shrinks and the fine estimate agrees
    within noise. Statistical gate only (SNIS is biased at finite M)."""

    (weights, means, covariances, observations, mask, noise, bandwidths,
     location, precision) = _small_selection_instance()
    analytic = oracle.convmmd_loss_selected(
        weights, means, covariances, observations, mask, noise, bandwidths,
        location, precision,
    ).loss
    omega = _gaussian_omega(location, precision)

    def error_at(num_samples):
        rng = np.random.Generator(np.random.PCG64(2026))
        estimates = [
            oracle.monte_carlo_loss_selected(
                weights, means, covariances, observations, mask, noise, bandwidths,
                omega, rng, num_samples,
            ).loss
            for _ in range(6)
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(96)
    fine_error, fine_se = error_at(768)
    assert fine_se < coarse_se               # spread shrinks with M
    assert fine_error < 4.0 * fine_se + 5e-3  # fine estimate agrees within noise


def test_snis_reference_diagnostics_gaussian():
    """CMMD-SEL-ESS-001 / CMMD-SEL-Z-001 (reference): ESS in (0, K*M], Zhat matches
    the closed-form Z_theta, and ESS collapses as Omega becomes more selective."""

    (weights, means, covariances, observations, mask, noise, bandwidths,
     location, precision) = _small_selection_instance()
    n_components = weights.shape[0]
    num_samples = 512
    z_theta = oracle.gaussian_window_transform(
        weights, means, covariances, location, precision
    ).effective_volume

    rng = np.random.Generator(np.random.PCG64(5))
    result = oracle.monte_carlo_loss_selected(
        weights, means, covariances, observations, mask, noise, bandwidths,
        _gaussian_omega(location, precision), rng, num_samples,
    )
    assert 0.0 < result.ess <= n_components * num_samples
    assert abs(result.effective_volume - z_theta) < 0.05

    rng2 = np.random.Generator(np.random.PCG64(5))
    selective = oracle.monte_carlo_loss_selected(
        weights, means, covariances, observations, mask, noise, bandwidths,
        _gaussian_omega(location, 25.0 * precision), rng2, num_samples,
    )
    assert selective.ess < result.ess


def test_snis_reference_nongaussian_finite_and_trends():
    """CMMD-SEL-MC-002 (reference): for a non-Gaussian (smooth logistic) Omega the
    SNIS reference is finite and its spread shrinks with M; no closed form, no
    machine-eps claim."""

    (weights, means, covariances, observations, mask, noise, bandwidths,
     location, precision) = _small_selection_instance()
    direction = np.array([1.0, -0.5])

    def omega(z):
        return 1.0 / (1.0 + np.exp(-(z @ direction)))  # smooth half-space in (0, 1)

    def spread_at(num_samples):
        rng = np.random.Generator(np.random.PCG64(11))
        estimates = [
            oracle.monte_carlo_loss_selected(
                weights, means, covariances, observations, mask, noise, bandwidths,
                omega, rng, num_samples,
            )
            for _ in range(6)
        ]
        values = np.array([e.loss for e in estimates])
        assert np.all(np.isfinite(values))
        assert np.all([0.0 < e.ess <= weights.shape[0] * num_samples for e in estimates])
        return float(values.std() / np.sqrt(len(values)))

    coarse = spread_at(96)
    fine = spread_at(768)
    assert fine < coarse  # spread shrinks with M (convergence trend)


def test_snis_denoiser_reference_converges_to_analytic_gaussian():
    """CMMD-SEL-DENOISE-003 (reference): for a Gaussian Omega the SNIS denoiser (sample
    the MAR posterior, weight by Omega) converges to the analytic selected denoiser;
    the spread shrinks and the fine estimate agrees within noise."""

    (weights, means, covariances, observations, mask, noise, _bandwidths,
     location, precision) = _small_selection_instance()
    analytic = oracle.denoise_selected(
        weights, means, covariances, observations, mask, noise, location, precision,
    ).posterior_mean
    omega = _gaussian_omega(location, precision)

    def error_at(num_samples):
        estimates = [
            oracle.snis_denoise_selected(
                weights, means, covariances, observations, mask, noise, omega,
                np.random.Generator(np.random.PCG64(seed)), num_samples,
            )
            for seed in range(6)
        ]
        stacked = np.stack(estimates)
        mean = stacked.mean(axis=0)
        row_error = float(np.max(np.linalg.norm(mean - analytic, axis=1)))
        row_se = float(np.max(np.linalg.norm(stacked.std(axis=0), axis=1) / np.sqrt(6)))
        return row_error, row_se

    _, coarse_se = error_at(256)
    fine_error, fine_se = error_at(4096)
    assert fine_se < coarse_se               # spread shrinks with M
    assert fine_error < 4.0 * fine_se + 5e-3  # fine estimate agrees within noise


def test_snis_denoiser_reference_m0_row_is_selected_prior_mean():
    """An M=0 row denoises to the selected prior mean (transformed-prior mean)."""

    (weights, means, covariances, _obs, _mask, noise, _bandwidths,
     location, precision) = _small_selection_instance()
    samples = 4
    observations = np.zeros((samples, means.shape[1]))
    mask = np.zeros((samples, means.shape[1]), dtype=bool)  # every row M=0
    noise = np.broadcast_to(noise[0], (samples,) + noise[0].shape).copy()

    transform = oracle.gaussian_window_transform(
        weights, means, covariances, location, precision
    )
    selected_prior_mean = transform.weights @ transform.means  # sum_k pi'_k mu'_k

    estimates = np.stack([
        oracle.snis_denoise_selected(
            weights, means, covariances, observations, mask, noise,
            _gaussian_omega(location, precision),
            np.random.Generator(np.random.PCG64(seed)), 8000,
        )
        for seed in range(6)
    ])
    mean_row0 = estimates.mean(axis=0)[0]
    assert np.linalg.norm(mean_row0 - selected_prior_mean) < 3e-2
