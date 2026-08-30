"""Self-checks for the independent NumPy convMMD oracle and its fixture.

Binds the stored ``convmmd_recovery_001`` fixture (pinned archive and per-payload
SHA-256), verifies the oracle reproduces the stored outputs from the stored
inputs, checks the analytic closed form's invariants and reductions, and binds
the ``Monte-Carlo -> analytic`` convergence property that justifies using the
closed form as the exact oracle for the stochastic estimator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tests.reference.convmmd import (
    convmmd_loss,
    denoise,
    expected_rbf_kernel,
    monte_carlo_loss,
    softmax,
    unconstrained_to_canonical,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_recovery_001.npz"
METADATA = FIXTURE_DIR / "convmmd_recovery_001.metadata.json"
ARCHIVE_SHA256 = "bcc64b0d61da5e3454cec328788d18ab752709af983e661a726cb8c31186ff64"


def _load_fixture() -> dict[str, np.ndarray]:
    with np.load(ARCHIVE) as data:
        return {name: data[name] for name in data.files}


def test_fixture_custody_is_pinned():
    assert ARCHIVE.is_file() and METADATA.is_file()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256, (
        f"convMMD fixture digest changed to {digest}; regenerate the pin and "
        "custody metadata deliberately, never silently"
    )
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["archive_sha256"] == ARCHIVE_SHA256
    assert metadata["contract_id"] == "xdgmm-jax.convmmd"

    # Every stored payload matches its recorded per-array SHA-256.
    from scripts import deterministic_npz  # local import; not a runtime dep

    fixture = _load_fixture()
    for name, expected in metadata["payload_sha256"].items():
        payload = deterministic_npz.npy_bytes(fixture[name])
        assert hashlib.sha256(payload).hexdigest() == expected, name


def test_oracle_reproduces_stored_outputs():
    fixture = _load_fixture()
    loss = convmmd_loss(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["observations"],
        fixture["measurement_covariances"],
        fixture["bandwidths"],
    )
    np.testing.assert_allclose(
        loss.per_scale_loss, fixture["oracle_per_scale_loss"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        loss.loss, float(fixture["oracle_loss"]), rtol=1e-12, atol=1e-12
    )

    denoised = denoise(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["observations"],
        fixture["measurement_covariances"],
    )
    np.testing.assert_allclose(
        denoised.responsibilities,
        fixture["oracle_responsibilities"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        denoised.component_posterior_means,
        fixture["oracle_component_means"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        denoised.posterior_mean,
        fixture["oracle_posterior_mean"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_denoiser_invariants():
    fixture = _load_fixture()
    denoised = denoise(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["observations"],
        fixture["measurement_covariances"],
    )
    row_sums = denoised.responsibilities.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0.0, atol=1e-12)
    assert np.all(denoised.responsibilities >= 0.0)
    reconstructed = np.einsum(
        "nk,nkd->nd", denoised.responsibilities, denoised.component_posterior_means
    )
    np.testing.assert_allclose(
        reconstructed, denoised.posterior_mean, rtol=1e-12, atol=1e-12
    )
    assert np.all(np.isfinite(denoised.posterior_mean))


def test_loss_aggregate_is_mean_over_scales():
    fixture = _load_fixture()
    loss = convmmd_loss(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["observations"],
        fixture["measurement_covariances"],
        fixture["bandwidths"],
    )
    np.testing.assert_allclose(
        loss.loss, loss.per_scale_loss.mean(), rtol=1e-13, atol=1e-13
    )
    assert np.isfinite(loss.loss)


def test_expected_rbf_kernel_reductions():
    gamma = 1.3
    # Zero mean, zero covariance -> kernel of a point with itself.
    assert abs(expected_rbf_kernel([0.0, 0.0], np.zeros((2, 2)), gamma) - 1.0) < 1e-14
    # Zero covariance -> the ordinary RBF kernel exp(-||delta||^2 / (2 gamma^2)).
    delta = np.array([0.4, -0.9, 0.2])
    plain = np.exp(-(delta @ delta) / (2.0 * gamma * gamma))
    assert abs(expected_rbf_kernel(delta, np.zeros((3, 3)), gamma) - plain) < 1e-13
    # Adding positive covariance shrinks the expected kernel below the plain one.
    omega = 0.5 * np.eye(3)
    assert expected_rbf_kernel(delta, omega, gamma) < plain


def test_parameterization_round_trip_matches_fixture():
    fixture = _load_fixture()
    weights, means, covariances = unconstrained_to_canonical(
        fixture["eval_alphas"], fixture["eval_means"], fixture["eval_unconstrained_L"]
    )
    np.testing.assert_allclose(weights, fixture["eval_weights"], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        covariances, fixture["eval_covariances"], rtol=1e-13, atol=1e-13
    )
    assert abs(weights.sum() - 1.0) < 1e-13
    for cov in covariances:
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues > 0.0)


def test_softmax_matches_definition():
    alphas = np.array([0.2, -1.1, 2.4, 0.0])
    reference = np.exp(alphas) / np.exp(alphas).sum()
    np.testing.assert_allclose(softmax(alphas), reference, rtol=1e-14, atol=1e-14)


def _small_instance():
    rng = np.random.Generator(np.random.PCG64(7))
    dimension, components, samples = 2, 2, 3
    means = rng.normal(size=(components, dimension))
    raw = rng.normal(size=(components, dimension, dimension))
    covariances = np.stack([a @ a.T + 0.5 * np.eye(dimension) for a in raw])
    weights = softmax(rng.normal(size=components))
    observations = rng.normal(size=(samples, dimension))
    raw_noise = rng.normal(size=(samples, dimension, dimension))
    noise = np.stack([a @ a.T + 0.3 * np.eye(dimension) for a in raw_noise])
    bandwidths = np.array([0.7, 1.5])
    return weights, means, covariances, observations, noise, bandwidths


def test_monte_carlo_converges_to_analytic():
    weights, means, covariances, observations, noise, bandwidths = _small_instance()
    analytic = convmmd_loss(
        weights, means, covariances, observations, noise, bandwidths
    ).loss

    def error_at(num_samples: int) -> tuple[float, float]:
        rng = np.random.Generator(np.random.PCG64(2026))
        estimates = [
            monte_carlo_loss(
                weights, means, covariances, observations, noise, bandwidths,
                rng, num_samples,
            )
            for _ in range(8)
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(200)
    fine_error, fine_se = error_at(8000)
    # The estimator spread (standard error) shrinks toward zero with M...
    assert fine_se < coarse_se
    # ...and the fine estimate agrees with the analytic oracle within noise.
    assert fine_error < 4.0 * fine_se + 1e-9
