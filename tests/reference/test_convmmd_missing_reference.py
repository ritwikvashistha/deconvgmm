"""Self-checks for the masked (MAR) convMMD oracle and its fixture.

Binds the stored ``convmmd_missing_001`` fixture (pinned archive and per-payload
SHA-256), verifies the masked oracle reproduces the stored outputs, and checks the
contract-§16 invariants purely inside the independent NumPy oracle (before any JAX
implementation exists): fully-observed reduction to the base oracle, the ``M=0``
inertness / prior-mean semantics, the masked bandwidth reduction and failure mode,
and the masked ``Monte-Carlo -> analytic`` convergence property.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tests.reference.convmmd import (
    convmmd_loss,
    convmmd_loss_masked,
    denoise,
    denoise_masked,
    median_bandwidths_masked,
    monte_carlo_loss_masked,
    softmax,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_missing_001.npz"
METADATA = FIXTURE_DIR / "convmmd_missing_001.metadata.json"
ARCHIVE_SHA256 = "d70ecc1ca0a1901fccca2bcaae4420516926aa81f1575a5a42f5b455aceb427f"


def _load_fixture() -> dict[str, np.ndarray]:
    with np.load(ARCHIVE) as data:
        return {name: data[name] for name in data.files}


def _params(fixture):
    return (
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
    )


def test_fixture_custody_is_pinned():
    assert ARCHIVE.is_file() and METADATA.is_file()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256, (
        f"masked convMMD fixture digest changed to {digest}; regenerate the pin "
        "and custody metadata deliberately, never silently"
    )
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["archive_sha256"] == ARCHIVE_SHA256
    assert metadata["contract_id"] == "xdgmm-jax.convmmd"
    assert metadata["contract_version"] == "0.2.0-draft.1"

    from scripts import deterministic_npz  # local import; not a runtime dep

    fixture = _load_fixture()
    for name, expected in metadata["payload_sha256"].items():
        payload = deterministic_npz.npy_bytes(fixture[name])
        assert hashlib.sha256(payload).hexdigest() == expected, name


def test_masked_oracle_reproduces_stored_outputs():
    fixture = _load_fixture()
    weights, means, covariances = _params(fixture)
    loss = convmmd_loss_masked(
        weights,
        means,
        covariances,
        fixture["observations"],
        fixture["observed_mask"],
        fixture["measurement_covariances"],
        fixture["bandwidths"],
    )
    np.testing.assert_allclose(
        loss.per_scale_loss, fixture["oracle_per_scale_loss"], rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        loss.loss, float(fixture["oracle_loss"]), rtol=1e-12, atol=1e-12
    )

    denoised = denoise_masked(
        weights,
        means,
        covariances,
        fixture["observations"],
        fixture["observed_mask"],
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


def test_fully_observed_masked_reduces_to_base_oracle():
    fixture = _load_fixture()
    weights, means, covariances = _params(fixture)
    observations = fixture["observations"]
    noise = fixture["measurement_covariances"]
    bandwidths = fixture["bandwidths"]
    full_mask = np.ones(observations.shape, dtype=bool)

    base = convmmd_loss(weights, means, covariances, observations, noise, bandwidths)
    masked = convmmd_loss_masked(
        weights, means, covariances, observations, full_mask, noise, bandwidths
    )
    np.testing.assert_allclose(
        masked.per_scale_loss, base.per_scale_loss, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(masked.loss, base.loss, rtol=1e-12, atol=1e-12)

    base_denoise = denoise(weights, means, covariances, observations, noise)
    masked_denoise = denoise_masked(
        weights, means, covariances, observations, full_mask, noise
    )
    np.testing.assert_allclose(
        masked_denoise.posterior_mean,
        base_denoise.posterior_mean,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        masked_denoise.responsibilities,
        base_denoise.responsibilities,
        rtol=1e-12,
        atol=1e-12,
    )


def test_m0_rows_are_inert_in_loss_and_return_prior_mean():
    fixture = _load_fixture()
    weights, means, covariances = _params(fixture)
    observations = fixture["observations"]
    mask = fixture["observed_mask"]
    noise = fixture["measurement_covariances"]
    bandwidths = fixture["bandwidths"]

    informative = mask.any(axis=1)
    assert (~informative).any(), "fixture must contain at least one M=0 row"

    full = convmmd_loss_masked(
        weights, means, covariances, observations, mask, noise, bandwidths
    )
    dropped = convmmd_loss_masked(
        weights,
        means,
        covariances,
        observations[informative],
        mask[informative],
        noise[informative],
        bandwidths,
    )
    # Dropping every M=0 row changes no loss value (informative-row normalization).
    np.testing.assert_allclose(
        dropped.per_scale_loss, full.per_scale_loss, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(dropped.loss, full.loss, rtol=1e-13, atol=1e-13)

    # Appending an extra M=0 row (arbitrary finite payload) also changes nothing.
    extra_obs = np.vstack([observations, observations[:1] + 7.0])
    extra_mask = np.vstack([mask, np.zeros((1, mask.shape[1]), dtype=bool)])
    extra_noise = np.concatenate([noise, noise[:1]], axis=0)
    added = convmmd_loss_masked(
        weights, means, covariances, extra_obs, extra_mask, extra_noise, bandwidths
    )
    np.testing.assert_allclose(added.loss, full.loss, rtol=1e-13, atol=1e-13)

    # The denoiser returns the prior mean and prior weights for every M=0 row.
    denoised = denoise_masked(
        weights, means, covariances, observations, mask, noise
    )
    prior_mean = np.einsum("k,kd->d", weights, means)
    for row in np.flatnonzero(~informative):
        np.testing.assert_allclose(
            denoised.posterior_mean[row], prior_mean, rtol=1e-13, atol=1e-13
        )
        np.testing.assert_allclose(
            denoised.responsibilities[row], weights, rtol=1e-13, atol=1e-13
        )


def test_all_m0_collection_has_zero_loss():
    fixture = _load_fixture()
    weights, means, covariances = _params(fixture)
    observations = fixture["observations"][:5]
    noise = fixture["measurement_covariances"][:5]
    mask = np.zeros(observations.shape, dtype=bool)
    loss = convmmd_loss_masked(
        weights, means, covariances, observations, mask, noise, fixture["bandwidths"]
    )
    assert loss.loss == 0.0
    np.testing.assert_array_equal(loss.per_scale_loss, 0.0)


def test_masked_denoiser_invariants():
    fixture = _load_fixture()
    weights, means, covariances = _params(fixture)
    denoised = denoise_masked(
        weights,
        means,
        covariances,
        fixture["observations"],
        fixture["observed_mask"],
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
    assert denoised.posterior_mean.shape == fixture["observations"].shape


def test_masked_bandwidths_reduce_to_full_pairwise_median():
    fixture = _load_fixture()
    observations = fixture["observations"]
    full_mask = np.ones(observations.shape, dtype=bool)
    gammas = median_bandwidths_masked(observations, full_mask)

    # Independent full pairwise-distance median (the §5 heuristic).
    differences = observations[:, None, :] - observations[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1))
    upper = distances[np.triu_indices(observations.shape[0], k=1)]
    expected = float(np.median(upper)) * np.logspace(-2.0, 2.0, 9)
    np.testing.assert_allclose(gammas, expected, rtol=1e-13, atol=1e-13)


def test_masked_bandwidths_use_shared_coordinates_and_match_fixture():
    fixture = _load_fixture()
    gammas = median_bandwidths_masked(
        fixture["observations"], fixture["observed_mask"]
    )
    np.testing.assert_allclose(gammas, fixture["bandwidths"], rtol=1e-13, atol=1e-13)


def test_masked_bandwidths_raise_without_shared_coordinate():
    observations = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    disjoint = np.array([[True, False], [False, True]], dtype=bool)
    with pytest.raises(ValueError):
        median_bandwidths_masked(observations, disjoint)


def _small_masked_instance():
    rng = np.random.Generator(np.random.PCG64(11))
    dimension, components = 3, 2
    means = rng.normal(size=(components, dimension))
    raw = rng.normal(size=(components, dimension, dimension))
    covariances = np.stack([a @ a.T + 0.5 * np.eye(dimension) for a in raw])
    weights = softmax(rng.normal(size=components))
    observations = rng.normal(size=(5, dimension))
    raw_noise = rng.normal(size=(5, dimension, dimension))
    noise = np.stack([a @ a.T + 0.3 * np.eye(dimension) for a in raw_noise])
    mask = np.array(
        [
            [True, True, True],
            [True, False, True],
            [False, True, True],
            [True, True, False],
            [False, False, False],  # an M=0 row both paths skip
        ],
        dtype=bool,
    )
    bandwidths = np.array([0.7, 1.5])
    return weights, means, covariances, observations, mask, noise, bandwidths


def test_masked_monte_carlo_converges_to_masked_analytic():
    weights, means, covariances, observations, mask, noise, bandwidths = (
        _small_masked_instance()
    )
    analytic = convmmd_loss_masked(
        weights, means, covariances, observations, mask, noise, bandwidths
    ).loss

    def error_at(num_samples: int) -> tuple[float, float]:
        rng = np.random.Generator(np.random.PCG64(2026))
        estimates = [
            monte_carlo_loss_masked(
                weights, means, covariances, observations, mask, noise,
                bandwidths, rng, num_samples,
            )
            for _ in range(8)
        ]
        mean = float(np.mean(estimates))
        standard_error = float(np.std(estimates) / np.sqrt(len(estimates)))
        return abs(mean - analytic), standard_error

    _, coarse_se = error_at(200)
    fine_error, fine_se = error_at(8000)
    assert fine_se < coarse_se
    assert fine_error < 4.0 * fine_se + 1e-9
