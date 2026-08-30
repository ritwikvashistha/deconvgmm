"""Generate the stored ``convmmd_missing_001`` fixture deterministically.

This script is not run by the test suite. It records one Gaussian-GMM
deconvolution case with **per-coordinate missing-at-random (MAR)** observations
and known latent ground truth: full-covariance heteroscedastic measurement noise,
a deterministic boolean ``observed_mask`` with mixed patterns (including one
fully-observed group and one ``M=0`` group), the predeclared masked bandwidth set
(contract §16.6), a candidate evaluation parameter point (unconstrained and
canonical), and the independent NumPy-oracle masked outputs at that point. The
archive is written with the deterministic ZIP_STORED writer so identical arrays
produce identical bytes, and every payload plus the whole archive carries a pinned
SHA-256.

Run:  ~/anaconda3/envs/cv/bin/python scripts/generate_convmmd_missing_fixture.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SEED = 20260831
FIXTURE_ID = "convmmd-missing-001"
CONTRACT_ID = "xdgmm-jax.convmmd"
CONTRACT_VERSION = "0.2.0-draft.1"
FIXTURE_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "tests" / "fixtures" / "convmmd_missing_001.npz"
DEFAULT_METADATA = (
    REPO_ROOT / "tests" / "fixtures" / "convmmd_missing_001.metadata.json"
)

N_SAMPLES = 48
N_COMPONENTS = 3
DIMENSION = 4
N_SCALES = 9
BANDWIDTH_LOG10_LO = -2.0
BANDWIDTH_LOG10_HI = 2.0

# Deterministic mask patterns over the D=4 coordinates, assigned round-robin so
# rows of one pattern are non-contiguous (exercises grouping and row
# restoration). Includes a fully-observed group (1111) and an M=0 group (0000).
MASK_PATTERNS = (
    (True, True, True, True),    # M=4 (fully observed)
    (True, True, False, True),   # M=3
    (True, False, True, False),  # M=2
    (False, True, True, False),  # M=2
    (True, False, False, False), # M=1
    (False, False, False, False),# M=0
)


def _load(module_name: str, relative: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


oracle = _load("_convmmd_oracle", "tests/reference/convmmd.py")
det_npz = _load("_convmmd_det_npz", "scripts/deterministic_npz.py")


def _random_spd(
    rng: np.random.Generator, dimension: int, scale: float, floor: float
) -> np.ndarray:
    raw = rng.standard_normal((dimension, dimension))
    covariance = raw @ raw.T
    covariance *= scale / np.trace(covariance)
    covariance += np.diag(rng.uniform(floor, 2.0 * floor, size=dimension))
    return 0.5 * (covariance + covariance.T)


def build_fixture() -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(SEED))

    # --- known latent ground truth (full-covariance GMM) ---
    true_weights = np.asarray([0.30, 0.45, 0.25], dtype=np.float64)
    true_means = np.asarray(
        [
            [-1.60, 0.35, -0.40, 0.50],
            [0.25, -0.95, 0.70, -0.30],
            [1.70, 1.05, 1.20, 0.10],
        ],
        dtype=np.float64,
    )
    true_covariances = np.stack(
        [
            _random_spd(rng, DIMENSION, scale=1.1, floor=0.15)
            for _ in range(N_COMPONENTS)
        ]
    )

    labels = rng.choice(N_COMPONENTS, size=N_SAMPLES, p=true_weights)
    latent_true = np.empty((N_SAMPLES, DIMENSION), dtype=np.float64)
    for sample, component in enumerate(labels):
        latent_true[sample] = rng.multivariate_normal(
            true_means[component], true_covariances[component]
        )

    # --- full-covariance heteroscedastic measurement noise (full D) ---
    measurement_covariances = np.empty(
        (N_SAMPLES, DIMENSION, DIMENSION), dtype=np.float64
    )
    observations = np.empty_like(latent_true)
    for sample in range(N_SAMPLES):
        noise_cov = _random_spd(
            rng, DIMENSION, scale=rng.uniform(0.08, 0.30), floor=0.01
        )
        measurement_covariances[sample] = noise_cov
        observations[sample] = latent_true[sample] + rng.multivariate_normal(
            np.zeros(DIMENSION), noise_cov
        )
    # Observations are stored at full width and are finite at every entry,
    # including masked positions; missingness is expressed only through the mask.

    # --- deterministic per-coordinate MAR mask (round-robin patterns) ---
    observed_mask = np.asarray(
        [MASK_PATTERNS[sample % len(MASK_PATTERNS)] for sample in range(N_SAMPLES)],
        dtype=bool,
    )
    observed_counts = observed_mask.sum(axis=1).astype(np.int64)

    # --- predeclared masked bandwidth set (§16.6) ---
    bandwidths = oracle.median_bandwidths_masked(
        observations,
        observed_mask,
        n_scales=N_SCALES,
        log10_low=BANDWIDTH_LOG10_LO,
        log10_high=BANDWIDTH_LOG10_HI,
    ).astype(np.float64)
    masked_median_distance = float(bandwidths[N_SCALES // 2])

    # --- candidate evaluation parameters (unconstrained, then canonical) ---
    eval_alphas = rng.normal(scale=0.5, size=N_COMPONENTS).astype(np.float64)
    eval_means = (
        true_means + rng.normal(scale=0.25, size=true_means.shape)
    ).astype(np.float64)
    eval_unconstrained_L = rng.normal(
        scale=0.6, size=(N_COMPONENTS, DIMENSION, DIMENSION)
    ).astype(np.float64)
    eval_weights, eval_means_canonical, eval_covariances = (
        oracle.unconstrained_to_canonical(
            eval_alphas, eval_means, eval_unconstrained_L
        )
    )
    assert np.array_equal(eval_means, eval_means_canonical)

    # --- masked oracle outputs at the candidate parameter point ---
    loss = oracle.convmmd_loss_masked(
        eval_weights,
        eval_means,
        eval_covariances,
        observations,
        observed_mask,
        measurement_covariances,
        bandwidths,
    )
    denoised = oracle.denoise_masked(
        eval_weights,
        eval_means,
        eval_covariances,
        observations,
        observed_mask,
        measurement_covariances,
    )

    return {
        "true_weights": true_weights,
        "true_means": true_means,
        "true_covariances": true_covariances,
        "latent_true": latent_true,
        "observations": observations,
        "observed_mask": observed_mask,
        "observed_counts": observed_counts,
        "measurement_covariances": measurement_covariances,
        "bandwidths": bandwidths,
        "masked_median_distance": np.asarray(
            masked_median_distance, dtype=np.float64
        ),
        "eval_alphas": eval_alphas,
        "eval_means": eval_means,
        "eval_unconstrained_L": eval_unconstrained_L,
        "eval_weights": eval_weights,
        "eval_covariances": eval_covariances,
        "oracle_per_scale_loss": loss.per_scale_loss,
        "oracle_loss": np.asarray(loss.loss, dtype=np.float64),
        "oracle_responsibilities": denoised.responsibilities,
        "oracle_component_means": denoised.component_posterior_means,
        "oracle_posterior_mean": denoised.posterior_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "metadata", nargs="?", type=Path, default=DEFAULT_METADATA
    )
    arguments = parser.parse_args()

    arrays = build_fixture()
    payload_hashes = det_npz.write_deterministic_npz(arguments.archive, arrays)
    archive_sha256 = hashlib.sha256(arguments.archive.read_bytes()).hexdigest()

    metadata = {
        "fixture_id": FIXTURE_ID,
        "fixture_version": FIXTURE_VERSION,
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "archive": arguments.archive.name,
        "archive_sha256": archive_sha256,
        "generator": "scripts/generate_convmmd_missing_fixture.py",
        "generator_numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_components": N_COMPONENTS,
        "dimension": DIMENSION,
        "n_scales": N_SCALES,
        "mask_patterns": [
            "".join("1" if flag else "0" for flag in pattern)
            for pattern in MASK_PATTERNS
        ],
        "bandwidth_protocol": (
            "masked median pairwise distance (shared observed coordinates) x "
            f"logspace({BANDWIDTH_LOG10_LO}, {BANDWIDTH_LOG10_HI}, {N_SCALES})"
        ),
        "masked_median_distance": float(arrays["masked_median_distance"]),
        "oracle_loss": float(arrays["oracle_loss"]),
        "purpose": (
            "Analytic masked convMMD loss, projected empirical-Bayes denoiser, "
            "and masked bandwidth evidence under per-coordinate MAR missingness "
            "(mixed masks including a fully-observed and an M=0 group) with known "
            "latent ground truth and full-covariance heteroscedastic noise."
        ),
        "stored_arrays": sorted(arrays),
        "payload_sha256": payload_hashes,
    }
    arguments.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"wrote {arguments.archive}")
    print(f"archive sha256 {archive_sha256}")
    print(f"wrote {arguments.metadata}")
    print(f"numpy {np.__version__}; PCG64 seed {SEED}")
    print(f"masked_median_distance {float(arrays['masked_median_distance']):.12f}")
    print(f"oracle_loss {float(arrays['oracle_loss']):.12f}")
    counts = arrays["observed_counts"]
    print(f"observed-count histogram {np.bincount(counts, minlength=DIMENSION+1)}")


if __name__ == "__main__":
    main()
