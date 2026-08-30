"""Generate the stored ``convmmd_recovery_001`` fixture deterministically.

This script is not run by the test suite. It records one PCG64-generated
Gaussian-GMM deconvolution case with known latent ground truth, full-covariance
heteroscedastic measurement noise, the predeclared bandwidth set, a candidate
evaluation parameter point (in both unconstrained and canonical form), and the
independent NumPy-oracle outputs at that point. The archive is written with the
deterministic ZIP_STORED writer so identical arrays produce identical bytes, and
every payload plus the whole archive carries a pinned SHA-256.

Run:  ~/anaconda3/envs/cv/bin/python scripts/generate_convmmd_fixture.py
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SEED = 20260830
FIXTURE_ID = "convmmd-recovery-001"
CONTRACT_ID = "xdgmm-jax.convmmd"
CONTRACT_VERSION = "0.1.0-draft.1"
FIXTURE_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "tests" / "fixtures" / "convmmd_recovery_001.npz"
DEFAULT_METADATA = (
    REPO_ROOT / "tests" / "fixtures" / "convmmd_recovery_001.metadata.json"
)

N_SAMPLES = 64
N_COMPONENTS = 3
DIMENSION = 3
N_SCALES = 9
BANDWIDTH_LOG10_LO = -2.0
BANDWIDTH_LOG10_HI = 2.0


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


def median_pairwise_distance(points: np.ndarray) -> float:
    """Median Euclidean distance over distinct point pairs (i < j)."""

    differences = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1))
    upper = distances[np.triu_indices(points.shape[0], k=1)]
    return float(np.median(upper))


def build_fixture() -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(SEED))

    # --- known latent ground truth (full-covariance GMM) ---
    true_weights = np.asarray([0.30, 0.45, 0.25], dtype=np.float64)
    true_means = np.asarray(
        [
            [-1.60, 0.35, -0.40],
            [0.25, -0.95, 0.70],
            [1.70, 1.05, 1.20],
        ],
        dtype=np.float64,
    )
    true_covariances = np.stack(
        [_random_spd(rng, DIMENSION, scale=1.1, floor=0.15) for _ in range(N_COMPONENTS)]
    )

    labels = rng.choice(N_COMPONENTS, size=N_SAMPLES, p=true_weights)
    latent_true = np.empty((N_SAMPLES, DIMENSION), dtype=np.float64)
    for sample, component in enumerate(labels):
        latent_true[sample] = rng.multivariate_normal(
            true_means[component], true_covariances[component]
        )

    # --- full-covariance heteroscedastic measurement noise ---
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

    # --- predeclared bandwidth set (median heuristic x log grid) ---
    median_distance = median_pairwise_distance(observations)
    scale_factors = np.logspace(
        BANDWIDTH_LOG10_LO, BANDWIDTH_LOG10_HI, N_SCALES
    )
    bandwidths = (median_distance * scale_factors).astype(np.float64)

    # --- candidate evaluation parameters (unconstrained, then canonical) ---
    eval_alphas = rng.normal(scale=0.5, size=N_COMPONENTS).astype(np.float64)
    eval_means = (true_means + rng.normal(scale=0.25, size=true_means.shape)).astype(
        np.float64
    )
    eval_unconstrained_L = rng.normal(
        scale=0.6, size=(N_COMPONENTS, DIMENSION, DIMENSION)
    ).astype(np.float64)
    eval_weights, eval_means_canonical, eval_covariances = (
        oracle.unconstrained_to_canonical(
            eval_alphas, eval_means, eval_unconstrained_L
        )
    )
    assert np.array_equal(eval_means, eval_means_canonical)

    # --- oracle outputs at the candidate parameter point ---
    loss = oracle.convmmd_loss(
        eval_weights,
        eval_means,
        eval_covariances,
        observations,
        measurement_covariances,
        bandwidths,
    )
    denoised = oracle.denoise(
        eval_weights,
        eval_means,
        eval_covariances,
        observations,
        measurement_covariances,
    )

    return {
        "true_weights": true_weights,
        "true_means": true_means,
        "true_covariances": true_covariances,
        "latent_true": latent_true,
        "observations": observations,
        "measurement_covariances": measurement_covariances,
        "bandwidths": bandwidths,
        "median_pairwise_distance": np.asarray(median_distance, dtype=np.float64),
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
        "generator": "scripts/generate_convmmd_fixture.py",
        "generator_numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_components": N_COMPONENTS,
        "dimension": DIMENSION,
        "n_scales": N_SCALES,
        "bandwidth_protocol": (
            "median pairwise distance of observations x "
            f"logspace({BANDWIDTH_LOG10_LO}, {BANDWIDTH_LOG10_HI}, {N_SCALES})"
        ),
        "median_pairwise_distance": float(arrays["median_pairwise_distance"]),
        "oracle_loss": float(arrays["oracle_loss"]),
        "purpose": (
            "Analytic convMMD loss, empirical-Bayes denoiser, and "
            "parameterization round-trip evidence with known latent ground "
            "truth and full-covariance heteroscedastic noise."
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
    print(f"oracle_loss {float(arrays['oracle_loss']):.12f}")


if __name__ == "__main__":
    main()
