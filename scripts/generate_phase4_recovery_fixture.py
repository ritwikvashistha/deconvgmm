"""Generate the deterministic Phase 4 final-fit recovery fixture.

This evidence fixture is deliberately independent of the JAX implementation.
It contains two statistically generated workloads:

* an identity-projection mixture with heterogeneous correlated errors; and
* a dense-projection, variable-observed-dimension mixture prepared through the
  boolean-mask adapter.

The fixture is generated once and stored so ordinary tests never depend on a
particular NumPy random implementation.  Recovery is intentionally conditioned
on one fixed, perturbed initialization.  It is not evidence that EM finds a
global optimum from arbitrary starting values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SEED = 20260825
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deterministic_npz import write_deterministic_npz


DEFAULT_ARCHIVE = (
    PROJECT_ROOT / "tests" / "fixtures" / "phase4_recovery_001.npz"
)
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "phase4_recovery_001.metadata.json"
)
RECOVERY_HELPER = PROJECT_ROOT / "tests" / "reference" / "recovery.py"
DETERMINISTIC_ARCHIVE_HELPER = (
    PROJECT_ROOT / "scripts" / "deterministic_npz.py"
)


def _sample_mixture(
    rng: np.random.Generator,
    *,
    n_samples: int,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    labels = rng.choice(len(weights), size=n_samples, p=weights).astype(
        np.int32
    )
    samples = np.empty((n_samples, means.shape[1]), dtype=np.float64)
    for component in range(len(weights)):
        selected = np.flatnonzero(labels == component)
        samples[selected] = rng.multivariate_normal(
            means[component], covariances[component], size=len(selected)
        )
    return labels, samples


def _identity_arrays() -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    n_samples = 1_200
    holdout_size = 4_096

    true_weights = np.asarray([0.42, 0.58], dtype=np.float64)
    true_means = np.asarray(
        [[-2.0, -0.8], [1.7, 1.1]], dtype=np.float64
    )
    true_covariances = np.asarray(
        [
            [[0.45, 0.12], [0.12, 0.70]],
            [[0.65, -0.18], [-0.18, 0.50]],
        ],
        dtype=np.float64,
    )
    labels, latent = _sample_mixture(
        rng,
        n_samples=n_samples,
        weights=true_weights,
        means=true_means,
        covariances=true_covariances,
    )

    variance_0 = rng.uniform(0.12, 0.38, size=n_samples)
    variance_1 = rng.uniform(0.10, 0.34, size=n_samples)
    correlation = rng.uniform(-0.40, 0.40, size=n_samples)
    noise = np.zeros((n_samples, 2, 2), dtype=np.float64)
    noise[:, 0, 0] = variance_0
    noise[:, 1, 1] = variance_1
    noise[:, 0, 1] = noise[:, 1, 0] = correlation * np.sqrt(
        variance_0 * variance_1
    )
    observations = np.empty_like(latent)
    for sample in range(n_samples):
        observations[sample] = latent[sample] + rng.multivariate_normal(
            np.zeros(2, dtype=np.float64), noise[sample]
        )

    initial_weights = np.asarray([0.55, 0.45], dtype=np.float64)
    initial_means = np.asarray(
        [[1.25, 0.70], [-1.55, -0.35]], dtype=np.float64
    )
    initial_covariances = np.asarray(
        [
            [[0.95, -0.04], [-0.04, 0.85]],
            [[0.85, 0.03], [0.03, 1.00]],
        ],
        dtype=np.float64,
    )
    holdout_labels, latent_holdout = _sample_mixture(
        rng,
        n_samples=holdout_size,
        weights=true_weights,
        means=true_means,
        covariances=true_covariances,
    )
    return {
        "identity_initial_covariances": initial_covariances,
        "identity_initial_means": initial_means,
        "identity_initial_weights": initial_weights,
        "identity_labels": labels,
        "identity_latent_holdout": latent_holdout,
        "identity_latent_holdout_labels": holdout_labels,
        "identity_measurement_covariances": noise,
        "identity_observations": observations,
        "identity_true_covariances": true_covariances,
        "identity_true_means": true_means,
        "identity_true_weights": true_weights,
    }


def _general_arrays() -> dict[str, np.ndarray]:
    # Resetting the recorded generator gives this workload its own stable stream
    # without coupling its bytes to the identity fixture's sample count.
    rng = np.random.Generator(np.random.PCG64(SEED))
    n_samples = 600
    holdout_size = 4_096
    latent_dimension = 3
    potential_observed_dimension = 3

    true_weights = np.asarray([0.46, 0.54], dtype=np.float64)
    true_means = np.asarray(
        [[-1.8, -0.9, 0.65], [1.55, 0.85, -0.55]], dtype=np.float64
    )
    true_covariances = np.asarray(
        [
            [
                [0.50, 0.10, -0.05],
                [0.10, 0.65, 0.08],
                [-0.05, 0.08, 0.42],
            ],
            [
                [0.68, -0.14, 0.06],
                [-0.14, 0.52, -0.09],
                [0.06, -0.09, 0.58],
            ],
        ],
        dtype=np.float64,
    )
    labels, latent = _sample_mixture(
        rng,
        n_samples=n_samples,
        weights=true_weights,
        means=true_means,
        covariances=true_covariances,
    )

    base_projection = np.asarray(
        [
            [1.00, 0.18, -0.12],
            [-0.22, 0.90, 0.16],
            [0.14, -0.11, 1.05],
        ],
        dtype=np.float64,
    )
    projection = base_projection[None, :, :] + rng.normal(
        scale=0.025,
        size=(n_samples, potential_observed_dimension, latent_dimension),
    )

    diagonal = rng.uniform(
        0.08,
        0.28,
        size=(n_samples, potential_observed_dimension),
    )
    correlations = rng.uniform(-0.18, 0.18, size=(n_samples, 3))
    noise = np.zeros(
        (
            n_samples,
            potential_observed_dimension,
            potential_observed_dimension,
        ),
        dtype=np.float64,
    )
    noise[:, np.arange(3), np.arange(3)] = diagonal
    noise[:, 0, 1] = noise[:, 1, 0] = correlations[:, 0] * np.sqrt(
        diagonal[:, 0] * diagonal[:, 1]
    )
    noise[:, 0, 2] = noise[:, 2, 0] = correlations[:, 1] * np.sqrt(
        diagonal[:, 0] * diagonal[:, 2]
    )
    noise[:, 1, 2] = noise[:, 2, 1] = correlations[:, 2] * np.sqrt(
        diagonal[:, 1] * diagonal[:, 2]
    )

    observations = np.empty(
        (n_samples, potential_observed_dimension), dtype=np.float64
    )
    for sample in range(n_samples):
        observations[sample] = (
            projection[sample] @ latent[sample]
            + rng.multivariate_normal(np.zeros(3, dtype=np.float64), noise[sample])
        )

    mask_patterns = np.asarray(
        [
            [True, True, True],
            [True, True, False],
            [True, False, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    pattern_probabilities = np.asarray(
        [0.55, 0.15, 0.15, 0.15], dtype=np.float64
    )
    observed_mask = mask_patterns[
        rng.choice(
            len(mask_patterns), size=n_samples, p=pattern_probabilities
        )
    ]
    sample_weight = rng.uniform(0.75, 1.25, size=n_samples)

    initial_weights = np.asarray([0.56, 0.44], dtype=np.float64)
    initial_means = np.asarray(
        [[1.20, 0.55, -0.25], [-1.45, -0.55, 0.35]],
        dtype=np.float64,
    )
    initial_covariances = np.asarray(
        [
            [
                [0.95, -0.03, 0.02],
                [-0.03, 0.85, -0.02],
                [0.02, -0.02, 0.90],
            ],
            [
                [0.82, 0.04, -0.01],
                [0.04, 1.00, 0.03],
                [-0.01, 0.03, 0.78],
            ],
        ],
        dtype=np.float64,
    )
    holdout_labels, latent_holdout = _sample_mixture(
        rng,
        n_samples=holdout_size,
        weights=true_weights,
        means=true_means,
        covariances=true_covariances,
    )
    return {
        "general_initial_covariances": initial_covariances,
        "general_initial_means": initial_means,
        "general_initial_weights": initial_weights,
        "general_labels": labels,
        "general_latent_holdout": latent_holdout,
        "general_latent_holdout_labels": holdout_labels,
        "general_measurement_covariances": noise,
        "general_observations": observations,
        "general_observed_mask": observed_mask,
        "general_projection_matrices": projection,
        "general_sample_weight": sample_weight,
        "general_true_covariances": true_covariances,
        "general_true_means": true_means,
        "general_true_weights": true_weights,
    }


def build_fixture() -> dict[str, np.ndarray]:
    """Return both deterministic recovery workloads."""

    arrays = {**_identity_arrays(), **_general_arrays()}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or not np.all(np.isfinite(array)):
            raise ValueError(f"fixture array {name!r} is not finite numeric data")
    return arrays


def _maximum_effective_condition(
    *,
    covariances: np.ndarray,
    projection: np.ndarray,
    noise: np.ndarray,
    observed_mask: np.ndarray | None = None,
) -> float:
    maximum = 0.0
    for sample in range(len(projection)):
        selected = (
            np.ones(projection.shape[1], dtype=bool)
            if observed_mask is None
            else observed_mask[sample]
        )
        row_projection = projection[sample, selected]
        row_noise = noise[sample][np.ix_(selected, selected)]
        for covariance in covariances:
            effective = row_projection @ covariance @ row_projection.T + row_noise
            maximum = max(maximum, float(np.linalg.cond(effective)))
    return maximum


def _metadata(
    *,
    archive: Path,
    arrays: dict[str, np.ndarray],
    payload_hashes: dict[str, str],
) -> dict[str, object]:
    identity_projection = np.broadcast_to(
        np.eye(2, dtype=np.float64),
        (len(arrays["identity_observations"]), 2, 2),
    )
    projection_gram = np.zeros((3, 3), dtype=np.float64)
    for projection, mask in zip(
        arrays["general_projection_matrices"],
        arrays["general_observed_mask"],
        strict=True,
    ):
        selected_projection = projection[mask]
        projection_gram += selected_projection.T @ selected_projection
    projection_gram /= len(arrays["general_observations"])
    projection_gram_eigenvalues = np.linalg.eigvalsh(projection_gram)
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return {
        "fixture_id": "xd-phase4-final-fit-recovery-001",
        "fixture_version": 1,
        "formal_matrix_status": "Pending",
        "archive": archive.name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "generator": "scripts/generate_phase4_recovery_fixture.py",
        "generator_sha256": script_digest,
        "generator_numpy_version": np.__version__,
        "evidence_source_sha256": {
            "scripts/deterministic_npz.py": hashlib.sha256(
                DETERMINISTIC_ARCHIVE_HELPER.read_bytes()
            ).hexdigest(),
            "tests/reference/recovery.py": hashlib.sha256(
                RECOVERY_HELPER.read_bytes()
            ).hexdigest(),
        },
        "bit_generator": "PCG64",
        "seed": SEED,
        "array_schema": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in sorted(arrays.items())
        },
        "npy_payload_sha256": dict(sorted(payload_hashes.items())),
        "fit_controls": {
            "max_iter": 100,
            "tol": 1e-5,
            "decrease_tol": 1e-8,
            "factor_jitter": 0.0,
            "covariance_ridge": 0.0,
        },
        "acceptance_thresholds": {
            "identity": {
                "max_absolute_weight_error": 0.05,
                "max_mean_mahalanobis_error": 0.20,
                "max_relative_covariance_frobenius_error": 0.25,
                "latent_log_density_rms_error": 0.30,
                "absolute_latent_log_density_mean_gap": 0.08,
                "mixture_mean_l2_error": 0.10,
                "mixture_covariance_relative_frobenius_error": 0.10,
                "total_symmetric_gaussian_kl": 0.15,
                "maximum_density_rms_fraction_of_initial": 0.50,
                "maximum_alignment_kl_fraction_of_initial": 0.25,
            },
            "general": {
                "max_absolute_weight_error": 0.06,
                "max_mean_mahalanobis_error": 0.30,
                "max_relative_covariance_frobenius_error": 0.35,
                "latent_log_density_rms_error": 0.45,
                "absolute_latent_log_density_mean_gap": 0.12,
                "mixture_mean_l2_error": 0.12,
                "mixture_covariance_relative_frobenius_error": 0.15,
                "total_symmetric_gaussian_kl": 0.25,
                "maximum_density_rms_fraction_of_initial": 0.65,
                "maximum_alignment_kl_fraction_of_initial": 0.35,
            },
        },
        "acceptance_rationale": {
            "parameter_scale": (
                "Weight, standardized-mean, and covariance bounds are broad "
                "finite-sample envelopes for component counts of order 275 "
                "or larger; they are not confidence intervals or exact-truth "
                "requirements."
            ),
            "distribution_scale": (
                "Stored 4096-draw latent holdouts provide label-invariant "
                "log-density and global-mixture-moment checks independent of "
                "the observations used for fitting."
            ),
            "discrimination": (
                "Absolute bounds are paired with improvement fractions, and "
                "the stored initial parameters fail the final-fit envelope, "
                "so returning the initialization cannot pass."
            ),
            "generality": (
                "The general envelope is wider because only two or three "
                "projected coordinates are observed per row and weighted "
                "effective component counts are smaller."
            ),
        },
        "identity": {
            "contract_id": "xdgmm-jax.identity-xd",
            "contract_version": "0.1.0-draft.1",
            "n_samples": int(len(arrays["identity_observations"])),
            "n_components": 2,
            "latent_dimension": 2,
            "component_counts": np.bincount(
                arrays["identity_labels"], minlength=2
            ).tolist(),
            "maximum_true_effective_condition": _maximum_effective_condition(
                covariances=arrays["identity_true_covariances"],
                projection=identity_projection,
                noise=arrays["identity_measurement_covariances"],
            ),
        },
        "general": {
            "contract_id": "xdgmm-jax.general-xd",
            "contract_version": "0.2.0-draft.1",
            "n_samples": int(len(arrays["general_observations"])),
            "n_components": 2,
            "latent_dimension": 3,
            "component_counts": np.bincount(
                arrays["general_labels"], minlength=2
            ).tolist(),
            "weighted_component_mass": np.bincount(
                arrays["general_labels"],
                weights=arrays["general_sample_weight"],
                minlength=2,
            ).tolist(),
            "observed_dimensions": sorted(
                {
                    int(value)
                    for value in arrays["general_observed_mask"].sum(axis=1)
                }
            ),
            "mask_counts": {
                "".join("1" if item else "0" for item in pattern): int(
                    np.sum(
                        np.all(
                            arrays["general_observed_mask"] == pattern,
                            axis=1,
                        )
                    )
                )
                for pattern in np.unique(
                    arrays["general_observed_mask"], axis=0
                )
            },
            "maximum_true_effective_condition": _maximum_effective_condition(
                covariances=arrays["general_true_covariances"],
                projection=arrays["general_projection_matrices"],
                noise=arrays["general_measurement_covariances"],
                observed_mask=arrays["general_observed_mask"],
            ),
            "aggregate_projection_gram_eigenvalues": (
                projection_gram_eigenvalues.tolist()
            ),
            "aggregate_projection_gram_condition": float(
                np.linalg.cond(projection_gram)
            ),
        },
        "scientific_scope": {
            "claim": (
                "basin-conditioned statistical recovery from one fixed, "
                "perturbed user-supplied initialization"
            ),
            "comparison": (
                "permutation-invariant generating-parameter and independent "
                "latent-holdout density metrics after a complete fit"
            ),
            "not_claimed": [
                "one-step or implementation parity",
                "global-optimum recovery",
                "robustness to arbitrary initialization",
                "multiple-restart selection",
                "finite-sample equality to generating parameters",
                "GPU or performance support",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", nargs="?", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--metadata", type=Path, default=DEFAULT_METADATA
    )
    arguments = parser.parse_args()

    arrays = build_fixture()
    payload_hashes = write_deterministic_npz(arguments.archive, arrays)
    metadata = _metadata(
        archive=arguments.archive,
        arrays=arrays,
        payload_hashes=payload_hashes,
    )
    arguments.metadata.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {arguments.archive}")
    print(f"sha256 {metadata['archive_sha256']}")
    print(f"wrote {arguments.metadata}")


if __name__ == "__main__":
    main()
