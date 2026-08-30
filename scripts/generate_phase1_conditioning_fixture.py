"""Generate the stored ``XD-IP-COV-001`` conditioning fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_phase1_em_fixture import write_deterministic_npz


DEFAULT_OUTPUT = (
    PROJECT_ROOT / "tests" / "fixtures" / "identity_cov_001.npz"
)


DCT_ORTHOGONAL = np.asarray(
    [
        [
            0.4472135954999579,
            0.6015009550075456,
            0.5116672736016927,
            0.3717480344601845,
            0.19543950758485482,
        ],
        [
            0.4472135954999579,
            0.3717480344601845,
            -0.19543950758485476,
            -0.6015009550075456,
            -0.5116672736016927,
        ],
        [
            0.4472135954999579,
            3.8726732145403873e-17,
            -0.6324555320336759,
            -1.1618019643621161e-16,
            0.6324555320336759,
        ],
        [
            0.4472135954999579,
            -0.37174803446018445,
            -0.1954395075848549,
            0.6015009550075457,
            -0.5116672736016926,
        ],
        [
            0.4472135954999579,
            -0.6015009550075456,
            0.5116672736016927,
            -0.37174803446018434,
            0.19543950758485454,
        ],
    ],
    dtype=np.float64,
)


def _problem_arrays(dtype: np.dtype, kappa: float) -> tuple[np.ndarray, ...]:
    selected = np.dtype(dtype)
    q = DCT_ORTHOGONAL.astype(selected)
    eigenvalues = np.geomspace(1.0, kappa, 5).astype(selected)
    total = (q * eigenvalues[None, :]) @ q.T
    total = (selected.type(0.5) * total) + (selected.type(0.5) * total.T)
    model_covariances = np.broadcast_to(
        selected.type(0.5) * np.eye(5, dtype=selected), (3, 5, 5)
    ).copy()
    measurement = total - model_covariances[0]
    measurement = (selected.type(0.5) * measurement) + (
        selected.type(0.5) * measurement.T
    )
    measurement_covariances = np.broadcast_to(
        measurement, (32, 5, 5)
    ).copy()

    u = np.linspace(-1.4, 1.4, 32, dtype=selected)
    eigen_coordinates = np.stack(
        (
            u,
            selected.type(0.7) * np.sin(selected.type(1.3) * u),
            selected.type(0.6) * np.cos(selected.type(0.9) * u),
            selected.type(0.25) * u * u - selected.type(0.15),
            selected.type(0.4) * np.sin(selected.type(0.6) + u),
        ),
        axis=-1,
    )
    observations = eigen_coordinates @ q.T
    mean_coordinates = np.asarray(
        [
            [-0.8, 0.25, 0.10, -0.05, 0.20],
            [0.0, -0.35, 0.30, 0.15, -0.10],
            [0.85, 0.20, -0.25, 0.05, 0.15],
        ],
        dtype=selected,
    )
    means = mean_coordinates @ q.T
    weights = np.asarray([0.25, 0.45, 0.30], dtype=selected)
    return (
        observations,
        measurement_covariances,
        weights,
        means,
        model_covariances,
        total,
    )


def build_fixture() -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {"dct_orthogonal": DCT_ORTHOGONAL}
    profiles = (
        ("float64_in", np.float64, 1e8),
        ("float64_out", np.float64, 1e12),
        ("float32_in", np.float32, 1e4),
        ("float32_out", np.float32, 1e7),
    )
    for label, dtype, kappa in profiles:
        (
            observations,
            measurement_covariances,
            weights,
            means,
            model_covariances,
            total,
        ) = _problem_arrays(np.dtype(dtype), kappa)
        arrays[f"{label}_observations"] = observations
        arrays[f"{label}_measurement_covariances"] = measurement_covariances
        arrays[f"{label}_weights"] = weights
        arrays[f"{label}_means"] = means
        arrays[f"{label}_model_covariances"] = model_covariances
        arrays[f"{label}_effective_covariance"] = total
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    write_deterministic_npz(arguments.output, build_fixture())
    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    print(f"wrote {arguments.output}")
    print(f"sha256 {digest}")
    print(f"numpy {np.__version__}; literal DCT orthogonal matrix")


if __name__ == "__main__":
    main()

