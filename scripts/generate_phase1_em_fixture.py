"""Generate the stored ``XD-IP-EM-002`` fixture deterministically.

This script is not run by the test suite.  It records one PCG64-generated
dataset as a deterministic NPZ archive so ordinary CI never depends on NumPy's
random-number implementation.  The fixed ZIP metadata makes identical arrays
produce identical archive bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np


SEED = 20260825
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "identity_em_002.npz"
)


def build_fixture() -> dict[str, np.ndarray]:
    """Return the fixed ``N=128, K=3, D=3`` synthetic identity-XD case."""

    rng = np.random.Generator(np.random.PCG64(SEED))
    n_samples = 128
    n_components = 3
    dimension = 3

    generating_weights = np.asarray([0.28, 0.44, 0.28], dtype=np.float64)
    generating_means = np.asarray(
        [
            [-1.35, 0.45, -0.30],
            [0.20, -0.80, 0.65],
            [1.55, 0.90, 1.15],
        ],
        dtype=np.float64,
    )
    generating_covariances = np.asarray(
        [
            [[0.55, 0.10, -0.04], [0.10, 0.72, 0.08], [-0.04, 0.08, 0.46]],
            [[0.83, -0.12, 0.07], [-0.12, 0.61, -0.05], [0.07, -0.05, 0.69]],
            [[0.48, 0.06, 0.09], [0.06, 0.88, 0.14], [0.09, 0.14, 0.77]],
        ],
        dtype=np.float64,
    )

    labels = rng.choice(
        n_components, size=n_samples, p=generating_weights
    ).astype(np.int32)
    latent = np.empty((n_samples, dimension), dtype=np.float64)
    for sample, component in enumerate(labels):
        latent[sample] = rng.multivariate_normal(
            generating_means[component], generating_covariances[component]
        )

    measurement_covariances = np.empty(
        (n_samples, dimension, dimension), dtype=np.float64
    )
    observed = np.empty_like(latent)
    for sample in range(n_samples):
        raw = rng.normal(size=(dimension, dimension))
        covariance = raw @ raw.T
        covariance *= rng.uniform(0.04, 0.22) / np.trace(covariance)
        covariance += np.diag(rng.uniform(0.005, 0.025, size=dimension))
        measurement_covariances[sample] = covariance
        observed[sample] = latent[sample] + rng.multivariate_normal(
            np.zeros(dimension), covariance
        )

    initial_weights = np.asarray([0.31, 0.39, 0.30], dtype=np.float64)
    initial_means = np.asarray(
        [
            [-1.05, 0.10, -0.05],
            [0.00, -0.45, 0.35],
            [1.20, 0.55, 0.85],
        ],
        dtype=np.float64,
    )
    initial_covariances = np.asarray(
        [
            [[0.95, 0.08, 0.00], [0.08, 0.90, 0.04], [0.00, 0.04, 0.82]],
            [[1.10, -0.06, 0.03], [-0.06, 0.92, -0.02], [0.03, -0.02, 0.96]],
            [[0.88, 0.04, 0.06], [0.04, 1.08, 0.09], [0.06, 0.09, 1.02]],
        ],
        dtype=np.float64,
    )

    return {
        "initial_covariances": initial_covariances,
        "initial_means": initial_means,
        "initial_weights": initial_weights,
        "measurement_covariances": measurement_covariances,
        "observations": observed,
    }


def _npy_bytes(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def write_deterministic_npz(
    destination: Path, arrays: dict[str, np.ndarray]
) -> None:
    """Write arrays with stable entry order, timestamps, and permissions."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(
                filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(
                info,
                _npy_bytes(np.asarray(arrays[name])),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    arrays = build_fixture()
    write_deterministic_npz(arguments.output, arrays)
    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    print(f"wrote {arguments.output}")
    print(f"sha256 {digest}")
    print(f"numpy {np.__version__}; PCG64 seed {SEED}")


if __name__ == "__main__":
    main()

