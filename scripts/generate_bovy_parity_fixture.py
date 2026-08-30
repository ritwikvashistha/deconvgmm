"""Generate the pending pinned Bovy general-XD parity fixture.

This release-evidence tool accepts an externally supplied source archive.  It
never downloads upstream source, and it refuses to build until the complete
archive, safe single-root layout, tracked-file manifest, and critical source
members match the audited commit.  The reference build happens only in a
temporary directory.

Ordinary tests exercise custody and schema helpers with tiny local archives.
They do not build Bovy or create numeric results claimed to be Bovy output.
The authoritative numeric fixture remains pending until this script is run in
the pinned Linux/amd64 CPU container defined under ``scripts/reference/bovy``.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deterministic_npz import write_deterministic_npz
from tests.reference.general_xd import general_e_step, general_m_step


BOVY_COMMIT = "a8a5988d2ab3ceeecbe7f0c23e0554d8a3a4222c"
BOVY_TREE = "d8cd1cb3dbe024d872992e461ba1290f22f722a8"
BOVY_SOURCE_ROOT = f"extreme-deconvolution-{BOVY_COMMIT}"
BOVY_SOURCE_ARCHIVE_SHA256 = (
    "c1882ea6be58c4f08a9d66c539504f1a1c4bc892fa2a5adad7abe47fcaf165fc"
)
BOVY_SOURCE_ARCHIVE_SIZE = 536_082
BOVY_SOURCE_FILE_COUNT = 66
BOVY_SOURCE_MANIFEST_SHA256 = (
    "7111a901ba02094f16f3fb44f748da4b831ab9fa5c4e6dea75ce336b624081b8"
)
BOVY_SOURCE_MANIFEST_FORMAT = (
    "For each regular file: lowercase SHA-256, two ASCII spaces, relative "
    "POSIX path, LF; byte-sort complete UTF-8 lines, concatenate, SHA-256"
)
BOVY_MEMBER_SHA256 = {
    "LICENSE": "e52808797a9bd901b30bbd0a42d2189090f9390803fa8102c1afbb9919f3c18e",
    "Makefile": "7b027b483c67d685f40efbc822b429d00bf6ab83aa8035ffea0013f11b77c4f3",
    "py/extreme_deconvolution.py": (
        "0340e31ab4d3fd2652cbf847c61e6c36888add630a5c151fefd0d226bcb07a49"
    ),
    "py/extreme_deconvolution_TEMPLATE.py": (
        "18a972372f9822960ed39890dc078944079dc18489f9b44ed9d99666733ecc2b"
    ),
    "src/proj_EM.c": (
        "b3428152ab4555655a66820be9dc8da20876a03b90de09945a5ead1ca2c4bfac"
    ),
    "src/proj_EM_step.c": (
        "53b37ca9a1baca8908e1f64c25ce8ef622a380bdd93f608a8274d1d6d4fe9d10"
    ),
    "src/proj_gauss_mixtures.c": (
        "354456bf061168b3e5f54d770c5ca17a9790612b2d46fe2ff593535c0c5b1e99"
    ),
    "src/proj_gauss_mixtures_IDL.c": (
        "4dabb1f9c75334939c96068e9749fa245fe40d275c220257939796de116f675d"
    ),
}

DEBIAN_AMD64_MANIFEST_DIGEST = (
    "sha256:5ae3c39ebd15e229dcedd5cee596b2497182493d41ff162e824ba13fc1b2b867"
)
DEBIAN_MULTIARCH_INDEX_DIGEST = (
    "sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171"
)
DEBIAN_SNAPSHOT = "20260824T000000Z"
EXPECTED_DEBIAN_PACKAGES = {
    "gcc": "4:12.2.0-3",
    "make": "4.3-4.1",
    "libgsl-dev": "2.7.1+dfsg-5+deb12u1",
    "python3": "3.11.2-1+b1",
    "python3-numpy": "1:1.24.2-1+deb12u1",
}

SEED = 20260825
N_SAMPLES = 19
N_COMPONENTS = 3
LATENT_DIMENSION = 4
OBSERVED_DIMENSION = 2
REFERENCE_RTOL = 5e-8
REFERENCE_ATOL = 5e-10
MAX_ABSOLUTE_LOG_ERROR = 5e-8
MAX_EFFECTIVE_CONDITION = 1e4
MAX_ABSOLUTE_COMPONENT_LOG_DENSITY = 1e3
MAX_EXTRACTED_BYTES = 16 * 1024 * 1024

FIXTURE_ARRAY_SCHEMA: dict[str, tuple[int, ...]] = {
    "observations": (N_SAMPLES, OBSERVED_DIMENSION),
    "projection_matrices": (
        N_SAMPLES,
        OBSERVED_DIMENSION,
        LATENT_DIMENSION,
    ),
    "measurement_covariances": (
        N_SAMPLES,
        OBSERVED_DIMENSION,
        OBSERVED_DIMENSION,
    ),
    "sample_weight": (N_SAMPLES,),
    "initial_weights": (N_COMPONENTS,),
    "initial_means": (N_COMPONENTS, LATENT_DIMENSION),
    "initial_covariances": (
        N_COMPONENTS,
        LATENT_DIMENSION,
        LATENT_DIMENSION,
    ),
    "bovy_initial_score_samples": (N_SAMPLES,),
    "bovy_initial_log_likelihood": (),
    "bovy_initial_mean_objective": (),
    "bovy_returned_preupdate_mean_objective": (),
    "bovy_updated_weights": (N_COMPONENTS,),
    "bovy_updated_means": (N_COMPONENTS, LATENT_DIMENSION),
    "bovy_updated_covariances": (
        N_COMPONENTS,
        LATENT_DIMENSION,
        LATENT_DIMENSION,
    ),
    "bovy_candidate_score_samples": (N_SAMPLES,),
    "bovy_candidate_log_likelihood": (),
    "bovy_candidate_mean_objective": (),
    "oracle_effective_covariances": (
        N_SAMPLES,
        N_COMPONENTS,
        OBSERVED_DIMENSION,
        OBSERVED_DIMENSION,
    ),
    "oracle_residuals": (
        N_SAMPLES,
        N_COMPONENTS,
        OBSERVED_DIMENSION,
    ),
    "oracle_gains": (
        N_SAMPLES,
        N_COMPONENTS,
        LATENT_DIMENSION,
        OBSERVED_DIMENSION,
    ),
    "oracle_component_log_density": (N_SAMPLES, N_COMPONENTS),
    "oracle_component_log_joint": (N_SAMPLES, N_COMPONENTS),
    "oracle_score_samples": (N_SAMPLES,),
    "oracle_responsibilities": (N_SAMPLES, N_COMPONENTS),
    "oracle_conditional_mean": (
        N_SAMPLES,
        N_COMPONENTS,
        LATENT_DIMENSION,
    ),
    "oracle_conditional_covariance": (
        N_SAMPLES,
        N_COMPONENTS,
        LATENT_DIMENSION,
        LATENT_DIMENSION,
    ),
    "oracle_stat_mass": (N_COMPONENTS,),
    "oracle_stat_first_moment": (N_COMPONENTS, LATENT_DIMENSION),
    "oracle_stat_second_moment": (
        N_COMPONENTS,
        LATENT_DIMENSION,
        LATENT_DIMENSION,
    ),
    "oracle_updated_weights": (N_COMPONENTS,),
    "oracle_updated_means": (N_COMPONENTS, LATENT_DIMENSION),
    "oracle_updated_covariances": (
        N_COMPONENTS,
        LATENT_DIMENSION,
        LATENT_DIMENSION,
    ),
    "oracle_candidate_component_log_density": (N_SAMPLES, N_COMPONENTS),
    "oracle_candidate_component_log_joint": (N_SAMPLES, N_COMPONENTS),
    "oracle_candidate_score_samples": (N_SAMPLES,),
    "oracle_candidate_log_likelihood": (),
    "oracle_candidate_mean_objective": (),
}


class CustodyError(RuntimeError):
    """Raised when a source artifact fails its pinned custody contract."""


class FixtureSchemaError(RuntimeError):
    """Raised when a would-be stored fixture violates its exact schema."""


class ReferenceDisagreement(RuntimeError):
    """Raised before writing when Bovy and the independent oracle disagree."""


class BuildEnvironmentError(RuntimeError):
    """Raised when authoritative generation is attempted outside the pin."""


@dataclass(frozen=True)
class SourceFileRecord:
    """One regular file in the source-archive custody manifest."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class VerifiedSourceArchive:
    """Immutable facts established before any archive extraction."""

    archive_path: Path
    archive_size: int
    archive_sha256: str
    root_name: str
    file_count: int
    manifest_sha256: str
    files: tuple[SourceFileRecord, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_bytes(records: Sequence[SourceFileRecord]) -> bytes:
    lines = [f"{record.sha256}  {record.path}\n" for record in records]
    return "".join(sorted(lines)).encode("utf-8")


def _validated_member_path(
    name: str, expected_root: str, *, is_dir: bool
) -> PurePosixPath:
    if not name:
        raise CustodyError("tar member has an empty name")
    if "\\" in name or any(ord(character) < 32 for character in name):
        raise CustodyError(f"unsafe tar member path: {name!r}")
    canonical_name = name[:-1] if is_dir and name.endswith("/") else name
    path = PurePosixPath(canonical_name)
    raw_parts = canonical_name.split("/")
    if path.is_absolute():
        raise CustodyError(f"absolute tar member path is forbidden: {name!r}")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise CustodyError(f"unsafe tar member path: {name!r}")
    if not path.parts or path.parts[0] != expected_root:
        raise CustodyError(
            "source archive must contain exactly the pinned single root "
            f"{expected_root!r}; received member {name!r}"
        )
    return path


def _inspect_tar_members(
    archive_path: Path,
    *,
    expected_root: str,
) -> tuple[tuple[SourceFileRecord, ...], tuple[str, ...]]:
    records: list[SourceFileRecord] = []
    directories: list[str] = []
    seen: set[str] = set()
    total_size = 0
    root_directory_seen = False

    try:
        archive = tarfile.open(archive_path, mode="r:gz")
    except (tarfile.TarError, OSError) as error:
        raise CustodyError(
            f"verified path is not a readable gzip tar: {error}"
        ) from error

    with archive:
        for member in archive.getmembers():
            if not (member.isfile() or member.isdir()):
                raise CustodyError(
                    "unsupported tar member type; links and special files are "
                    f"forbidden: {member.name!r}"
                )
            member_path = _validated_member_path(
                member.name, expected_root, is_dir=member.isdir()
            )
            canonical_name = member_path.as_posix()
            if canonical_name in seen:
                raise CustodyError(f"duplicate tar member: {canonical_name!r}")
            seen.add(canonical_name)

            if member.isdir():
                directories.append(canonical_name)
                root_directory_seen |= canonical_name == expected_root
                continue
            if len(member_path.parts) == 1:
                raise CustodyError("the pinned root member must be a directory")
            total_size += member.size
            if total_size > MAX_EXTRACTED_BYTES:
                raise CustodyError(
                    "source archive exceeds the bounded extracted-size policy"
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise CustodyError(f"cannot read regular tar member {member.name!r}")
            payload = stream.read()
            if len(payload) != member.size:
                raise CustodyError(f"truncated tar member {member.name!r}")
            relative = PurePosixPath(*member_path.parts[1:]).as_posix()
            records.append(
                SourceFileRecord(
                    path=relative,
                    size=len(payload),
                    sha256=_sha256_bytes(payload),
                )
            )

    if not root_directory_seen:
        raise CustodyError(
            f"source archive is missing its root directory {expected_root!r}"
        )
    return tuple(sorted(records, key=lambda record: record.path)), tuple(
        sorted(directories)
    )


def verify_source_archive(
    source_archive: Path,
    *,
    expected_sha256: str = BOVY_SOURCE_ARCHIVE_SHA256,
    expected_size: int = BOVY_SOURCE_ARCHIVE_SIZE,
    expected_root: str = BOVY_SOURCE_ROOT,
    expected_file_count: int = BOVY_SOURCE_FILE_COUNT,
    expected_manifest_sha256: str = BOVY_SOURCE_MANIFEST_SHA256,
    expected_member_sha256: Mapping[str, str] = BOVY_MEMBER_SHA256,
) -> VerifiedSourceArchive:
    """Verify the complete external source archive without extracting it."""

    path = source_archive.expanduser().resolve(strict=True)
    if not path.is_file():
        raise CustodyError(f"source archive is not a regular file: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise CustodyError(
            "source archive size mismatch: "
            f"expected {expected_size}, received {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise CustodyError(
            "source archive SHA-256 mismatch: "
            f"expected {expected_sha256}, received {actual_sha256}"
        )

    records, _ = _inspect_tar_members(path, expected_root=expected_root)
    if len(records) != expected_file_count:
        raise CustodyError(
            "source archive regular-file count mismatch: "
            f"expected {expected_file_count}, received {len(records)}"
        )
    manifest_sha256 = _sha256_bytes(_manifest_bytes(records))
    if manifest_sha256 != expected_manifest_sha256:
        raise CustodyError(
            "source manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, received {manifest_sha256}"
        )

    by_path = {record.path: record.sha256 for record in records}
    for member, expected_digest in expected_member_sha256.items():
        if member not in by_path:
            raise CustodyError(f"missing critical member {member!r}")
        if by_path[member] != expected_digest:
            raise CustodyError(
                f"critical member SHA-256 mismatch for {member}: "
                f"expected {expected_digest}, received {by_path[member]}"
            )

    return VerifiedSourceArchive(
        archive_path=path,
        archive_size=actual_size,
        archive_sha256=actual_sha256,
        root_name=expected_root,
        file_count=len(records),
        manifest_sha256=manifest_sha256,
        files=records,
    )


def verify_extracted_source_tree(
    source_root: Path,
    verified: VerifiedSourceArchive,
) -> None:
    """Compare every tracked extracted source file with the verified archive."""

    root = source_root.resolve(strict=True)
    if root.name != verified.root_name or not root.is_dir():
        raise CustodyError(
            f"unexpected extracted source root: {root}; expected {verified.root_name}"
        )
    for record in verified.files:
        candidate = root.joinpath(*PurePosixPath(record.path).parts)
        if candidate.is_symlink():
            raise CustodyError(
                f"extracted source member is a symbolic link: {record.path}"
            )
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise CustodyError(
                f"extracted source member escaped its root: {record.path}"
            ) from error
        if not resolved.is_file():
            raise CustodyError(
                f"extracted source member is not a regular file: {record.path}"
            )
        if (
            resolved.stat().st_size != record.size
            or _sha256_file(resolved) != record.sha256
        ):
            raise CustodyError(
                f"extracted source member differs from archive: {record.path}"
            )


def safe_extract_verified_archive(
    verified: VerifiedSourceArchive,
    destination: Path,
) -> Path:
    """Safely extract a byte-identical verified archive into an empty directory."""

    if (
        verified.archive_path.stat().st_size != verified.archive_size
        or _sha256_file(verified.archive_path) != verified.archive_sha256
    ):
        raise CustodyError("source archive changed after verification")

    records, directories = _inspect_tar_members(
        verified.archive_path, expected_root=verified.root_name
    )
    if records != verified.files:
        raise CustodyError("source archive members changed after verification")

    target = destination.expanduser().resolve()
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise CustodyError(
                f"extraction destination must be an empty directory: {target}"
            )
    else:
        target.mkdir(parents=True)

    for directory in directories:
        target.joinpath(*PurePosixPath(directory).parts).mkdir(
            parents=True, exist_ok=True
        )

    expected = {record.path: record for record in verified.files}
    with tarfile.open(verified.archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            member_path = _validated_member_path(
                member.name, verified.root_name, is_dir=False
            )
            relative = PurePosixPath(*member_path.parts[1:]).as_posix()
            record = expected[relative]
            stream = archive.extractfile(member)
            if stream is None:
                raise CustodyError(f"cannot read regular tar member {member.name!r}")
            payload = stream.read()
            if len(payload) != record.size or _sha256_bytes(payload) != record.sha256:
                raise CustodyError(f"tar member changed during extraction: {relative}")
            output = target.joinpath(*member_path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as destination_stream:
                destination_stream.write(payload)
            output.chmod(member.mode & 0o777)

    source_root = target / verified.root_name
    verify_extracted_source_tree(source_root, verified)
    return source_root


def validate_fixture_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Fail closed unless fixture arrays exactly match the versioned schema."""

    actual_names = set(arrays)
    expected_names = set(FIXTURE_ARRAY_SCHEMA)
    if actual_names != expected_names:
        raise FixtureSchemaError(
            "fixture array names differ from the exact schema: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}"
        )
    for name, expected_shape in FIXTURE_ARRAY_SCHEMA.items():
        value = np.asarray(arrays[name])
        if value.shape != expected_shape:
            raise FixtureSchemaError(
                f"fixture array {name!r} has shape {value.shape}; "
                f"expected {expected_shape}"
            )
        if value.dtype != np.dtype(np.float64):
            raise FixtureSchemaError(
                f"fixture array {name!r} must have dtype float64; received {value.dtype}"
            )
        if np.any(~np.isfinite(value)):
            raise FixtureSchemaError(f"fixture array {name!r} must be finite")


def assert_reference_agreement(
    name: str,
    bovy_value: np.ndarray,
    oracle_value: np.ndarray,
    *,
    log_quantity: bool = False,
) -> None:
    """Apply the matrix tolerances before any fixture bytes are written."""

    actual = np.asarray(bovy_value, dtype=np.float64)
    expected = np.asarray(oracle_value, dtype=np.float64)
    if actual.shape != expected.shape or not np.allclose(
        actual,
        expected,
        rtol=REFERENCE_RTOL,
        atol=REFERENCE_ATOL,
    ):
        maximum = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape and actual.size
            else float("inf")
        )
        raise ReferenceDisagreement(
            f"{name} parameter mismatch between Bovy and oracle; "
            f"maximum absolute error {maximum}"
        )
    if log_quantity and actual.size:
        maximum_log_error = float(np.max(np.abs(actual - expected)))
        if maximum_log_error > MAX_ABSOLUTE_LOG_ERROR:
            raise ReferenceDisagreement(
                f"{name} log error {maximum_log_error} exceeds "
                f"{MAX_ABSOLUTE_LOG_ERROR}"
            )


def build_inputs() -> tuple[np.ndarray, ...]:
    """Build the deterministic ordinary ``N=19,K=3,D=4,M=2`` workload."""

    rng = np.random.Generator(np.random.PCG64(SEED))
    initial_weights = np.asarray([0.24, 0.41, 0.35], dtype=np.float64)
    initial_means = np.asarray(
        [
            [-1.10, 0.45, -0.30, 0.75],
            [0.20, -0.80, 0.95, -0.25],
            [1.25, 0.65, -0.55, 0.10],
        ],
        dtype=np.float64,
    )
    factors = np.asarray(
        [
            [
                [0.90, 0, 0, 0],
                [0.12, 0.75, 0, 0],
                [-0.08, 0.10, 0.68, 0],
                [0.05, -0.07, 0.09, 0.82],
            ],
            [
                [0.78, 0, 0, 0],
                [-0.10, 0.92, 0, 0],
                [0.06, 0.12, 0.73, 0],
                [-0.04, 0.08, -0.11, 0.88],
            ],
            [
                [0.84, 0, 0, 0],
                [0.09, 0.81, 0, 0],
                [0.13, -0.06, 0.89, 0],
                [0.07, 0.04, 0.10, 0.71],
            ],
        ],
        dtype=np.float64,
    )
    initial_covariances = factors @ np.swapaxes(factors, -1, -2)

    base_projection = np.asarray(
        [[1.00, 0.25, -0.35, 0.15], [-0.20, 0.80, 0.30, -0.45]],
        dtype=np.float64,
    )
    projection_matrices = base_projection + rng.normal(
        scale=0.12,
        size=(N_SAMPLES, OBSERVED_DIMENSION, LATENT_DIMENSION),
    )
    measurement_covariances = np.empty(
        (N_SAMPLES, OBSERVED_DIMENSION, OBSERVED_DIMENSION), dtype=np.float64
    )
    for sample in range(N_SAMPLES):
        variances = rng.uniform(0.04, 0.30, size=OBSERVED_DIMENSION)
        correlation = rng.uniform(0.12, 0.48) * (-1.0 if sample % 2 else 1.0)
        off_diagonal = correlation * np.sqrt(variances[0] * variances[1])
        measurement_covariances[sample] = np.asarray(
            [[variances[0], off_diagonal], [off_diagonal, variances[1]]],
            dtype=np.float64,
        )

    labels = np.arange(N_SAMPLES) % N_COMPONENTS
    rng.shuffle(labels)
    latent = np.stack(
        [
            rng.multivariate_normal(
                initial_means[component], initial_covariances[component]
            )
            for component in labels
        ]
    )
    observations = np.empty((N_SAMPLES, OBSERVED_DIMENSION), dtype=np.float64)
    for sample in range(N_SAMPLES):
        observations[sample] = projection_matrices[sample] @ latent[
            sample
        ] + rng.multivariate_normal(
            np.zeros(OBSERVED_DIMENSION), measurement_covariances[sample]
        )
    sample_weight = np.ones(N_SAMPLES, dtype=np.float64)

    values = (
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight,
        initial_weights,
        initial_means,
        initial_covariances,
    )
    if any(np.any(~np.isfinite(value)) for value in values):
        raise RuntimeError("deterministic Bovy workload contains nonfinite input")
    if not np.allclose(initial_weights.sum(), 1.0, rtol=0.0, atol=5e-13):
        raise RuntimeError("deterministic Bovy weights are not normalized")
    if any(
        np.linalg.matrix_rank(matrix) != OBSERVED_DIMENSION
        for matrix in projection_matrices
    ):
        raise RuntimeError("deterministic Bovy projection is not full row rank")
    if any(np.linalg.eigvalsh(matrix).min() <= 0.0 for matrix in initial_covariances):
        raise RuntimeError("deterministic Bovy model covariance is not SPD")
    if any(
        np.linalg.eigvalsh(matrix).min() < 0.0 for matrix in measurement_covariances
    ):
        raise RuntimeError("deterministic Bovy noise covariance is not PSD")
    return values


def _oracle_diagnostics(
    observations: np.ndarray,
    projection_matrices: np.ndarray,
    measurement_covariances: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    effective_covariances = np.empty(
        (
            N_SAMPLES,
            N_COMPONENTS,
            OBSERVED_DIMENSION,
            OBSERVED_DIMENSION,
        ),
        dtype=np.float64,
    )
    residuals = np.empty(
        (N_SAMPLES, N_COMPONENTS, OBSERVED_DIMENSION), dtype=np.float64
    )
    gains = np.empty(
        (
            N_SAMPLES,
            N_COMPONENTS,
            LATENT_DIMENSION,
            OBSERVED_DIMENSION,
        ),
        dtype=np.float64,
    )
    maximum_condition = 0.0
    for sample in range(N_SAMPLES):
        projection = projection_matrices[sample]
        for component in range(N_COMPONENTS):
            covariance = covariances[component]
            total = (
                projection @ covariance @ projection.T + measurement_covariances[sample]
            )
            residual = observations[sample] - projection @ means[component]
            gain = np.linalg.solve(total, projection @ covariance).T
            effective_covariances[sample, component] = total
            residuals[sample, component] = residual
            gains[sample, component] = gain
            maximum_condition = max(maximum_condition, float(np.linalg.cond(total)))
    return effective_covariances, residuals, gains, maximum_condition


def _score_rows_with_bovy(
    reference: Any,
    observations: np.ndarray,
    measurement_covariances: np.ndarray,
    projection_matrices: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> np.ndarray:
    scores = np.empty(N_SAMPLES, dtype=np.float64)
    for sample in range(N_SAMPLES):
        scores[sample] = reference(
            observations[sample : sample + 1],
            measurement_covariances[sample : sample + 1],
            weights.copy(),
            means.copy(),
            covariances.copy(),
            projection=projection_matrices[sample : sample + 1],
            tol=1e-12,
            maxiter=1,
            w=0.0,
            splitnmerge=0,
            likeonly=True,
        )
    return scores


def _run_checked(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        env=None if env is None else dict(env),
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "command": list(command),
        "cwd": None if cwd is None else str(cwd.resolve()),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        raise RuntimeError(
            f"reference command failed ({completed.returncode}): {list(command)!r}\n"
            f"{completed.stderr}"
        )
    return record


def _authoritative_environment() -> dict[str, Any]:
    expected_environment = {
        "XDGMM_BOVY_REFERENCE_CONTAINER": "1",
        "XDGMM_BOVY_BASE_AMD64_DIGEST": DEBIAN_AMD64_MANIFEST_DIGEST,
        "XDGMM_BOVY_BASE_INDEX_DIGEST": DEBIAN_MULTIARCH_INDEX_DIGEST,
        "XDGMM_BOVY_DEBIAN_SNAPSHOT": DEBIAN_SNAPSHOT,
        "XDGMM_BOVY_RUNTIME_NETWORK": "none",
        "OMP_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "TZ": "UTC",
    }
    for name, expected in expected_environment.items():
        if os.environ.get(name) != expected:
            raise BuildEnvironmentError(
                f"authoritative generation requires {name}={expected!r}"
            )
    image_id = os.environ.get("XDGMM_BOVY_CONTAINER_IMAGE_ID", "")
    if not image_id.startswith("sha256:"):
        raise BuildEnvironmentError(
            "authoritative generation requires a recorded container image ID"
        )
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise BuildEnvironmentError(
            "authoritative Bovy generation requires Linux/amd64"
        )

    package_versions: dict[str, str] = {}
    for package, expected_version in EXPECTED_DEBIAN_PACKAGES.items():
        completed = subprocess.run(
            ["dpkg-query", "-W", "-f=${Version}", package],
            capture_output=True,
            text=True,
            check=False,
        )
        actual_version = completed.stdout.strip()
        if completed.returncode != 0 or actual_version != expected_version:
            raise BuildEnvironmentError(
                f"expected Debian package {package}={expected_version}; "
                f"received {actual_version or 'unavailable'}"
            )
        package_versions[package] = actual_version

    manifest = _run_checked(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"])[
        "stdout"
    ].splitlines()
    return {
        "qualified": True,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_full": sys.version.splitlines()[0],
        "numpy": np.__version__,
        "container_image_id": image_id,
        "base_amd64_manifest_digest": DEBIAN_AMD64_MANIFEST_DIGEST,
        "base_multiarch_index_digest": DEBIAN_MULTIARCH_INDEX_DIGEST,
        "debian_snapshot": DEBIAN_SNAPSHOT,
        "runtime_network": "none",
        "required_packages": package_versions,
        "resolved_package_manifest": manifest,
        "thread_controls": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OMP_DYNAMIC",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "locale": os.environ.get("LC_ALL"),
        "language": os.environ.get("LANG"),
        "timezone": os.environ.get("TZ"),
    }


def _load_reference_module(import_root: Path) -> tuple[Any, dict[str, str]]:
    wrapper = (import_root / "extreme_deconvolution.py").resolve(strict=True)
    module_name = "_xdgmm_pinned_bovy_reference"
    if module_name in sys.modules:
        raise RuntimeError("pinned Bovy reference module was imported more than once")
    specification = importlib.util.spec_from_file_location(module_name, wrapper)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot construct import specification for {wrapper}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    actual_wrapper = Path(module.__file__).resolve(strict=True)
    actual_library = Path(module._lib._name).resolve(strict=True)
    import_root_resolved = import_root.resolve(strict=True)
    for value in (actual_wrapper, actual_library):
        try:
            value.relative_to(import_root_resolved)
        except ValueError as error:
            raise RuntimeError(
                f"Bovy reference import escaped the isolated root: {value}"
            ) from error
    return module.extreme_deconvolution, {
        "wrapper_path": str(actual_wrapper),
        "wrapper_sha256": _sha256_file(actual_wrapper),
        "library_path": str(actual_library),
        "library_sha256": _sha256_file(actual_library),
    }


def _build_reference(
    source_root: Path,
    verified: VerifiedSourceArchive,
) -> tuple[Any, dict[str, Any]]:
    build_environment = os.environ.copy()
    build_environment.update(
        {
            "CC": "gcc",
            "CFLAGS": "-O2",
            "LDFLAGS": "",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
        }
    )
    command = ["make", "-j1", "CC=gcc", "CFLAGS=-O2", "LDFLAGS="]
    build_command = _run_checked(command, cwd=source_root, env=build_environment)
    verify_extracted_source_tree(source_root, verified)

    built_library = (source_root / "build" / "libextremedeconvolution.so").resolve(
        strict=True
    )
    import_root = source_root.parent / "isolated-reference-import"
    import_root.mkdir()
    isolated_library = import_root / built_library.name
    shutil.copyfile(built_library, isolated_library)

    template = source_root / "py" / "extreme_deconvolution_TEMPLATE.py"
    template_text = template.read_text(encoding="utf-8")
    if template_text.count("TEMPLATE_LIBRARY_PATH") != 1:
        raise RuntimeError("unexpected Bovy wrapper-template substitution count")
    generated_wrapper = template_text.replace(
        "TEMPLATE_LIBRARY_PATH", repr(str(import_root.resolve()))
    )
    wrapper_path = import_root / "extreme_deconvolution.py"
    wrapper_path.write_text(generated_wrapper, encoding="utf-8")

    reference, import_record = _load_reference_module(import_root)
    tool_records = {
        name: _run_checked(command_values)["stdout"].splitlines()
        for name, command_values in (
            ("gcc", ["gcc", "--version"]),
            ("make", ["make", "--version"]),
            ("gsl", ["gsl-config", "--version"]),
            ("ldd", ["ldd", "--version"]),
        )
    }
    dynamic_libraries = _run_checked(["ldd", str(isolated_library)])[
        "stdout"
    ].splitlines()
    return reference, {
        "build_command": build_command,
        "effective_build_flags": {
            "CC": "gcc",
            "CFLAGS": "-O2 -fopenmp -fcommon",
            "LDFLAGS": "-fopenmp -fcommon",
            "fast_math": False,
            "make_parallelism": 1,
        },
        "tool_versions": tool_records,
        "dynamic_libraries": dynamic_libraries,
        "upstream_built_library_sha256": _sha256_file(built_library),
        "import": import_record,
        "generated_wrapper": {
            "source_template": "py/extreme_deconvolution_TEMPLATE.py",
            "source_template_sha256": BOVY_MEMBER_SHA256[
                "py/extreme_deconvolution_TEMPLATE.py"
            ],
            "operation": (
                "replace the single TEMPLATE_LIBRARY_PATH token with the "
                "temporary isolated import directory"
            ),
        },
    }


def _reference_and_oracle_arrays(
    reference: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    (
        observations,
        projection_matrices,
        measurement_covariances,
        sample_weight,
        initial_weights,
        initial_means,
        initial_covariances,
    ) = build_inputs()

    oracle_e_step = general_e_step(
        observations,
        projection_matrices,
        measurement_covariances,
        initial_weights,
        initial_means,
        initial_covariances,
    )
    oracle_parameters, oracle_statistics = general_m_step(oracle_e_step)
    oracle_candidate = general_e_step(
        observations,
        projection_matrices,
        measurement_covariances,
        oracle_parameters.weights,
        oracle_parameters.means,
        oracle_parameters.covariances,
    )
    effective_covariances, residuals, gains, maximum_condition = _oracle_diagnostics(
        observations,
        projection_matrices,
        measurement_covariances,
        initial_means,
        initial_covariances,
    )
    if maximum_condition > MAX_EFFECTIVE_CONDITION:
        raise RuntimeError(
            f"deterministic workload condition {maximum_condition} exceeds "
            f"{MAX_EFFECTIVE_CONDITION}"
        )
    maximum_component_log = float(np.max(np.abs(oracle_e_step.component_log_density)))
    if maximum_component_log >= MAX_ABSOLUTE_COMPONENT_LOG_DENSITY:
        raise RuntimeError(
            f"deterministic component log magnitude {maximum_component_log} is too large"
        )

    bovy_initial_scores = _score_rows_with_bovy(
        reference,
        observations,
        measurement_covariances,
        projection_matrices,
        initial_weights,
        initial_means,
        initial_covariances,
    )
    updated_weights = initial_weights.copy()
    updated_means = initial_means.copy()
    updated_covariances = initial_covariances.copy()
    returned_preupdate_mean = np.float64(
        reference(
            observations,
            measurement_covariances,
            updated_weights,
            updated_means,
            updated_covariances,
            projection=projection_matrices,
            tol=1e-12,
            maxiter=1,
            w=0.0,
            splitnmerge=0,
            likeonly=False,
        )
    )
    bovy_candidate_scores = _score_rows_with_bovy(
        reference,
        observations,
        measurement_covariances,
        projection_matrices,
        updated_weights,
        updated_means,
        updated_covariances,
    )

    bovy_initial_total = np.asarray(bovy_initial_scores.sum(), dtype=np.float64)
    bovy_initial_mean = np.asarray(bovy_initial_scores.mean(), dtype=np.float64)
    bovy_candidate_total = np.asarray(bovy_candidate_scores.sum(), dtype=np.float64)
    bovy_candidate_mean = np.asarray(bovy_candidate_scores.mean(), dtype=np.float64)
    oracle_initial_total = np.asarray(
        oracle_e_step.score_samples.sum(), dtype=np.float64
    )
    oracle_initial_mean = np.asarray(
        oracle_e_step.score_samples.mean(), dtype=np.float64
    )
    oracle_candidate_total = np.asarray(
        oracle_candidate.score_samples.sum(), dtype=np.float64
    )
    oracle_candidate_mean = np.asarray(
        oracle_candidate.score_samples.mean(), dtype=np.float64
    )

    for name, actual, expected, is_log in (
        (
            "initial row log scores",
            bovy_initial_scores,
            oracle_e_step.score_samples,
            True,
        ),
        (
            "initial total log likelihood",
            bovy_initial_total,
            oracle_initial_total,
            True,
        ),
        ("initial mean objective", bovy_initial_mean, oracle_initial_mean, True),
        (
            "returned pre-update mean objective",
            returned_preupdate_mean,
            oracle_initial_mean,
            True,
        ),
        ("updated weights", updated_weights, oracle_parameters.weights, False),
        ("updated means", updated_means, oracle_parameters.means, False),
        (
            "updated covariances",
            updated_covariances,
            oracle_parameters.covariances,
            False,
        ),
        (
            "candidate row log scores",
            bovy_candidate_scores,
            oracle_candidate.score_samples,
            True,
        ),
        (
            "candidate total log likelihood",
            bovy_candidate_total,
            oracle_candidate_total,
            True,
        ),
        ("candidate mean objective", bovy_candidate_mean, oracle_candidate_mean, True),
    ):
        assert_reference_agreement(name, actual, expected, log_quantity=is_log)

    arrays = {
        "observations": observations,
        "projection_matrices": projection_matrices,
        "measurement_covariances": measurement_covariances,
        "sample_weight": sample_weight,
        "initial_weights": initial_weights,
        "initial_means": initial_means,
        "initial_covariances": initial_covariances,
        "bovy_initial_score_samples": bovy_initial_scores,
        "bovy_initial_log_likelihood": bovy_initial_total,
        "bovy_initial_mean_objective": bovy_initial_mean,
        "bovy_returned_preupdate_mean_objective": np.asarray(
            returned_preupdate_mean, dtype=np.float64
        ),
        "bovy_updated_weights": np.asarray(updated_weights, dtype=np.float64),
        "bovy_updated_means": np.asarray(updated_means, dtype=np.float64),
        "bovy_updated_covariances": np.asarray(updated_covariances, dtype=np.float64),
        "bovy_candidate_score_samples": bovy_candidate_scores,
        "bovy_candidate_log_likelihood": bovy_candidate_total,
        "bovy_candidate_mean_objective": bovy_candidate_mean,
        "oracle_effective_covariances": effective_covariances,
        "oracle_residuals": residuals,
        "oracle_gains": gains,
        "oracle_component_log_density": oracle_e_step.component_log_density,
        "oracle_component_log_joint": oracle_e_step.component_log_joint,
        "oracle_score_samples": oracle_e_step.score_samples,
        "oracle_responsibilities": oracle_e_step.responsibilities,
        "oracle_conditional_mean": oracle_e_step.conditional_mean,
        "oracle_conditional_covariance": oracle_e_step.conditional_covariance,
        "oracle_stat_mass": oracle_statistics.mass,
        "oracle_stat_first_moment": oracle_statistics.first_moment,
        "oracle_stat_second_moment": oracle_statistics.second_moment,
        "oracle_updated_weights": oracle_parameters.weights,
        "oracle_updated_means": oracle_parameters.means,
        "oracle_updated_covariances": oracle_parameters.covariances,
        "oracle_candidate_component_log_density": oracle_candidate.component_log_density,
        "oracle_candidate_component_log_joint": oracle_candidate.component_log_joint,
        "oracle_candidate_score_samples": oracle_candidate.score_samples,
        "oracle_candidate_log_likelihood": oracle_candidate_total,
        "oracle_candidate_mean_objective": oracle_candidate_mean,
    }
    arrays = {
        name: np.asarray(value, dtype=np.float64) for name, value in arrays.items()
    }
    validate_fixture_arrays(arrays)
    return arrays, {
        "maximum_effective_covariance_condition_2": maximum_condition,
        "maximum_absolute_component_log_density": maximum_component_log,
    }


def _repository_file_record(path: Path) -> dict[str, str]:
    project_root = PROJECT_ROOT.resolve(strict=True)
    resolved = path.resolve(strict=True)
    return {
        "path": resolved.relative_to(project_root).as_posix(),
        "actual_path": str(resolved),
        "sha256": _sha256_file(resolved),
    }


def _run_source_paths() -> dict[str, Path]:
    """Resolve run-input locations dynamically for testable source custody."""

    return {
        "generator": Path(__file__),
        "archive_helper": PROJECT_ROOT / "scripts" / "deterministic_npz.py",
        "independent_oracle": (PROJECT_ROOT / "tests" / "reference" / "general_xd.py"),
    }


def _capture_run_source_records() -> dict[str, dict[str, str]]:
    """Capture the paths and bytes of repository code used by this run."""

    return {
        name: _repository_file_record(path)
        for name, path in _run_source_paths().items()
    }


def _verify_run_source_records(
    captured: Mapping[str, Mapping[str, str]],
) -> None:
    """Fail if a captured run input moved or changed after work began."""

    paths = _run_source_paths()
    if set(captured) != set(paths):
        raise CustodyError("captured repository run-source set changed")
    for name, path in paths.items():
        try:
            current = _repository_file_record(path)
        except (OSError, ValueError) as error:
            raise CustodyError(
                f"repository run source {name!r} changed or became unavailable"
            ) from error
        if current != dict(captured[name]):
            raise CustodyError(f"repository run source {name!r} changed after capture")


def _new_staged_path(destination: Path) -> Path:
    """Create a private sibling file for same-filesystem atomic publication."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


@contextmanager
def _staged_output_pair(fixture: Path, metadata: Path) -> Iterator[tuple[Path, Path]]:
    """Provide private sibling files and remove them if staging is interrupted."""

    staged_fixture: Path | None = None
    staged_metadata: Path | None = None
    try:
        staged_fixture = _new_staged_path(fixture)
        staged_metadata = _new_staged_path(metadata)
        yield staged_fixture, staged_metadata
    finally:
        if staged_fixture is not None:
            staged_fixture.unlink(missing_ok=True)
        if staged_metadata is not None:
            staged_metadata.unlink(missing_ok=True)


def _publish_staged_pair(
    *,
    staged_fixture: Path,
    fixture: Path,
    staged_metadata: Path,
    metadata: Path,
) -> None:
    """Atomically publish metadata first and its numeric fixture last.

    Hard-link publication is atomic, stays on the destination filesystem, and
    fails rather than replacing a raced or pre-existing output.  Publishing the
    numeric archive last means an interruption cannot leave that archive without
    its custody metadata.  A failed second link rolls the metadata link back.
    Staging names are removed on every exit path.
    """

    staged_pairs = (
        (staged_fixture, fixture, "fixture"),
        (staged_metadata, metadata, "metadata"),
    )
    published_metadata = False
    try:
        for staged, destination, label in staged_pairs:
            if staged.parent.resolve() != destination.parent.resolve():
                raise ValueError(f"staged {label} must be a sibling of its destination")
            if staged.is_symlink() or not staged.is_file():
                raise ValueError(f"staged {label} is not a regular file: {staged}")
        if fixture.exists() or metadata.exists():
            raise FileExistsError(
                "refusing to overwrite an existing fixture or metadata record"
            )

        os.link(staged_metadata, metadata)
        published_metadata = True
        os.link(staged_fixture, fixture)
    except BaseException:
        if published_metadata:
            try:
                metadata.unlink()
            except OSError as cleanup_error:
                raise RuntimeError(
                    "fixture publication failed and metadata rollback also failed"
                ) from cleanup_error
        raise
    finally:
        staged_fixture.unlink(missing_ok=True)
        staged_metadata.unlink(missing_ok=True)


def _source_record(verified: VerifiedSourceArchive) -> dict[str, Any]:
    return {
        "repository": "https://github.com/jobovy/extreme-deconvolution",
        "commit": BOVY_COMMIT,
        "git_tree": BOVY_TREE,
        "archive_path": str(verified.archive_path),
        "archive_size": verified.archive_size,
        "archive_sha256": verified.archive_sha256,
        "archive_root": verified.root_name,
        "regular_file_count": verified.file_count,
        "manifest_sha256": verified.manifest_sha256,
        "manifest_format": BOVY_SOURCE_MANIFEST_FORMAT,
        "critical_member_sha256": dict(sorted(BOVY_MEMBER_SHA256.items())),
        "license_spdx": "BSD-3-Clause",
        "license_notice": "THIRD_PARTY_NOTICES.md",
        "endorsement": "none",
    }


def generate_fixture(
    verified: VerifiedSourceArchive,
    *,
    output: Path,
    metadata_output: Path,
) -> None:
    """Build the isolated reference, cross-check it, then write exact evidence."""

    run_source_records = _capture_run_source_records()
    qualified_environment = _authoritative_environment()
    output = output.expanduser().resolve()
    metadata_output = metadata_output.expanduser().resolve()
    if output == metadata_output:
        raise ValueError("fixture and metadata outputs must be different paths")
    if output.exists() or metadata_output.exists():
        raise FileExistsError(
            "refusing to overwrite an existing fixture or metadata record"
        )

    with tempfile.TemporaryDirectory(prefix="xdgmm-bovy-reference-") as temporary:
        extraction = Path(temporary) / "source"
        source_root = safe_extract_verified_archive(verified, extraction)
        reference, build_record = _build_reference(source_root, verified)
        arrays, workload_record = _reference_and_oracle_arrays(reference)

        with _staged_output_pair(output, metadata_output) as (
            staged_output,
            staged_metadata,
        ):
            array_hashes = write_deterministic_npz(staged_output, arrays)
            archive_sha256 = _sha256_file(staged_output)
            array_manifest = json.dumps(
                dict(sorted(array_hashes.items())),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

            metadata = {
                "fixture_id": "xd-gen-ref-bovy-001",
                "fixture_version": 1,
                "status": "authoritative pinned CPU reference evidence",
                "contract_id": "xdgmm-jax.general-xd",
                "contract_version": "0.2.0-draft.1",
                "matrix_id": "xdgmm-jax.general-xd.matrix",
                "matrix_version": "0.2.0-draft.1",
                "matrix_row": "XD-GEN-REF-BOVY-001",
                "archive": output.name,
                "archive_sha256": archive_sha256,
                "archive_format": "NumPy NPZ / ZIP",
                "archive_compression": "ZIP_STORED",
                "archive_npy_version": "1.0",
                "array_npy_sha256": dict(sorted(array_hashes.items())),
                "array_manifest_sha256": _sha256_bytes(array_manifest),
                "stored_arrays": sorted(arrays),
                "array_schema": {
                    name: list(shape) for name, shape in FIXTURE_ARRAY_SCHEMA.items()
                },
                "dtype": "float64",
                "n_samples": N_SAMPLES,
                "n_components": N_COMPONENTS,
                "latent_dimension": LATENT_DIMENSION,
                "observed_dimension": OBSERVED_DIMENSION,
                "sample_weight": "unit weights; Bovy weight argument omitted",
                "seed": SEED,
                "bit_generator": "PCG64",
                "reference_tolerances": {
                    "rtol": REFERENCE_RTOL,
                    "atol": REFERENCE_ATOL,
                    "maximum_absolute_log_error": MAX_ABSOLUTE_LOG_ERROR,
                },
                "workload": workload_record,
                "source": _source_record(verified),
                "build": build_record,
                "generation_environment": qualified_environment,
                "generator": run_source_records["generator"],
                "archive_helper": run_source_records["archive_helper"],
                "independent_oracle": run_source_records["independent_oracle"],
                "reference_controls": {
                    "tol": 1e-12,
                    "maxiter": 1,
                    "likeonly_for_row_scores": True,
                    "weight": None,
                    "fixed_parameters": None,
                    "prior_regularization_w": 0.0,
                    "splitnmerge": 0,
                    "omp_threads": 1,
                },
                "evidence_scope": {
                    "direct_bovy": sorted(
                        name for name in arrays if name.startswith("bovy_")
                    ),
                    "independent_oracle": sorted(
                        name for name in arrays if name.startswith("oracle_")
                    ),
                    "interpretation": (
                        "Composite evidence: Bovy supplies initial/candidate observed "
                        "likelihood endpoints and one-step parameters; the independent "
                        "NumPy oracle supplies intermediate quantities. Fixture writing "
                        "is conditional on agreement at every shared endpoint."
                    ),
                },
                "distribution_policy": (
                    "Only deterministic numeric arrays and custody metadata are stored. "
                    "No Bovy/GSL source, shared library, or container image is distributed."
                ),
                "endorsement": "No endorsement by upstream authors is claimed.",
            }
            staged_metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staged_output.chmod(0o644)
            staged_metadata.chmod(0o644)
            _verify_run_source_records(run_source_records)
            _publish_staged_pair(
                staged_fixture=staged_output,
                fixture=output,
                staged_metadata=staged_metadata,
                metadata=metadata_output,
            )

    print(f"wrote {output}")
    print(f"sha256 {archive_sha256}")
    print(f"wrote {metadata_output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        required=True,
        type=Path,
        help="externally supplied official archive for the pinned Bovy commit",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify source custody without building or writing numeric evidence",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    return parser


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        verified = verify_source_archive(arguments.source_archive)
        if arguments.verify_only:
            print(f"verified {verified.archive_path}")
            print(f"sha256 {verified.archive_sha256}")
            print(f"manifest {verified.manifest_sha256} ({verified.file_count} files)")
            return
        if arguments.output is None:
            parser.error("--output is required unless --verify-only is used")
        metadata_output = arguments.metadata_output
        if metadata_output is None:
            metadata_output = arguments.output.with_suffix(".metadata.json")
        generate_fixture(
            verified,
            output=arguments.output,
            metadata_output=metadata_output,
        )
    except (
        BuildEnvironmentError,
        CustodyError,
        FileExistsError,
        FixtureSchemaError,
        OSError,
        ReferenceDisagreement,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
