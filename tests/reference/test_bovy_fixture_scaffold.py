"""Custody and schema tests for the pending Bovy parity fixture scaffold."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile

import numpy as np
import pytest

from scripts import generate_bovy_parity_fixture as generator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_bovy_parity_fixture.py"
DOCKERFILE_PATH = PROJECT_ROOT / "scripts" / "reference" / "bovy" / "Dockerfile"
INVOCATION_PATH = PROJECT_ROOT / "scripts" / "reference" / "bovy" / "generate.sh"


def _tar_bytes(
    entries: list[tuple[str, bytes, str]],
) -> bytes:
    """Return a tiny gzip-compressed tar for local custody tests.

    ``kind`` is ``file``, ``dir``, ``symlink``, or ``hardlink``.
    """

    payload = io.BytesIO()
    with tarfile.open(
        fileobj=payload, mode="w:gz", format=tarfile.PAX_FORMAT
    ) as archive:
        for name, value, kind in entries:
            member = tarfile.TarInfo(name)
            member.mtime = 0
            if kind == "dir":
                member.type = tarfile.DIRTYPE
                member.mode = 0o755
                archive.addfile(member)
            elif kind in {"symlink", "hardlink"}:
                member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
                member.linkname = value.decode("utf-8")
                archive.addfile(member)
            else:
                member.type = tarfile.REGTYPE
                member.mode = 0o644
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    return payload.getvalue()


def _write_archive(path: Path, entries: list[tuple[str, bytes, str]]) -> bytes:
    payload = _tar_bytes(entries)
    path.write_bytes(payload)
    return payload


def _manifest_digest(files: dict[str, bytes]) -> str:
    lines = [
        f"{hashlib.sha256(value).hexdigest()}  {name}\n"
        for name, value in files.items()
    ]
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8")).hexdigest()


def _verified_tiny_archive(tmp_path: Path):
    root = "reference-deadbeef"
    files = {
        "LICENSE": b"test license\n",
        "src/core.c": b"int answer(void) { return 42; }\n",
    }
    entries = [(root + "/", b"", "dir")]
    entries.extend((f"{root}/{name}", value, "file") for name, value in files.items())
    archive_path = tmp_path / "source.tar.gz"
    payload = _write_archive(archive_path, entries)
    critical = {
        name: hashlib.sha256(value).hexdigest() for name, value in files.items()
    }
    verified = generator.verify_source_archive(
        archive_path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        expected_root=root,
        expected_file_count=len(files),
        expected_manifest_sha256=_manifest_digest(files),
        expected_member_sha256=critical,
    )
    return verified, files


def test_tiny_archive_sha_manifest_and_safe_extraction_round_trip(tmp_path: Path):
    verified, files = _verified_tiny_archive(tmp_path)

    assert verified.file_count == 2
    assert verified.root_name == "reference-deadbeef"
    assert verified.manifest_sha256 == _manifest_digest(files)
    assert [record.path for record in verified.files] == sorted(files)

    destination = tmp_path / "extract"
    root = generator.safe_extract_verified_archive(verified, destination)
    assert root == destination / verified.root_name
    assert {
        record.path: (root / record.path).read_bytes() for record in verified.files
    } == files

    # The extracted tree is checked independently against archive custody.
    generator.verify_extracted_source_tree(root, verified)


def test_archive_digest_is_checked_before_tar_parsing(tmp_path: Path):
    archive_path = tmp_path / "not-even-a-tar.tar.gz"
    archive_path.write_bytes(b"not a tar")

    with pytest.raises(generator.CustodyError, match="archive SHA-256 mismatch"):
        generator.verify_source_archive(
            archive_path,
            expected_sha256="0" * 64,
            expected_size=len(b"not a tar"),
            expected_root="root",
            expected_file_count=0,
            expected_manifest_sha256=hashlib.sha256(b"").hexdigest(),
            expected_member_sha256={},
        )


@pytest.mark.parametrize(
    ("bad_name", "kind", "link_target", "message"),
    [
        ("/absolute", "file", b"payload", "absolute"),
        ("root/../escape", "file", b"payload", "unsafe"),
        ("other/file", "file", b"payload", "single root"),
        ("root/link", "symlink", b"../../escape", "unsupported tar member"),
        ("root/hard", "hardlink", b"root/file", "unsupported tar member"),
    ],
)
def test_unsafe_or_multi_root_archives_are_rejected_before_extraction(
    tmp_path: Path,
    bad_name: str,
    kind: str,
    link_target: bytes,
    message: str,
):
    entries = [
        ("root/", b"", "dir"),
        ("root/file", b"ok", "file"),
        (bad_name, link_target, kind),
    ]
    archive_path = tmp_path / "malicious.tar.gz"
    payload = _write_archive(archive_path, entries)

    with pytest.raises(generator.CustodyError, match=message):
        generator.verify_source_archive(
            archive_path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            expected_root="root",
            expected_file_count=2,
            expected_manifest_sha256="unused",
            expected_member_sha256={},
        )
    assert not (tmp_path / "escape").exists()


def test_duplicate_tar_member_is_rejected(tmp_path: Path):
    entries = [
        ("root/", b"", "dir"),
        ("root/file", b"first", "file"),
        ("root/file", b"second", "file"),
    ]
    archive_path = tmp_path / "duplicate.tar.gz"
    payload = _write_archive(archive_path, entries)

    with pytest.raises(generator.CustodyError, match="duplicate tar member"):
        generator.verify_source_archive(
            archive_path,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            expected_root="root",
            expected_file_count=2,
            expected_manifest_sha256="unused",
            expected_member_sha256={},
        )


def test_critical_member_hash_and_presence_are_enforced(tmp_path: Path):
    verified, files = _verified_tiny_archive(tmp_path)

    with pytest.raises(generator.CustodyError, match="critical member SHA-256"):
        generator.verify_source_archive(
            verified.archive_path,
            expected_sha256=verified.archive_sha256,
            expected_size=verified.archive_size,
            expected_root=verified.root_name,
            expected_file_count=verified.file_count,
            expected_manifest_sha256=verified.manifest_sha256,
            expected_member_sha256={"LICENSE": "0" * 64},
        )
    with pytest.raises(generator.CustodyError, match="missing critical member"):
        generator.verify_source_archive(
            verified.archive_path,
            expected_sha256=verified.archive_sha256,
            expected_size=verified.archive_size,
            expected_root=verified.root_name,
            expected_file_count=verified.file_count,
            expected_manifest_sha256=verified.manifest_sha256,
            expected_member_sha256={"missing.c": hashlib.sha256(b"").hexdigest()},
        )
    assert files["LICENSE"] == b"test license\n"


def test_extracted_member_tampering_and_symlink_substitution_are_rejected(
    tmp_path: Path,
):
    verified, _ = _verified_tiny_archive(tmp_path)
    root = generator.safe_extract_verified_archive(verified, tmp_path / "extract")
    core = root / "src" / "core.c"
    core.write_bytes(b"changed after extraction\n")
    with pytest.raises(generator.CustodyError, match="differs from archive"):
        generator.verify_extracted_source_tree(root, verified)

    core.unlink()
    core.symlink_to(root / "LICENSE")
    with pytest.raises(generator.CustodyError, match="symbolic link"):
        generator.verify_extracted_source_tree(root, verified)


def _valid_schema_arrays() -> dict[str, np.ndarray]:
    return {
        name: np.zeros(shape, dtype=np.float64)
        for name, shape in generator.FIXTURE_ARRAY_SCHEMA.items()
    }


def test_fixture_schema_accepts_exact_finite_float64_arrays():
    arrays = _valid_schema_arrays()
    generator.validate_fixture_arrays(arrays)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan"])
def test_fixture_schema_fails_closed(mutation: str):
    arrays = _valid_schema_arrays()
    first = next(iter(generator.FIXTURE_ARRAY_SCHEMA))
    if mutation == "missing":
        del arrays[first]
        message = "array names"
    elif mutation == "extra":
        arrays["uncontracted"] = np.asarray(0.0, dtype=np.float64)
        message = "array names"
    elif mutation == "shape":
        arrays[first] = np.zeros((1,), dtype=np.float64)
        message = "shape"
    elif mutation == "dtype":
        arrays[first] = arrays[first].astype(np.float32)
        message = "float64"
    else:
        arrays[first] = arrays[first].copy()
        arrays[first].flat[0] = np.nan
        message = "finite"

    with pytest.raises(generator.FixtureSchemaError, match=message):
        generator.validate_fixture_arrays(arrays)


def test_reference_agreement_checks_allclose_and_absolute_log_error():
    generator.assert_reference_agreement(
        "ordinary parameter", np.asarray([1.0]), np.asarray([1.0 + 1e-10])
    )
    with pytest.raises(generator.ReferenceDisagreement, match="parameter mismatch"):
        generator.assert_reference_agreement(
            "parameter", np.asarray([1.0]), np.asarray([1.1])
        )
    with pytest.raises(generator.ReferenceDisagreement, match="log error"):
        generator.assert_reference_agreement(
            "candidate log score",
            np.asarray([100.0]),
            np.asarray([100.0 + 5.1e-8]),
            log_quantity=True,
        )


def test_cli_refuses_unpinned_archive_without_creating_fixture(tmp_path: Path):
    archive_path = tmp_path / "untrusted.tar.gz"
    archive_path.write_bytes(b"untrusted")
    output = tmp_path / "must-not-exist.npz"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--source-archive",
            str(archive_path),
            "--verify-only",
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "archive" in completed.stderr.lower()
    assert "mismatch" in completed.stderr.lower()
    assert not output.exists()


def test_paired_publish_is_all_or_nothing_and_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fixture = tmp_path / "fixture.npz"
    metadata = tmp_path / "fixture.metadata.json"
    staged_fixture = tmp_path / ".fixture.npz.staged"
    staged_metadata = tmp_path / ".fixture.metadata.json.staged"
    staged_fixture.write_bytes(b"fixture")
    staged_metadata.write_bytes(b"metadata")

    real_link = generator.os.link

    def fail_fixture_publish(source, destination):
        if Path(destination) == fixture:
            raise OSError("injected fixture publish failure")
        return real_link(source, destination)

    monkeypatch.setattr(generator.os, "link", fail_fixture_publish)
    with pytest.raises(OSError, match="injected fixture publish failure"):
        generator._publish_staged_pair(
            staged_fixture=staged_fixture,
            fixture=fixture,
            staged_metadata=staged_metadata,
            metadata=metadata,
        )

    assert not fixture.exists()
    assert not metadata.exists()
    assert not staged_fixture.exists()
    assert not staged_metadata.exists()


def test_staging_context_cleans_unpublished_files_after_write_failure(
    tmp_path: Path,
):
    fixture = tmp_path / "fixture.npz"
    metadata = tmp_path / "fixture.metadata.json"

    with pytest.raises(OSError, match="injected metadata serialization failure"):
        with generator._staged_output_pair(fixture, metadata) as (
            staged_fixture,
            staged_metadata,
        ):
            staged_fixture.write_bytes(b"complete numeric bytes")
            staged_metadata.write_bytes(b"partial metadata")
            raise OSError("injected metadata serialization failure")

    assert not fixture.exists()
    assert not metadata.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_paired_publish_succeeds_without_overwriting(tmp_path: Path):
    fixture = tmp_path / "fixture.npz"
    metadata = tmp_path / "fixture.metadata.json"
    staged_fixture = tmp_path / ".fixture.npz.staged"
    staged_metadata = tmp_path / ".fixture.metadata.json.staged"
    staged_fixture.write_bytes(b"fixture")
    staged_metadata.write_bytes(b"metadata")

    generator._publish_staged_pair(
        staged_fixture=staged_fixture,
        fixture=fixture,
        staged_metadata=staged_metadata,
        metadata=metadata,
    )

    assert fixture.read_bytes() == b"fixture"
    assert metadata.read_bytes() == b"metadata"
    assert not staged_fixture.exists()
    assert not staged_metadata.exists()

    replacement_fixture = tmp_path / ".replacement.npz.staged"
    replacement_metadata = tmp_path / ".replacement.json.staged"
    replacement_fixture.write_bytes(b"replacement fixture")
    replacement_metadata.write_bytes(b"replacement metadata")
    with pytest.raises(FileExistsError):
        generator._publish_staged_pair(
            staged_fixture=replacement_fixture,
            fixture=fixture,
            staged_metadata=replacement_metadata,
            metadata=metadata,
        )
    assert fixture.read_bytes() == b"fixture"
    assert metadata.read_bytes() == b"metadata"
    assert not replacement_fixture.exists()
    assert not replacement_metadata.exists()


def test_container_scaffold_is_pinned_offline_and_does_not_fetch_bovy():
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    invocation = INVOCATION_PATH.read_text(encoding="utf-8")

    assert "debian:bookworm-20260824-slim@sha256:" in dockerfile
    assert generator.DEBIAN_AMD64_MANIFEST_DIGEST in dockerfile
    assert generator.DEBIAN_SNAPSHOT in dockerfile
    for package, version in generator.EXPECTED_DEBIAN_PACKAGES.items():
        assert f"{package}={version}" in dockerfile
    assert "--network none" in invocation
    assert "--security-opt no-new-privileges=true" in invocation
    assert "/input/source.tar.gz" in invocation
    assert "github.com/jobovy" not in dockerfile + invocation
    assert "git clone" not in dockerfile + invocation
    assert "curl " not in dockerfile + invocation

    build_block = invocation.split('"${CONTAINER_ENGINE_COMMAND}" build', maxsplit=1)[
        1
    ].split("REFERENCE_IMAGE_ID=", maxsplit=1)[0]
    assert build_block.rstrip().endswith('"${SCRIPT_DIRECTORY_PATH}"')
    assert '"${PROJECT_ROOT_PATH}"' not in build_block

    syntax = subprocess.run(
        ["bash", "-n", str(INVOCATION_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr


def test_authoritative_environment_rejects_nonpinned_openmp_threads(
    monkeypatch: pytest.MonkeyPatch,
):
    """The hard-coded one-thread evidence claim must be enforced, not trusted."""

    required_environment = {
        "XDGMM_BOVY_REFERENCE_CONTAINER": "1",
        "XDGMM_BOVY_BASE_AMD64_DIGEST": generator.DEBIAN_AMD64_MANIFEST_DIGEST,
        "XDGMM_BOVY_BASE_INDEX_DIGEST": generator.DEBIAN_MULTIARCH_INDEX_DIGEST,
        "XDGMM_BOVY_DEBIAN_SNAPSHOT": generator.DEBIAN_SNAPSHOT,
        "XDGMM_BOVY_RUNTIME_NETWORK": "none",
        "XDGMM_BOVY_CONTAINER_IMAGE_ID": "sha256:" + "1" * 64,
        "OMP_NUM_THREADS": "2",
        "OMP_DYNAMIC": "FALSE",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
    }
    for name, value in required_environment.items():
        monkeypatch.setenv(name, value)

    monkeypatch.setattr(generator.platform, "system", lambda: "Linux")
    monkeypatch.setattr(generator.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(generator.platform, "platform", lambda: "Linux-test-x86_64")

    def pinned_dpkg_query(command, **kwargs):
        del kwargs
        if command[-1] in generator.EXPECTED_DEBIAN_PACKAGES:
            stdout = generator.EXPECTED_DEBIAN_PACKAGES[command[-1]]
        else:
            stdout = "gcc\t4:12.2.0-3\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(generator.subprocess, "run", pinned_dpkg_query)

    with pytest.raises(generator.BuildEnvironmentError, match="OMP_NUM_THREADS"):
        generator._authoritative_environment()


def test_generation_rejects_repository_source_change_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Metadata hashes must describe the code that was loaded and executed."""

    project_root = tmp_path / "project"
    generator_path = project_root / "scripts" / "generate_bovy_parity_fixture.py"
    helper_path = project_root / "scripts" / "deterministic_npz.py"
    oracle_path = project_root / "tests" / "reference" / "general_xd.py"
    for path in (generator_path, helper_path, oracle_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"loaded source for {path.name}\n", encoding="utf-8")

    monkeypatch.setattr(generator, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(generator, "__file__", str(generator_path))
    monkeypatch.setattr(generator, "_authoritative_environment", lambda: {})
    monkeypatch.setattr(
        generator,
        "safe_extract_verified_archive",
        lambda verified, destination: destination / verified.root_name,
    )
    monkeypatch.setattr(
        generator,
        "_build_reference",
        lambda source_root, verified: (object(), {}),
    )

    arrays = {
        name: np.zeros(shape, dtype=np.float64)
        for name, shape in generator.FIXTURE_ARRAY_SCHEMA.items()
    }

    def run_loaded_oracle_then_change_its_source(reference):
        del reference
        oracle_path.write_text(
            "different source that was not loaded or executed\n",
            encoding="utf-8",
        )
        return arrays, {}

    monkeypatch.setattr(
        generator,
        "_reference_and_oracle_arrays",
        run_loaded_oracle_then_change_its_source,
    )

    source_archive = tmp_path / "source.tar.gz"
    source_archive.write_bytes(b"fixture test source")
    verified = generator.VerifiedSourceArchive(
        archive_path=source_archive,
        archive_size=source_archive.stat().st_size,
        archive_sha256=hashlib.sha256(source_archive.read_bytes()).hexdigest(),
        root_name="reference-deadbeef",
        file_count=0,
        manifest_sha256=hashlib.sha256(b"").hexdigest(),
        files=(),
    )
    output = tmp_path / "fixture.npz"
    metadata = tmp_path / "fixture.metadata.json"

    with pytest.raises((generator.CustodyError, RuntimeError), match="changed"):
        generator.generate_fixture(
            verified,
            output=output,
            metadata_output=metadata,
        )

    assert not output.exists()
    assert not metadata.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_custody_document_does_not_claim_pending_numbers_exist():
    document = (
        PROJECT_ROOT / "docs" / "reference-fixtures" / "bovy-a8a5988.md"
    ).read_text(encoding="utf-8")

    assert "authoritative numeric fixture: **pending**" in document.lower()
    assert "does not yet establish" in document.lower()
    assert "no endorsement" in document.lower()
    assert generator.BOVY_COMMIT in document
    assert generator.BOVY_SOURCE_ARCHIVE_SHA256 in document
    assert generator.BOVY_SOURCE_MANIFEST_SHA256 in document
