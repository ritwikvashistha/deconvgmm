"""Fail-closed and resource-bound tests for numerical-artifact readers.

Every malformed archive is derived from a small valid CPU artifact.  The suite
uses deliberately low caller-supplied limits rather than allocating large test
payloads, and it never extracts a ZIP member to the filesystem.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
import json
from pathlib import Path
import stat
import struct
import warnings
import zipfile

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitMode, FitStatus
from development.general_grouped import GroupedFailureStage
from development.metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
)
from tests.development._serialization_test_helpers import (
    ZipMember,
    grouped_fit_result,
    identity_fit_result,
    manifest_from,
    mutate_manifest,
    patch_declared_uncompressed_size,
    patch_encrypted_flags,
    read_members,
    replace_manifest_bytes,
    valid_parameters,
    write_members,
)


@pytest.fixture(scope="module")
def serialization():
    return importlib.import_module("development.serialization")


@pytest.fixture
def parameter_artifact(serialization, tmp_path) -> Path:
    path = tmp_path / "valid-parameters.artifact"
    serialization.save_parameters(
        path,
        valid_parameters(jnp.float64),
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )
    return path


@pytest.fixture
def identity_artifact(serialization, tmp_path) -> Path:
    path = tmp_path / "valid-identity.artifact"
    serialization.save_identity_fit_result(
        path, identity_fit_result(jnp.float64, "converged")
    )
    return path


@pytest.fixture
def grouped_artifact(serialization, tmp_path) -> Path:
    path = tmp_path / "valid-grouped.artifact"
    serialization.save_grouped_general_fit_result(
        path, grouped_fit_result(jnp.float64, "converged")
    )
    return path


def _assert_message(error: BaseException, *terms: str) -> None:
    message = str(error).lower().replace("_", " ").replace("-", " ")
    for term in terms:
        normalized_term = term.lower().replace("_", " ").replace("-", " ")
        assert normalized_term in message


def _load_rejects(serialization, path: Path, *, terms=(), limits=None) -> None:
    kwargs = {} if limits is None else {"limits": limits}
    with pytest.raises(serialization.ArtifactFormatError) as error:
        serialization.load_parameters(path, **kwargs)
    _assert_message(error.value, *terms)


def _load_limit_rejects(serialization, path: Path, limits, *terms: str) -> None:
    with pytest.raises(serialization.ArtifactLimitError) as error:
        serialization.load_parameters(path, limits=limits)
    _assert_message(error.value, *terms)


def _replace_named_member(
    source: Path,
    target: Path,
    old_name: str,
    new_name: str,
) -> None:
    members = read_members(source)
    write_members(
        target,
        [
            replace(member, name=new_name)
            if member.name == old_name
            else member
            for member in members
        ],
    )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("artifact_kind", "future_artifact"),
        ("format_id", "unknown.format"),
        ("format_version", "0.1.0-draft.2"),
        ("record_id", "unknown.record"),
        ("record_version", "0.1.0-draft.2"),
        ("contract_id", "unknown.contract"),
        ("contract_version", "0.1.0-draft.999"),
    ],
)
def test_reader_rejects_unknown_exact_identifiers_and_versions(
    serialization, parameter_artifact, tmp_path, field, replacement
):
    target = tmp_path / f"unknown-{field}.artifact"
    mutate_manifest(
        parameter_artifact,
        target,
        lambda manifest: manifest.__setitem__(field, replacement),
    )
    _load_rejects(serialization, target, terms=(field.replace("_", " "),))


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_reader_rejects_missing_or_unknown_top_level_manifest_field(
    serialization, parameter_artifact, tmp_path, operation
):
    target = tmp_path / f"top-{operation}.artifact"

    def mutation(manifest):
        if operation == "missing":
            del manifest["model"]
        else:
            manifest["future_field"] = 1

    mutate_manifest(parameter_artifact, target, mutation)
    _load_rejects(serialization, target, terms=("field",))


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_reader_rejects_missing_or_unknown_nested_model_field(
    serialization, parameter_artifact, tmp_path, operation
):
    target = tmp_path / f"model-{operation}.artifact"

    def mutation(manifest):
        if operation == "missing":
            del manifest["model"]["latent_dimension"]
        else:
            manifest["model"]["observed_dimension"] = 2

    mutate_manifest(parameter_artifact, target, mutation)
    _load_rejects(serialization, target, terms=("model", "field"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("artifact_kind", 1),
        ("package_version", 1),
        ("model", []),
        ("arrays", []),
    ],
)
def test_reader_rejects_wrong_manifest_field_types(
    serialization, parameter_artifact, tmp_path, field, value
):
    target = tmp_path / f"wrong-type-{field}.artifact"
    mutate_manifest(
        parameter_artifact,
        target,
        lambda manifest: manifest.__setitem__(field, value),
    )
    _load_rejects(serialization, target, terms=(field.replace("_", " "),))


@pytest.mark.parametrize("location", ["top", "nested"])
def test_reader_rejects_duplicate_json_object_keys_at_every_depth(
    serialization, parameter_artifact, tmp_path, location
):
    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        original = archive.read("manifest.json")
    if location == "top":
        duplicate = b'{"format_id":"duplicate",' + original[1:]
    else:
        needle = b'"parameters.weights":{'
        duplicate = original.replace(needle, needle + b'"shape":[2],')
        assert duplicate != original
    target = tmp_path / f"duplicate-json-key-{location}.artifact"
    replace_manifest_bytes(parameter_artifact, target, duplicate)
    _load_rejects(serialization, target, terms=("duplicate",))


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_reader_rejects_nonstandard_json_constants(
    serialization, parameter_artifact, tmp_path, constant
):
    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        original = archive.read("manifest.json")
    payload = original.replace(
        b'"latent_dimension":2', b'"latent_dimension":' + constant
    )
    assert payload != original
    target = tmp_path / f"constant-{constant.decode()}.artifact"
    replace_manifest_bytes(parameter_artifact, target, payload)
    _load_rejects(serialization, target, terms=("json",))


@pytest.mark.parametrize(
    "label,transform",
    [
        ("bom", lambda payload: b"\xef\xbb\xbf" + payload),
        ("invalid-utf8", lambda payload: b"\xff" + payload),
        ("no-newline", lambda payload: payload.rstrip(b"\n")),
        ("extra-newline", lambda payload: payload + b"\n"),
        (
            "pretty",
            lambda payload: json.dumps(
                json.loads(payload), sort_keys=True, indent=2
            ).encode("utf-8")
            + b"\n",
        ),
        (
            "unsorted",
            lambda payload: (
                json.dumps(
                    dict(reversed(tuple(json.loads(payload).items()))),
                    sort_keys=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
        ),
    ],
)
def test_reader_rejects_noncanonical_or_non_utf8_manifest_bytes(
    serialization, parameter_artifact, tmp_path, label, transform
):
    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        original = archive.read("manifest.json")
    target = tmp_path / f"manifest-{label}.artifact"
    replace_manifest_bytes(parameter_artifact, target, transform(original))
    _load_rejects(serialization, target, terms=("manifest",))


def test_reader_rejects_non_object_json_manifest(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "manifest-array.artifact"
    replace_manifest_bytes(parameter_artifact, target, b"[]\n")
    _load_rejects(serialization, target, terms=("object",))


def test_reader_wraps_bounded_deep_json_recursion_as_format_error(
    serialization, parameter_artifact, tmp_path
):
    depth = 2_000
    payload = b'{"nested":' * depth + b"0" + b"}" * depth + b"\n"
    assert len(payload) <= serialization.DEFAULT_LIMITS.max_manifest_bytes
    target = tmp_path / "deep-json.artifact"
    replace_manifest_bytes(parameter_artifact, target, payload)
    # The reader must fail closed on this byte-bounded, deeply nested manifest.
    # The exact rejection reason is environment-sensitive: on some interpreters
    # the JSON decoder exceeds its recursion limit and the reader wraps that as
    # a JSON format error, while on others (e.g. Python 3.12) the decoder parses
    # the nesting and the manifest structure check rejects it. Either way the
    # reader raises ArtifactFormatError rather than crashing with a RecursionError
    # or accepting the payload, which is the guarantee under test.
    _load_rejects(serialization, target)


@pytest.mark.parametrize("payload", [b"not a zip", b"", b"PK\x03\x04"])
def test_reader_wraps_invalid_or_truncated_container_as_format_error(
    serialization, tmp_path, payload
):
    target = tmp_path / "invalid.artifact"
    target.write_bytes(payload)
    _load_rejects(serialization, target, terms=("zip",))


@pytest.mark.parametrize("location", ["prefix", "suffix"])
def test_reader_rejects_opaque_bytes_outside_the_canonical_zip_container(
    serialization, parameter_artifact, tmp_path, location
):
    """A self-extracting prefix or trailing overlay is outside this format."""

    canonical = parameter_artifact.read_bytes()
    opaque = b"opaque-unlisted-bytes"
    target = tmp_path / f"opaque-{location}.artifact"
    target.write_bytes(
        opaque + canonical if location == "prefix" else canonical + opaque
    )

    _load_rejects(serialization, target)


def test_reader_rejects_opaque_gap_between_local_zip_members(
    serialization, parameter_artifact, tmp_path
):
    """Canonical local members must be contiguous, not only centrally listed."""

    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        infos = archive.infolist()
        insertion_offset = infos[1].header_offset
        central_offset = archive.start_dir
    opaque = b"opaque-internal-gap"
    payload = bytearray(parameter_artifact.read_bytes())
    payload[insertion_offset:insertion_offset] = opaque
    delta = len(opaque)
    new_central_offset = central_offset + delta

    central_entry_offset = new_central_offset
    for _info in infos:
        assert payload[central_entry_offset : central_entry_offset + 4] == (
            b"PK\x01\x02"
        )
        name_length, extra_length, comment_length = struct.unpack_from(
            "<3H", payload, central_entry_offset + 28
        )
        local_offset = struct.unpack_from(
            "<L", payload, central_entry_offset + 42
        )[0]
        if local_offset >= insertion_offset:
            struct.pack_into(
                "<L", payload, central_entry_offset + 42, local_offset + delta
            )
        central_entry_offset += (
            46 + name_length + extra_length + comment_length
        )

    end_record_offset = len(payload) - 22
    assert payload[end_record_offset : end_record_offset + 4] == b"PK\x05\x06"
    struct.pack_into(
        "<L", payload, end_record_offset + 16, new_central_offset
    )
    target = tmp_path / "opaque-internal-gap.artifact"
    target.write_bytes(payload)

    _load_rejects(serialization, target)


def test_reader_rejects_directory_instead_of_ordinary_file(serialization, tmp_path):
    with pytest.raises(serialization.ArtifactFormatError) as error:
        serialization.load_parameters(tmp_path)
    _assert_message(error.value, "file")


def test_reader_rejects_duplicate_zip_member_names(
    serialization, parameter_artifact, tmp_path
):
    members = read_members(parameter_artifact)
    target = tmp_path / "duplicate-member.artifact"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        write_members(target, [*members, members[-1]])
    _load_rejects(serialization, target, terms=("duplicate",))


def test_reader_rejects_unlisted_extra_zip_member(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "extra-member.artifact"
    write_members(
        target,
        [*read_members(parameter_artifact), ZipMember("arrays/extra.npy", b"x")],
    )
    _load_rejects(serialization, target, terms=("extra", "member"))


def test_reader_rejects_missing_listed_zip_member(
    serialization, parameter_artifact, tmp_path
):
    members = read_members(parameter_artifact)
    target = tmp_path / "missing-member.artifact"
    write_members(target, members[:-1])
    _load_rejects(serialization, target, terms=("missing", "member"))


def test_reader_rejects_directory_entry(serialization, parameter_artifact, tmp_path):
    target = tmp_path / "directory-entry.artifact"
    directory = ZipMember(
        "arrays/",
        b"",
        mode=stat.S_IFDIR | 0o755,
    )
    write_members(target, [*read_members(parameter_artifact), directory])
    _load_rejects(serialization, target, terms=("directory",))


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "/arrays/parameters.weights.npy",
        "../parameters.weights.npy",
        "arrays/../parameters.weights.npy",
        "arrays\\parameters.weights.npy",
    ],
)
def test_reader_rejects_forbidden_member_paths_without_extracting(
    serialization, parameter_artifact, tmp_path, forbidden_name
):
    manifest = manifest_from(parameter_artifact)
    old_name = manifest["arrays"]["parameters.weights"]["path"]
    target = tmp_path / "forbidden-path.artifact"
    _replace_named_member(parameter_artifact, target, old_name, forbidden_name)
    outside = tmp_path.parent / "parameters.weights.npy"

    _load_rejects(serialization, target, terms=("path",))

    assert not outside.exists()


def test_reader_rejects_nonstored_compression(
    serialization, parameter_artifact, tmp_path
):
    members = read_members(parameter_artifact)
    target = tmp_path / "compressed.artifact"
    write_members(
        target,
        [
            members[0],
            replace(members[1], compress_type=zipfile.ZIP_DEFLATED),
            *members[2:],
        ],
    )
    _load_rejects(serialization, target, terms=("compression",))


def test_reader_rejects_encrypted_member_flag(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "encrypted.artifact"
    target.write_bytes(parameter_artifact.read_bytes())
    patch_encrypted_flags(target)
    _load_rejects(serialization, target, terms=("encrypted",))


@pytest.mark.parametrize(
    "label,member_update,archive_comment,expected_term",
    [
        (
            "timestamp",
            lambda member: replace(member, date_time=(2026, 8, 26, 0, 0, 0)),
            b"",
            "timestamp",
        ),
        (
            "extra",
            lambda member: replace(member, extra=b"\x01\x00\x00\x00"),
            b"",
            "extra",
        ),
        (
            "comment",
            lambda member: replace(member, comment=b"comment"),
            b"",
            "comment",
        ),
        (
            "permissions",
            lambda member: replace(member, mode=stat.S_IFREG | 0o600),
            b"",
            "permission",
        ),
        (
            "symlink",
            lambda member: replace(member, mode=stat.S_IFLNK | 0o777),
            b"",
            "regular",
        ),
        ("archive-comment", lambda member: member, b"comment", "comment"),
    ],
)
def test_reader_rejects_noncanonical_zip_metadata(
    serialization,
    parameter_artifact,
    tmp_path,
    label,
    member_update,
    archive_comment,
    expected_term,
):
    members = read_members(parameter_artifact)
    target = tmp_path / f"metadata-{label}.artifact"
    write_members(
        target,
        [member_update(members[0]), *members[1:]],
        archive_comment=archive_comment,
    )
    _load_rejects(serialization, target, terms=(expected_term,))


@pytest.mark.parametrize(
    "label,field_offset,replacement,expected_term",
    [
        ("local-timestamp", 10, 1, "timestamp"),
        ("local-compression", 8, zipfile.ZIP_DEFLATED, "compression"),
    ],
)
def test_reader_rejects_noncanonical_local_zip_header_metadata(
    serialization,
    parameter_artifact,
    tmp_path,
    label,
    field_offset,
    replacement,
    expected_term,
):
    """Central metadata must not hide a contradictory local-file header."""

    target = tmp_path / f"metadata-{label}.artifact"
    payload = bytearray(parameter_artifact.read_bytes())
    assert payload[:4] == b"PK\x03\x04"
    struct.pack_into("<H", payload, field_offset, replacement)
    target.write_bytes(payload)

    _load_rejects(serialization, target, terms=(expected_term,))


def test_reader_rejects_per_member_multidisk_metadata(
    serialization, parameter_artifact, tmp_path
):
    """A central entry cannot claim another start disk in a one-file format."""

    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        central_offset = archive.start_dir
    payload = bytearray(parameter_artifact.read_bytes())
    assert payload[central_offset : central_offset + 4] == b"PK\x01\x02"
    # The two-byte disk-number-start field begins at offset 34 of a central
    # directory entry.
    struct.pack_into("<H", payload, central_offset + 34, 1)
    target = tmp_path / "member-start-disk.artifact"
    target.write_bytes(payload)

    _load_rejects(serialization, target, terms=("disk",))


def test_reader_rejects_absurd_declared_member_size_before_reading(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "declared-size.artifact"
    target.write_bytes(parameter_artifact.read_bytes())
    member_name = manifest_from(parameter_artifact)["arrays"][
        "parameters.weights"
    ]["path"]
    patch_declared_uncompressed_size(
        target,
        member_name=member_name,
        size=64 * 1024 * 1024 + 1,
    )
    _load_limit_rejects(serialization, target, serialization.DEFAULT_LIMITS, "member")


def test_reader_rejects_corrupted_member_crc_or_payload(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "crc-corrupt.artifact"
    payload = bytearray(parameter_artifact.read_bytes())
    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        info = archive.getinfo("manifest.json")
    data_offset = (
        info.header_offset
        + 30
        + len(info.filename.encode())
        + len(info.extra)
    )
    payload[data_offset] ^= 0x01
    target.write_bytes(payload)
    _load_rejects(serialization, target, terms=("crc",))


@pytest.mark.parametrize(
    "limit_field,expected_term",
    [
        ("max_manifest_bytes", "manifest"),
        ("max_npy_header_bytes", "header"),
        ("max_members", "member"),
        ("max_member_bytes", "member"),
        ("max_total_bytes", "total"),
    ],
)
def test_every_resource_limit_is_enforced_before_array_construction(
    serialization, parameter_artifact, limit_field, expected_term
):
    with zipfile.ZipFile(parameter_artifact, "r") as archive:
        infos = archive.infolist()
        manifest_size = archive.getinfo("manifest.json").file_size
        largest_array_member = max(
            info.file_size for info in infos if info.filename != "manifest.json"
        )
        array_total_size = sum(
            info.file_size for info in infos if info.filename != "manifest.json"
        )
    values = {
        "max_manifest_bytes": manifest_size - 1,
        "max_npy_header_bytes": 8,
        "max_members": len(infos) - 1,
        "max_member_bytes": largest_array_member - 1,
        "max_total_bytes": array_total_size - 1,
    }
    limits = serialization.DEFAULT_LIMITS._replace(
        **{limit_field: values[limit_field]}
    )
    _load_limit_rejects(serialization, parameter_artifact, limits, expected_term)


@pytest.mark.parametrize("bad_value", [-1, True, 1.5, "32"])
def test_reader_rejects_invalid_caller_supplied_limit_values(
    serialization, parameter_artifact, bad_value
):
    limits = serialization.DEFAULT_LIMITS._replace(max_members=bad_value)
    with pytest.raises((TypeError, ValueError)) as error:
        serialization.load_parameters(parameter_artifact, limits=limits)
    _assert_message(error.value, "limit", "member")


def test_reader_rejects_wrong_limits_container(serialization, parameter_artifact):
    with pytest.raises(TypeError) as error:
        serialization.load_parameters(parameter_artifact, limits={})
    _assert_message(error.value, "limits")


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_array_descriptor_has_exact_fields(
    serialization, parameter_artifact, tmp_path, operation
):
    target = tmp_path / f"descriptor-{operation}.artifact"

    def mutation(manifest):
        descriptor = manifest["arrays"]["parameters.weights"]
        if operation == "missing":
            del descriptor["sha256"]
        else:
            descriptor["offset"] = 0

    mutate_manifest(parameter_artifact, target, mutation)
    _load_rejects(serialization, target, terms=("descriptor", "field"))


@pytest.mark.parametrize(
    "field,mutation",
    [
        ("path", lambda value: value + ".other"),
        ("dtype", lambda _value: "float32"),
        ("shape", lambda value: [*value, 1]),
        ("data_nbytes", lambda value: value + 1),
        ("member_nbytes", lambda value: value + 1),
        ("sha256", lambda _value: "0" * 64),
    ],
)
def test_descriptor_must_match_logical_schema_and_complete_npy_member(
    serialization, parameter_artifact, tmp_path, field, mutation
):
    target = tmp_path / f"descriptor-mismatch-{field}.artifact"

    def change(manifest):
        descriptor = manifest["arrays"]["parameters.weights"]
        descriptor[field] = mutation(descriptor[field])

    mutate_manifest(parameter_artifact, target, change)
    _load_rejects(serialization, target, terms=(field.replace("_", " "),))


def test_unknown_or_missing_logical_array_name_is_rejected(
    serialization, parameter_artifact, tmp_path
):
    for operation in ("unknown", "missing"):
        target = tmp_path / f"logical-{operation}.artifact"

        def change(manifest, operation=operation):
            descriptor = manifest["arrays"].pop("parameters.weights")
            if operation == "unknown":
                manifest["arrays"]["parameters.alpha"] = descriptor

        mutate_manifest(parameter_artifact, target, change)
        _load_rejects(serialization, target, terms=("array",))


@pytest.mark.parametrize("field", ["n_components", "latent_dimension"])
def test_model_dimensions_must_match_parameter_array_shapes(
    serialization, parameter_artifact, tmp_path, field
):
    target = tmp_path / f"model-shape-{field}.artifact"
    mutate_manifest(
        parameter_artifact,
        target,
        lambda manifest: manifest["model"].__setitem__(
            field, manifest["model"][field] + 1
        ),
    )
    _load_rejects(serialization, target, terms=(field,))


def test_payload_hash_is_checked_before_npy_decode(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "payload-hash.artifact"
    manifest = manifest_from(parameter_artifact)
    member_path = manifest["arrays"]["parameters.weights"]["path"]
    members = read_members(parameter_artifact)
    write_members(
        target,
        [
            replace(member, data=member.data[:-1] + bytes([member.data[-1] ^ 1]))
            if member.name == member_path
            else member
            for member in members
        ],
    )
    _load_rejects(serialization, target, terms=("sha256",))
