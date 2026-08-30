"""Test-only builders for the draft numerical-artifact contract.

The helpers in this module intentionally do not import
``development.serialization``.  That keeps the complete tests-first inventory
collectable before the implementation module exists.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
from typing import Iterable
import zipfile

import jax.numpy as jnp
import numpy as np

from development.fit_control import FitMode, FitResult, FitStatus
from development.general_fit_control import GroupedGeneralFitResult
from development.general_grouped import GroupedFailureStage
from development.identity_xd import Params
from development.metadata import (
    current_general_result_metadata,
    current_result_metadata,
    user_supplied_initialization,
)


FORMAT_ID = "xdgmm-jax.numeric-artifact"
FORMAT_VERSION = "0.1.0-draft.1"
PARAMETERS_RECORD_ID = "xdgmm-jax.parameters"
PARAMETERS_RECORD_VERSION = "0.1.0-draft.1"
IDENTITY_FIT_RECORD_ID = "xdgmm-jax.identity-fit-result"
IDENTITY_FIT_RECORD_VERSION = "0.1.0-draft.1"
GROUPED_FIT_RECORD_ID = "xdgmm-jax.grouped-general-fit-result"
GROUPED_FIT_RECORD_VERSION = "0.1.0-draft.1"

TERMINAL_CASES = (
    "converged",
    "max_iter",
    "fixed_steps_complete",
    "objective_decreased",
    "numerical_failure",
    "component_collapsed",
    "invalid_initial_objective",
)


def numpy_dtype(dtype: object) -> np.dtype:
    return np.dtype(dtype)


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def valid_parameters(dtype: object, *, source: str = "device") -> Params:
    """Return nontrivial, valid ``K=2,D=2`` parameters with signed zero."""

    target = numpy_dtype(dtype)
    arrays = (
        np.asarray([0.375, 0.625], dtype=target),
        np.asarray([[-0.0, 0.75], [1.25, -0.5]], dtype=target),
        np.asarray(
            [
                [[1.1, 0.2], [0.2, 0.7]],
                [[0.6, -0.1], [-0.1, 1.3]],
            ],
            dtype=target,
        ),
    )
    if source == "host":
        return Params(*arrays)
    if source == "device":
        return Params(*(jnp.asarray(array) for array in arrays))
    raise AssertionError(f"unknown source {source!r}")


def updated_parameters(dtype: object) -> Params:
    target = numpy_dtype(dtype)
    return Params(
        weights=jnp.asarray([0.4, 0.6], dtype=target),
        means=jnp.asarray([[-0.1, 0.7], [1.1, -0.4]], dtype=target),
        covariances=jnp.asarray(
            [
                [[1.0, 0.15], [0.15, 0.8]],
                [[0.7, -0.05], [-0.05, 1.2]],
            ],
            dtype=target,
        ),
    )


def _scalar(value: object, dtype: object):
    return jnp.asarray(value, dtype=numpy_dtype(dtype))


def _integer(value: int):
    return jnp.asarray(value, dtype=jnp.int32)


def _boolean(value: bool):
    return jnp.asarray(value, dtype=jnp.bool_)


def identity_fit_result(dtype: object, case: str) -> FitResult:
    """Build one bounded, internally coherent identity terminal record."""

    if case not in TERMINAL_CASES:
        raise AssertionError(f"unknown terminal case {case!r}")
    initial = valid_parameters(dtype)
    updated = updated_parameters(dtype)
    nan = _scalar(np.nan, dtype)
    initial_objective = _scalar(-11.0, dtype)
    final_objective = _scalar(-10.0, dtype)
    no_components = jnp.zeros((2,), dtype=bool)
    collapsed_components = jnp.asarray([False, True], dtype=bool)

    mode = FitMode.CONVERGED
    status = FitStatus.CONVERGED
    parameters = updated
    objective = final_objective
    objective_valid = True
    history = jnp.asarray([-11.0, -10.0], dtype=numpy_dtype(dtype))
    n_iter = 1
    iteration_limit = 3
    converged = True
    attempted_iteration = 1
    attempted_objective = final_objective
    attempted_objective_valid = True
    numerical_failure = False
    collapsed = False
    component_mask = no_components
    # The accepted normalized change is 1 / 11, so 0.2 makes the converged
    # terminal state arithmetically consistent rather than merely tagged.
    tol = _scalar(0.2, dtype)
    decrease_tol = _scalar(1e-7, dtype)

    if case == "max_iter":
        status = FitStatus.MAX_ITER
        parameters = initial
        objective = initial_objective
        history = jnp.asarray([-11.0], dtype=numpy_dtype(dtype))
        n_iter = 0
        converged = False
        iteration_limit = 0
        attempted_iteration = 0
        attempted_objective = nan
        attempted_objective_valid = False
        tol = _scalar(1e-4, dtype)
    elif case == "fixed_steps_complete":
        mode = FitMode.FIXED_STEPS
        status = FitStatus.FIXED_STEPS_COMPLETE
        converged = False
        iteration_limit = 1
        tol = None
        decrease_tol = None
    elif case == "objective_decreased":
        status = FitStatus.OBJECTIVE_DECREASED
        parameters = initial
        objective = initial_objective
        history = jnp.asarray([-11.0], dtype=numpy_dtype(dtype))
        n_iter = 0
        converged = False
        attempted_objective = _scalar(-12.0, dtype)
    elif case == "numerical_failure":
        status = FitStatus.NUMERICAL_FAILURE
        parameters = initial
        objective = initial_objective
        history = jnp.asarray([-11.0], dtype=numpy_dtype(dtype))
        n_iter = 0
        converged = False
        attempted_objective = nan
        attempted_objective_valid = False
        numerical_failure = True
    elif case == "component_collapsed":
        mode = FitMode.FIXED_STEPS
        status = FitStatus.COMPONENT_COLLAPSED
        parameters = initial
        objective = initial_objective
        history = jnp.asarray([-11.0], dtype=numpy_dtype(dtype))
        n_iter = 0
        converged = False
        attempted_objective = nan
        attempted_objective_valid = False
        collapsed = True
        component_mask = collapsed_components
        tol = None
        decrease_tol = None
    elif case == "invalid_initial_objective":
        status = FitStatus.NUMERICAL_FAILURE
        parameters = initial
        objective = nan
        objective_valid = False
        history = jnp.empty((0,), dtype=numpy_dtype(dtype))
        n_iter = 0
        converged = False
        attempted_iteration = 0
        attempted_objective = nan
        attempted_objective_valid = False
        numerical_failure = True

    return FitResult(
        parameters=parameters,
        initial_parameters=initial,
        objective=objective,
        objective_valid=_boolean(objective_valid),
        history=history,
        n_iter=_integer(n_iter),
        iteration_limit=_integer(iteration_limit),
        converged=_boolean(converged),
        status=_integer(int(status)),
        mode=_integer(int(mode)),
        attempted_iteration=_integer(attempted_iteration),
        attempted_objective=attempted_objective,
        attempted_objective_valid=_boolean(attempted_objective_valid),
        numerical_failure=_boolean(numerical_failure),
        collapsed=_boolean(collapsed),
        collapsed_components=component_mask,
        factor_jitter=_scalar(2.0**-12, dtype),
        covariance_ridge=_scalar(2.0**-10, dtype),
        tol=tol,
        decrease_tol=decrease_tol,
        initialization=user_supplied_initialization(),
        metadata=current_result_metadata(),
    )


def grouped_fit_result(dtype: object, case: str) -> GroupedGeneralFitResult:
    """Build one bounded, internally coherent grouped-general terminal record."""

    identity = identity_fit_result(dtype, case)
    n_samples = 3
    n_groups = 2
    failure_stage = GroupedFailureStage.NONE
    group_failure = jnp.zeros((n_groups,), dtype=bool)
    failed_pairs = jnp.zeros((n_samples, 2), dtype=bool)

    if case == "numerical_failure":
        failure_stage = GroupedFailureStage.CANDIDATE_OBJECTIVE
        group_failure = jnp.asarray([False, True], dtype=bool)
        failed_pairs = jnp.asarray(
            [[False, False], [False, False], [False, True]], dtype=bool
        )
    elif case == "invalid_initial_objective":
        failure_stage = GroupedFailureStage.CURRENT_STATISTICS
        group_failure = jnp.asarray([True, False], dtype=bool)
        failed_pairs = jnp.asarray(
            [[True, False], [False, False], [False, False]], dtype=bool
        )
    elif case == "component_collapsed":
        failure_stage = GroupedFailureStage.M_STEP

    return GroupedGeneralFitResult(
        parameters=identity.parameters,
        initial_parameters=identity.initial_parameters,
        objective=identity.objective,
        objective_valid=identity.objective_valid,
        history=identity.history,
        n_iter=identity.n_iter,
        iteration_limit=identity.iteration_limit,
        converged=identity.converged,
        status=identity.status,
        mode=identity.mode,
        attempted_iteration=identity.attempted_iteration,
        attempted_objective=identity.attempted_objective,
        attempted_objective_valid=identity.attempted_objective_valid,
        numerical_failure=identity.numerical_failure,
        collapsed=identity.collapsed,
        collapsed_components=identity.collapsed_components,
        failure_stage=_integer(int(failure_stage)),
        group_numerical_failure=group_failure,
        failed_pairs=failed_pairs,
        informative_weight=_scalar(2.5, dtype),
        factor_jitter=identity.factor_jitter,
        covariance_ridge=identity.covariance_ridge,
        tol=identity.tol,
        decrease_tol=identity.decrease_tol,
        initialization=identity.initialization,
        metadata=current_general_result_metadata(),
    )


@dataclass(frozen=True)
class ZipMember:
    """Complete test representation of one ZIP member and its metadata."""

    name: str
    data: bytes
    compress_type: int = zipfile.ZIP_STORED
    date_time: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)
    extra: bytes = b""
    comment: bytes = b""
    mode: int = stat.S_IFREG | 0o644


def read_members(path: Path) -> list[ZipMember]:
    with zipfile.ZipFile(path, "r") as archive:
        return [
            ZipMember(
                name=info.filename,
                data=archive.read(info),
                compress_type=info.compress_type,
                date_time=info.date_time,
                extra=info.extra,
                comment=info.comment,
                mode=info.external_attr >> 16,
            )
            for info in archive.infolist()
        ]


def write_members(
    path: Path,
    members: Iterable[ZipMember],
    *,
    archive_comment: bytes = b"",
) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = archive_comment
        for member in members:
            info = zipfile.ZipInfo(member.name, date_time=member.date_time)
            info.compress_type = member.compress_type
            info.create_system = 3
            info.external_attr = member.mode << 16
            info.extra = member.extra
            info.comment = member.comment
            archive.writestr(info, member.data)


def manifest_from(path: Path) -> dict:
    with zipfile.ZipFile(path, "r") as archive:
        return json.loads(archive.read("manifest.json"))


def replace_manifest_bytes(source: Path, target: Path, payload: bytes) -> None:
    members = read_members(source)
    members = [
        replace(member, data=payload) if member.name == "manifest.json" else member
        for member in members
    ]
    write_members(target, members)


def mutate_manifest(source: Path, target: Path, mutation) -> dict:
    manifest = manifest_from(source)
    mutation(manifest)
    replace_manifest_bytes(source, target, canonical_json(manifest))
    return manifest


def npy_bytes(
    array: np.ndarray,
    *,
    version: tuple[int, int] = (1, 0),
    allow_pickle: bool = False,
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(array),
        version=version,
        allow_pickle=allow_pickle,
    )
    return stream.getvalue()


def descriptor_dtype(array: np.ndarray) -> str:
    dtype = np.asarray(array).dtype
    if dtype == np.dtype(np.float32):
        return "float32"
    if dtype == np.dtype(np.float64):
        return "float64"
    if dtype == np.dtype(bool):
        return "bool"
    return dtype.str


def replace_array_payload(
    source: Path,
    target: Path,
    logical_name: str,
    payload: bytes,
    *,
    descriptor_array: np.ndarray | None = None,
    update_descriptor: bool = True,
) -> None:
    manifest = manifest_from(source)
    descriptor = manifest["arrays"][logical_name]
    member_path = descriptor["path"]
    if update_descriptor:
        if descriptor_array is None:
            raise AssertionError("descriptor_array is required")
        array = np.asarray(descriptor_array)
        descriptor.update(
            {
                "data_nbytes": int(array.nbytes),
                "dtype": descriptor_dtype(array),
                "member_nbytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(array.shape),
            }
        )
    members = read_members(source)
    replaced_members = []
    for member in members:
        if member.name == "manifest.json":
            replaced_members.append(replace(member, data=canonical_json(manifest)))
        elif member.name == member_path:
            replaced_members.append(replace(member, data=payload))
        else:
            replaced_members.append(member)
    write_members(target, replaced_members)


def replace_array(
    source: Path,
    target: Path,
    logical_name: str,
    array: np.ndarray,
    *,
    version: tuple[int, int] = (1, 0),
    allow_pickle: bool = False,
) -> None:
    payload = npy_bytes(array, version=version, allow_pickle=allow_pickle)
    replace_array_payload(
        source,
        target,
        logical_name,
        payload,
        descriptor_array=np.asarray(array),
    )


def assert_arrays_bits_equal(actual: Params, expected: Params) -> None:
    for actual_array, expected_array in zip(actual, expected, strict=True):
        actual_host = np.asarray(actual_array)
        expected_host = np.asarray(expected_array)
        assert actual_host.shape == expected_host.shape
        assert actual_host.dtype == expected_host.dtype
        assert actual_host.tobytes(order="C") == expected_host.tobytes(order="C")


def assert_fit_result_bits_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    assert actual._fields == expected._fields
    for field in actual._fields:
        actual_value = getattr(actual, field)
        expected_value = getattr(expected, field)
        if isinstance(expected_value, Params):
            assert_arrays_bits_equal(actual_value, expected_value)
        elif hasattr(expected_value, "_fields"):
            assert actual_value == expected_value
        elif expected_value is None:
            assert actual_value is None
        else:
            actual_array = np.asarray(actual_value)
            expected_array = np.asarray(expected_value)
            assert actual_array.shape == expected_array.shape, field
            assert actual_array.dtype == expected_array.dtype, field
            assert (
                actual_array.tobytes(order="C")
                == expected_array.tobytes(order="C")
            ), field


def patch_encrypted_flags(path: Path) -> None:
    payload = bytearray(path.read_bytes())
    offset = 0
    while True:
        offset = payload.find(b"PK\x03\x04", offset)
        if offset < 0:
            break
        flags = struct.unpack_from("<H", payload, offset + 6)[0]
        struct.pack_into("<H", payload, offset + 6, flags | 0x1)
        offset += 4
    offset = 0
    while True:
        offset = payload.find(b"PK\x01\x02", offset)
        if offset < 0:
            break
        flags = struct.unpack_from("<H", payload, offset + 8)[0]
        struct.pack_into("<H", payload, offset + 8, flags | 0x1)
        offset += 4
    path.write_bytes(payload)


def patch_declared_uncompressed_size(
    path: Path, *, member_name: str, size: int
) -> None:
    """Patch one named member's local and central uncompressed sizes."""

    payload = bytearray(path.read_bytes())
    encoded_name = member_name.encode("utf-8")
    patched_local = False
    offset = 0
    while True:
        offset = payload.find(b"PK\x03\x04", offset)
        if offset < 0:
            break
        name_length = struct.unpack_from("<H", payload, offset + 26)[0]
        extra_length = struct.unpack_from("<H", payload, offset + 28)[0]
        name_start = offset + 30
        name = bytes(payload[name_start : name_start + name_length])
        compressed_size = struct.unpack_from("<I", payload, offset + 18)[0]
        if name == encoded_name:
            struct.pack_into("<I", payload, offset + 22, size)
            patched_local = True
        offset = name_start + name_length + extra_length + compressed_size

    patched_central = False
    offset = 0
    while True:
        offset = payload.find(b"PK\x01\x02", offset)
        if offset < 0:
            break
        name_length = struct.unpack_from("<H", payload, offset + 28)[0]
        extra_length = struct.unpack_from("<H", payload, offset + 30)[0]
        comment_length = struct.unpack_from("<H", payload, offset + 32)[0]
        name_start = offset + 46
        name = bytes(payload[name_start : name_start + name_length])
        if name == encoded_name:
            struct.pack_into("<I", payload, offset + 24, size)
            patched_central = True
        offset = name_start + name_length + extra_length + comment_length

    if not patched_local or not patched_central:
        raise AssertionError(f"member headers not found for {member_name!r}")
    path.write_bytes(payload)
