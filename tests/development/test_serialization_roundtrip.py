"""Tests-first record and byte contract for numerical-artifact serialization.

The temporary API frozen here is path based.  Parameter records load through a
tagged wrapper because the shared ``Params`` tuple cannot identify identity
versus general semantics.  Existing identity and grouped fit-result types
already carry strict contract metadata and therefore load directly.

These are CPU tests.  Cross-device GPU evidence remains explicitly deferred by
the serialization contract.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitResult
from development.general_fit_control import GroupedGeneralFitResult
from development.identity_xd import Params
from development.metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    GENERAL_CONTRACT_ID,
    GENERAL_CONTRACT_VERSION,
)
from tests.development._serialization_test_helpers import (
    FORMAT_ID,
    FORMAT_VERSION,
    GROUPED_FIT_RECORD_ID,
    GROUPED_FIT_RECORD_VERSION,
    IDENTITY_FIT_RECORD_ID,
    IDENTITY_FIT_RECORD_VERSION,
    PARAMETERS_RECORD_ID,
    PARAMETERS_RECORD_VERSION,
    TERMINAL_CASES,
    assert_arrays_bits_equal,
    assert_fit_result_bits_equal,
    canonical_json,
    grouped_fit_result,
    identity_fit_result,
    manifest_from,
    mutate_manifest,
    valid_parameters,
)


DTYPES = (
    pytest.param(jnp.float32, id="float32"),
    pytest.param(jnp.float64, id="float64"),
)

CONTRACTS = (
    pytest.param(CONTRACT_ID, CONTRACT_VERSION, id="identity"),
    pytest.param(
        GENERAL_CONTRACT_ID, GENERAL_CONTRACT_VERSION, id="general"
    ),
)

EXPECTED_LIMIT_FIELDS = (
    "max_manifest_bytes",
    "max_npy_header_bytes",
    "max_members",
    "max_member_bytes",
    "max_total_bytes",
)

EXPECTED_ARRAY_DESCRIPTOR_FIELDS = {
    "data_nbytes",
    "dtype",
    "member_nbytes",
    "path",
    "sha256",
    "shape",
}

COMMON_FIT_FIELDS = {
    "attempted_iteration",
    "attempted_objective_valid",
    "collapsed",
    "converged",
    "initialization",
    "iteration_limit",
    "mode",
    "n_iter",
    "numerical_failure",
    "objective_semantics",
    "objective_valid",
    "ridge_application",
    "status",
}

STATUS_STRINGS = {
    "converged": "converged",
    "max_iter": "max_iter",
    "fixed_steps_complete": "fixed_steps_complete",
    "objective_decreased": "objective_decreased",
    "numerical_failure": "numerical_failure",
    "component_collapsed": "component_collapsed",
    "invalid_initial_objective": "numerical_failure",
}

FAILURE_STAGE_STRINGS = {
    "converged": "none",
    "max_iter": "none",
    "fixed_steps_complete": "none",
    "objective_decreased": "none",
    "numerical_failure": "candidate_objective",
    "component_collapsed": "m_step",
    "invalid_initial_objective": "current_statistics",
}


@pytest.fixture(scope="module")
def serialization():
    """Import lazily so every intended red case remains collectable."""

    return importlib.import_module("development.serialization")


@pytest.fixture(scope="module")
def version_module():
    return importlib.import_module("development.version")


def _assert_signature(
    function,
    positional: tuple[str, ...],
    required_keywords: tuple[str, ...],
    keyword_defaults: dict[str, object],
) -> None:
    parameters = inspect.signature(function).parameters
    assert tuple(parameters) == (
        *positional,
        *required_keywords,
        *keyword_defaults,
    )
    for name in positional:
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameter.default is inspect.Parameter.empty
    for name in required_keywords:
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    for name, expected_default in keyword_defaults.items():
        parameter = parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        if expected_default is _ANY_DEFAULT:
            assert parameter.default is not inspect.Parameter.empty
        else:
            assert parameter.default == expected_default


_ANY_DEFAULT = object()


def _save_parameters(
    serialization,
    path: Path,
    parameters: Params,
    contract_id: str,
    contract_version: str,
    *,
    overwrite: bool = False,
) -> None:
    serialization.save_parameters(
        path,
        parameters,
        contract_id=contract_id,
        contract_version=contract_version,
        overwrite=overwrite,
    )


def _canonical_manifest_bytes(path: Path) -> tuple[dict, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        payload = archive.read("manifest.json")
    return json.loads(payload), payload


def _npy_header(payload: bytes):
    stream = io.BytesIO(payload)
    assert np.lib.format.read_magic(stream) == (1, 0)
    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    return shape, fortran_order, dtype, stream.tell()


def _assert_container_and_array_metadata(path: Path) -> None:
    with zipfile.ZipFile(path, "r") as archive:
        assert archive.comment == b""
        infos = archive.infolist()
        manifest = json.loads(archive.read("manifest.json"))
        expected_names = ["manifest.json"] + sorted(
            descriptor["path"]
            for descriptor in manifest["arrays"].values()
        )
        assert [info.filename for info in infos] == expected_names
        assert len({info.filename for info in infos}) == len(infos)

        for info in infos:
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.extra == b""
            assert info.comment == b""
            assert not info.is_dir()
            assert not (info.flag_bits & 0x1)
            assert info.create_system == 3
            mode = info.external_attr >> 16
            assert stat.S_IFMT(mode) == stat.S_IFREG
            assert stat.S_IMODE(mode) == 0o644

        for logical_name, descriptor in manifest["arrays"].items():
            assert set(descriptor) == EXPECTED_ARRAY_DESCRIPTOR_FIELDS
            assert descriptor["path"] == f"arrays/{logical_name}.npy"
            payload = archive.read(descriptor["path"])
            assert descriptor["member_nbytes"] == len(payload)
            assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
            shape, fortran_order, dtype, header_nbytes = _npy_header(payload)
            assert not fortran_order
            assert list(shape) == descriptor["shape"]
            assert header_nbytes <= 16 * 1024
            if dtype == np.dtype(bool):
                assert descriptor["dtype"] == "bool"
                assert dtype.str == "|b1"
            else:
                assert descriptor["dtype"] in {"float32", "float64"}
                assert dtype.str in {"<f4", "<f8"}
            assert int(np.prod(shape, dtype=np.int64)) * dtype.itemsize == (
                descriptor["data_nbytes"]
            )


def test_temporary_serialization_api_is_exact(serialization, version_module):
    assert version_module.__version__ == "0.2.0b1"

    expected_constants = {
        "FORMAT_ID": FORMAT_ID,
        "FORMAT_VERSION": FORMAT_VERSION,
        "PARAMETERS_RECORD_ID": PARAMETERS_RECORD_ID,
        "PARAMETERS_RECORD_VERSION": PARAMETERS_RECORD_VERSION,
        "IDENTITY_FIT_RECORD_ID": IDENTITY_FIT_RECORD_ID,
        "IDENTITY_FIT_RECORD_VERSION": IDENTITY_FIT_RECORD_VERSION,
        "GROUPED_GENERAL_FIT_RECORD_ID": GROUPED_FIT_RECORD_ID,
        "GROUPED_GENERAL_FIT_RECORD_VERSION": GROUPED_FIT_RECORD_VERSION,
    }
    for name, expected in expected_constants.items():
        assert getattr(serialization, name) == expected

    assert serialization.ArtifactLimits._fields == EXPECTED_LIMIT_FIELDS
    assert serialization.ParameterArtifact._fields == (
        "parameters",
        "contract_id",
        "contract_version",
        "package_version",
    )
    assert issubclass(serialization.ArtifactFormatError, ValueError)
    assert issubclass(
        serialization.ArtifactLimitError, serialization.ArtifactFormatError
    )
    assert serialization.DEFAULT_LIMITS == serialization.ArtifactLimits(
        max_manifest_bytes=256 * 1024,
        max_npy_header_bytes=16 * 1024,
        max_members=32,
        max_member_bytes=64 * 1024 * 1024,
        max_total_bytes=128 * 1024 * 1024,
    )

    required_exports = {
        *expected_constants,
        "ArtifactFormatError",
        "ArtifactLimitError",
        "ArtifactLimits",
        "DEFAULT_LIMITS",
        "ParameterArtifact",
        "load_grouped_general_fit_result",
        "load_identity_fit_result",
        "load_parameters",
        "save_grouped_general_fit_result",
        "save_identity_fit_result",
        "save_parameters",
    }
    assert required_exports <= set(serialization.__all__)

    _assert_signature(
        serialization.save_parameters,
        ("path", "parameters"),
        ("contract_id", "contract_version"),
        {"overwrite": False},
    )
    _assert_signature(
        serialization.load_parameters,
        ("path",),
        (),
        {"device": None, "limits": _ANY_DEFAULT},
    )
    for save_name in (
        "save_identity_fit_result",
        "save_grouped_general_fit_result",
    ):
        _assert_signature(
            getattr(serialization, save_name),
            ("path", "result"),
            (),
            {"overwrite": False},
        )
    for load_name in (
        "load_identity_fit_result",
        "load_grouped_general_fit_result",
    ):
        _assert_signature(
            getattr(serialization, load_name),
            ("path",),
            (),
            {"device": None, "limits": _ANY_DEFAULT},
        )
    for load_name in (
        "load_parameters",
        "load_identity_fit_result",
        "load_grouped_general_fit_result",
    ):
        assert (
            inspect.signature(getattr(serialization, load_name))
            .parameters["limits"]
            .default
            is serialization.DEFAULT_LIMITS
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("contract_id,contract_version", CONTRACTS)
@pytest.mark.parametrize("source", ["host", "device"])
def test_parameter_round_trip_preserves_tag_dtype_values_and_cpu_placement(
    serialization,
    tmp_path,
    dtype,
    contract_id,
    contract_version,
    source,
):
    parameters = valid_parameters(dtype, source=source)
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization,
        path,
        parameters,
        contract_id,
        contract_version,
    )

    loaded = serialization.load_parameters(path)

    assert isinstance(loaded, serialization.ParameterArtifact)
    assert isinstance(loaded.parameters, Params)
    assert loaded.contract_id == contract_id
    assert loaded.contract_version == contract_version
    assert loaded.package_version == "0.2.0b1"
    assert_arrays_bits_equal(loaded.parameters, parameters)
    manifest = manifest_from(path)
    assert manifest["contract_id"] == contract_id
    assert manifest["contract_version"] == contract_version
    assert manifest["model"]["dtype"] == np.dtype(dtype).name
    for array in loaded.parameters:
        assert np.asarray(array).dtype == np.dtype(dtype)
        assert {device.platform for device in array.devices()} == {"cpu"}


@pytest.mark.parametrize("dtype", DTYPES)
def test_explicit_single_cpu_device_load_preserves_parameters(
    serialization, tmp_path, dtype
):
    parameters = valid_parameters(dtype)
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization, path, parameters, CONTRACT_ID, CONTRACT_VERSION
    )
    cpu = jax.devices("cpu")[0]

    loaded = serialization.load_parameters(path, device=cpu)

    assert_arrays_bits_equal(loaded.parameters, parameters)
    for array in loaded.parameters:
        assert array.devices() == {cpu}


def test_device_argument_must_be_one_explicit_jax_device(serialization, tmp_path):
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization,
        path,
        valid_parameters(jnp.float32),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    with pytest.raises((TypeError, serialization.ArtifactFormatError)) as error:
        serialization.load_parameters(path, device=[jax.devices("cpu")[0]])
    message = str(error.value).lower()
    assert "device" in message
    assert "single" in message


@pytest.mark.parametrize("family", ["identity", "grouped"])
def test_explicit_cpu_device_places_every_loaded_fit_array(
    serialization, tmp_path, family
):
    cpu = jax.devices("cpu")[0]
    path = tmp_path / f"{family}.artifact"
    if family == "identity":
        serialization.save_identity_fit_result(
            path, identity_fit_result(jnp.float64, "converged")
        )
        loaded = serialization.load_identity_fit_result(path, device=cpu)
    else:
        serialization.save_grouped_general_fit_result(
            path, grouped_fit_result(jnp.float64, "converged")
        )
        loaded = serialization.load_grouped_general_fit_result(path, device=cpu)

    array_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(loaded)
        if isinstance(leaf, jax.Array)
    ]
    assert array_leaves
    assert all(leaf.devices() == {cpu} for leaf in array_leaves)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", TERMINAL_CASES)
def test_identity_fit_round_trip_covers_every_bounded_terminal_state(
    serialization, tmp_path, dtype, case
):
    result = identity_fit_result(dtype, case)
    path = tmp_path / f"identity-{case}.artifact"

    serialization.save_identity_fit_result(path, result)
    loaded = serialization.load_identity_fit_result(path)

    assert isinstance(loaded, FitResult)
    assert_fit_result_bits_equal(loaded, result)
    manifest = manifest_from(path)
    assert manifest["fit"]["status"] == STATUS_STRINGS[case]
    expected_mode = (
        "fixed_steps"
        if case in {"fixed_steps_complete", "component_collapsed"}
        else "converged"
    )
    assert manifest["fit"]["mode"] == expected_mode
    assert manifest["fit"]["initialization"] == {"kind": "user_supplied"}
    tolerance_names = {"fit.tol", "fit.decrease_tol"}
    if expected_mode == "converged":
        assert tolerance_names <= set(manifest["arrays"])
    else:
        assert tolerance_names.isdisjoint(manifest["arrays"])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("case", TERMINAL_CASES)
def test_grouped_fit_round_trip_covers_every_bounded_terminal_state(
    serialization, tmp_path, dtype, case
):
    result = grouped_fit_result(dtype, case)
    path = tmp_path / f"grouped-{case}.artifact"

    serialization.save_grouped_general_fit_result(path, result)
    loaded = serialization.load_grouped_general_fit_result(path)

    assert isinstance(loaded, GroupedGeneralFitResult)
    assert_fit_result_bits_equal(loaded, result)
    manifest = manifest_from(path)
    assert manifest["fit"]["status"] == STATUS_STRINGS[case]
    assert manifest["fit"]["failure_stage"] == FAILURE_STAGE_STRINGS[case]
    expected_mode = (
        "fixed_steps"
        if case in {"fixed_steps_complete", "component_collapsed"}
        else "converged"
    )
    assert manifest["fit"]["mode"] == expected_mode
    assert manifest["fit"]["initialization"] == {"kind": "user_supplied"}
    tolerance_names = {"fit.tol", "fit.decrease_tol"}
    if expected_mode == "converged":
        assert tolerance_names <= set(manifest["arrays"])
    else:
        assert tolerance_names.isdisjoint(manifest["arrays"])


def test_parameter_manifest_has_exact_schema(serialization, tmp_path):
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization,
        path,
        valid_parameters(jnp.float64),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    manifest, payload = _canonical_manifest_bytes(path)

    assert payload == canonical_json(manifest)
    assert set(manifest) == {
        "arrays",
        "artifact_kind",
        "contract_id",
        "contract_version",
        "format_id",
        "format_version",
        "model",
        "package_version",
        "record_id",
        "record_version",
    }
    assert manifest["artifact_kind"] == "parameters"
    assert manifest["format_id"] == FORMAT_ID
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["record_id"] == PARAMETERS_RECORD_ID
    assert manifest["record_version"] == PARAMETERS_RECORD_VERSION
    assert manifest["contract_id"] == CONTRACT_ID
    assert manifest["contract_version"] == CONTRACT_VERSION
    assert manifest["package_version"] == "0.2.0b1"
    assert manifest["model"] == {
        "dtype": "float64",
        "latent_dimension": 2,
        "n_components": 2,
    }
    assert set(manifest["arrays"]) == {
        "parameters.covariances",
        "parameters.means",
        "parameters.weights",
    }
    _assert_container_and_array_metadata(path)


@pytest.mark.parametrize(
    "family,case,expected_kind,expected_record_id,expected_record_version",
    [
        (
            "identity",
            "fixed_steps_complete",
            "identity_fit_result",
            IDENTITY_FIT_RECORD_ID,
            IDENTITY_FIT_RECORD_VERSION,
        ),
        (
            "grouped",
            "numerical_failure",
            "grouped_general_fit_result",
            GROUPED_FIT_RECORD_ID,
            GROUPED_FIT_RECORD_VERSION,
        ),
    ],
)
def test_fit_manifest_has_exact_schema_and_semantic_strings(
    serialization,
    tmp_path,
    family,
    case,
    expected_kind,
    expected_record_id,
    expected_record_version,
):
    path = tmp_path / f"{family}.artifact"
    if family == "identity":
        result = identity_fit_result(jnp.float64, case)
        serialization.save_identity_fit_result(path, result)
        expected_contract = (CONTRACT_ID, CONTRACT_VERSION)
        expected_model = {
            "dtype": "float64",
            "latent_dimension": 2,
            "n_components": 2,
        }
        expected_extra_fit_fields = set()
        expected_extra_arrays = set()
        expected_failure_stage = None
    else:
        result = grouped_fit_result(jnp.float64, case)
        serialization.save_grouped_general_fit_result(path, result)
        expected_contract = (GENERAL_CONTRACT_ID, GENERAL_CONTRACT_VERSION)
        expected_model = {
            "dtype": "float64",
            "latent_dimension": 2,
            "n_components": 2,
            "n_groups": 2,
            "n_samples": 3,
        }
        expected_extra_fit_fields = {"failure_stage"}
        expected_extra_arrays = {
            "fit.failed_pairs",
            "fit.group_numerical_failure",
            "fit.informative_weight",
        }
        expected_failure_stage = "candidate_objective"

    manifest, payload = _canonical_manifest_bytes(path)
    assert payload == canonical_json(manifest)
    assert set(manifest) == {
        "arrays",
        "artifact_kind",
        "contract_id",
        "contract_version",
        "fit",
        "format_id",
        "format_version",
        "model",
        "package_version",
        "record_id",
        "record_version",
    }
    assert manifest["artifact_kind"] == expected_kind
    assert manifest["record_id"] == expected_record_id
    assert manifest["record_version"] == expected_record_version
    assert (manifest["contract_id"], manifest["contract_version"]) == (
        expected_contract
    )
    assert manifest["model"] == expected_model
    assert set(manifest["fit"]) == COMMON_FIT_FIELDS | expected_extra_fit_fields
    assert manifest["fit"]["mode"] == (
        "fixed_steps" if family == "identity" else "converged"
    )
    assert manifest["fit"]["status"] == STATUS_STRINGS[case]
    assert manifest["fit"]["objective_semantics"] == (
        "identity_fixed_jitter_effective_observed_mean"
        if family == "identity"
        else "general_fixed_jitter_effective_informative_weighted_observed_mean"
    )
    assert (
        manifest["fit"]["ridge_application"]
        == "post_em_latent_covariance"
    )
    assert manifest["fit"]["initialization"] == {"kind": "user_supplied"}
    if expected_failure_stage is not None:
        assert manifest["fit"]["failure_stage"] == expected_failure_stage

    expected_arrays = {
        "fit.attempted_objective",
        "fit.collapsed_components",
        "fit.covariance_ridge",
        "fit.factor_jitter",
        "fit.history",
        "fit.objective",
        "initial_parameters.covariances",
        "initial_parameters.means",
        "initial_parameters.weights",
        "parameters.covariances",
        "parameters.means",
        "parameters.weights",
    } | expected_extra_arrays
    if family == "grouped":
        expected_arrays |= {"fit.decrease_tol", "fit.tol"}
    assert set(manifest["arrays"]) == expected_arrays
    _assert_container_and_array_metadata(path)


@pytest.mark.parametrize(
    "family,exact_semantics,effective_semantics",
    [
        (
            "identity",
            "identity_exact_observed_mean",
            "identity_fixed_jitter_effective_observed_mean",
        ),
        (
            "grouped",
            "general_exact_informative_weighted_observed_mean",
            "general_fixed_jitter_effective_informative_weighted_observed_mean",
        ),
    ],
)
def test_objective_semantics_depends_only_on_nonzero_factor_jitter(
    serialization,
    tmp_path,
    family,
    exact_semantics,
    effective_semantics,
):
    result = (
        identity_fit_result(jnp.float64, "converged")
        if family == "identity"
        else grouped_fit_result(jnp.float64, "converged")
    )
    save = (
        serialization.save_identity_fit_result
        if family == "identity"
        else serialization.save_grouped_general_fit_result
    )
    zero_path = tmp_path / f"{family}-zero.artifact"
    nonzero_path = tmp_path / f"{family}-nonzero.artifact"
    zero_jitter = result._replace(factor_jitter=jnp.asarray(0.0, dtype=jnp.float64))

    save(zero_path, zero_jitter)
    save(nonzero_path, result)

    assert manifest_from(zero_path)["fit"]["objective_semantics"] == exact_semantics
    assert (
        manifest_from(nonzero_path)["fit"]["objective_semantics"]
        == effective_semantics
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_identical_canonical_inputs_produce_identical_container_bytes(
    serialization, tmp_path, dtype
):
    device_path = tmp_path / "device.artifact"
    host_path = tmp_path / "host.artifact"

    _save_parameters(
        serialization,
        device_path,
        valid_parameters(dtype, source="device"),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    _save_parameters(
        serialization,
        host_path,
        valid_parameters(dtype, source="host"),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )

    assert device_path.read_bytes() == host_path.read_bytes()


def test_identical_fit_results_produce_identical_container_bytes(
    serialization, tmp_path
):
    result = grouped_fit_result(jnp.float64, "numerical_failure")
    first = tmp_path / "first.artifact"
    second = tmp_path / "second.artifact"
    serialization.save_grouped_general_fit_result(first, result)
    serialization.save_grouped_general_fit_result(second, result)
    assert first.read_bytes() == second.read_bytes()


def test_default_save_refuses_overwrite_and_preserves_existing_bytes(
    serialization, tmp_path
):
    path = tmp_path / "parameters.artifact"
    original = valid_parameters(jnp.float64)
    replacement = valid_parameters(jnp.float32)
    _save_parameters(
        serialization, path, original, CONTRACT_ID, CONTRACT_VERSION
    )
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        _save_parameters(
            serialization,
            path,
            replacement,
            CONTRACT_ID,
            CONTRACT_VERSION,
        )

    assert path.read_bytes() == before


def test_explicit_overwrite_atomically_replaces_complete_artifact(
    serialization, tmp_path
):
    path = tmp_path / "parameters.artifact"
    original = valid_parameters(jnp.float64)
    replacement = valid_parameters(jnp.float32)
    _save_parameters(
        serialization, path, original, CONTRACT_ID, CONTRACT_VERSION
    )

    _save_parameters(
        serialization,
        path,
        replacement,
        GENERAL_CONTRACT_ID,
        GENERAL_CONTRACT_VERSION,
        overwrite=True,
    )

    loaded = serialization.load_parameters(path)
    assert loaded.contract_id == GENERAL_CONTRACT_ID
    assert_arrays_bits_equal(loaded.parameters, replacement)


def test_writer_flushes_and_synchronizes_before_publish(
    serialization, tmp_path, monkeypatch
):
    calls = []
    real_fsync = os.fsync

    def recording_fsync(fd):
        calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization,
        path,
        valid_parameters(jnp.float32),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )

    assert calls
    serialization.load_parameters(path)


def test_failed_atomic_replace_keeps_destination_and_cleans_sibling_temp(
    serialization, tmp_path, monkeypatch
):
    path = tmp_path / "parameters.artifact"
    _save_parameters(
        serialization,
        path,
        valid_parameters(jnp.float64),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    before = path.read_bytes()
    before_names = {entry.name for entry in tmp_path.iterdir()}
    replace_calls = []

    def fail_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        _save_parameters(
            serialization,
            path,
            valid_parameters(jnp.float32),
            CONTRACT_ID,
            CONTRACT_VERSION,
            overwrite=True,
        )

    assert path.read_bytes() == before
    assert {entry.name for entry in tmp_path.iterdir()} == before_names
    assert len(replace_calls) == 1
    temporary, destination = replace_calls[0]
    assert temporary.parent == path.parent
    assert destination == path


class _NonAddressableArray:
    is_fully_addressable = False
    shape = (2, 2)
    dtype = np.dtype(np.float32)

    def __array__(self, *_args, **_kwargs):
        raise AssertionError("a non-addressable array must be rejected first")


def test_writer_rejects_non_fully_addressable_array_before_transfer(
    serialization, tmp_path
):
    parameters = valid_parameters(jnp.float32)._replace(
        means=_NonAddressableArray()
    )
    path = tmp_path / "parameters.artifact"

    with pytest.raises(serialization.ArtifactFormatError) as error:
        _save_parameters(
            serialization, path, parameters, CONTRACT_ID, CONTRACT_VERSION
        )

    message = str(error.value).lower()
    assert "addressable" in message or "shard" in message
    assert not path.exists()


def test_float64_load_fails_actionably_when_jax_x64_is_disabled(
    serialization, tmp_path
):
    path = tmp_path / "float64.artifact"
    _save_parameters(
        serialization,
        path,
        valid_parameters(jnp.float64),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    project_root = Path(__file__).resolve().parents[2]
    code = """
import jax
assert not jax.config.x64_enabled
from development.serialization import load_parameters
from development.validation import PrecisionError
import sys

try:
    load_parameters(sys.argv[1])
except PrecisionError as error:
    message = str(error).lower()
    if "float64" in message and "x64" in message:
        raise SystemExit(0)
    print(message)
    raise SystemExit(2)
raise SystemExit(3)
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_package_version_is_informational_not_a_compatibility_override(
    serialization, tmp_path
):
    source = tmp_path / "source.artifact"
    mutated = tmp_path / "mutated.artifact"
    _save_parameters(
        serialization,
        source,
        valid_parameters(jnp.float32),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    mutate_manifest(
        source,
        mutated,
        lambda manifest: manifest.__setitem__("package_version", "9999.0"),
    )

    loaded = serialization.load_parameters(mutated)

    assert loaded.package_version == "9999.0"
    assert loaded.contract_id == CONTRACT_ID


def test_record_specific_loaders_reject_other_supported_record_kinds(
    serialization, tmp_path
):
    parameters_path = tmp_path / "parameters.artifact"
    identity_path = tmp_path / "identity.artifact"
    grouped_path = tmp_path / "grouped.artifact"
    _save_parameters(
        serialization,
        parameters_path,
        valid_parameters(jnp.float32),
        CONTRACT_ID,
        CONTRACT_VERSION,
    )
    serialization.save_identity_fit_result(
        identity_path, identity_fit_result(jnp.float32, "converged")
    )
    serialization.save_grouped_general_fit_result(
        grouped_path, grouped_fit_result(jnp.float32, "converged")
    )

    calls = (
        (serialization.load_parameters, identity_path),
        (serialization.load_parameters, grouped_path),
        (serialization.load_identity_fit_result, parameters_path),
        (serialization.load_identity_fit_result, grouped_path),
        (serialization.load_grouped_general_fit_result, parameters_path),
        (serialization.load_grouped_general_fit_result, identity_path),
    )
    for load, path in calls:
        with pytest.raises(serialization.ArtifactFormatError):
            load(path)
