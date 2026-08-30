"""Strict temporary numerical-artifact serialization.

This module implements the draft contract in ``docs/serialization-contract.md``
without selecting a future public package name.  Artifacts contain canonical
JSON and no-pickle NPY 1.0 members in a deterministic, uncompressed ZIP.  The
reader validates all host bytes and numerical domains before creating JAX
arrays.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import struct
import tempfile
from typing import Any, NamedTuple
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from .fit_control import FitMode, FitResult, FitStatus
from .general_fit_control import GroupedGeneralFitResult
from .general_grouped import GroupedFailureStage
from .identity_xd import Params
from .metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    GENERAL_CONTRACT_ID,
    GENERAL_CONTRACT_VERSION,
    InitializationProvenance,
    current_general_result_metadata,
    current_result_metadata,
    user_supplied_initialization,
)
from .validation import PrecisionError
from .version import __version__


FORMAT_ID = "xdgmm-jax.numeric-artifact"
FORMAT_VERSION = "0.1.0-draft.1"
PARAMETERS_RECORD_ID = "xdgmm-jax.parameters"
PARAMETERS_RECORD_VERSION = "0.1.0-draft.1"
IDENTITY_FIT_RECORD_ID = "xdgmm-jax.identity-fit-result"
IDENTITY_FIT_RECORD_VERSION = "0.1.0-draft.1"
GROUPED_GENERAL_FIT_RECORD_ID = "xdgmm-jax.grouped-general-fit-result"
GROUPED_GENERAL_FIT_RECORD_VERSION = "0.1.0-draft.1"


class ArtifactFormatError(ValueError):
    """The artifact or proposed in-memory record violates the wire contract."""


class ArtifactLimitError(ArtifactFormatError):
    """A declared or observed artifact resource exceeds caller limits."""


class ArtifactLimits(NamedTuple):
    """Reader resource ceilings, all expressed in bytes except member count."""

    max_manifest_bytes: int
    max_npy_header_bytes: int
    max_members: int
    max_member_bytes: int
    max_total_bytes: int


DEFAULT_LIMITS = ArtifactLimits(
    max_manifest_bytes=256 * 1024,
    max_npy_header_bytes=16 * 1024,
    max_members=32,
    max_member_bytes=64 * 1024 * 1024,
    max_total_bytes=128 * 1024 * 1024,
)


class ParameterArtifact(NamedTuple):
    """Contract-tagged parameters returned by the generic parameter loader."""

    parameters: Params
    contract_id: str
    contract_version: str
    package_version: str


_SUPPORTED_CONTRACTS = {
    (CONTRACT_ID, CONTRACT_VERSION),
    (GENERAL_CONTRACT_ID, GENERAL_CONTRACT_VERSION),
}

_STATUS_TO_STRING = {
    int(FitStatus.CONVERGED): "converged",
    int(FitStatus.MAX_ITER): "max_iter",
    int(FitStatus.OBJECTIVE_DECREASED): "objective_decreased",
    int(FitStatus.NUMERICAL_FAILURE): "numerical_failure",
    int(FitStatus.COMPONENT_COLLAPSED): "component_collapsed",
    int(FitStatus.FIXED_STEPS_COMPLETE): "fixed_steps_complete",
}
_STRING_TO_STATUS = {value: key for key, value in _STATUS_TO_STRING.items()}
_MODE_TO_STRING = {
    int(FitMode.CONVERGED): "converged",
    int(FitMode.FIXED_STEPS): "fixed_steps",
}
_STRING_TO_MODE = {value: key for key, value in _MODE_TO_STRING.items()}
_STAGE_TO_STRING = {
    int(GroupedFailureStage.NONE): "none",
    int(GroupedFailureStage.CURRENT_STATISTICS): "current_statistics",
    int(GroupedFailureStage.M_STEP): "m_step",
    int(GroupedFailureStage.CANDIDATE_OBJECTIVE): "candidate_objective",
}
_STRING_TO_STAGE = {value: key for key, value in _STAGE_TO_STRING.items()}

_DESCRIPTOR_FIELDS = {
    "data_nbytes",
    "dtype",
    "member_nbytes",
    "path",
    "sha256",
    "shape",
}
_COMMON_FIT_FIELDS = {
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


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactFormatError("manifest contains a non-JSON value") from error
    return (text + "\n").encode("utf-8")


def _field_error(field: str, detail: str) -> ArtifactFormatError:
    return ArtifactFormatError(f"{field}: {detail}")


def _check_addressable(value: object, *, field: str) -> None:
    addressable = getattr(value, "is_fully_addressable", True)
    if not bool(addressable):
        raise _field_error(
            field,
            "array is not fully addressable; sharded or remote shards cannot be saved",
        )
    devices = getattr(value, "devices", None)
    if callable(devices):
        try:
            if len(devices()) > 1:
                raise _field_error(
                    field, "a multi-device sharded array cannot be saved"
                )
        except TypeError:
            pass


def _host_array(value: object, *, field: str) -> np.ndarray:
    _check_addressable(value, field=field)
    blocker = getattr(value, "block_until_ready", None)
    if callable(blocker):
        blocker()
    try:
        return np.asarray(jax.device_get(value))
    except ArtifactFormatError:
        raise
    except Exception as error:
        raise _field_error(field, "could not transfer array to the host") from error


def _little_c_array(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.dtype(np.float32):
        dtype = np.dtype("<f4")
    elif array.dtype == np.dtype(np.float64):
        dtype = np.dtype("<f8")
    elif array.dtype == np.dtype(bool):
        dtype = np.dtype(bool)
    else:
        raise ArtifactFormatError(f"unsupported artifact array dtype {array.dtype}")
    converted = array.astype(dtype, copy=False)
    if converted.ndim == 0:
        return np.asarray(converted).reshape(())
    return np.ascontiguousarray(converted)


def _normalization_tolerance(dtype: np.dtype) -> float:
    return 5e-13 if dtype == np.dtype(np.float64) else 2e-5


def _symmetry_tolerance(dtype: np.dtype) -> float:
    return 2e-13 if dtype == np.dtype(np.float64) else 2e-6


def _validate_parameter_domain(parameters: Params, *, field: str) -> Params:
    if not isinstance(parameters, Params):
        raise _field_error(field, "must be a Params instance")
    weights = np.asarray(parameters.weights)
    means = np.asarray(parameters.means)
    covariances = np.asarray(parameters.covariances)
    arrays = (weights, means, covariances)
    if any(array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)) for array in arrays):
        raise _field_error(field, "all parameter arrays must use float32 or float64 dtype")
    if not (weights.dtype == means.dtype == covariances.dtype):
        raise _field_error(field, "parameter arrays must use one common dtype")
    if weights.ndim != 1:
        raise _field_error(f"{field}.weights", "shape must be (K,)")
    if means.ndim != 2:
        raise _field_error(f"{field}.means", "shape must be (K,D)")
    if covariances.ndim != 3:
        raise _field_error(
            f"{field}.covariances", "shape must be (K,D,D)"
        )
    k = weights.shape[0]
    if k <= 0:
        raise _field_error(field, "n_components must be positive")
    if means.shape[0] != k or covariances.shape[0] != k:
        raise _field_error(field, "parameter component shapes do not agree")
    d = means.shape[1]
    if d <= 0:
        raise _field_error(field, "latent_dimension must be positive")
    if covariances.shape[1:] != (d, d):
        raise _field_error(f"{field}.covariances", "shape must be (K,D,D)")
    if not np.all(np.isfinite(weights)) or not np.all(weights > 0.0):
        raise _field_error(f"{field}.weights", "weights must be finite and strictly positive")
    with np.errstate(over="ignore", invalid="ignore"):
        weight_sum = np.sum(weights, dtype=weights.dtype)
    if not np.isfinite(weight_sum) or abs(float(weight_sum) - 1.0) > _normalization_tolerance(weights.dtype):
        raise _field_error(f"{field}.weights", "weights must be normalized")
    if not np.all(np.isfinite(means)):
        raise _field_error(f"{field}.means", "means must be finite")
    if not np.all(np.isfinite(covariances)):
        raise _field_error(f"{field}.covariances", "covariances must be finite")
    symmetry_tolerance = _symmetry_tolerance(weights.dtype)
    for index, covariance in enumerate(covariances):
        metric_covariance = covariance.astype(np.float64, copy=False)
        entry_scale = max(
            1.0, float(np.max(np.abs(metric_covariance), initial=0.0))
        )
        scaled_covariance = metric_covariance / entry_scale
        scaled_spectral_norm = float(
            np.linalg.norm(scaled_covariance, ord=2)
        )
        scaled_spectral_scale = max(
            1.0 / entry_scale, scaled_spectral_norm
        )
        scaled_symmetry_norm = float(
            np.linalg.norm(
                scaled_covariance - scaled_covariance.T, ord=np.inf
            )
        )
        symmetry_residual = scaled_symmetry_norm / scaled_spectral_scale
        metrics = (
            entry_scale,
            scaled_spectral_norm,
            scaled_spectral_scale,
            scaled_symmetry_norm,
            symmetry_residual,
        )
        if not np.all(np.isfinite(metrics)) or symmetry_residual > symmetry_tolerance:
            raise _field_error(
                f"{field}.covariances",
                f"covariance {index} is materially asymmetric",
            )
        symmetric = covariance * weights.dtype.type(0.5)
        symmetric = symmetric + covariance.T * weights.dtype.type(0.5)
        if not np.all(np.isfinite(symmetric)):
            raise _field_error(
                f"{field}.covariances",
                f"covariance {index} became nonfinite during symmetrization",
            )
        scaled_symmetric = symmetric.astype(np.float64, copy=False) / entry_scale
        if not np.all(np.isfinite(scaled_symmetric)):
            raise _field_error(
                f"{field}.covariances",
                f"covariance {index} has invalid scaled metrics",
            )
        try:
            factor = np.linalg.cholesky(scaled_symmetric)
        except np.linalg.LinAlgError as error:
            raise _field_error(
                f"{field}.covariances",
                f"covariance {index} is not positive definite",
            ) from error
        if not np.all(np.isfinite(factor)):
            raise _field_error(
                f"{field}.covariances",
                f"covariance {index} is not positive definite",
            )
        if weights.dtype == np.dtype(np.float32) or jax.config.x64_enabled:
            selected_factor = jax.lax.linalg.cholesky(
                jnp.asarray(symmetric, dtype=weights.dtype),
                symmetrize_input=False,
            )
            selected_host_factor = np.asarray(jax.device_get(selected_factor))
            if not np.all(np.isfinite(selected_host_factor)) or not np.all(
                np.diag(selected_host_factor) > 0.0
            ):
                raise _field_error(
                    f"{field}.covariances",
                    f"covariance {index} is not positive definite in the selected JAX dtype",
                )
    return Params(*(_little_c_array(array) for array in arrays))


def _host_parameters(parameters: object, *, field: str) -> Params:
    if not isinstance(parameters, Params):
        raise _field_error(field, "must be a Params instance")
    host = Params(
        _host_array(parameters.weights, field=f"{field}.weights"),
        _host_array(parameters.means, field=f"{field}.means"),
        _host_array(parameters.covariances, field=f"{field}.covariances"),
    )
    return _validate_parameter_domain(host, field=field)


def _npy_payload(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        _little_c_array(array),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _dtype_name(array: np.ndarray) -> str:
    if array.dtype == np.dtype(np.float32):
        return "float32"
    if array.dtype == np.dtype(np.float64):
        return "float64"
    if array.dtype == np.dtype(bool):
        return "bool"
    raise ArtifactFormatError(f"unsupported artifact array dtype {array.dtype}")


def _array_members(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, dict[str, object]], dict[str, bytes]]:
    descriptors: dict[str, dict[str, object]] = {}
    payloads: dict[str, bytes] = {}
    for logical_name in sorted(arrays):
        array = _little_c_array(arrays[logical_name])
        payload = _npy_payload(array)
        path = f"arrays/{logical_name}.npy"
        descriptors[logical_name] = {
            "data_nbytes": int(array.nbytes),
            "dtype": _dtype_name(array),
            "member_nbytes": len(payload),
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "shape": list(array.shape),
        }
        payloads[path] = payload
    return descriptors, payloads


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.extra = b""
    info.comment = b""
    return info


def _write_artifact(
    path: os.PathLike[str] | str,
    manifest: dict[str, object],
    payloads: dict[str, bytes],
    *,
    overwrite: bool,
) -> None:
    destination = Path(path)
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be bool")
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    manifest_payload = _canonical_json(manifest)
    destination.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "w+b") as raw:
            with zipfile.ZipFile(
                raw,
                "w",
                compression=zipfile.ZIP_STORED,
                allowZip64=False,
            ) as archive:
                archive.comment = b""
                archive.writestr(_zip_info("manifest.json"), manifest_payload)
                for member_path in sorted(payloads):
                    archive.writestr(_zip_info(member_path), payloads[member_path])
            raw.flush()
            os.fsync(raw.fileno())
        if overwrite:
            os.replace(temporary, destination)
            published = True
        else:
            os.link(temporary, destination)
            published = True
            temporary.unlink()
    finally:
        if not published or temporary.exists():
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _parameter_record(
    parameters: object,
    *,
    contract_id: object,
    contract_version: object,
) -> tuple[dict[str, object], dict[str, bytes]]:
    if not isinstance(contract_id, str):
        raise ArtifactFormatError("contract_id must be a string")
    if not isinstance(contract_version, str):
        raise ArtifactFormatError("contract_version must be a string")
    if (contract_id, contract_version) not in _SUPPORTED_CONTRACTS:
        raise ArtifactFormatError(
            "contract_id and contract_version must be one exact supported pair"
        )
    canonical = _host_parameters(parameters, field="parameters")
    arrays = {
        "parameters.weights": canonical.weights,
        "parameters.means": canonical.means,
        "parameters.covariances": canonical.covariances,
    }
    descriptors, payloads = _array_members(arrays)
    k, d = canonical.means.shape
    manifest: dict[str, object] = {
        "arrays": descriptors,
        "artifact_kind": "parameters",
        "contract_id": contract_id,
        "contract_version": contract_version,
        "format_id": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "model": {
            "dtype": canonical.means.dtype.name,
            "latent_dimension": d,
            "n_components": k,
        },
        "package_version": __version__,
        "record_id": PARAMETERS_RECORD_ID,
        "record_version": PARAMETERS_RECORD_VERSION,
    }
    return manifest, payloads


def save_parameters(
    path,
    parameters,
    *,
    contract_id,
    contract_version,
    overwrite=False,
):
    """Save one exact-contract parameter record."""

    manifest, payloads = _parameter_record(
        parameters,
        contract_id=contract_id,
        contract_version=contract_version,
    )
    _write_artifact(path, manifest, payloads, overwrite=overwrite)


def _host_float_scalar(value: object, *, field: str, dtype: np.dtype) -> np.ndarray:
    array = _host_array(value, field=field)
    if array.shape != ():
        raise _field_error(field, "must be a scalar")
    if array.dtype != dtype:
        raise _field_error(field, f"dtype must be {dtype.name}")
    return _little_c_array(array)


def _host_bool_scalar(value: object, *, field: str) -> bool:
    array = _host_array(value, field=field)
    if array.shape != () or array.dtype != np.dtype(bool):
        raise _field_error(field, "must be a boolean scalar")
    return bool(array)


def _host_int_scalar(value: object, *, field: str) -> int:
    array = _host_array(value, field=field)
    if array.shape != () or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise _field_error(field, "must be an integer scalar")
    return int(array)


def _host_bool_array(value: object, *, field: str) -> np.ndarray:
    array = _host_array(value, field=field)
    if array.dtype != np.dtype(bool):
        raise _field_error(field, "must use bool dtype")
    return _little_c_array(array)


def _params_bits_equal(first: Params, second: Params) -> bool:
    return all(
        np.asarray(left).shape == np.asarray(right).shape
        and np.asarray(left).dtype == np.asarray(right).dtype
        and np.asarray(left).tobytes(order="C")
        == np.asarray(right).tobytes(order="C")
        for left, right in zip(first, second, strict=True)
    )


def _float_bits_equal(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        first.shape == second.shape
        and first.dtype == second.dtype
        and first.tobytes(order="C") == second.tobytes(order="C")
    )


def _validate_nonnegative_scalar(value: np.ndarray, *, field: str) -> None:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise _field_error(field, "must be finite and nonnegative")


def _validate_fit_state(state: dict[str, Any], *, family: str) -> None:
    parameters: Params = state["parameters"]
    initial_parameters: Params = state["initial_parameters"]
    dtype = np.asarray(parameters.means).dtype
    if np.asarray(initial_parameters.means).dtype != dtype:
        raise _field_error("initial_parameters", "dtype must match final parameters")
    if any(
        np.asarray(left).shape != np.asarray(right).shape
        for left, right in zip(parameters, initial_parameters, strict=True)
    ):
        raise _field_error(
            "initial_parameters", "shapes must match final parameters"
        )

    mode = state["mode"]
    status = state["status"]
    if mode not in _MODE_TO_STRING:
        raise _field_error("mode", "is unknown")
    if status not in _STATUS_TO_STRING:
        raise _field_error("status", "must be a terminal status")

    n_iter = state["n_iter"]
    iteration_limit = state["iteration_limit"]
    attempted_iteration = state["attempted_iteration"]
    for field, value in (
        ("n_iter", n_iter),
        ("iteration_limit", iteration_limit),
        ("attempted_iteration", attempted_iteration),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise _field_error(field, "must be a nonnegative integer")
        if value > np.iinfo(np.int32).max:
            raise _field_error(field, "must fit the loaded int32 schema")
    if n_iter > iteration_limit:
        raise _field_error("n_iter", "cannot exceed iteration_limit")
    if attempted_iteration < n_iter or attempted_iteration > iteration_limit:
        raise _field_error(
            "attempted_iteration", "must be between n_iter and iteration_limit"
        )

    objective = state["objective"]
    history = state["history"]
    attempted_objective = state["attempted_objective"]
    if objective.shape != () or objective.dtype != dtype:
        raise _field_error("fit.objective", "must be a computation-dtype scalar")
    if history.ndim != 1 or history.dtype != dtype:
        raise _field_error("fit.history", "must be a rank-one computation-dtype array")
    if attempted_objective.shape != () or attempted_objective.dtype != dtype:
        raise _field_error(
            "fit.attempted_objective", "must be a computation-dtype scalar"
        )

    objective_valid = state["objective_valid"]
    objective_is_finite = bool(np.isfinite(objective))
    history_is_valid = (
        history.shape == (n_iter + 1,)
        and bool(np.all(np.isfinite(history)))
        and _float_bits_equal(history[-1], objective)
    )
    if objective_valid != (objective_is_finite and history_is_valid):
        raise _field_error(
            "objective_valid", "does not match fit.objective and fit.history"
        )
    if not objective_valid:
        if status != int(FitStatus.NUMERICAL_FAILURE):
            raise _field_error(
                "status",
                "an invalid initial objective requires numerical_failure",
            )
        if history.shape != (0,):
            raise _field_error("fit.history", "must be empty for an invalid initial objective")
        if n_iter != 0:
            raise _field_error("n_iter", "must be zero for an invalid initial objective")
        if objective_is_finite:
            raise _field_error("fit.objective", "must be nonfinite when objective_valid is false")
        if not _params_bits_equal(parameters, initial_parameters):
            raise _field_error(
                "parameters", "must equal initial_parameters after an invalid initial objective"
            )
        if attempted_iteration != 0:
            raise _field_error(
                "attempted_iteration", "must be zero for an invalid initial objective"
            )

    attempted_valid = state["attempted_objective_valid"]
    attempted_is_finite = bool(np.isfinite(attempted_objective))
    if attempted_valid != attempted_is_finite:
        raise _field_error(
            "attempted_objective_valid",
            "does not match fit.attempted_objective finiteness",
        )

    for field in ("fit.factor_jitter", "fit.covariance_ridge"):
        value = state[field]
        if value.shape != () or value.dtype != dtype:
            raise _field_error(field, "must be a computation-dtype scalar")
        _validate_nonnegative_scalar(value, field=field)

    if mode == int(FitMode.CONVERGED):
        if state["tol"] is None:
            raise _field_error("fit.tol", "is required in converged mode")
        if state["decrease_tol"] is None:
            raise _field_error(
                "fit.decrease_tol", "is required in converged mode"
            )
        for field, value in (
            ("fit.tol", state["tol"]),
            ("fit.decrease_tol", state["decrease_tol"]),
        ):
            if value.shape != () or value.dtype != dtype:
                raise _field_error(field, "must be a computation-dtype scalar")
            _validate_nonnegative_scalar(value, field=field)
    else:
        if state["tol"] is not None or state["decrease_tol"] is not None:
            raise _field_error(
                "fit.tol and fit.decrease_tol", "are forbidden in fixed_steps mode"
            )

    converged = state["converged"]
    numerical_failure = state["numerical_failure"]
    collapsed = state["collapsed"]
    if converged != (status == int(FitStatus.CONVERGED)):
        raise _field_error("converged", "is inconsistent with status")
    if numerical_failure != (status == int(FitStatus.NUMERICAL_FAILURE)):
        raise _field_error("numerical_failure", "is inconsistent with status")
    if collapsed != (status == int(FitStatus.COMPONENT_COLLAPSED)):
        raise _field_error("collapsed", "is inconsistent with status")
    if mode == int(FitMode.FIXED_STEPS) and converged:
        raise _field_error("converged", "fixed_steps mode cannot converge")
    if status in (int(FitStatus.CONVERGED), int(FitStatus.MAX_ITER), int(FitStatus.OBJECTIVE_DECREASED)) and mode != int(FitMode.CONVERGED):
        raise _field_error("mode", "is inconsistent with status")
    if status == int(FitStatus.FIXED_STEPS_COMPLETE) and mode != int(FitMode.FIXED_STEPS):
        raise _field_error("mode", "is inconsistent with fixed_steps_complete status")
    if status == int(FitStatus.MAX_ITER) and n_iter != iteration_limit:
        raise _field_error("n_iter", "max_iter status requires n_iter == iteration_limit")
    if status == int(FitStatus.FIXED_STEPS_COMPLETE) and n_iter != iteration_limit:
        raise _field_error(
            "n_iter", "fixed_steps_complete requires n_iter == iteration_limit"
        )
    if status == int(FitStatus.CONVERGED) and n_iter == 0:
        raise _field_error("n_iter", "converged status requires an accepted update")

    collapsed_components = state["collapsed_components"]
    k = np.asarray(parameters.weights).shape[0]
    if collapsed_components.dtype != np.dtype(bool) or collapsed_components.shape != (k,):
        raise _field_error(
            "fit.collapsed_components", f"must have bool shape ({k},)"
        )
    has_collapsed_component = bool(np.any(collapsed_components))
    if collapsed and not has_collapsed_component:
        raise _field_error(
            "fit.collapsed_components", "component collapse requires at least one true entry"
        )
    if not collapsed and has_collapsed_component:
        raise _field_error(
            "fit.collapsed_components", "must be all false without component collapse"
        )

    if status in (
        int(FitStatus.OBJECTIVE_DECREASED),
        int(FitStatus.NUMERICAL_FAILURE),
        int(FitStatus.COMPONENT_COLLAPSED),
    ) and n_iter == 0 and not _params_bits_equal(parameters, initial_parameters):
        raise _field_error(
            "parameters", "terminal failure at iteration zero must roll back exactly"
        )

    if status in (int(FitStatus.CONVERGED), int(FitStatus.FIXED_STEPS_COMPLETE)):
        if attempted_iteration != n_iter:
            raise _field_error(
                "attempted_iteration", "successful terminal status must identify n_iter"
            )
    elif status == int(FitStatus.MAX_ITER) and attempted_iteration != n_iter:
        raise _field_error("attempted_iteration", "max_iter must identify n_iter")
    elif objective_valid and status in (
        int(FitStatus.OBJECTIVE_DECREASED),
        int(FitStatus.NUMERICAL_FAILURE),
        int(FitStatus.COMPONENT_COLLAPSED),
    ) and attempted_iteration != n_iter + 1:
        raise _field_error(
            "attempted_iteration", "failed candidate must identify n_iter + 1"
        )

    if status == int(FitStatus.OBJECTIVE_DECREASED):
        if not attempted_valid:
            raise _field_error(
                "attempted_objective_valid", "objective_decreased requires a finite attempt"
            )
        decrease_tol = float(state["decrease_tol"])
        change = (float(attempted_objective) - float(objective)) / max(
            1.0, abs(float(objective))
        )
        if not change < -decrease_tol:
            raise _field_error(
                "status", "objective_decreased is inconsistent with the recorded objective"
            )

    if family == "grouped":
        informative_weight = state["informative_weight"]
        if informative_weight.shape != () or informative_weight.dtype != dtype:
            raise _field_error(
                "fit.informative_weight", "must be a computation-dtype scalar"
            )
        if not np.isfinite(float(informative_weight)) or not float(informative_weight) > 0.0:
            raise _field_error(
                "fit.informative_weight", "must be finite and strictly positive"
            )
        group_failure = state["group_numerical_failure"]
        failed_pairs = state["failed_pairs"]
        if group_failure.dtype != np.dtype(bool) or group_failure.ndim != 1:
            raise _field_error(
                "fit.group_numerical_failure", "must be a rank-one bool array"
            )
        if failed_pairs.dtype != np.dtype(bool) or failed_pairs.ndim != 2:
            raise _field_error("fit.failed_pairs", "must be a rank-two bool array")
        if failed_pairs.shape[1] != k:
            raise _field_error(
                "fit.failed_pairs", "component dimension must equal n_components"
            )
        if not numerical_failure:
            if bool(np.any(group_failure)):
                raise _field_error(
                    "fit.group_numerical_failure",
                    "must be all false without a numerical failure",
                )
            if bool(np.any(failed_pairs)):
                raise _field_error(
                    "fit.failed_pairs",
                    "must be all false without a numerical failure",
                )
        stage = state["failure_stage"]
        if stage not in _STAGE_TO_STRING:
            raise _field_error("failure_stage", "is unknown")
        if status == int(FitStatus.COMPONENT_COLLAPSED):
            if stage != int(GroupedFailureStage.M_STEP):
                raise _field_error(
                    "failure_stage", "component collapse must occur at m_step"
                )
        elif status == int(FitStatus.NUMERICAL_FAILURE):
            if stage == int(GroupedFailureStage.NONE):
                raise _field_error(
                    "failure_stage", "numerical failure requires a terminal stage"
                )
        elif stage != int(GroupedFailureStage.NONE):
            raise _field_error(
                "failure_stage", "is inconsistent with the terminal status"
            )


def _normalize_fit_result(result: object, *, family: str) -> dict[str, Any]:
    expected_type = FitResult if family == "identity" else GroupedGeneralFitResult
    if not isinstance(result, expected_type):
        raise ArtifactFormatError(
            f"result must be a {expected_type.__name__} instance"
        )
    parameters = _host_parameters(result.parameters, field="parameters")
    initial_parameters = _host_parameters(
        result.initial_parameters, field="initial_parameters"
    )
    dtype = np.asarray(parameters.means).dtype

    expected_metadata = (
        current_result_metadata()
        if family == "identity"
        else current_general_result_metadata()
    )
    if result.metadata != expected_metadata:
        raise _field_error("metadata", "contract ID/version is not supported")
    if not isinstance(result.initialization, InitializationProvenance):
        raise _field_error("initialization", "has an unsupported type")
    if result.initialization != user_supplied_initialization():
        raise _field_error(
            "initialization", "kind must be user_supplied"
        )

    state: dict[str, Any] = {
        "parameters": parameters,
        "initial_parameters": initial_parameters,
        "objective": _host_float_scalar(
            result.objective, field="fit.objective", dtype=dtype
        ),
        "objective_valid": _host_bool_scalar(
            result.objective_valid, field="objective_valid"
        ),
        "history": _little_c_array(_host_array(result.history, field="fit.history")),
        "n_iter": _host_int_scalar(result.n_iter, field="n_iter"),
        "iteration_limit": _host_int_scalar(
            result.iteration_limit, field="iteration_limit"
        ),
        "converged": _host_bool_scalar(result.converged, field="converged"),
        "status": _host_int_scalar(result.status, field="status"),
        "mode": _host_int_scalar(result.mode, field="mode"),
        "attempted_iteration": _host_int_scalar(
            result.attempted_iteration, field="attempted_iteration"
        ),
        "attempted_objective": _host_float_scalar(
            result.attempted_objective,
            field="fit.attempted_objective",
            dtype=dtype,
        ),
        "attempted_objective_valid": _host_bool_scalar(
            result.attempted_objective_valid,
            field="attempted_objective_valid",
        ),
        "numerical_failure": _host_bool_scalar(
            result.numerical_failure, field="numerical_failure"
        ),
        "collapsed": _host_bool_scalar(result.collapsed, field="collapsed"),
        "collapsed_components": _host_bool_array(
            result.collapsed_components, field="fit.collapsed_components"
        ),
        "fit.factor_jitter": _host_float_scalar(
            result.factor_jitter, field="fit.factor_jitter", dtype=dtype
        ),
        "fit.covariance_ridge": _host_float_scalar(
            result.covariance_ridge,
            field="fit.covariance_ridge",
            dtype=dtype,
        ),
        "tol": None,
        "decrease_tol": None,
    }
    if result.tol is not None:
        state["tol"] = _host_float_scalar(
            result.tol, field="fit.tol", dtype=dtype
        )
    if result.decrease_tol is not None:
        state["decrease_tol"] = _host_float_scalar(
            result.decrease_tol, field="fit.decrease_tol", dtype=dtype
        )
    if family == "grouped":
        state.update(
            {
                "failure_stage": _host_int_scalar(
                    result.failure_stage, field="failure_stage"
                ),
                "group_numerical_failure": _host_bool_array(
                    result.group_numerical_failure,
                    field="fit.group_numerical_failure",
                ),
                "failed_pairs": _host_bool_array(
                    result.failed_pairs, field="fit.failed_pairs"
                ),
                "informative_weight": _host_float_scalar(
                    result.informative_weight,
                    field="fit.informative_weight",
                    dtype=dtype,
                ),
            }
        )
    _validate_fit_state(state, family=family)
    return state


def _objective_semantics(family: str, factor_jitter: np.ndarray) -> str:
    nonzero = bool(float(factor_jitter) != 0.0)
    if family == "identity":
        return (
            "identity_fixed_jitter_effective_observed_mean"
            if nonzero
            else "identity_exact_observed_mean"
        )
    return (
        "general_fixed_jitter_effective_informative_weighted_observed_mean"
        if nonzero
        else "general_exact_informative_weighted_observed_mean"
    )


def _fit_record(
    result: object, *, family: str
) -> tuple[dict[str, object], dict[str, bytes]]:
    state = _normalize_fit_result(result, family=family)
    parameters = state["parameters"]
    initial_parameters = state["initial_parameters"]
    arrays: dict[str, np.ndarray] = {
        "parameters.weights": parameters.weights,
        "parameters.means": parameters.means,
        "parameters.covariances": parameters.covariances,
        "initial_parameters.weights": initial_parameters.weights,
        "initial_parameters.means": initial_parameters.means,
        "initial_parameters.covariances": initial_parameters.covariances,
        "fit.objective": state["objective"],
        "fit.history": state["history"],
        "fit.attempted_objective": state["attempted_objective"],
        "fit.factor_jitter": state["fit.factor_jitter"],
        "fit.covariance_ridge": state["fit.covariance_ridge"],
        "fit.collapsed_components": state["collapsed_components"],
    }
    if state["mode"] == int(FitMode.CONVERGED):
        arrays["fit.tol"] = state["tol"]
        arrays["fit.decrease_tol"] = state["decrease_tol"]
    if family == "grouped":
        arrays.update(
            {
                "fit.group_numerical_failure": state["group_numerical_failure"],
                "fit.failed_pairs": state["failed_pairs"],
                "fit.informative_weight": state["informative_weight"],
            }
        )
    descriptors, payloads = _array_members(arrays)
    k, d = parameters.means.shape
    if family == "identity":
        artifact_kind = "identity_fit_result"
        record_id = IDENTITY_FIT_RECORD_ID
        record_version = IDENTITY_FIT_RECORD_VERSION
        contract_id = CONTRACT_ID
        contract_version = CONTRACT_VERSION
        model: dict[str, object] = {
            "dtype": parameters.means.dtype.name,
            "latent_dimension": d,
            "n_components": k,
        }
    else:
        artifact_kind = "grouped_general_fit_result"
        record_id = GROUPED_GENERAL_FIT_RECORD_ID
        record_version = GROUPED_GENERAL_FIT_RECORD_VERSION
        contract_id = GENERAL_CONTRACT_ID
        contract_version = GENERAL_CONTRACT_VERSION
        model = {
            "dtype": parameters.means.dtype.name,
            "latent_dimension": d,
            "n_components": k,
            "n_groups": int(state["group_numerical_failure"].shape[0]),
            "n_samples": int(state["failed_pairs"].shape[0]),
        }
    fit: dict[str, object] = {
        "attempted_iteration": state["attempted_iteration"],
        "attempted_objective_valid": state["attempted_objective_valid"],
        "collapsed": state["collapsed"],
        "converged": state["converged"],
        "initialization": {"kind": "user_supplied"},
        "iteration_limit": state["iteration_limit"],
        "mode": _MODE_TO_STRING[state["mode"]],
        "n_iter": state["n_iter"],
        "numerical_failure": state["numerical_failure"],
        "objective_semantics": _objective_semantics(
            family, state["fit.factor_jitter"]
        ),
        "objective_valid": state["objective_valid"],
        "ridge_application": "post_em_latent_covariance",
        "status": _STATUS_TO_STRING[state["status"]],
    }
    if family == "grouped":
        fit["failure_stage"] = _STAGE_TO_STRING[state["failure_stage"]]
    manifest: dict[str, object] = {
        "arrays": descriptors,
        "artifact_kind": artifact_kind,
        "contract_id": contract_id,
        "contract_version": contract_version,
        "fit": fit,
        "format_id": FORMAT_ID,
        "format_version": FORMAT_VERSION,
        "model": model,
        "package_version": __version__,
        "record_id": record_id,
        "record_version": record_version,
    }
    return manifest, payloads


def save_identity_fit_result(path, result, *, overwrite=False):
    """Save one validated temporary identity fit result."""

    manifest, payloads = _fit_record(result, family="identity")
    _write_artifact(path, manifest, payloads, overwrite=overwrite)


def save_grouped_general_fit_result(path, result, *, overwrite=False):
    """Save one validated temporary grouped-general fit result."""

    manifest, payloads = _fit_record(result, family="grouped")
    _write_artifact(path, manifest, payloads, overwrite=overwrite)


def _validate_limits(limits: object) -> ArtifactLimits:
    if not isinstance(limits, ArtifactLimits):
        raise TypeError("limits must be an ArtifactLimits instance")
    for field, value in zip(limits._fields, limits, strict=True):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            readable = field.replace("max_", "").replace("_", " ")
            raise ValueError(f"limit {readable} must be a nonnegative integer")
    return limits


def _safe_member_path(name: object) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        return False
    pieces = name.split("/")
    return all(piece not in ("", ".", "..") for piece in pieces)


def _validate_zip_boundaries(
    archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo]
) -> None:
    """Reject self-extracting prefixes, overlays, and noncanonical EOCD data."""

    stream = archive.fp
    if stream is None:
        raise ArtifactFormatError("ZIP archive stream is unavailable")
    try:
        previous_offset = stream.tell()
        stream.seek(0, os.SEEK_END)
        archive_size = stream.tell()
        if archive_size < 22 or not infos or infos[0].header_offset != 0:
            raise ArtifactFormatError(
                "opaque bytes outside the canonical ZIP container are forbidden"
            )
        stream.seek(archive_size - 22)
        eocd = stream.read(22)
        if len(eocd) != 22:
            raise ArtifactFormatError("ZIP end-of-central-directory is truncated")
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_length,
        ) = struct.unpack("<4s4H2LH", eocd)
        if signature != b"PK\x05\x06" or comment_length != 0:
            raise ArtifactFormatError(
                "opaque bytes outside the canonical ZIP container are forbidden"
            )
        if (
            disk_number != 0
            or central_disk != 0
            or disk_entries != len(infos)
            or total_entries != len(infos)
            or central_offset != archive.start_dir
            or central_offset + central_size != archive_size - 22
        ):
            raise ArtifactFormatError(
                "ZIP central-directory boundaries are not canonical"
            )
    except ArtifactFormatError:
        raise
    except (OSError, struct.error, ValueError) as error:
        raise ArtifactFormatError("ZIP container boundaries are invalid") from error
    finally:
        try:
            stream.seek(previous_offset)
        except (OSError, UnboundLocalError):
            pass


def _validate_local_zip_header(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> int:
    """Validate metadata that the central ZIP directory can otherwise hide."""

    stream = archive.fp
    if stream is None:
        raise ArtifactFormatError("ZIP archive stream is unavailable")
    try:
        previous_offset = stream.tell()
        stream.seek(info.header_offset)
        header = stream.read(30)
        if len(header) != 30:
            raise ArtifactFormatError(
                f"ZIP local header for {info.filename} is truncated"
            )
        (
            signature,
            extract_version,
            flag_bits,
            compress_type,
            dos_time,
            dos_date,
            crc,
            compress_size,
            file_size,
            filename_length,
            extra_length,
        ) = struct.unpack("<4s5H3L2H", header)
        if signature != b"PK\x03\x04":
            raise ArtifactFormatError(
                f"ZIP local header for {info.filename} has an invalid signature"
            )
        if compress_type != zipfile.ZIP_STORED:
            raise ArtifactFormatError(
                "ZIP local-header compression must be ZIP_STORED"
            )
        if (dos_time, dos_date) != (0, 33):
            raise ArtifactFormatError(
                "ZIP local-header member timestamp is not canonical"
            )
        if extract_version != info.extract_version or flag_bits != info.flag_bits:
            raise ArtifactFormatError(
                f"ZIP local-header flags or version for {info.filename} "
                "disagree with the central directory"
            )
        if (crc, compress_size, file_size) != (
            info.CRC,
            info.compress_size,
            info.file_size,
        ):
            raise ArtifactFormatError(
                f"ZIP local-header sizes or CRC for {info.filename} "
                "disagree with the central directory"
            )
        filename = stream.read(filename_length)
        extra = stream.read(extra_length)
        if len(filename) != filename_length or len(extra) != extra_length:
            raise ArtifactFormatError(
                f"ZIP local header for {info.filename} is truncated"
            )
        encoding = "utf-8" if flag_bits & 0x800 else "cp437"
        try:
            expected_filename = info.filename.encode(encoding)
        except UnicodeEncodeError as error:
            raise ArtifactFormatError(
                f"ZIP member name {info.filename!r} has inconsistent encoding"
            ) from error
        if filename != expected_filename:
            raise ArtifactFormatError(
                f"ZIP local-header member name for {info.filename} "
                "disagrees with the central directory"
            )
        if extra != b"":
            raise ArtifactFormatError(
                f"ZIP local-header extra field for {info.filename} is forbidden"
            )
        local_record_end = (
            info.header_offset
            + 30
            + filename_length
            + extra_length
            + info.compress_size
        )
    except ArtifactFormatError:
        raise
    except (OSError, struct.error, ValueError) as error:
        raise ArtifactFormatError(
            f"ZIP local header for {info.filename} is invalid"
        ) from error
    finally:
        try:
            stream.seek(previous_offset)
        except (OSError, UnboundLocalError):
            pass
    return local_record_end


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    try:
        return archive.read(info)
    except zipfile.BadZipFile as error:
        message = str(error).lower()
        if "crc" in message:
            raise ArtifactFormatError(
                f"ZIP CRC validation failed for member {info.filename}"
            ) from error
        raise ArtifactFormatError(
            f"ZIP member {info.filename} is corrupt or truncated"
        ) from error
    except (EOFError, OSError, RuntimeError) as error:
        raise ArtifactFormatError(
            f"ZIP member {info.filename} is corrupt or unreadable"
        ) from error


class _DuplicateJSONKey(ValueError):
    pass


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"nonstandard JSON constant {value}")


def _parse_manifest(payload: bytes) -> dict[str, object]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ArtifactFormatError("manifest is not valid UTF-8") from error
    if text.startswith("\ufeff"):
        raise ArtifactFormatError("manifest must not contain a byte-order mark")
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except _DuplicateJSONKey as error:
        raise ArtifactFormatError(str(error)) from error
    except (json.JSONDecodeError, ValueError, TypeError, RecursionError) as error:
        raise ArtifactFormatError("manifest is not canonical JSON") from error
    if not isinstance(decoded, dict):
        raise ArtifactFormatError("manifest JSON must contain one object")
    if _canonical_json(decoded) != payload:
        raise ArtifactFormatError("manifest bytes are not canonical")
    return decoded


def _exact_fields(value: object, expected: set[str], *, field: str) -> dict:
    if not isinstance(value, dict):
        raise _field_error(field, "must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = "has an invalid field set"
        if missing:
            detail += f"; missing {', '.join(missing)}"
        if extra:
            detail += f"; extra {', '.join(extra)}"
        raise _field_error(field, detail)
    return value


def _manifest_string(manifest: dict, field: str) -> str:
    value = manifest[field]
    if not isinstance(value, str):
        raise _field_error(field, "must be a string")
    return value


def _manifest_bool(record: dict, field: str) -> bool:
    value = record[field]
    if not isinstance(value, bool):
        raise _field_error(field, "must be boolean")
    return value


def _manifest_int(
    record: dict,
    field: str,
    *,
    positive: bool = False,
    maximum: int | None = None,
) -> int:
    value = record[field]
    if not isinstance(value, int) or isinstance(value, bool):
        raise _field_error(field, "must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise _field_error(field, f"must be {qualifier}")
    if maximum is not None and value > maximum:
        raise _field_error(field, f"must not exceed {maximum}")
    return value


def _validate_fit_manifest(fit_value: object, *, family: str) -> dict[str, Any]:
    expected_fields = set(_COMMON_FIT_FIELDS)
    if family == "grouped":
        expected_fields.add("failure_stage")
    fit = _exact_fields(fit_value, expected_fields, field="fit")
    initialization = _exact_fields(
        fit["initialization"], {"kind"}, field="fit.initialization"
    )
    if initialization["kind"] != "user_supplied":
        raise _field_error(
            "fit.initialization", "kind must be user_supplied"
        )
    for field in ("mode", "status", "objective_semantics", "ridge_application"):
        if not isinstance(fit[field], str):
            raise _field_error(field, "must be a string")
    if fit["mode"] not in _STRING_TO_MODE:
        raise _field_error("mode", "is unknown")
    if fit["status"] not in _STRING_TO_STATUS:
        raise _field_error("status", "must be a supported terminal string")
    result: dict[str, Any] = {
        "mode": _STRING_TO_MODE[fit["mode"]],
        "status": _STRING_TO_STATUS[fit["status"]],
        "n_iter": _manifest_int(fit, "n_iter", maximum=np.iinfo(np.int32).max),
        "iteration_limit": _manifest_int(
            fit, "iteration_limit", maximum=np.iinfo(np.int32).max
        ),
        "attempted_iteration": _manifest_int(
            fit, "attempted_iteration", maximum=np.iinfo(np.int32).max
        ),
        "converged": _manifest_bool(fit, "converged"),
        "objective_valid": _manifest_bool(fit, "objective_valid"),
        "attempted_objective_valid": _manifest_bool(
            fit, "attempted_objective_valid"
        ),
        "numerical_failure": _manifest_bool(fit, "numerical_failure"),
        "collapsed": _manifest_bool(fit, "collapsed"),
        "objective_semantics": fit["objective_semantics"],
        "ridge_application": fit["ridge_application"],
    }
    if family == "grouped":
        if not isinstance(fit["failure_stage"], str):
            raise _field_error("failure_stage", "must be a string")
        if fit["failure_stage"] not in _STRING_TO_STAGE:
            raise _field_error("failure_stage", "is unknown")
        result["failure_stage"] = _STRING_TO_STAGE[fit["failure_stage"]]
    if result["n_iter"] > result["iteration_limit"]:
        raise _field_error("n_iter", "cannot exceed iteration_limit")
    if not result["objective_valid"] and result["n_iter"] != 0:
        raise _field_error(
            "objective_valid", "an invalid initial objective requires n_iter zero"
        )
    if not result["objective_valid"] and result["status"] != int(
        FitStatus.NUMERICAL_FAILURE
    ):
        raise _field_error(
            "status", "an invalid initial objective requires numerical_failure"
        )
    if result["converged"] != (result["status"] == int(FitStatus.CONVERGED)):
        raise _field_error("converged", "is inconsistent with status")
    if result["numerical_failure"] != (
        result["status"] == int(FitStatus.NUMERICAL_FAILURE)
    ):
        raise _field_error("numerical_failure", "is inconsistent with status")
    if result["collapsed"] != (
        result["status"] == int(FitStatus.COMPONENT_COLLAPSED)
    ):
        raise _field_error("collapsed", "is inconsistent with status")
    return result


def _record_header(
    manifest: dict[str, object], *, expected_kind: str
) -> tuple[str, dict[str, Any], dict[str, tuple[str, tuple[int, ...]]]]:
    if "artifact_kind" not in manifest:
        raise _field_error("manifest field", "artifact_kind is missing")
    artifact_kind = _manifest_string(manifest, "artifact_kind")
    known_kinds = {
        "parameters",
        "identity_fit_result",
        "grouped_general_fit_result",
    }
    if artifact_kind not in known_kinds:
        raise _field_error("artifact_kind", "is unknown")
    if artifact_kind != expected_kind:
        raise _field_error(
            "artifact_kind", f"expected {expected_kind}, received {artifact_kind}"
        )
    top_fields = {
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
    if artifact_kind != "parameters":
        top_fields.add("fit")
    _exact_fields(manifest, top_fields, field="manifest field")
    for field in (
        "format_id",
        "format_version",
        "record_id",
        "record_version",
        "contract_id",
        "contract_version",
        "package_version",
    ):
        _manifest_string(manifest, field)
    if manifest["format_id"] != FORMAT_ID:
        raise _field_error("format_id", "is unknown")
    if manifest["format_version"] != FORMAT_VERSION:
        raise _field_error("format_version", "is unsupported")

    expected_records = {
        "parameters": (PARAMETERS_RECORD_ID, PARAMETERS_RECORD_VERSION),
        "identity_fit_result": (
            IDENTITY_FIT_RECORD_ID,
            IDENTITY_FIT_RECORD_VERSION,
        ),
        "grouped_general_fit_result": (
            GROUPED_GENERAL_FIT_RECORD_ID,
            GROUPED_GENERAL_FIT_RECORD_VERSION,
        ),
    }
    expected_record_id, expected_record_version = expected_records[artifact_kind]
    if manifest["record_id"] != expected_record_id:
        raise _field_error("record_id", "is unknown for artifact_kind")
    if manifest["record_version"] != expected_record_version:
        raise _field_error("record_version", "is unsupported")

    contract = (manifest["contract_id"], manifest["contract_version"])
    if artifact_kind == "parameters":
        if contract not in _SUPPORTED_CONTRACTS:
            if manifest["contract_id"] not in {pair[0] for pair in _SUPPORTED_CONTRACTS}:
                raise _field_error("contract_id", "is unknown")
            raise _field_error("contract_version", "is unsupported")
    else:
        expected_contract = (
            (CONTRACT_ID, CONTRACT_VERSION)
            if artifact_kind == "identity_fit_result"
            else (GENERAL_CONTRACT_ID, GENERAL_CONTRACT_VERSION)
        )
        if contract != expected_contract:
            raise _field_error(
                "contract", "fit record cannot be relabelled to another contract"
            )

    model_fields = {"dtype", "latent_dimension", "n_components"}
    family = "parameters"
    if artifact_kind == "identity_fit_result":
        family = "identity"
    elif artifact_kind == "grouped_general_fit_result":
        family = "grouped"
        model_fields |= {"n_groups", "n_samples"}
    model = _exact_fields(manifest["model"], model_fields, field="model field")
    if not isinstance(model["dtype"], str) or model["dtype"] not in {
        "float32",
        "float64",
    }:
        raise _field_error("model dtype", "must be float32 or float64")
    k = _manifest_int(model, "n_components", positive=True)
    d = _manifest_int(model, "latent_dimension", positive=True)
    n_groups = n_samples = None
    if family == "grouped":
        n_groups = _manifest_int(model, "n_groups", positive=True)
        n_samples = _manifest_int(model, "n_samples", positive=True)

    fit_state: dict[str, Any] = {}
    if family in {"identity", "grouped"}:
        fit_state = _validate_fit_manifest(manifest["fit"], family=family)

    dtype_name = model["dtype"]
    schemas: dict[str, tuple[str, tuple[int, ...]]] = {
        "parameters.weights": (dtype_name, (k,)),
        "parameters.means": (dtype_name, (k, d)),
        "parameters.covariances": (dtype_name, (k, d, d)),
    }
    if family in {"identity", "grouped"}:
        history_length = (
            fit_state["n_iter"] + 1 if fit_state["objective_valid"] else 0
        )
        schemas.update(
            {
                "initial_parameters.weights": (dtype_name, (k,)),
                "initial_parameters.means": (dtype_name, (k, d)),
                "initial_parameters.covariances": (dtype_name, (k, d, d)),
                "fit.objective": (dtype_name, ()),
                "fit.history": (dtype_name, (history_length,)),
                "fit.attempted_objective": (dtype_name, ()),
                "fit.factor_jitter": (dtype_name, ()),
                "fit.covariance_ridge": (dtype_name, ()),
                "fit.collapsed_components": ("bool", (k,)),
            }
        )
        if fit_state["mode"] == int(FitMode.CONVERGED):
            schemas["fit.tol"] = (dtype_name, ())
            schemas["fit.decrease_tol"] = (dtype_name, ())
    if family == "grouped":
        schemas.update(
            {
                "fit.informative_weight": (dtype_name, ()),
                "fit.group_numerical_failure": ("bool", (n_groups,)),
                "fit.failed_pairs": ("bool", (n_samples, k)),
            }
        )

    arrays = manifest["arrays"]
    if not isinstance(arrays, dict):
        raise _field_error("arrays", "must be an object")
    actual_names = set(arrays)
    expected_names = set(schemas)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        detail = "logical array set is invalid"
        if missing:
            detail += f"; missing {', '.join(missing)}"
        if extra:
            detail += f"; extra {', '.join(extra)}"
        raise _field_error("arrays", detail)
    return family, fit_state, schemas


def _checked_array_size(
    shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    field: str,
    max_data_bytes: int,
) -> tuple[int, int]:
    """Return ``(element_count, nbytes)`` without fixed-width overflow."""

    if any(dimension == 0 for dimension in shape):
        return 0, 0
    max_elements = max_data_bytes // dtype.itemsize
    element_count = 1
    for dimension in shape:
        if element_count > max_elements // dimension:
            raise ArtifactLimitError(
                f"{field} array data exceeds the member byte limit"
            )
        element_count *= dimension
    return element_count, element_count * dtype.itemsize


def _validate_descriptor(
    logical_name: str,
    value: object,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    limits: ArtifactLimits,
) -> dict[str, object]:
    descriptor = _exact_fields(
        value, _DESCRIPTOR_FIELDS, field=f"array descriptor {logical_name} field"
    )
    expected_path = f"arrays/{logical_name}.npy"
    if descriptor["path"] != expected_path:
        raise _field_error(
            f"{logical_name} path", f"must be {expected_path}"
        )
    if not isinstance(descriptor["dtype"], str):
        raise _field_error(f"{logical_name} dtype", "must be a string")
    if descriptor["dtype"].startswith(">"):
        raise _field_error(f"{logical_name} endian", "must be little-endian")
    if descriptor["dtype"] != expected_dtype:
        raise _field_error(
            f"{logical_name} dtype", f"must be {expected_dtype}"
        )
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in shape)
        or tuple(shape) != expected_shape
    ):
        dimension_terms = "n_components or latent_dimension"
        if logical_name == "fit.group_numerical_failure":
            dimension_terms = "n_groups"
        elif logical_name == "fit.failed_pairs":
            dimension_terms = "n_samples or n_components"
        raise _field_error(
            f"{logical_name} shape",
            f"must be {expected_shape} and agree with {dimension_terms}",
        )
    dtype = {
        "float32": np.dtype("<f4"),
        "float64": np.dtype("<f8"),
        "bool": np.dtype(bool),
    }[expected_dtype]
    _element_count, expected_data_nbytes = _checked_array_size(
        expected_shape,
        dtype,
        field=logical_name,
        max_data_bytes=limits.max_member_bytes,
    )
    if not isinstance(descriptor["data_nbytes"], int) or isinstance(
        descriptor["data_nbytes"], bool
    ) or descriptor["data_nbytes"] != expected_data_nbytes:
        raise _field_error(
            f"{logical_name} data_nbytes", "does not match dtype and shape"
        )
    if not isinstance(descriptor["member_nbytes"], int) or isinstance(
        descriptor["member_nbytes"], bool
    ) or descriptor["member_nbytes"] < 0:
        raise _field_error(f"{logical_name} member_nbytes", "must be nonnegative")
    digest = descriptor["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise _field_error(f"{logical_name} sha256", "must be a lowercase SHA-256")
    return descriptor


def _decode_npy(
    payload: bytes,
    *,
    logical_name: str,
    descriptor: dict[str, object],
    expected_dtype: str,
    expected_shape: tuple[int, ...],
    limits: ArtifactLimits,
) -> np.ndarray:
    if not payload.startswith(b"\x93NUMPY"):
        raise _field_error(logical_name, "NPY magic is invalid")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
    except (EOFError, ValueError) as error:
        raise _field_error(logical_name, "NPY header is invalid") from error
    if version != (1, 0):
        raise _field_error(logical_name, "NPY version must be 1.0")
    if len(payload) < 10:
        raise _field_error(logical_name, "NPY header is truncated")
    header_length = struct.unpack_from("<H", payload, 8)[0]
    if header_length > limits.max_npy_header_bytes:
        raise ArtifactLimitError(
            f"{logical_name} NPY header exceeds the header limit"
        )
    raw_header = payload[10 : 10 + header_length]
    if len(raw_header) != header_length:
        raise _field_error(logical_name, "NPY header is truncated")
    try:
        raw_header_dict = ast.literal_eval(
            raw_header.decode("latin1").strip()
        )
    except (SyntaxError, ValueError, TypeError, UnicodeDecodeError, RecursionError) as error:
        raise _field_error(logical_name, "NPY header is invalid") from error
    raw_descriptor = (
        raw_header_dict.get("descr")
        if isinstance(raw_header_dict, dict)
        else None
    )
    canonical_descriptor = {
        "float32": "<f4",
        "float64": "<f8",
        "bool": "|b1",
    }[expected_dtype]
    if raw_descriptor != canonical_descriptor:
        if isinstance(raw_descriptor, str) and raw_descriptor.startswith((">", "=")):
            raise _field_error(
                logical_name, "NPY endian marker must be explicitly little-endian"
            )
        raise _field_error(logical_name, "NPY dtype is forbidden or inconsistent")
    try:
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
            stream, max_header_size=limits.max_npy_header_bytes
        )
    except (EOFError, ValueError, TypeError, SyntaxError, RecursionError) as error:
        raise _field_error(logical_name, "NPY header is invalid") from error
    dtype = np.dtype(dtype)
    if dtype.byteorder == ">":
        raise _field_error(logical_name, "NPY endian must be little-endian")
    if fortran_order:
        raise _field_error(logical_name, "NPY Fortran order is forbidden")
    expected_np_dtype = {
        "float32": np.dtype("<f4"),
        "float64": np.dtype("<f8"),
        "bool": np.dtype(bool),
    }[expected_dtype]
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.kind not in ("f", "b")
        or dtype != expected_np_dtype
    ):
        raise _field_error(logical_name, "NPY dtype is forbidden or inconsistent")
    if tuple(shape) != expected_shape:
        raise _field_error(logical_name, "NPY shape is inconsistent with the record")
    data_offset = stream.tell()
    element_count, data_nbytes = _checked_array_size(
        tuple(shape),
        dtype,
        field=logical_name,
        max_data_bytes=limits.max_member_bytes,
    )
    expected_end = data_offset + data_nbytes
    if len(payload) < expected_end:
        raise _field_error(logical_name, "NPY data payload is truncated")
    if len(payload) > expected_end:
        raise _field_error(logical_name, "NPY payload contains trailing bytes")
    if descriptor["data_nbytes"] != data_nbytes:
        raise _field_error(logical_name, "data_nbytes does not match NPY data")
    try:
        array = np.frombuffer(
            payload,
            dtype=dtype,
            count=element_count,
            offset=data_offset,
        ).reshape(shape, order="C")
    except (TypeError, ValueError) as error:
        raise _field_error(logical_name, "NPY data payload is invalid") from error
    copied = array.copy()
    if expected_shape == ():
        return np.asarray(copied).reshape(())
    return np.ascontiguousarray(copied)


def _read_artifact(
    path: os.PathLike[str] | str,
    *,
    expected_kind: str,
    limits: object,
) -> tuple[dict[str, object], str, dict[str, Any], dict[str, np.ndarray]]:
    checked_limits = _validate_limits(limits)
    source = Path(path)
    try:
        mode = source.lstat().st_mode
    except (FileNotFoundError, OSError) as error:
        raise ArtifactFormatError("artifact path must name an ordinary file") from error
    if not stat.S_ISREG(mode):
        raise ArtifactFormatError("artifact path must name an ordinary file")
    try:
        archive = zipfile.ZipFile(source, "r")
    except (zipfile.BadZipFile, OSError, EOFError) as error:
        raise ArtifactFormatError("artifact is not a valid ZIP container") from error
    with archive:
        infos = archive.infolist()
        if len(infos) > checked_limits.max_members:
            raise ArtifactLimitError("ZIP member count exceeds the member limit")
        names = [info.filename for info in infos]
        if len(set(names)) != len(names):
            raise ArtifactFormatError("ZIP contains a duplicate member name")
        if archive.comment != b"":
            raise ArtifactFormatError("ZIP archive comment is forbidden")
        _validate_zip_boundaries(archive, infos)
        expected_local_offset = 0
        for info in infos:
            if info.header_offset != expected_local_offset:
                raise ArtifactFormatError(
                    "opaque bytes between canonical ZIP members are forbidden"
                )
            if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFDIR:
                raise ArtifactFormatError("ZIP directory entries are forbidden")
            if not _safe_member_path(info.filename):
                raise ArtifactFormatError(
                    f"ZIP member path {info.filename!r} is forbidden"
                )
            if info.file_size > checked_limits.max_member_bytes:
                raise ArtifactLimitError(
                    f"ZIP member {info.filename} exceeds the member byte limit"
                )
            if info.compress_type != zipfile.ZIP_STORED:
                raise ArtifactFormatError("ZIP compression must be ZIP_STORED")
            if info.flag_bits & 0x1:
                raise ArtifactFormatError("encrypted ZIP members are forbidden")
            if info.volume != 0:
                raise ArtifactFormatError(
                    "ZIP member start disk must be zero"
                )
            if info.date_time != (1980, 1, 1, 0, 0, 0):
                raise ArtifactFormatError("ZIP member timestamp is not canonical")
            if info.extra != b"":
                raise ArtifactFormatError("ZIP member extra field is forbidden")
            if info.comment != b"":
                raise ArtifactFormatError("ZIP member comment is forbidden")
            member_mode = info.external_attr >> 16
            if stat.S_IFMT(member_mode) != stat.S_IFREG:
                raise ArtifactFormatError("ZIP member must be a regular file")
            if stat.S_IMODE(member_mode) != 0o644:
                raise ArtifactFormatError("ZIP member permission must be 0644")
            if info.create_system != 3:
                raise ArtifactFormatError("ZIP member creator system is not canonical")
            if info.compress_size != info.file_size:
                raise ArtifactFormatError("stored ZIP member size metadata is inconsistent")
            expected_local_offset = _validate_local_zip_header(archive, info)
        if expected_local_offset != archive.start_dir:
            raise ArtifactFormatError(
                "opaque bytes before the ZIP central directory are forbidden"
            )
        total_size = sum(info.file_size for info in infos)
        if total_size > checked_limits.max_total_bytes:
            raise ArtifactLimitError("ZIP total uncompressed bytes exceed the total limit")
        if "manifest.json" not in names:
            raise ArtifactFormatError("ZIP is missing member manifest.json")
        manifest_info = archive.getinfo("manifest.json")
        if manifest_info.file_size > checked_limits.max_manifest_bytes:
            raise ArtifactLimitError("manifest bytes exceed the manifest limit")
        manifest_payload = _read_member(archive, manifest_info)
        manifest = _parse_manifest(manifest_payload)
        family, fit_state, schemas = _record_header(
            manifest, expected_kind=expected_kind
        )
        descriptors: dict[str, dict[str, object]] = {}
        for logical_name, (expected_dtype, expected_shape) in schemas.items():
            descriptors[logical_name] = _validate_descriptor(
                logical_name,
                manifest["arrays"][logical_name],
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
                limits=checked_limits,
            )
        expected_member_names = ["manifest.json"] + sorted(
            descriptor["path"] for descriptor in descriptors.values()
        )
        missing = sorted(set(expected_member_names) - set(names))
        extra = sorted(set(names) - set(expected_member_names))
        if missing:
            raise ArtifactFormatError(
                f"ZIP is missing listed member {', '.join(missing)}"
            )
        if extra:
            raise ArtifactFormatError(
                f"ZIP contains extra member {', '.join(extra)}"
            )
        if names != expected_member_names:
            raise ArtifactFormatError("ZIP member order is not canonical")

        info_by_name = {info.filename: info for info in infos}
        arrays: dict[str, np.ndarray] = {}
        for logical_name in sorted(schemas):
            expected_dtype, expected_shape = schemas[logical_name]
            descriptor = descriptors[logical_name]
            member_path = descriptor["path"]
            info = info_by_name[member_path]
            if descriptor["member_nbytes"] != info.file_size:
                raise _field_error(
                    f"{logical_name} member_nbytes",
                    "does not match the ZIP member",
                )
            payload = _read_member(archive, info)
            if len(payload) != descriptor["member_nbytes"]:
                raise _field_error(
                    f"{logical_name} member_nbytes", "does not match bytes read"
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest != descriptor["sha256"]:
                raise _field_error(
                    f"{logical_name} sha256", "does not match the complete NPY member"
                )
            arrays[logical_name] = _decode_npy(
                payload,
                logical_name=logical_name,
                descriptor=descriptor,
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
                limits=checked_limits,
            )
    return manifest, family, fit_state, arrays


def _parameters_from_arrays(
    arrays: dict[str, np.ndarray], *, prefix: str
) -> Params:
    parameters = Params(
        arrays[f"{prefix}.weights"],
        arrays[f"{prefix}.means"],
        arrays[f"{prefix}.covariances"],
    )
    return _validate_parameter_domain(parameters, field=prefix)


def _validate_device(device: object) -> None:
    if device is not None and not isinstance(device, jax.Device):
        raise TypeError("device must be one explicit single JAX device or None")


def _ensure_x64_available(dtype: np.dtype) -> None:
    if dtype == np.dtype(np.float64) and not jax.config.x64_enabled:
        raise PrecisionError(
            "cannot load a float64 artifact while JAX x64 support is disabled"
        )


def _device_array(array: np.ndarray, *, device: object):
    if device is None:
        return jax.device_put(array)
    return jax.device_put(array, device=device)


def _device_parameters(parameters: Params, *, device: object) -> Params:
    return Params(
        *(_device_array(np.asarray(array), device=device) for array in parameters)
    )


def load_parameters(path, *, device=None, limits=DEFAULT_LIMITS):
    """Load and validate one tagged parameter artifact."""

    _validate_device(device)
    manifest, _family, _fit_state, arrays = _read_artifact(
        path, expected_kind="parameters", limits=limits
    )
    parameters = _parameters_from_arrays(arrays, prefix="parameters")
    dtype = np.asarray(parameters.means).dtype
    _ensure_x64_available(dtype)
    placed = _device_parameters(parameters, device=device)
    return ParameterArtifact(
        parameters=placed,
        contract_id=manifest["contract_id"],
        contract_version=manifest["contract_version"],
        package_version=manifest["package_version"],
    )


def _fit_state_from_artifact(
    manifest: dict[str, object],
    family: str,
    fit_manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    parameters = _parameters_from_arrays(arrays, prefix="parameters")
    initial_parameters = _parameters_from_arrays(
        arrays, prefix="initial_parameters"
    )
    state: dict[str, Any] = dict(fit_manifest)
    state.update(
        {
            "parameters": parameters,
            "initial_parameters": initial_parameters,
            "objective": arrays["fit.objective"],
            "history": arrays["fit.history"],
            "attempted_objective": arrays["fit.attempted_objective"],
            "collapsed_components": arrays["fit.collapsed_components"],
            "fit.factor_jitter": arrays["fit.factor_jitter"],
            "fit.covariance_ridge": arrays["fit.covariance_ridge"],
            "tol": arrays.get("fit.tol"),
            "decrease_tol": arrays.get("fit.decrease_tol"),
        }
    )
    if fit_manifest["ridge_application"] != "post_em_latent_covariance":
        raise _field_error("ridge_application", "is unsupported")
    expected_semantics = _objective_semantics(
        family, arrays["fit.factor_jitter"]
    )
    if fit_manifest["objective_semantics"] != expected_semantics:
        raise _field_error(
            "objective_semantics",
            f"must be {expected_semantics} for the recorded factor jitter",
        )
    if family == "grouped":
        state.update(
            {
                "group_numerical_failure": arrays[
                    "fit.group_numerical_failure"
                ],
                "failed_pairs": arrays["fit.failed_pairs"],
                "informative_weight": arrays["fit.informative_weight"],
            }
        )
        model = manifest["model"]
        if state["group_numerical_failure"].shape != (model["n_groups"],):
            raise _field_error(
                "n_groups", "does not match fit.group_numerical_failure"
            )
        if state["failed_pairs"].shape[0] != model["n_samples"]:
            raise _field_error("n_samples", "does not match fit.failed_pairs")
    _validate_fit_state(state, family=family)
    return state


def _placed_fit_result(state: dict[str, Any], *, family: str, device: object):
    parameters = _device_parameters(state["parameters"], device=device)
    initial_parameters = _device_parameters(
        state["initial_parameters"], device=device
    )

    def put(name: str):
        return _device_array(np.asarray(state[name]), device=device)

    common = dict(
        parameters=parameters,
        initial_parameters=initial_parameters,
        objective=put("objective"),
        objective_valid=_device_array(
            np.asarray(state["objective_valid"], dtype=bool), device=device
        ),
        history=put("history"),
        n_iter=_device_array(
            np.asarray(state["n_iter"], dtype=np.int32), device=device
        ),
        iteration_limit=_device_array(
            np.asarray(state["iteration_limit"], dtype=np.int32), device=device
        ),
        converged=_device_array(
            np.asarray(state["converged"], dtype=bool), device=device
        ),
        status=_device_array(
            np.asarray(state["status"], dtype=np.int32), device=device
        ),
        mode=_device_array(
            np.asarray(state["mode"], dtype=np.int32), device=device
        ),
        attempted_iteration=_device_array(
            np.asarray(state["attempted_iteration"], dtype=np.int32),
            device=device,
        ),
        attempted_objective=put("attempted_objective"),
        attempted_objective_valid=_device_array(
            np.asarray(state["attempted_objective_valid"], dtype=bool),
            device=device,
        ),
        numerical_failure=_device_array(
            np.asarray(state["numerical_failure"], dtype=bool), device=device
        ),
        collapsed=_device_array(
            np.asarray(state["collapsed"], dtype=bool), device=device
        ),
        collapsed_components=put("collapsed_components"),
    )
    controls = dict(
        factor_jitter=put("fit.factor_jitter"),
        covariance_ridge=put("fit.covariance_ridge"),
        tol=None if state["tol"] is None else put("tol"),
        decrease_tol=(
            None if state["decrease_tol"] is None else put("decrease_tol")
        ),
        initialization=user_supplied_initialization(),
    )
    if family == "identity":
        return FitResult(
            **common,
            **controls,
            metadata=current_result_metadata(),
        )
    return GroupedGeneralFitResult(
        **common,
        failure_stage=_device_array(
            np.asarray(state["failure_stage"], dtype=np.int32), device=device
        ),
        group_numerical_failure=put("group_numerical_failure"),
        failed_pairs=put("failed_pairs"),
        informative_weight=put("informative_weight"),
        **controls,
        metadata=current_general_result_metadata(),
    )


def load_identity_fit_result(path, *, device=None, limits=DEFAULT_LIMITS):
    """Load and validate one temporary identity fit-result artifact."""

    _validate_device(device)
    manifest, family, fit_manifest, arrays = _read_artifact(
        path, expected_kind="identity_fit_result", limits=limits
    )
    state = _fit_state_from_artifact(
        manifest, family, fit_manifest, arrays
    )
    dtype = np.asarray(state["parameters"].means).dtype
    _ensure_x64_available(dtype)
    return _placed_fit_result(state, family=family, device=device)


def load_grouped_general_fit_result(
    path, *, device=None, limits=DEFAULT_LIMITS
):
    """Load and validate one temporary grouped-general fit-result artifact."""

    _validate_device(device)
    manifest, family, fit_manifest, arrays = _read_artifact(
        path, expected_kind="grouped_general_fit_result", limits=limits
    )
    state = _fit_state_from_artifact(
        manifest, family, fit_manifest, arrays
    )
    dtype = np.asarray(state["parameters"].means).dtype
    _ensure_x64_available(dtype)
    return _placed_fit_result(state, family=family, device=device)


__all__ = [
    "FORMAT_ID",
    "FORMAT_VERSION",
    "PARAMETERS_RECORD_ID",
    "PARAMETERS_RECORD_VERSION",
    "IDENTITY_FIT_RECORD_ID",
    "IDENTITY_FIT_RECORD_VERSION",
    "GROUPED_GENERAL_FIT_RECORD_ID",
    "GROUPED_GENERAL_FIT_RECORD_VERSION",
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
]
