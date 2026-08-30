"""NPY, parameter-domain, and fit-invariant serialization red gates."""

from __future__ import annotations

from dataclasses import replace
import importlib
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitMode, FitStatus
from development.general_grouped import GroupedFailureStage
from development.identity_xd import Params
from development.metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    GENERAL_CONTRACT_ID,
    GENERAL_CONTRACT_VERSION,
    InitializationProvenance,
    ResultMetadata,
    current_result_metadata,
)
from tests.development._serialization_test_helpers import (
    assert_arrays_bits_equal,
    canonical_json,
    grouped_fit_result,
    identity_fit_result,
    manifest_from,
    mutate_manifest,
    npy_bytes,
    read_members,
    replace_array,
    replace_array_payload,
    valid_parameters,
    write_members,
)


@pytest.fixture(scope="module")
def serialization():
    return importlib.import_module("development.serialization")


@pytest.fixture
def parameter_artifact(serialization, tmp_path):
    path = tmp_path / "parameters.artifact"
    serialization.save_parameters(
        path,
        valid_parameters(jnp.float64),
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )
    return path


@pytest.fixture
def identity_artifact(serialization, tmp_path):
    path = tmp_path / "identity.artifact"
    serialization.save_identity_fit_result(
        path, identity_fit_result(jnp.float64, "converged")
    )
    return path


@pytest.fixture
def grouped_artifact(serialization, tmp_path):
    path = tmp_path / "grouped.artifact"
    serialization.save_grouped_general_fit_result(
        path, grouped_fit_result(jnp.float64, "converged")
    )
    return path


def _assert_terms(error: BaseException, *terms: str) -> None:
    message = str(error).lower().replace("_", " ").replace("-", " ")
    for term in terms:
        normalized_term = term.lower().replace("_", " ").replace("-", " ")
        assert normalized_term in message


def _reject_parameters(serialization, path, *terms: str) -> None:
    with pytest.raises(serialization.ArtifactFormatError) as error:
        serialization.load_parameters(path)
    _assert_terms(error.value, *terms)


def _reject_identity(serialization, path, *terms: str) -> None:
    with pytest.raises(serialization.ArtifactFormatError) as error:
        serialization.load_identity_fit_result(path)
    _assert_terms(error.value, *terms)


def _reject_grouped(serialization, path, *terms: str) -> None:
    with pytest.raises(serialization.ArtifactFormatError) as error:
        serialization.load_grouped_general_fit_result(path)
    _assert_terms(error.value, *terms)


def _replace_array_case(
    source,
    target,
    logical_name,
    array,
    *,
    version=(1, 0),
    allow_pickle=False,
):
    replace_array(
        source,
        target,
        logical_name,
        np.asarray(array),
        version=version,
        allow_pickle=allow_pickle,
    )


@pytest.mark.parametrize(
    "case,expected_term",
    [
        ("object", "dtype"),
        ("structured", "dtype"),
        ("string", "dtype"),
        ("complex", "dtype"),
        ("integer", "dtype"),
        ("big_endian", "endian"),
        ("fortran_order", "fortran"),
        ("unsupported_rank", "shape"),
        ("npy_v2", "version"),
        ("trailing_bytes", "trailing"),
        ("truncated_payload", "npy"),
        ("bad_magic", "npy"),
    ],
)
def test_reader_rejects_forbidden_or_noncanonical_parameter_npy_payloads(
    serialization, parameter_artifact, tmp_path, case, expected_term
):
    target = tmp_path / f"npy-{case}.artifact"
    logical_name = "parameters.weights"
    array = np.asarray([0.375, 0.625], dtype=np.float64)
    version = (1, 0)
    allow_pickle = False

    if case == "object":
        array = np.asarray(["a", "b"], dtype=object)
        allow_pickle = True
    elif case == "structured":
        array = np.asarray([(0.375,), (0.625,)], dtype=[("weight", "<f8")])
    elif case == "string":
        array = np.asarray(["0.375", "0.625"], dtype="U5")
    elif case == "complex":
        array = np.asarray([0.375 + 0j, 0.625 + 0j], dtype=np.complex128)
    elif case == "integer":
        array = np.asarray([0, 1], dtype=np.int64)
    elif case == "big_endian":
        array = np.asarray([0.375, 0.625], dtype=">f8")
    elif case == "fortran_order":
        logical_name = "parameters.means"
        array = np.asfortranarray(
            np.asarray([[-0.0, 0.75], [1.25, -0.5]], dtype=np.float64)
        )
        assert array.flags.f_contiguous and not array.flags.c_contiguous
    elif case == "unsupported_rank":
        array = np.asarray([[0.375, 0.625]], dtype=np.float64)
    elif case == "npy_v2":
        version = (2, 0)

    payload = npy_bytes(array, version=version, allow_pickle=allow_pickle)
    if case == "trailing_bytes":
        payload += b"trailing"
    elif case == "truncated_payload":
        payload = payload[:-1]
    elif case == "bad_magic":
        payload = b"NOTNPY" + payload[6:]

    replace_array_payload(
        parameter_artifact,
        target,
        logical_name,
        payload,
        descriptor_array=array,
    )
    _reject_parameters(serialization, target, expected_term)


def test_fit_boolean_status_member_must_use_npy_bool_dtype(
    serialization, identity_artifact, tmp_path
):
    target = tmp_path / "bool-as-uint8.artifact"
    _replace_array_case(
        identity_artifact,
        target,
        "fit.collapsed_components",
        np.asarray([0, 0], dtype=np.uint8),
    )
    _reject_identity(serialization, target, "collapsed", "bool")


def test_fit_computation_arrays_must_match_the_model_dtype(
    serialization, identity_artifact, tmp_path
):
    target = tmp_path / "mixed-fit-dtype.artifact"
    _replace_array_case(
        identity_artifact,
        target,
        "fit.objective",
        np.asarray(-10.0, dtype=np.float32),
    )
    _reject_identity(serialization, target, "objective", "dtype")


def test_parameter_arrays_cannot_mix_float32_and_float64(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "mixed-parameter-dtype.artifact"
    _replace_array_case(
        parameter_artifact,
        target,
        "parameters.weights",
        np.asarray([0.375, 0.625], dtype=np.float32),
    )
    _reject_parameters(serialization, target, "dtype")


def test_absurd_manifest_dimensions_fail_closed_without_integer_overflow(
    serialization, parameter_artifact, tmp_path
):
    """Untrusted JSON integers must not escape the artifact error boundary."""

    target = tmp_path / "absurd-model-dimension.artifact"
    absurd_dimension = 2**100

    def change(manifest):
        manifest["model"]["n_components"] = absurd_dimension
        manifest["arrays"]["parameters.weights"]["shape"] = [
            absurd_dimension
        ]

    mutate_manifest(parameter_artifact, target, change)

    with pytest.raises(serialization.ArtifactFormatError):
        serialization.load_parameters(target)


def test_npy_shape_in_header_must_match_descriptor_and_record_shape(
    serialization, parameter_artifact, tmp_path
):
    target = tmp_path / "shape-mismatch.artifact"
    array = np.asarray([[0.375, 0.625]], dtype=np.float64)
    _replace_array_case(
        parameter_artifact, target, "parameters.weights", array
    )
    _reject_parameters(serialization, target, "shape")


def test_reader_rejects_native_endian_marker_instead_of_canonical_little_endian(
    serialization, parameter_artifact, tmp_path
):
    """NPY float descriptors must spell the wire endian explicitly as ``<``."""

    logical_name = "parameters.weights"
    manifest = manifest_from(parameter_artifact)
    member_path = manifest["arrays"][logical_name]["path"]
    member = next(
        member
        for member in read_members(parameter_artifact)
        if member.name == member_path
    )
    payload = member.data.replace(b"'<f8'", b"'=f8'", 1)
    assert payload != member.data
    target = tmp_path / "native-endian-descriptor.artifact"
    replace_array_payload(
        parameter_artifact,
        target,
        logical_name,
        payload,
        descriptor_array=np.asarray([0.375, 0.625], dtype=np.float64),
    )

    _reject_parameters(serialization, target, "endian")


def test_npy_header_limit_is_checked_before_loading_array_data(
    serialization, parameter_artifact
):
    limits = serialization.DEFAULT_LIMITS._replace(max_npy_header_bytes=8)
    with pytest.raises(serialization.ArtifactLimitError) as error:
        serialization.load_parameters(parameter_artifact, limits=limits)
    _assert_terms(error.value, "header")


def _invalid_parameters(case: str) -> Params:
    parameters = valid_parameters(jnp.float64, source="host")
    if case == "zero_weight":
        return parameters._replace(weights=np.asarray([0.0, 1.0]))
    if case == "negative_weight":
        return parameters._replace(weights=np.asarray([-0.1, 1.1]))
    if case == "nonnormalized_weight":
        return parameters._replace(weights=np.asarray([0.3, 0.6]))
    if case == "nonfinite_weight":
        return parameters._replace(weights=np.asarray([np.nan, np.nan]))
    if case == "nonfinite_mean":
        means = np.asarray(parameters.means).copy()
        means[0, 0] = np.inf
        return parameters._replace(means=means)
    if case == "asymmetric_covariance":
        covariances = np.asarray(parameters.covariances).copy()
        covariances[0, 0, 1] = 0.7
        return parameters._replace(covariances=covariances)
    if case == "indefinite_covariance":
        covariances = np.asarray(parameters.covariances).copy()
        covariances[0] = np.asarray([[1.0, 2.0], [2.0, 1.0]])
        return parameters._replace(covariances=covariances)
    if case == "singular_covariance":
        covariances = np.asarray(parameters.covariances).copy()
        covariances[0] = np.asarray([[1.0, 1.0], [1.0, 1.0]])
        return parameters._replace(covariances=covariances)
    if case == "nonfinite_covariance":
        covariances = np.asarray(parameters.covariances).copy()
        covariances[0, 0, 0] = np.nan
        return parameters._replace(covariances=covariances)
    if case == "integer_parameters":
        return Params(
            np.asarray([1], dtype=np.int64),
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([[[1, 0], [0, 1]]], dtype=np.int64),
        )
    if case == "mixed_dtype":
        return parameters._replace(
            weights=np.asarray(parameters.weights, dtype=np.float32)
        )
    if case == "empty_components":
        return Params(
            np.empty((0,), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            np.empty((0, 2, 2), dtype=np.float64),
        )
    if case == "zero_dimension":
        return Params(
            np.asarray([1.0], dtype=np.float64),
            np.empty((1, 0), dtype=np.float64),
            np.empty((1, 0, 0), dtype=np.float64),
        )
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "zero_weight",
        "negative_weight",
        "nonnormalized_weight",
        "nonfinite_weight",
        "nonfinite_mean",
        "asymmetric_covariance",
        "indefinite_covariance",
        "singular_covariance",
        "nonfinite_covariance",
        "integer_parameters",
        "mixed_dtype",
        "empty_components",
        "zero_dimension",
    ],
)
def test_writer_rejects_invalid_parameter_domain_without_creating_artifact(
    serialization, tmp_path, case
):
    path = tmp_path / f"invalid-{case}.artifact"
    before = {entry.name for entry in tmp_path.iterdir()}

    with pytest.raises(serialization.ArtifactFormatError):
        serialization.save_parameters(
            path,
            _invalid_parameters(case),
            contract_id=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
        )

    assert not path.exists()
    assert {entry.name for entry in tmp_path.iterdir()} == before


@pytest.mark.parametrize("direction", ["write", "read"])
def test_overflowing_finite_weight_sum_is_rejected_without_runtime_warning(
    serialization, tmp_path, direction
):
    """An invalid finite domain must still produce the documented error type."""

    maximum = np.finfo(np.float64).max
    invalid_weights = np.asarray([maximum, maximum], dtype=np.float64)
    path = tmp_path / f"overflowing-weight-sum-{direction}.artifact"

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        if direction == "write":
            invalid = valid_parameters(
                jnp.float64, source="host"
            )._replace(weights=invalid_weights)
            with pytest.raises(serialization.ArtifactFormatError):
                serialization.save_parameters(
                    path,
                    invalid,
                    contract_id=CONTRACT_ID,
                    contract_version=CONTRACT_VERSION,
                )
        else:
            source = tmp_path / "valid-source.artifact"
            serialization.save_parameters(
                source,
                valid_parameters(jnp.float64),
                contract_id=CONTRACT_ID,
                contract_version=CONTRACT_VERSION,
            )
            _replace_array_case(
                source,
                path,
                "parameters.weights",
                invalid_weights,
            )
            with pytest.raises(serialization.ArtifactFormatError):
                serialization.load_parameters(path)


@pytest.mark.parametrize(
    "contract_id,contract_version",
    [
        ("unknown.contract", CONTRACT_VERSION),
        (CONTRACT_ID, "0.1.0-draft.999"),
        (GENERAL_CONTRACT_ID, CONTRACT_VERSION),
        (CONTRACT_ID, GENERAL_CONTRACT_VERSION),
        (1, CONTRACT_VERSION),
        (CONTRACT_ID, 1),
    ],
)
def test_parameter_writer_accepts_only_exact_supported_contract_pairs(
    serialization, tmp_path, contract_id, contract_version
):
    path = tmp_path / "wrong-contract.artifact"
    with pytest.raises(serialization.ArtifactFormatError):
        serialization.save_parameters(
            path,
            valid_parameters(jnp.float64),
            contract_id=contract_id,
            contract_version=contract_version,
        )
    assert not path.exists()


@pytest.mark.parametrize(
    "case,logical_name,array",
    [
        (
            "zero_weight",
            "parameters.weights",
            np.asarray([0.0, 1.0], dtype=np.float64),
        ),
        (
            "negative_weight",
            "parameters.weights",
            np.asarray([-0.1, 1.1], dtype=np.float64),
        ),
        (
            "nonnormalized_weight",
            "parameters.weights",
            np.asarray([0.3, 0.6], dtype=np.float64),
        ),
        (
            "nonfinite_mean",
            "parameters.means",
            np.asarray([[-0.0, np.inf], [1.25, -0.5]], dtype=np.float64),
        ),
        (
            "asymmetric_covariance",
            "parameters.covariances",
            np.asarray(
                [
                    [[1.1, 0.7], [0.2, 0.7]],
                    [[0.6, -0.1], [-0.1, 1.3]],
                ],
                dtype=np.float64,
            ),
        ),
        (
            "indefinite_covariance",
            "parameters.covariances",
            np.asarray(
                [
                    [[1.0, 2.0], [2.0, 1.0]],
                    [[0.6, -0.1], [-0.1, 1.3]],
                ],
                dtype=np.float64,
            ),
        ),
        (
            "singular_covariance",
            "parameters.covariances",
            np.asarray(
                [
                    [[1.0, 1.0], [1.0, 1.0]],
                    [[0.6, -0.1], [-0.1, 1.3]],
                ],
                dtype=np.float64,
            ),
        ),
    ],
)
def test_reader_rejects_invalid_parameter_numerical_domain_without_repair(
    serialization,
    parameter_artifact,
    tmp_path,
    case,
    logical_name,
    array,
):
    target = tmp_path / f"domain-{case}.artifact"
    _replace_array_case(parameter_artifact, target, logical_name, array)
    _reject_parameters(serialization, target)


def test_parameter_domain_validation_finishes_before_any_device_placement(
    serialization, parameter_artifact, tmp_path, monkeypatch
):
    target = tmp_path / "invalid-before-device.artifact"
    _replace_array_case(
        parameter_artifact,
        target,
        "parameters.weights",
        np.asarray([0.0, 1.0], dtype=np.float64),
    )
    calls = []

    def forbidden_device_put(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("device placement occurred before validation")

    monkeypatch.setattr(jax, "device_put", forbidden_device_put)
    _reject_parameters(serialization, target)
    assert calls == []


def _selected_dtype_cholesky_failure_parameters() -> Params:
    covariance = np.asarray(
        [
            [2.0119961e16, 1.6779924e9],
            [1.6779924e9, 1.3994356e2],
        ],
        dtype=np.float32,
    )
    assert np.all(np.isfinite(np.linalg.cholesky(covariance)))
    selected_factor = np.asarray(
        jax.lax.linalg.cholesky(
            jnp.asarray(covariance), symmetrize_input=False
        )
    )
    # This fixture depends on the selected-dtype (XLA float32) Cholesky failing
    # for a covariance that NumPy still factors. Whether XLA produces a non-finite
    # factor for this borderline, extremely ill-conditioned matrix is sensitive to
    # the compiled jaxlib build (for example it fails on the cp310 build but stays
    # finite on the cp312 build, even at the same jax/jaxlib version). When the
    # current build factors it finitely, the selected-dtype rejection scenario
    # cannot be exercised, so skip rather than assert a build-specific outcome.
    if np.all(np.isfinite(selected_factor)):
        pytest.skip(
            "selected-dtype (XLA float32) Cholesky factors this borderline "
            "covariance finitely on this jaxlib build; the selected-dtype "
            "rejection scenario is not reproducible here"
        )
    return Params(
        weights=np.asarray([1.0], dtype=np.float32),
        means=np.zeros((1, 2), dtype=np.float32),
        covariances=covariance[None, ...],
    )


def test_writer_rejects_covariance_that_fails_selected_jax_dtype_cholesky(
    serialization, tmp_path
):
    path = tmp_path / "selected-dtype-failure.artifact"
    with pytest.raises(serialization.ArtifactFormatError):
        serialization.save_parameters(
            path,
            _selected_dtype_cholesky_failure_parameters(),
            contract_id=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
        )
    assert not path.exists()


def test_reader_rejects_covariance_that_fails_selected_jax_dtype_cholesky(
    serialization, tmp_path
):
    source = tmp_path / "source.artifact"
    target = tmp_path / "selected-dtype-failure.artifact"
    serialization.save_parameters(
        source,
        Params(
            weights=np.asarray([1.0], dtype=np.float32),
            means=np.zeros((1, 2), dtype=np.float32),
            covariances=np.eye(2, dtype=np.float32)[None, ...],
        ),
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )
    covariance = np.asarray(
        _selected_dtype_cholesky_failure_parameters().covariances
    )
    _replace_array_case(
        source, target, "parameters.covariances", covariance
    )
    _reject_parameters(serialization, target, "positive definite", "selected")


def test_near_float32_limit_covariance_round_trips_without_metric_overflow(
    serialization, tmp_path
):
    parameters = Params(
        weights=np.asarray([1.0], dtype=np.float32),
        means=np.zeros((1, 2), dtype=np.float32),
        covariances=(
            np.asarray(2.0e38, dtype=np.float32)
            * np.eye(2, dtype=np.float32)[None, ...]
        ),
    )
    path = tmp_path / "near-float32-limit.artifact"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        serialization.save_parameters(
            path,
            parameters,
            contract_id=CONTRACT_ID,
            contract_version=CONTRACT_VERSION,
        )
        loaded = serialization.load_parameters(path)
    assert_arrays_bits_equal(loaded.parameters, parameters)


def test_scaled_symmetry_rejection_near_dtype_limit_emits_no_overflow_warning(
    serialization, tmp_path
):
    maximum = np.finfo(np.float32).max
    covariance = np.asarray(
        [[0.75 * maximum, 0.75 * maximum],
         [-0.75 * maximum, 0.75 * maximum]],
        dtype=np.float32,
    )
    parameters = Params(
        weights=np.asarray([1.0], dtype=np.float32),
        means=np.zeros((1, 2), dtype=np.float32),
        covariances=covariance[None, ...],
    )
    path = tmp_path / "near-limit-asymmetry.artifact"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(serialization.ArtifactFormatError, match="asymmetric"):
            serialization.save_parameters(
                path,
                parameters,
                contract_id=CONTRACT_ID,
                contract_version=CONTRACT_VERSION,
            )
    assert not path.exists()


def test_valid_within_tolerance_asymmetry_is_preserved_not_repaired(
    serialization, tmp_path
):
    parameters = valid_parameters(jnp.float64, source="host")
    covariances = np.asarray(parameters.covariances).copy()
    covariances[0, 0, 1] += 1e-14
    assert covariances[0, 0, 1] != covariances[0, 1, 0]
    parameters = parameters._replace(covariances=covariances)
    path = tmp_path / "within-tolerance.artifact"

    serialization.save_parameters(
        path,
        parameters,
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )
    loaded = serialization.load_parameters(path)

    assert_arrays_bits_equal(loaded.parameters, parameters)


def _invalid_identity_result(case: str):
    result = identity_fit_result(jnp.float64, "converged")
    nan = jnp.asarray(np.nan, dtype=jnp.float64)
    if case == "unknown_status":
        return result._replace(status=jnp.asarray(99, dtype=jnp.int32))
    if case == "nonterminal_status":
        return result._replace(
            status=jnp.asarray(int(FitStatus.CONTINUE), dtype=jnp.int32)
        )
    if case == "unknown_mode":
        return result._replace(mode=jnp.asarray(99, dtype=jnp.int32))
    if case == "wrong_metadata":
        return result._replace(
            metadata=ResultMetadata(CONTRACT_ID, "0.1.0-draft.999")
        )
    if case == "unknown_initialization":
        return result._replace(
            initialization=InitializationProvenance(kind="kmeans")
        )
    if case == "history_length":
        return result._replace(history=result.history[:1])
    if case == "history_final":
        return result._replace(
            history=jnp.asarray([-11.0, -9.0], dtype=jnp.float64)
        )
    if case == "history_nonfinite":
        return result._replace(
            history=jnp.asarray([-11.0, np.nan], dtype=jnp.float64)
        )
    if case == "objective_flag":
        return result._replace(objective_valid=jnp.asarray(False))
    if case == "attempt_flag":
        return result._replace(attempted_objective_valid=jnp.asarray(False))
    if case == "attempt_nonfinite":
        return result._replace(attempted_objective=nan)
    if case == "negative_jitter":
        return result._replace(factor_jitter=jnp.asarray(-1.0, dtype=jnp.float64))
    if case == "nonfinite_ridge":
        return result._replace(covariance_ridge=nan)
    if case == "missing_tol":
        return result._replace(tol=None)
    if case == "negative_decrease_tol":
        return result._replace(
            decrease_tol=jnp.asarray(-1.0, dtype=jnp.float64)
        )
    if case == "fixed_converged":
        return result._replace(
            mode=jnp.asarray(int(FitMode.FIXED_STEPS), dtype=jnp.int32),
            status=jnp.asarray(
                int(FitStatus.FIXED_STEPS_COMPLETE), dtype=jnp.int32
            ),
            converged=jnp.asarray(True),
            tol=None,
            decrease_tol=None,
        )
    if case == "success_failure_flag":
        return result._replace(numerical_failure=jnp.asarray(True))
    if case == "success_collapse_mask":
        return result._replace(
            collapsed_components=jnp.asarray([True, False])
        )
    if case == "collapse_without_mask":
        collapsed = identity_fit_result(jnp.float64, "component_collapsed")
        return collapsed._replace(
            collapsed_components=jnp.asarray([False, False])
        )
    if case == "collapse_without_rollback":
        collapsed = identity_fit_result(jnp.float64, "component_collapsed")
        return collapsed._replace(parameters=result.parameters)
    if case == "decrease_without_rollback":
        decreased = identity_fit_result(jnp.float64, "objective_decreased")
        return decreased._replace(parameters=result.parameters)
    if case == "invalid_initial_nonempty_history":
        invalid = identity_fit_result(
            jnp.float64, "invalid_initial_objective"
        )
        return invalid._replace(history=jnp.asarray([nan]))
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "unknown_status",
        "nonterminal_status",
        "unknown_mode",
        "wrong_metadata",
        "unknown_initialization",
        "history_length",
        "history_final",
        "history_nonfinite",
        "objective_flag",
        "attempt_flag",
        "attempt_nonfinite",
        "negative_jitter",
        "nonfinite_ridge",
        "missing_tol",
        "negative_decrease_tol",
        "fixed_converged",
        "success_failure_flag",
        "success_collapse_mask",
        "collapse_without_mask",
        "collapse_without_rollback",
        "decrease_without_rollback",
        "invalid_initial_nonempty_history",
    ],
)
def test_fit_writer_rejects_inconsistent_host_result_before_creating_file(
    serialization, tmp_path, case
):
    path = tmp_path / f"invalid-fit-{case}.artifact"
    before = {entry.name for entry in tmp_path.iterdir()}
    with pytest.raises(serialization.ArtifactFormatError):
        serialization.save_identity_fit_result(
            path, _invalid_identity_result(case)
        )
    assert not path.exists()
    assert {entry.name for entry in tmp_path.iterdir()} == before


@pytest.mark.parametrize("direction", ["write", "read"])
def test_fit_iteration_fields_cannot_overflow_the_loaded_int32_schema(
    serialization, tmp_path, direction
):
    """Wire integers must not silently wrap when reconstructed as JAX int32."""

    out_of_range = 2**31
    path = tmp_path / f"iteration-overflow-{direction}.artifact"
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        if direction == "write":
            result = identity_fit_result(
                jnp.float64, "converged"
            )._replace(
                iteration_limit=jnp.asarray(out_of_range, dtype=jnp.int64)
            )
            with pytest.raises(serialization.ArtifactFormatError):
                serialization.save_identity_fit_result(path, result)
        else:
            source = tmp_path / "valid-fit-source.artifact"
            serialization.save_identity_fit_result(
                source, identity_fit_result(jnp.float64, "converged")
            )
            mutate_manifest(
                source,
                path,
                lambda manifest: manifest["fit"].__setitem__(
                    "iteration_limit", out_of_range
                ),
            )
            with pytest.raises(serialization.ArtifactFormatError):
                serialization.load_identity_fit_result(path)


@pytest.mark.parametrize("direction", ["write", "read"])
def test_invalid_initial_objective_must_have_numerical_failure_status(
    serialization, tmp_path, direction
):
    """A nonfinite initial objective cannot be relabelled as max-iteration."""

    path = tmp_path / f"invalid-initial-status-{direction}.artifact"
    if direction == "write":
        result = identity_fit_result(
            jnp.float64, "invalid_initial_objective"
        )._replace(
            status=jnp.asarray(int(FitStatus.MAX_ITER), dtype=jnp.int32),
            numerical_failure=jnp.asarray(False),
            iteration_limit=jnp.asarray(0, dtype=jnp.int32),
        )
        with pytest.raises(serialization.ArtifactFormatError):
            serialization.save_identity_fit_result(path, result)
    else:
        source = tmp_path / "invalid-initial-source.artifact"
        serialization.save_identity_fit_result(
            source,
            identity_fit_result(jnp.float64, "invalid_initial_objective"),
        )

        def relabel(manifest):
            manifest["fit"].update(
                status="max_iter",
                numerical_failure=False,
                iteration_limit=0,
            )

        mutate_manifest(source, path, relabel)
        with pytest.raises(serialization.ArtifactFormatError):
            serialization.load_identity_fit_result(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "unknown"),
        ("status", "continue"),
        ("objective_semantics", "identity_exact_observed_mean"),
        ("ridge_application", "pre_em_observed_covariance"),
        ("n_iter", 99),
        ("iteration_limit", -1),
        ("converged", False),
        ("objective_valid", False),
        ("attempted_objective_valid", False),
        ("numerical_failure", True),
        ("collapsed", True),
    ],
)
def test_reader_rejects_inconsistent_or_unknown_fit_manifest_semantics(
    serialization, identity_artifact, tmp_path, field, value
):
    target = tmp_path / f"fit-field-{field}.artifact"
    mutate_manifest(
        identity_artifact,
        target,
        lambda manifest: manifest["fit"].__setitem__(field, value),
    )
    _reject_identity(serialization, target, field.replace("_", " "))


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_fit_object_and_initialization_use_closed_exact_field_sets(
    serialization, identity_artifact, tmp_path, operation
):
    for nested in (False, True):
        target = tmp_path / f"fit-fields-{operation}-{nested}.artifact"

        def change(manifest, nested=nested):
            record = (
                manifest["fit"]["initialization"]
                if nested
                else manifest["fit"]
            )
            if operation == "missing":
                del record["kind" if nested else "ridge_application"]
            else:
                record["seed" if nested else "future"] = 0

        mutate_manifest(identity_artifact, target, change)
        _reject_identity(serialization, target, "field")


def test_initialization_kind_is_closed_to_user_supplied(
    serialization, identity_artifact, tmp_path
):
    target = tmp_path / "initialization-kind.artifact"
    mutate_manifest(
        identity_artifact,
        target,
        lambda manifest: manifest["fit"]["initialization"].__setitem__(
            "kind", "kmeans"
        ),
    )
    _reject_identity(serialization, target, "initialization")


def _remove_logical_array(source, target, logical_name):
    manifest = manifest_from(source)
    member_path = manifest["arrays"].pop(logical_name)["path"]
    members = []
    for member in read_members(source):
        if member.name == "manifest.json":
            members.append(replace(member, data=canonical_json(manifest)))
        elif member.name != member_path:
            members.append(member)
    write_members(target, members)


def test_converged_fit_requires_tol_and_decrease_tol_members(
    serialization, identity_artifact, tmp_path
):
    for logical_name in ("fit.tol", "fit.decrease_tol"):
        target = tmp_path / f"missing-{logical_name}.artifact"
        _remove_logical_array(identity_artifact, target, logical_name)
        _reject_identity(serialization, target, logical_name)


def test_fixed_step_fit_forbids_tol_and_decrease_tol_members(
    serialization, identity_artifact, tmp_path
):
    target = tmp_path / "fixed-with-tolerances.artifact"

    def change(manifest):
        manifest["fit"]["mode"] = "fixed_steps"
        manifest["fit"]["status"] = "fixed_steps_complete"
        manifest["fit"]["converged"] = False
        manifest["fit"]["iteration_limit"] = 1

    mutate_manifest(identity_artifact, target, change)
    _reject_identity(serialization, target, "tol")


@pytest.mark.parametrize(
    "logical_name,array",
    [
        (
            "fit.history",
            np.asarray([-11.0], dtype=np.float64),
        ),
        (
            "fit.history",
            np.asarray([-11.0, -9.0], dtype=np.float64),
        ),
        (
            "fit.history",
            np.asarray([-11.0, np.nan], dtype=np.float64),
        ),
        (
            "fit.attempted_objective",
            np.asarray(np.nan, dtype=np.float64),
        ),
        (
            "fit.factor_jitter",
            np.asarray(-1.0, dtype=np.float64),
        ),
        (
            "fit.covariance_ridge",
            np.asarray(np.inf, dtype=np.float64),
        ),
        (
            "fit.tol",
            np.asarray(-1.0, dtype=np.float64),
        ),
        (
            "fit.decrease_tol",
            np.asarray(np.nan, dtype=np.float64),
        ),
    ],
)
def test_reader_rejects_fit_array_invariants(
    serialization,
    identity_artifact,
    tmp_path,
    logical_name,
    array,
):
    target = tmp_path / f"fit-array-{logical_name}-{array.size}.artifact"
    _replace_array_case(identity_artifact, target, logical_name, array)
    _reject_identity(serialization, target, logical_name)


def test_grouped_manifest_dimensions_are_derived_from_diagnostic_shapes(
    serialization, grouped_artifact, tmp_path
):
    for field in ("n_samples", "n_groups"):
        target = tmp_path / f"grouped-model-{field}.artifact"
        mutate_manifest(
            grouped_artifact,
            target,
            lambda manifest, field=field: manifest["model"].__setitem__(
                field, manifest["model"][field] + 1
            ),
        )
        _reject_grouped(serialization, target, field.replace("_", " "))


@pytest.mark.parametrize(
    "logical_name,array",
    [
        (
            "fit.group_numerical_failure",
            np.asarray([False], dtype=bool),
        ),
        (
            "fit.failed_pairs",
            np.zeros((2, 2), dtype=bool),
        ),
        (
            "fit.failed_pairs",
            np.zeros((3, 1), dtype=bool),
        ),
        (
            "fit.informative_weight",
            np.asarray(0.0, dtype=np.float64),
        ),
        (
            "fit.informative_weight",
            np.asarray(np.nan, dtype=np.float64),
        ),
    ],
)
def test_grouped_reader_rejects_diagnostic_shape_and_weight_inconsistency(
    serialization,
    grouped_artifact,
    tmp_path,
    logical_name,
    array,
):
    target = tmp_path / f"grouped-array-{logical_name}-{array.shape}.artifact"
    _replace_array_case(grouped_artifact, target, logical_name, array)
    _reject_grouped(serialization, target, logical_name)


@pytest.mark.parametrize(
    "field,value",
    [
        ("failure_stage", "unknown_stage"),
        ("failure_stage", "candidate_objective"),
        ("status", "numerical_failure"),
        ("numerical_failure", True),
    ],
)
def test_grouped_reader_rejects_inconsistent_failure_stage_and_terminal_status(
    serialization, grouped_artifact, tmp_path, field, value
):
    target = tmp_path / f"grouped-fit-{field}.artifact"
    mutate_manifest(
        grouped_artifact,
        target,
        lambda manifest: manifest["fit"].__setitem__(field, value),
    )
    _reject_grouped(serialization, target, field.replace("_", " "))


def test_grouped_writer_rejects_diagnostic_shapes_before_writing(
    serialization, tmp_path
):
    valid = grouped_fit_result(jnp.float64, "converged")
    invalid_results = (
        valid._replace(failed_pairs=jnp.zeros((3, 1), dtype=bool)),
        valid._replace(informative_weight=jnp.asarray(0.0, dtype=jnp.float64)),
        valid._replace(metadata=current_result_metadata()),
        valid._replace(
            failure_stage=jnp.asarray(
                int(GroupedFailureStage.CANDIDATE_OBJECTIVE), dtype=jnp.int32
            )
        ),
    )
    for index, result in enumerate(invalid_results):
        path = tmp_path / f"invalid-grouped-{index}.artifact"
        with pytest.raises(serialization.ArtifactFormatError):
            serialization.save_grouped_general_fit_result(path, result)
        assert not path.exists()


def test_grouped_writer_derives_n_and_g_from_the_two_diagnostic_shapes(
    serialization, tmp_path
):
    result = grouped_fit_result(jnp.float64, "converged")._replace(
        group_numerical_failure=jnp.zeros((1,), dtype=bool),
        failed_pairs=jnp.zeros((2, 2), dtype=bool),
    )
    path = tmp_path / "derived-group-shapes.artifact"

    serialization.save_grouped_general_fit_result(path, result)

    assert manifest_from(path)["model"] == {
        "dtype": "float64",
        "latent_dimension": 2,
        "n_components": 2,
        "n_groups": 1,
        "n_samples": 2,
    }


def test_successful_grouped_record_forbids_failure_masks_on_write_and_read(
    serialization, tmp_path
):
    valid = grouped_fit_result(jnp.float64, "converged")
    invalid = valid._replace(
        group_numerical_failure=jnp.asarray([True, False], dtype=bool),
        failed_pairs=jnp.asarray(
            [[True, False], [False, False], [False, False]], dtype=bool
        ),
    )
    rejected_path = tmp_path / "writer-failure-masks.artifact"
    with pytest.raises(serialization.ArtifactFormatError):
        serialization.save_grouped_general_fit_result(rejected_path, invalid)
    assert not rejected_path.exists()

    source = tmp_path / "source.artifact"
    target = tmp_path / "reader-failure-masks.artifact"
    serialization.save_grouped_general_fit_result(source, valid)
    _replace_array_case(
        source,
        target,
        "fit.group_numerical_failure",
        np.asarray([True, False], dtype=bool),
    )
    _reject_grouped(serialization, target, "group_numerical_failure")


def test_fit_contract_cannot_be_relabelled_to_other_supported_contract(
    serialization, identity_artifact, tmp_path
):
    target = tmp_path / "relabeled-contract.artifact"

    def change(manifest):
        manifest["contract_id"] = GENERAL_CONTRACT_ID
        manifest["contract_version"] = GENERAL_CONTRACT_VERSION

    mutate_manifest(identity_artifact, target, change)
    _reject_identity(serialization, target, "contract")
