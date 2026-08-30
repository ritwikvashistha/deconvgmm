"""Tests-first contract for ordered user-supplied restart selection.

The restart layer is host-only and intentionally separate from the frozen
single-fit metadata and serialization schemas.
"""

from __future__ import annotations

import importlib
import inspect

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitMode, FitResult, FitStatus
from development.identity_xd import Params
from development.metadata import (
    current_result_metadata,
    user_supplied_initialization,
)
from development.validation import ValidationError


@pytest.fixture
def restarts():
    """Import lazily so the tests collect before the new module exists."""

    return importlib.import_module("development.restarts")


def _params(dtype, *, shift=0.0, permutation=False) -> Params:
    weights = jnp.asarray([0.4, 0.6], dtype=dtype)
    means = jnp.asarray(
        [[-0.8 + shift, 0.2], [1.1 + shift, -0.35]], dtype=dtype
    )
    covariances = jnp.asarray(
        [
            [[0.7, 0.08], [0.08, 0.55]],
            [[0.5, -0.04], [-0.04, 0.8]],
        ],
        dtype=dtype,
    )
    if permutation:
        order = jnp.asarray([1, 0])
        return Params(weights[order], means[order], covariances[order])
    return Params(weights, means, covariances)


def _candidate_at(candidates, index: int) -> Params:
    return Params(
        candidates.weights[index],
        candidates.means[index],
        candidates.covariances[index],
    )


def _assert_params_equal(actual: Params, expected: Params) -> None:
    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_leaf), np.asarray(expected_leaf)
        )


def _fake_fit_result(
    params: Params,
    *,
    objective: float,
    status: FitStatus,
    mode: FitMode = FitMode.CONVERGED,
    objective_valid: bool = True,
) -> FitResult:
    dtype = params.means.dtype
    objective_array = jnp.asarray(
        objective if objective_valid else jnp.nan, dtype=dtype
    )
    converged = status == FitStatus.CONVERGED
    numerical_failure = status == FitStatus.NUMERICAL_FAILURE
    collapsed = status == FitStatus.COMPONENT_COLLAPSED
    n_iter = 1 if converged else 0
    history = (
        jnp.repeat(objective_array[None], n_iter + 1)
        if objective_valid
        else jnp.empty((0,), dtype=dtype)
    )
    iteration_limit = 0 if status in (
        FitStatus.MAX_ITER,
        FitStatus.FIXED_STEPS_COMPLETE,
    ) else max(1, n_iter)
    successful = status in (
        FitStatus.CONVERGED,
        FitStatus.FIXED_STEPS_COMPLETE,
    )
    failed_after_valid_state = objective_valid and status in (
        FitStatus.OBJECTIVE_DECREASED,
        FitStatus.NUMERICAL_FAILURE,
        FitStatus.COMPONENT_COLLAPSED,
    )
    attempted_iteration = (
        n_iter + 1 if failed_after_valid_state else n_iter
    )
    attempted_valid = successful or status == FitStatus.OBJECTIVE_DECREASED
    attempted_objective = jnp.asarray(
        objective - 1.0
        if status == FitStatus.OBJECTIVE_DECREASED
        else objective,
        dtype=dtype,
    ) if attempted_valid else jnp.asarray(jnp.nan, dtype=dtype)
    return FitResult(
        parameters=params,
        initial_parameters=params,
        objective=objective_array,
        objective_valid=jnp.asarray(objective_valid),
        history=history,
        n_iter=jnp.asarray(n_iter, dtype=jnp.int32),
        iteration_limit=jnp.asarray(iteration_limit, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        mode=jnp.asarray(int(mode), dtype=jnp.int32),
        attempted_iteration=jnp.asarray(
            attempted_iteration, dtype=jnp.int32
        ),
        attempted_objective=attempted_objective,
        attempted_objective_valid=jnp.asarray(attempted_valid),
        numerical_failure=jnp.asarray(numerical_failure),
        collapsed=jnp.asarray(collapsed),
        collapsed_components=jnp.asarray([collapsed, False]),
        factor_jitter=jnp.asarray(0.0, dtype=dtype),
        covariance_ridge=jnp.asarray(0.0, dtype=dtype),
        tol=(
            jnp.asarray(1e-6, dtype=dtype)
            if mode == FitMode.CONVERGED
            else None
        ),
        decrease_tol=(
            jnp.asarray(1e-10, dtype=dtype)
            if mode == FitMode.CONVERGED
            else None
        ),
        initialization=user_supplied_initialization(),
        metadata=current_result_metadata(),
    )


def test_restart_api_and_result_schemas_are_exact(restarts):
    assert restarts.RESTART_CONTRACT_ID == "xdgmm-jax.restart-selection"
    assert restarts.RESTART_CONTRACT_VERSION == "0.1.0-draft.1"
    assert restarts.RESTART_SELECTION_RULE_ID == (
        "xdgmm-jax.highest-eligible-objective"
    )
    assert restarts.RESTART_SELECTION_RULE_VERSION == "0.1.0-draft.1"
    assert restarts.RestartCandidates._fields == (
        "weights",
        "means",
        "covariances",
        "initialization_kind",
    )
    assert restarts.RestartDiagnostics._fields == (
        "objective",
        "objective_valid",
        "n_iter",
        "status",
        "converged",
        "numerical_failure",
        "collapsed",
        "eligible",
    )
    assert restarts.RestartSelection._fields == (
        "restart_count",
        "selected_restart",
        "success",
        "status",
        "contract_id",
        "contract_version",
        "rule_id",
        "rule_version",
    )
    assert restarts.IdentityRestartFitResult._fields == (
        "selected_result",
        "candidates",
        "diagnostics",
        "selection",
    )
    assert restarts.GroupedGeneralRestartFitResult._fields == (
        "selected_result",
        "candidates",
        "diagnostics",
        "selection",
    )

    constructor = inspect.signature(
        restarts.user_supplied_restart_candidates
    ).parameters
    assert tuple(constructor) == ("initial_parameters", "dtype")
    assert constructor["dtype"].kind is inspect.Parameter.KEYWORD_ONLY
    assert constructor["dtype"].default is inspect.Parameter.empty

    identity_converged = inspect.signature(
        restarts.fit_converged_restarts
    ).parameters
    assert tuple(identity_converged) == (
        "candidates",
        "observations",
        "measurement_covariances",
        "max_iter",
        "tol",
        "decrease_tol",
        "factor_jitter",
        "covariance_ridge",
    )
    identity_fixed = inspect.signature(
        restarts.fit_fixed_steps_restarts
    ).parameters
    assert tuple(identity_fixed) == (
        "candidates",
        "observations",
        "measurement_covariances",
        "n_steps",
        "factor_jitter",
        "covariance_ridge",
    )
    general_converged = inspect.signature(
        restarts.fit_converged_grouped_restarts
    ).parameters
    assert tuple(general_converged) == (
        "fit",
        "candidates",
        "max_iter",
        "tol",
        "decrease_tol",
    )
    general_fixed = inspect.signature(
        restarts.fit_fixed_steps_grouped_restarts
    ).parameters
    assert tuple(general_fixed) == ("fit", "candidates", "n_steps")


def test_user_candidates_are_canonical_stacked_and_ordered(restarts):
    first = _params(jnp.float64)
    duplicate = _params(jnp.float64)
    permuted = _params(jnp.float64, permutation=True)
    candidates = restarts.user_supplied_restart_candidates(
        [first, duplicate, permuted], dtype=jnp.float64
    )

    assert candidates.weights.shape == (3, 2)
    assert candidates.means.shape == (3, 2, 2)
    assert candidates.covariances.shape == (3, 2, 2, 2)
    assert candidates.weights.dtype == jnp.float64
    assert candidates.means.dtype == jnp.float64
    assert candidates.covariances.dtype == jnp.float64
    assert candidates.initialization_kind == "user_supplied_ordered"
    _assert_params_equal(_candidate_at(candidates, 0), first)
    _assert_params_equal(_candidate_at(candidates, 1), duplicate)
    _assert_params_equal(_candidate_at(candidates, 2), permuted)


def test_candidate_boundary_allows_mixed_floating_sources_under_explicit_dtype(
    restarts,
):
    first = _params(jnp.float32)
    second = _params(jnp.float64, shift=0.2)
    candidates = restarts.user_supplied_restart_candidates(
        [first, second], dtype=jnp.float32
    )
    assert candidates.weights.dtype == jnp.float32
    assert candidates.means.dtype == jnp.float32
    assert candidates.covariances.dtype == jnp.float32


@pytest.mark.parametrize(
    "initial_parameters,match",
    [
        pytest.param([], "nonempty|at least one", id="empty"),
        pytest.param(
            [
                _params(jnp.float64),
                Params(
                    jnp.asarray([1.0], dtype=jnp.float64),
                    jnp.asarray([[0.0, 0.0]], dtype=jnp.float64),
                    jnp.asarray([np.eye(2)], dtype=jnp.float64),
                ),
            ],
            "K|component|shape",
            id="mixed-k",
        ),
        pytest.param(
            [
                _params(jnp.float64),
                Params(
                    jnp.asarray([0.4, 0.6], dtype=jnp.float64),
                    jnp.asarray([[-0.8], [1.1]], dtype=jnp.float64),
                    jnp.asarray([[[0.7]], [[0.5]]], dtype=jnp.float64),
                ),
            ],
            "D|dimension|shape",
            id="mixed-d",
        ),
        pytest.param(
            [
                _params(jnp.float64),
                _params(jnp.float64)._replace(
                    weights=jnp.asarray([0.0, 1.0], dtype=jnp.float64)
                ),
            ],
            "weights|positive",
            id="invalid-weight",
        ),
        pytest.param(
            [
                _params(jnp.float64),
                _params(jnp.float64)._replace(
                    means=jnp.asarray(
                        [[jnp.nan, 0.0], [1.0, 0.0]], dtype=jnp.float64
                    )
                ),
            ],
            "means|finite",
            id="nonfinite-mean",
        ),
        pytest.param(
            [
                _params(jnp.float64),
                _params(jnp.float64)._replace(
                    covariances=jnp.asarray(
                        [
                            [[1.0, 2.0], [2.0, 1.0]],
                            [[0.5, 0.0], [0.0, 0.5]],
                        ],
                        dtype=jnp.float64,
                    )
                ),
            ],
            "covariance|positive definite|Cholesky",
            id="indefinite-covariance",
        ),
    ],
)
def test_candidate_boundary_rejects_invalid_whole_collections(
    restarts, initial_parameters, match
):
    with pytest.raises((TypeError, ValueError, ValidationError), match=match):
        restarts.user_supplied_restart_candidates(
            initial_parameters, dtype=jnp.float64
        )


def test_candidate_boundary_uses_selected_jax_factorization_policy(restarts):
    pathological = _params(jnp.float32)._replace(
        covariances=jnp.asarray(
            [
                [
                    [2.0119961e16, 1.6779924e9],
                    [1.6779924e9, 1.3994356e2],
                ],
                [[0.5, 0.0], [0.0, 0.5]],
            ],
            dtype=jnp.float32,
        )
    )
    with pytest.raises(ValidationError, match="selected|Cholesky|positive"):
        restarts.user_supplied_restart_candidates(
            [pathological], dtype=jnp.float32
        )


def test_converged_selection_runs_all_candidates_and_uses_status_eligibility(
    restarts, monkeypatch
):
    source = [_params(jnp.float64, shift=float(i)) for i in range(4)]
    candidates = restarts.user_supplied_restart_candidates(
        source, dtype=jnp.float64
    )
    outcomes = (
        (FitStatus.NUMERICAL_FAILURE, -1.0, True),
        (FitStatus.CONVERGED, -5.0, True),
        (FitStatus.MAX_ITER, -3.0, True),
        (FitStatus.OBJECTIVE_DECREASED, -2.0, True),
    )
    returned = []
    calls = []

    def fake_fit(params, observations, measurement_covariances, **controls):
        del observations, measurement_covariances
        index = len(calls)
        calls.append((params, controls))
        status, objective, valid = outcomes[index]
        result = _fake_fit_result(
            params,
            objective=objective,
            status=status,
            objective_valid=valid,
        )
        returned.append(result)
        return result

    monkeypatch.setattr(restarts, "fit_converged", fake_fit)
    result = restarts.fit_converged_restarts(
        candidates,
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros((1, 2, 2), dtype=jnp.float64),
    )

    assert len(calls) == 4
    for index, (called, _) in enumerate(calls):
        _assert_params_equal(called, _candidate_at(candidates, index))
    assert int(np.asarray(result.selection.selected_restart)) == 2
    assert bool(np.asarray(result.selection.success))
    assert int(np.asarray(result.selection.status)) == int(
        restarts.RestartSelectionStatus.SELECTED_ELIGIBLE
    )
    assert result.selected_result is returned[2]
    np.testing.assert_array_equal(
        np.asarray(result.diagnostics.eligible),
        np.asarray([False, True, True, False]),
    )
    _assert_params_equal(
        result.selected_result.initial_parameters,
        _candidate_at(result.candidates, 2),
    )


def test_exact_objective_ties_select_the_lowest_candidate_index(
    restarts, monkeypatch
):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64), _params(jnp.float64, permutation=True)],
        dtype=jnp.float64,
    )
    calls = 0

    def fake_fit(params, *args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return _fake_fit_result(
            params, objective=-2.0, status=FitStatus.CONVERGED
        )

    monkeypatch.setattr(restarts, "fit_converged", fake_fit)
    result = restarts.fit_converged_restarts(
        candidates,
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros((1, 2, 2), dtype=jnp.float64),
    )
    assert calls == 2
    assert int(np.asarray(result.selection.selected_restart)) == 0


@pytest.mark.parametrize(
    "objectives,valid,expected",
    [
        pytest.param((-4.0, -2.0, -3.0), (True, True, True), 1, id="best-valid"),
        pytest.param(
            (np.nan, np.nan, np.nan),
            (False, False, False),
            0,
            id="none-valid",
        ),
        pytest.param((-2.0, -2.0, -3.0), (True, True, True), 0, id="failed-tie"),
    ],
)
def test_all_failed_returns_an_unsuccessful_diagnostic_representative(
    restarts, monkeypatch, objectives, valid, expected
):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64, shift=float(i)) for i in range(3)],
        dtype=jnp.float64,
    )
    call_index = 0

    def fake_fit(params, *args, **kwargs):
        nonlocal call_index
        del args, kwargs
        index = call_index
        call_index += 1
        return _fake_fit_result(
            params,
            objective=objectives[index],
            status=FitStatus.NUMERICAL_FAILURE,
            objective_valid=valid[index],
        )

    monkeypatch.setattr(restarts, "fit_converged", fake_fit)
    result = restarts.fit_converged_restarts(
        candidates,
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros((1, 2, 2), dtype=jnp.float64),
    )
    assert not bool(np.asarray(result.selection.success))
    assert int(np.asarray(result.selection.status)) == int(
        restarts.RestartSelectionStatus.ALL_INITIALIZATIONS_FAILED
    )
    assert int(np.asarray(result.selection.selected_restart)) == expected
    assert int(np.asarray(result.selected_result.status)) == int(
        FitStatus.NUMERICAL_FAILURE
    )


def test_fixed_step_wrapper_accepts_only_fixed_steps_complete(
    restarts, monkeypatch
):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64), _params(jnp.float64, shift=0.2)],
        dtype=jnp.float64,
    )
    outcomes = (
        FitStatus.NUMERICAL_FAILURE,
        FitStatus.FIXED_STEPS_COMPLETE,
    )
    call_index = 0

    def fake_fit(params, *args, **kwargs):
        nonlocal call_index
        del args, kwargs
        status = outcomes[call_index]
        call_index += 1
        return _fake_fit_result(
            params,
            objective=-2.0 - call_index,
            status=status,
            mode=FitMode.FIXED_STEPS,
        )

    monkeypatch.setattr(restarts, "fit_fixed_steps", fake_fit)
    result = restarts.fit_fixed_steps_restarts(
        candidates,
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.zeros((1, 2, 2), dtype=jnp.float64),
        n_steps=0,
    )
    np.testing.assert_array_equal(
        np.asarray(result.diagnostics.eligible), np.asarray([False, True])
    )
    assert int(np.asarray(result.selection.selected_restart)) == 1


def test_contradictory_single_result_is_an_internal_error(restarts, monkeypatch):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64)], dtype=jnp.float64
    )
    invalid = _fake_fit_result(
        _candidate_at(candidates, 0),
        objective=-2.0,
        status=FitStatus.CONVERGED,
    )._replace(converged=jnp.asarray(False))
    monkeypatch.setattr(restarts, "fit_converged", lambda *args, **kwargs: invalid)
    with pytest.raises(RuntimeError, match="invariant|converged|status"):
        restarts.fit_converged_restarts(
            candidates,
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 2, 2), dtype=jnp.float64),
        )


def test_selected_result_must_retain_the_exact_candidate_initialization(
    restarts, monkeypatch
):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64)], dtype=jnp.float64
    )
    wrong = _params(jnp.float64, shift=3.0)
    invalid = _fake_fit_result(
        wrong, objective=-2.0, status=FitStatus.CONVERGED
    )
    monkeypatch.setattr(restarts, "fit_converged", lambda *args, **kwargs: invalid)
    with pytest.raises(RuntimeError, match="initial|candidate|invariant"):
        restarts.fit_converged_restarts(
            candidates,
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 2, 2), dtype=jnp.float64),
        )


@pytest.mark.parametrize("mutation", ["invalid-domain", "repairable-asymmetry"])
def test_forged_candidate_tokens_are_fully_validated_before_candidate_zero(
    restarts, monkeypatch, mutation
):
    valid = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64), _params(jnp.float64, shift=0.2)],
        dtype=jnp.float64,
    )
    covariances = valid.covariances
    if mutation == "invalid-domain":
        covariances = covariances.at[1, 0].set(
            jnp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float64)
        )
    else:
        covariances = covariances.at[1, 0, 0, 1].set(
            jnp.nextafter(
                covariances[1, 0, 0, 1],
                jnp.asarray(jnp.inf, dtype=jnp.float64),
            )
        )
    forged = restarts.RestartCandidates(
        valid.weights,
        valid.means,
        covariances,
        valid.initialization_kind,
    )
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("candidate zero must not run")

    monkeypatch.setattr(restarts, "fit_converged", fail_if_called)
    with pytest.raises(
        ValidationError,
        match=r"candidates\[1\]|canonical|covariance|positive",
    ):
        restarts.fit_converged_restarts(
            forged,
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 2, 2), dtype=jnp.float64),
            max_iter=0,
        )
    assert calls == 0


@pytest.mark.parametrize("mutation", ["invalid-domain", "repairable-asymmetry"])
def test_controller_final_parameters_must_be_bit_exact_canonical(
    restarts, monkeypatch, mutation
):
    candidates = restarts.user_supplied_restart_candidates(
        [_params(jnp.float64)], dtype=jnp.float64
    )
    candidate = _candidate_at(candidates, 0)
    covariances = candidate.covariances
    if mutation == "invalid-domain":
        covariances = covariances.at[0].set(
            jnp.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float64)
        )
    else:
        covariances = covariances.at[0, 0, 1].set(
            jnp.nextafter(
                covariances[0, 0, 1],
                jnp.asarray(jnp.inf, dtype=jnp.float64),
            )
        )
    invalid_parameters = candidate._replace(covariances=covariances)
    invalid_result = _fake_fit_result(
        candidate, objective=-2.0, status=FitStatus.CONVERGED
    )._replace(parameters=invalid_parameters)
    monkeypatch.setattr(
        restarts, "fit_converged", lambda *args, **kwargs: invalid_result
    )

    with pytest.raises(RuntimeError, match="result.*parameter|canonical|invariant"):
        restarts.fit_converged_restarts(
            candidates,
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros((1, 2, 2), dtype=jnp.float64),
        )
