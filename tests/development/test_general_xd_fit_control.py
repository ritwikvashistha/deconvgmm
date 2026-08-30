"""Red tests for host-controlled grouped general-XD fitting.

These tests settle accepted-state, rollback, weighting, and diagnostic semantics
before the temporary controller exists.  Expected numerical trajectories come
from both repeated calls to :func:`development.general_grouped.one_em_step_grouped`
and the independent NumPy oracle in :mod:`tests.reference.general_xd`.

Variable-``M`` grouping and the current global grouped update are eager host
orchestration.  This file deliberately requires no compiled whole-group fit
leaf and makes no whole-fit JIT or autodiff claim.
"""

from __future__ import annotations

from dataclasses import replace
import importlib
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
from development.fit_control import FitMode, FitStatus
from development.general_grouped import (
    GroupedFailureStage,
    GroupedStepStatus,
    one_em_step_grouped,
    sufficient_statistics_grouped,
)
from development.general_validation import (
    PerItemFullNoise,
    PerItemProjection,
    group_masked_general_fit_inputs,
)
from development.identity_xd import Params
from development.metadata import ResultMetadata as IdentityResultMetadata
from tests.reference.general_xd import (
    general_e_step,
    general_grouped_m_step,
    general_grouped_objective,
)


DTYPES = (
    pytest.param(jnp.float64, 8e-10, 8e-12, id="float64"),
    pytest.param(jnp.float32, 2e-4, 2e-5, id="float32"),
)

RESULT_FIELDS = (
    "parameters",
    "initial_parameters",
    "objective",
    "objective_valid",
    "history",
    "n_iter",
    "iteration_limit",
    "converged",
    "status",
    "mode",
    "attempted_iteration",
    "attempted_objective",
    "attempted_objective_valid",
    "numerical_failure",
    "collapsed",
    "collapsed_components",
    "failure_stage",
    "group_numerical_failure",
    "failed_pairs",
    "informative_weight",
    "factor_jitter",
    "covariance_ridge",
    "tol",
    "decrease_tol",
    "initialization",
    "metadata",
)


@pytest.fixture
def general_fit_control():
    """Import lazily so an absent implementation is a bounded red failure."""

    return importlib.import_module("development.general_fit_control")


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _ordinary_fixture(dtype):
    """Return a small weighted ``D=2`` collection with three observed sizes."""

    numpy_dtype = np.dtype(dtype)
    params = _params(
        dtype,
        [0.45, 0.55],
        [[-0.7, 0.25], [1.0, -0.35]],
        [
            [[0.8, 0.12], [0.12, 0.65]],
            [[0.55, -0.08], [-0.08, 0.9]],
        ],
    )
    observations = np.asarray(
        [
            [-1.1, 0.2, 0.5],
            [0.4, -0.7, 0.1],
            [1.3, 0.5, -0.4],
            [-0.2, 0.9, 0.7],
            [0.8, -0.1, 0.35],
            [1.7, 0.4, -0.6],
        ],
        dtype=numpy_dtype,
    )
    observed_mask = np.asarray(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [True, False, False],
            [True, True, True],
            [True, True, False],
        ],
        dtype=bool,
    )
    base_projection = np.asarray(
        [[1.0, 0.2], [-0.3, 0.8], [0.4, -0.5]], dtype=numpy_dtype
    )
    projection = np.stack(
        [base_projection + (sample - 2.5) * 0.015 for sample in range(6)]
    ).astype(numpy_dtype)
    noise = np.empty((6, 3, 3), dtype=numpy_dtype)
    for sample in range(6):
        diagonal = np.asarray(
            [
                0.18 + 0.01 * sample,
                0.24 + 0.015 * sample,
                0.31 + 0.02 * sample,
            ],
            dtype=numpy_dtype,
        )
        noise[sample] = np.diag(diagonal)
        noise[sample, 0, 1] = noise[sample, 1, 0] = 0.012
        noise[sample, 1, 2] = noise[sample, 2, 1] = -0.009
    sample_weight = np.asarray(
        [0.5, 1.0, 2.0, 0.0, 1.5, 0.75], dtype=numpy_dtype
    )
    return (
        params,
        observations,
        observed_mask,
        projection,
        noise,
        sample_weight,
    )


def _group_fit(
    fixture,
    *,
    dtype,
    factor_jitter=0.0,
    covariance_ridge=0.0,
):
    params, observations, mask, projection, noise, sample_weight = fixture
    return group_masked_general_fit_inputs(
        params,
        observations,
        mask,
        projection=PerItemProjection(projection),
        noise=PerItemFullNoise(noise),
        sample_weight=sample_weight,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
        dtype=dtype,
    )


def _with_parameters(fit, parameters):
    return replace(fit, grouped=replace(fit.grouped, parameters=parameters))


def _reference_group_steps(fit, parameters):
    jitter = float(np.asarray(fit.controls.factor_jitter))
    return [
        general_e_step(
            np.asarray(group.observations),
            np.asarray(group.projection_matrices),
            np.asarray(group.measurement_covariances),
            np.asarray(parameters.weights),
            np.asarray(parameters.means),
            np.asarray(parameters.covariances),
            factor_jitter=jitter,
        )
        for group in fit.grouped.groups
    ]


def _reference_objective(fit, parameters) -> float:
    steps = _reference_group_steps(fit, parameters)
    sample_weights = [
        np.asarray(group.sample_weight) for group in fit.grouped.groups
    ]
    return float(general_grouped_objective(steps, sample_weights)[2])


def _reference_trajectory(fit, n_steps):
    """Return a float64-oracle trajectory, casting each state like the fit."""

    dtype = fit.grouped.parameters.means.dtype
    current = fit.grouped.parameters
    history = [_reference_objective(fit, current)]
    sample_weights = [
        np.asarray(group.sample_weight) for group in fit.grouped.groups
    ]
    for _ in range(n_steps):
        steps = _reference_group_steps(fit, current)
        candidate, _ = general_grouped_m_step(
            steps,
            sample_weights,
            covariance_ridge=float(np.asarray(fit.controls.covariance_ridge)),
        )
        current = Params(
            weights=jnp.asarray(candidate.weights, dtype=dtype),
            means=jnp.asarray(candidate.means, dtype=dtype),
            covariances=jnp.asarray(candidate.covariances, dtype=dtype),
        )
        history.append(_reference_objective(fit, current))
    return current, np.asarray(history)


def _repeated_grouped_trajectory(fit, n_steps):
    """Return the exact eager trajectory from the already verified one-step."""

    current_fit = fit
    history = [sufficient_statistics_grouped(current_fit).objective]
    for _ in range(n_steps):
        step = one_em_step_grouped(current_fit)
        assert int(np.asarray(step.status)) == int(GroupedStepStatus.SUCCESS)
        history.append(step.objective)
        current_fit = _with_parameters(current_fit, step.parameters)
    return current_fit.grouped.parameters, jnp.stack(history)


def _assert_params_exact(actual, expected) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_value, expected_value)


def _assert_params_close(actual, expected, *, rtol, atol) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        np.testing.assert_allclose(
            actual_value, expected_value, rtol=rtol, atol=atol
        )


def _assert_scalar_equal(actual, expected) -> None:
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def _assert_status(actual, expected) -> None:
    assert int(np.asarray(actual)) == int(expected)


def _assert_general_metadata(module, result) -> None:
    assert isinstance(result.metadata, module.GeneralResultMetadata)
    assert not isinstance(result.metadata, IdentityResultMetadata)
    assert result.metadata._fields == ("contract_id", "contract_version")
    assert result.metadata.contract_id == "xdgmm-jax.general-xd"
    assert result.metadata.contract_version == "0.2.0-draft.1"


def _assert_success_diagnostics(result, fit, *, n_iter, mode) -> None:
    assert result._fields == RESULT_FIELDS
    assert int(np.asarray(result.n_iter)) == n_iter
    assert not bool(np.asarray(result.converged))
    _assert_status(result.mode, mode)
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        result.collapsed_components,
        np.zeros_like(np.asarray(result.parameters.weights), dtype=bool),
    )
    _assert_status(result.failure_stage, GroupedFailureStage.NONE)
    np.testing.assert_array_equal(
        result.group_numerical_failure,
        np.zeros(len(fit.grouped.groups), dtype=bool),
    )
    np.testing.assert_array_equal(
        result.failed_pairs,
        np.zeros(
            (fit.grouped.n_samples, fit.grouped.parameters.weights.shape[0]),
            dtype=bool,
        ),
    )
    _assert_scalar_equal(result.informative_weight, fit.informative_weight)
    _assert_scalar_equal(result.factor_jitter, fit.controls.factor_jitter)
    _assert_scalar_equal(result.covariance_ridge, fit.controls.covariance_ridge)


def test_grouped_fit_schema_exports_general_metadata_and_no_fake_compiled_leaf(
    general_fit_control,
):
    required = {
        "GENERAL_CONTRACT_ID",
        "GENERAL_CONTRACT_VERSION",
        "GeneralResultMetadata",
        "GroupedGeneralFitResult",
        "current_general_result_metadata",
        "fit_converged_grouped",
        "fit_fixed_steps_grouped",
    }
    assert required <= set(general_fit_control.__all__)
    for name in required:
        assert getattr(development, name) is getattr(general_fit_control, name)

    assert general_fit_control.FitStatus is FitStatus
    assert general_fit_control.FitMode is FitMode
    assert general_fit_control.GroupedGeneralFitResult._fields == RESULT_FIELDS
    assert general_fit_control.GeneralResultMetadata._fields == (
        "contract_id",
        "contract_version",
    )
    metadata = general_fit_control.current_general_result_metadata()
    assert metadata == general_fit_control.GeneralResultMetadata(
        "xdgmm-jax.general-xd", "0.2.0-draft.1"
    )
    assert general_fit_control.GENERAL_CONTRACT_ID == "xdgmm-jax.general-xd"
    assert general_fit_control.GENERAL_CONTRACT_VERSION == "0.2.0-draft.1"

    # Contract sections 12 and 15 explicitly exclude eager variable-M grouping
    # from whole-fit JIT/autodiff guarantees.
    assert not hasattr(general_fit_control, "fit_fixed_steps_grouped_kernel")


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
@pytest.mark.parametrize("n_steps", [0, 1, 5])
def test_grouped_fixed_steps_0_1_5_match_repeated_step_and_numpy_oracle(
    general_fit_control, dtype, rtol, atol, n_steps
):
    fit = _group_fit(_ordinary_fixture(dtype), dtype=dtype)
    repeated_parameters, repeated_history = _repeated_grouped_trajectory(
        fit, n_steps
    )
    reference_parameters, reference_history = _reference_trajectory(
        fit, n_steps
    )

    result = general_fit_control.fit_fixed_steps_grouped(
        fit, n_steps=n_steps
    )

    _assert_success_diagnostics(
        result, fit, n_iter=n_steps, mode=FitMode.FIXED_STEPS
    )
    _assert_status(result.status, FitStatus.FIXED_STEPS_COMPLETE)
    _assert_params_exact(result.parameters, repeated_parameters)
    np.testing.assert_array_equal(result.history, repeated_history)
    _assert_scalar_equal(result.objective, repeated_history[-1])
    assert np.asarray(result.history).shape == (n_steps + 1,)
    assert int(np.asarray(result.attempted_iteration)) == n_steps
    assert bool(np.asarray(result.attempted_objective_valid)) is (n_steps > 0)
    if n_steps > 0:
        _assert_scalar_equal(result.attempted_objective, result.objective)
    _assert_params_close(
        result.parameters, reference_parameters, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        result.history,
        reference_history,
        rtol=3 * rtol,
        atol=3 * atol,
    )
    _assert_general_metadata(general_fit_control, result)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_converged_max_iter_zero_returns_valid_initial_state(
    general_fit_control, dtype, rtol, atol
):
    fit = _group_fit(_ordinary_fixture(dtype), dtype=dtype)
    expected_objective = _reference_objective(fit, fit.grouped.parameters)

    result = general_fit_control.fit_converged_grouped(
        fit,
        max_iter=0,
        tol=1e-6,
        decrease_tol=1e-10,
    )

    _assert_params_exact(result.parameters, fit.grouped.parameters)
    assert result._fields == RESULT_FIELDS
    assert np.asarray(result.history).shape == (1,)
    np.testing.assert_allclose(
        result.history, [expected_objective], rtol=3 * rtol, atol=3 * atol
    )
    _assert_scalar_equal(result.objective, result.history[0])
    assert int(np.asarray(result.n_iter)) == 0
    assert not bool(np.asarray(result.converged))
    _assert_status(result.status, FitStatus.MAX_ITER)
    _assert_status(result.mode, FitMode.CONVERGED)
    assert int(np.asarray(result.attempted_iteration)) == 0
    assert not bool(np.asarray(result.attempted_objective_valid))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    _assert_status(result.failure_stage, GroupedFailureStage.NONE)
    _assert_general_metadata(general_fit_control, result)


def test_grouped_converged_large_tolerance_commits_exactly_one_global_update(
    general_fit_control,
):
    fit = _group_fit(_ordinary_fixture(jnp.float64), dtype=jnp.float64)
    step = one_em_step_grouped(fit)
    assert int(np.asarray(step.status)) == int(GroupedStepStatus.SUCCESS)

    result = general_fit_control.fit_converged_grouped(
        fit,
        max_iter=5,
        tol=1e6,
        decrease_tol=1e-10,
    )

    _assert_params_exact(result.parameters, step.parameters)
    np.testing.assert_array_equal(
        result.history, [step.previous_objective, step.objective]
    )
    assert int(np.asarray(result.n_iter)) == 1
    assert bool(np.asarray(result.converged))
    _assert_status(result.status, FitStatus.CONVERGED)
    _assert_status(result.mode, FitMode.CONVERGED)
    assert int(np.asarray(result.attempted_iteration)) == 1
    assert bool(np.asarray(result.attempted_objective_valid))
    _assert_scalar_equal(result.attempted_objective, result.objective)
    _assert_status(result.failure_stage, GroupedFailureStage.NONE)
    _assert_general_metadata(general_fit_control, result)


def _tagged_candidate(initial, shift: float) -> Params:
    return Params(
        weights=initial.weights,
        means=initial.means + jnp.asarray(shift, dtype=initial.means.dtype),
        covariances=initial.covariances,
    )


def _fake_step(
    fit,
    *,
    parameters,
    previous_objective,
    objective,
    status=GroupedStepStatus.SUCCESS,
    failure_stage=GroupedFailureStage.NONE,
    numerical_failure=False,
    collapsed=False,
    collapsed_components=None,
    attempted_objective=None,
    attempted_objective_valid=True,
    group_failure=None,
    failed_pairs=None,
):
    n_groups = len(fit.grouped.groups)
    n_samples = fit.grouped.n_samples
    n_components = fit.grouped.parameters.weights.shape[0]
    if collapsed_components is None:
        collapsed_components = np.zeros(n_components, dtype=bool)
    if group_failure is None:
        group_failure = np.zeros(n_groups, dtype=bool)
    if failed_pairs is None:
        failed_pairs = np.zeros((n_samples, n_components), dtype=bool)
    if attempted_objective is None:
        attempted_objective = objective
    return SimpleNamespace(
        parameters=parameters,
        objective=jnp.asarray(objective),
        previous_objective=jnp.asarray(previous_objective),
        attempted_objective=jnp.asarray(attempted_objective),
        attempted_objective_valid=jnp.asarray(attempted_objective_valid),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        failure_stage=jnp.asarray(int(failure_stage), dtype=jnp.int32),
        collapsed=jnp.asarray(collapsed),
        collapsed_components=jnp.asarray(collapsed_components),
        numerical_failure=jnp.asarray(numerical_failure),
        candidate_group_numerical_failure=jnp.asarray(group_failure),
        candidate_failed_pairs=jnp.asarray(failed_pairs),
        statistics=SimpleNamespace(
            objective=jnp.asarray(previous_objective),
            numerical_failure=jnp.asarray(False),
            group_numerical_failure=jnp.zeros((n_groups,), dtype=bool),
            failed_pairs=jnp.zeros((n_samples, n_components), dtype=bool),
        ),
    )


def _install_step_sequence(monkeypatch, module, expected_parameters, steps):
    iterator = iter(zip(expected_parameters, steps, strict=True))

    def fake_one_step(fit):
        expected, result = next(iterator)
        _assert_params_exact(fit.grouped.parameters, expected)
        return result

    monkeypatch.setattr(module, "one_em_step_grouped", fake_one_step)


def test_finite_decrease_is_accepted_by_fixed_and_rejected_by_converged(
    monkeypatch, general_fit_control
):
    fit = _group_fit(_ordinary_fixture(jnp.float64), dtype=jnp.float64)
    initial = fit.grouped.parameters
    theta1 = _tagged_candidate(initial, 0.1)
    theta2 = _tagged_candidate(initial, 0.2)
    initial_objective = float(
        np.asarray(sufficient_statistics_grouped(fit).objective)
    )
    objective1 = initial_objective + 1.0
    objective2 = objective1 - 0.5

    fixed_steps = [
        _fake_step(
            fit,
            parameters=theta1,
            previous_objective=initial_objective,
            objective=objective1,
        ),
        _fake_step(
            fit,
            parameters=theta2,
            previous_objective=objective1,
            objective=objective2,
        ),
    ]
    _install_step_sequence(
        monkeypatch,
        general_fit_control,
        [initial, theta1],
        fixed_steps,
    )
    fixed = general_fit_control.fit_fixed_steps_grouped(fit, n_steps=2)

    _assert_params_exact(fixed.parameters, theta2)
    np.testing.assert_array_equal(
        fixed.history, [initial_objective, objective1, objective2]
    )
    assert int(np.asarray(fixed.n_iter)) == 2
    _assert_status(fixed.status, FitStatus.FIXED_STEPS_COMPLETE)
    assert bool(np.asarray(fixed.attempted_objective_valid))
    _assert_scalar_equal(fixed.attempted_objective, objective2)

    converged_steps = [
        _fake_step(
            fit,
            parameters=theta1,
            previous_objective=initial_objective,
            objective=objective1,
        ),
        _fake_step(
            fit,
            parameters=theta2,
            previous_objective=objective1,
            objective=objective2,
        ),
    ]
    _install_step_sequence(
        monkeypatch,
        general_fit_control,
        [initial, theta1],
        converged_steps,
    )
    converged = general_fit_control.fit_converged_grouped(
        fit,
        max_iter=2,
        tol=0.0,
        decrease_tol=1e-10,
    )

    _assert_params_exact(converged.parameters, theta1)
    np.testing.assert_array_equal(
        converged.history, [initial_objective, objective1]
    )
    _assert_scalar_equal(converged.objective, objective1)
    assert int(np.asarray(converged.n_iter)) == 1
    assert not bool(np.asarray(converged.converged))
    _assert_status(converged.status, FitStatus.OBJECTIVE_DECREASED)
    assert int(np.asarray(converged.attempted_iteration)) == 2
    assert bool(np.asarray(converged.attempted_objective_valid))
    _assert_scalar_equal(converged.attempted_objective, objective2)
    _assert_status(converged.failure_stage, GroupedFailureStage.NONE)
    np.testing.assert_array_equal(
        converged.group_numerical_failure,
        np.zeros(len(fit.grouped.groups), dtype=bool),
    )
    np.testing.assert_array_equal(
        converged.failed_pairs,
        np.zeros((fit.grouped.n_samples, 2), dtype=bool),
    )


@pytest.mark.parametrize("mode", ["fixed", "converged"])
@pytest.mark.parametrize("failure_kind", ["numerical", "collapse"])
def test_failure_after_one_accept_rolls_back_with_terminating_diagnostics(
    monkeypatch, general_fit_control, mode, failure_kind
):
    fit = _group_fit(_ordinary_fixture(jnp.float64), dtype=jnp.float64)
    initial = fit.grouped.parameters
    theta1 = _tagged_candidate(initial, 0.1)
    initial_objective = float(
        np.asarray(sufficient_statistics_grouped(fit).objective)
    )
    objective1 = initial_objective + 1.0
    first = _fake_step(
        fit,
        parameters=theta1,
        previous_objective=initial_objective,
        objective=objective1,
    )

    if failure_kind == "numerical":
        group_failure = np.zeros(len(fit.grouped.groups), dtype=bool)
        group_failure[-1] = True
        failed_pairs = np.zeros((fit.grouped.n_samples, 2), dtype=bool)
        failed_pairs[1, 0] = True
        second = _fake_step(
            fit,
            parameters=theta1,
            previous_objective=objective1,
            objective=objective1,
            status=GroupedStepStatus.NUMERICAL_FAILURE,
            failure_stage=GroupedFailureStage.CANDIDATE_OBJECTIVE,
            numerical_failure=True,
            attempted_objective=jnp.nan,
            attempted_objective_valid=False,
            group_failure=group_failure,
            failed_pairs=failed_pairs,
        )
        expected_status = FitStatus.NUMERICAL_FAILURE
        expected_stage = GroupedFailureStage.CANDIDATE_OBJECTIVE
        expected_collapsed = False
        expected_components = [False, False]
    else:
        group_failure = np.zeros(len(fit.grouped.groups), dtype=bool)
        failed_pairs = np.zeros((fit.grouped.n_samples, 2), dtype=bool)
        second = _fake_step(
            fit,
            parameters=theta1,
            previous_objective=objective1,
            objective=objective1,
            status=GroupedStepStatus.COMPONENT_COLLAPSED,
            failure_stage=GroupedFailureStage.M_STEP,
            collapsed=True,
            collapsed_components=[False, True],
            attempted_objective=jnp.nan,
            attempted_objective_valid=False,
        )
        expected_status = FitStatus.COMPONENT_COLLAPSED
        expected_stage = GroupedFailureStage.M_STEP
        expected_collapsed = True
        expected_components = [False, True]

    _install_step_sequence(
        monkeypatch,
        general_fit_control,
        [initial, theta1],
        [first, second],
    )
    if mode == "fixed":
        result = general_fit_control.fit_fixed_steps_grouped(fit, n_steps=4)
        expected_mode = FitMode.FIXED_STEPS
    else:
        result = general_fit_control.fit_converged_grouped(
            fit,
            max_iter=4,
            tol=0.0,
            decrease_tol=0.0,
        )
        expected_mode = FitMode.CONVERGED

    _assert_params_exact(result.parameters, theta1)
    np.testing.assert_array_equal(
        result.history, [initial_objective, objective1]
    )
    _assert_scalar_equal(result.objective, objective1)
    assert int(np.asarray(result.n_iter)) == 1
    assert int(np.asarray(result.attempted_iteration)) == 2
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_status(result.status, expected_status)
    _assert_status(result.mode, expected_mode)
    _assert_status(result.failure_stage, expected_stage)
    assert bool(np.asarray(result.numerical_failure)) is (
        failure_kind == "numerical"
    )
    assert bool(np.asarray(result.collapsed)) is expected_collapsed
    np.testing.assert_array_equal(
        result.collapsed_components, expected_components
    )
    np.testing.assert_array_equal(result.group_numerical_failure, group_failure)
    np.testing.assert_array_equal(result.failed_pairs, failed_pairs)


def _current_failure_fixture(dtype):
    """Return a positive-weight row whose selected effective matrices fail."""

    numpy_dtype = np.dtype(dtype)
    params = _params(
        dtype,
        [0.4, 0.6],
        [[-0.2, 0.3], [0.7, -0.4]],
        [np.eye(2), np.asarray([[0.8, 0.05], [0.05, 1.1]])],
    )
    observations = np.asarray([[0.4, 3.0], [5.0, -2.0]], dtype=numpy_dtype)
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    projection = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=numpy_dtype,
    )
    noise = np.asarray(
        [
            [[0.2, 0.0], [0.0, 0.0]],
            [[0.2, 0.0], [0.0, 0.0]],
        ],
        dtype=numpy_dtype,
    )
    return (
        params,
        observations,
        mask,
        projection,
        noise,
        np.ones(2, dtype=numpy_dtype),
    )


def _finite_objective_statistics_failure_fixture():
    """Return a valid current score whose required raw moments overflow."""

    params = _params(jnp.float64, [1.0], [[0.0]], [[[1.0]]])
    return (
        params,
        np.asarray([[10.0]], dtype=np.float64),
        np.asarray([[True]], dtype=bool),
        np.asarray([[[1.0]]], dtype=np.float64),
        np.asarray([[[0.0]]], dtype=np.float64),
        np.asarray([1e308], dtype=np.float64),
    )


@pytest.mark.parametrize("mode", ["fixed", "converged"])
@pytest.mark.parametrize("iteration_limit", [0, 1])
def test_finite_initial_objective_is_retained_on_current_statistics_failure(
    general_fit_control, mode, iteration_limit
):
    fit = _group_fit(
        _finite_objective_statistics_failure_fixture(), dtype=jnp.float64
    )
    statistics = sufficient_statistics_grouped(fit)
    assert bool(np.asarray(statistics.numerical_failure))
    assert np.isfinite(float(np.asarray(statistics.objective)))

    if mode == "fixed":
        result = general_fit_control.fit_fixed_steps_grouped(
            fit, n_steps=iteration_limit
        )
        expected_mode = FitMode.FIXED_STEPS
        zero_status = FitStatus.FIXED_STEPS_COMPLETE
    else:
        result = general_fit_control.fit_converged_grouped(
            fit, max_iter=iteration_limit, tol=0.0, decrease_tol=0.0
        )
        expected_mode = FitMode.CONVERGED
        zero_status = FitStatus.MAX_ITER

    _assert_params_exact(result.parameters, fit.grouped.parameters)
    np.testing.assert_array_equal(result.history, [statistics.objective])
    _assert_scalar_equal(result.objective, statistics.objective)
    assert int(np.asarray(result.n_iter)) == 0
    assert int(np.asarray(result.attempted_iteration)) == iteration_limit
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_status(
        result.status,
        FitStatus.NUMERICAL_FAILURE if iteration_limit else zero_status,
    )
    _assert_status(result.mode, expected_mode)
    _assert_status(
        result.failure_stage,
        (
            GroupedFailureStage.CURRENT_STATISTICS
            if iteration_limit
            else GroupedFailureStage.NONE
        ),
    )
    assert bool(np.asarray(result.numerical_failure)) is bool(iteration_limit)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_initial_current_objective_failure_has_no_accepted_history(
    monkeypatch, general_fit_control, mode
):
    fit = _group_fit(_current_failure_fixture(jnp.float64), dtype=jnp.float64)
    statistics = sufficient_statistics_grouped(fit)
    assert bool(np.asarray(statistics.numerical_failure))
    assert not np.isfinite(float(np.asarray(statistics.objective)))

    def must_not_step(_fit):
        raise AssertionError("an invalid initial state must not attempt an update")

    monkeypatch.setattr(
        general_fit_control, "one_em_step_grouped", must_not_step
    )
    if mode == "fixed":
        result = general_fit_control.fit_fixed_steps_grouped(fit, n_steps=3)
        expected_mode = FitMode.FIXED_STEPS
    else:
        result = general_fit_control.fit_converged_grouped(
            fit, max_iter=3, tol=0.0, decrease_tol=0.0
        )
        expected_mode = FitMode.CONVERGED

    _assert_params_exact(result.parameters, fit.grouped.parameters)
    assert np.asarray(result.history).shape == (0,)
    assert not np.isfinite(float(np.asarray(result.objective)))
    assert int(np.asarray(result.n_iter)) == 0
    assert int(np.asarray(result.attempted_iteration)) == 0
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    _assert_status(result.mode, expected_mode)
    _assert_status(result.failure_stage, GroupedFailureStage.CURRENT_STATISTICS)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(
        result.group_numerical_failure, statistics.group_numerical_failure
    )
    np.testing.assert_array_equal(result.failed_pairs, statistics.failed_pairs)


def _candidate_failure_fixture():
    params = _params(jnp.float64, [1.0], [[0.0]], [[[1e-300]]])
    observations = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    projection = np.asarray(
        [
            [[1e155], [1.0]],
            [[1e155], [1.0]],
        ],
        dtype=np.float64,
    )
    noise = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1e-300]],
            [[1.0, 0.0], [0.0, 1e-300]],
        ],
        dtype=np.float64,
    )
    return params, observations, mask, projection, noise, np.ones(2)


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_real_candidate_objective_failure_rolls_back_first_attempt(
    general_fit_control, mode
):
    fit = _group_fit(_candidate_failure_fixture(), dtype=jnp.float64)
    step = one_em_step_grouped(fit)
    assert int(np.asarray(step.failure_stage)) == int(
        GroupedFailureStage.CANDIDATE_OBJECTIVE
    )
    assert bool(np.asarray(step.numerical_failure))
    if mode == "fixed":
        result = general_fit_control.fit_fixed_steps_grouped(fit, n_steps=3)
        expected_mode = FitMode.FIXED_STEPS
    else:
        result = general_fit_control.fit_converged_grouped(
            fit, max_iter=3, tol=0.0, decrease_tol=0.0
        )
        expected_mode = FitMode.CONVERGED

    _assert_params_exact(result.parameters, fit.grouped.parameters)
    np.testing.assert_array_equal(result.history, [step.previous_objective])
    _assert_scalar_equal(result.objective, step.previous_objective)
    assert int(np.asarray(result.n_iter)) == 0
    assert int(np.asarray(result.attempted_iteration)) == 1
    assert not bool(np.asarray(result.attempted_objective_valid))
    _assert_status(result.status, FitStatus.NUMERICAL_FAILURE)
    _assert_status(result.mode, expected_mode)
    _assert_status(
        result.failure_stage, GroupedFailureStage.CANDIDATE_OBJECTIVE
    )
    np.testing.assert_array_equal(
        result.group_numerical_failure,
        step.candidate_group_numerical_failure,
    )
    np.testing.assert_array_equal(result.failed_pairs, step.candidate_failed_pairs)


def test_nonzero_controls_drive_every_step_and_are_recorded(
    general_fit_control,
):
    factor_jitter = 1e-6
    covariance_ridge = 2e-3
    fit = _group_fit(
        _ordinary_fixture(jnp.float64),
        dtype=jnp.float64,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    repeated_parameters, repeated_history = _repeated_grouped_trajectory(fit, 2)
    reference_parameters, reference_history = _reference_trajectory(fit, 2)

    result = general_fit_control.fit_fixed_steps_grouped(fit, n_steps=2)

    _assert_params_exact(result.parameters, repeated_parameters)
    np.testing.assert_array_equal(result.history, repeated_history)
    _assert_params_close(
        result.parameters, reference_parameters, rtol=8e-10, atol=8e-12
    )
    np.testing.assert_allclose(
        result.history, reference_history, rtol=2.4e-9, atol=2.4e-11
    )
    _assert_scalar_equal(result.factor_jitter, fit.controls.factor_jitter)
    _assert_scalar_equal(result.covariance_ridge, fit.controls.covariance_ridge)
    _assert_scalar_equal(result.informative_weight, fit.informative_weight)
    _assert_general_metadata(general_fit_control, result)


@pytest.mark.parametrize(
    "invalid_count",
    [-1, 1.5, True, pytest.param(np.bool_(True), id="numpy-boolean")],
)
def test_grouped_fit_counts_reject_negative_noninteger_and_boolean(
    general_fit_control, invalid_count
):
    fit = _group_fit(_ordinary_fixture(jnp.float64), dtype=jnp.float64)
    with pytest.raises((TypeError, ValueError)):
        general_fit_control.fit_fixed_steps_grouped(
            fit, n_steps=invalid_count
        )
    with pytest.raises((TypeError, ValueError)):
        general_fit_control.fit_converged_grouped(
            fit, max_iter=invalid_count
        )


@pytest.mark.parametrize(
    "field,value",
    [
        pytest.param("tol", -1.0, id="negative-tol"),
        pytest.param("tol", np.inf, id="infinite-tol"),
        pytest.param("tol", True, id="boolean-tol"),
        pytest.param("decrease_tol", np.nan, id="nan-decrease-tol"),
        pytest.param("decrease_tol", jnp.asarray([0.0]), id="vector-decrease-tol"),
        pytest.param("decrease_tol", 1.0 + 0.0j, id="complex-decrease-tol"),
    ],
)
def test_grouped_converged_tolerances_fail_before_iteration(
    monkeypatch, general_fit_control, field, value
):
    fit = _group_fit(_ordinary_fixture(jnp.float64), dtype=jnp.float64)

    def must_not_step(_fit):
        raise AssertionError("invalid host controls must fail before fitting")

    monkeypatch.setattr(
        general_fit_control, "one_em_step_grouped", must_not_step
    )
    arguments = {"tol": 1e-6, "decrease_tol": 1e-10, field: value}
    with pytest.raises((TypeError, ValueError)):
        general_fit_control.fit_converged_grouped(
            fit, max_iter=1, **arguments
        )


@pytest.mark.parametrize("field", ["tol", "decrease_tol"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.float64(1e300), id="finite-source-overflow"),
        pytest.param(np.float64(1e-300), id="positive-source-underflow"),
    ],
)
def test_grouped_converged_tolerances_must_survive_float32_conversion(
    monkeypatch, general_fit_control, field, value
):
    fit = _group_fit(_ordinary_fixture(jnp.float32), dtype=jnp.float32)
    calls = 0

    def must_not_step(_fit):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid tolerance must fail before fitting")

    monkeypatch.setattr(
        general_fit_control, "one_em_step_grouped", must_not_step
    )
    arguments = {"tol": 1e-6, "decrease_tol": 1e-10, field: value}
    with pytest.raises(
        (ValueError, OverflowError),
        match=rf"{field}|float32|finite|zero|underflow",
    ):
        general_fit_control.fit_converged_grouped(
            fit, max_iter=1, **arguments
        )
    assert calls == 0


def _with_empty_rows(fixture, *, empty_weight):
    params, observations, mask, projection, noise, sample_weight = fixture
    empty_observations = np.asarray(
        [[9.0, 8.0, 7.0], [-8.0, -7.0, -6.0]], dtype=observations.dtype
    )
    empty_mask = np.zeros((2, mask.shape[1]), dtype=bool)
    empty_projection = np.broadcast_to(
        np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.25, -0.5]],
            dtype=projection.dtype,
        ),
        (2, 3, 2),
    ).copy()
    empty_noise = np.broadcast_to(
        np.eye(3, dtype=noise.dtype) * np.asarray(0.2, dtype=noise.dtype),
        (2, 3, 3),
    ).copy()
    return (
        params,
        np.concatenate([empty_observations[:1], observations, empty_observations[1:]]),
        np.concatenate([empty_mask[:1], mask, empty_mask[1:]]),
        np.concatenate([empty_projection[:1], projection, empty_projection[1:]]),
        np.concatenate([empty_noise[:1], noise, empty_noise[1:]]),
        np.concatenate(
            [
                np.asarray([empty_weight], dtype=sample_weight.dtype),
                sample_weight,
                np.asarray([empty_weight], dtype=sample_weight.dtype),
            ]
        ),
    )


@pytest.mark.parametrize("mode", ["fixed", "converged"])
def test_m_zero_rows_and_global_weight_scale_do_not_change_fit_trajectory(
    general_fit_control, mode
):
    dtype = jnp.float64
    base_fixture = _ordinary_fixture(dtype)
    full_fixture = _with_empty_rows(base_fixture, empty_weight=1e200)
    scaled_fixture = (*full_fixture[:-1], full_fixture[-1] * 13.5)
    base_fit = _group_fit(base_fixture, dtype=dtype)
    full_fit = _group_fit(full_fixture, dtype=dtype)
    scaled_fit = _group_fit(scaled_fixture, dtype=dtype)

    if mode == "fixed":
        run = lambda fit: general_fit_control.fit_fixed_steps_grouped(
            fit, n_steps=2
        )
    else:
        run = lambda fit: general_fit_control.fit_converged_grouped(
            fit,
            max_iter=5,
            tol=1e6,
            decrease_tol=1e-10,
        )
    base = run(base_fit)
    full = run(full_fit)
    scaled = run(scaled_fit)

    for actual in (full, scaled):
        _assert_params_close(
            actual.parameters, base.parameters, rtol=8e-10, atol=8e-12
        )
        np.testing.assert_allclose(
            actual.history, base.history, rtol=8e-10, atol=8e-12
        )
        _assert_scalar_equal(actual.status, base.status)
        _assert_scalar_equal(actual.n_iter, base.n_iter)
        _assert_scalar_equal(actual.converged, base.converged)
    np.testing.assert_allclose(
        np.asarray(scaled.informative_weight),
        13.5 * np.asarray(full.informative_weight),
        rtol=8e-10,
        atol=8e-12,
    )
