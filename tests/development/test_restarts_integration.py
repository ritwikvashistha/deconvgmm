"""Real CPU integrations for the temporary sequential restart wrappers."""

from __future__ import annotations

from dataclasses import replace
import importlib

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import (
    FitStatus,
    fit_converged,
    fit_fixed_steps,
)
from development.general_fit_control import (
    fit_converged_grouped,
    fit_fixed_steps_grouped,
)
from development.general_validation import (
    GroupedGeneralFitInputs,
    IdentityProjection,
    PerItemFullNoise,
    group_masked_general_fit_inputs,
)
from development.identity_xd import Params
from development.serialization import (
    ArtifactFormatError,
    save_grouped_general_fit_result,
    save_identity_fit_result,
)


@pytest.fixture
def restarts():
    return importlib.import_module("development.restarts")


def _parameters(dtype, *, shifted: bool) -> Params:
    return Params(
        weights=jnp.asarray([0.5, 0.5], dtype=dtype),
        means=jnp.asarray(
            [[-1.4, 0.1], [1.3, -0.2]]
            if not shifted
            else [[-0.2, 0.8], [0.35, -0.9]],
            dtype=dtype,
        ),
        covariances=jnp.asarray(
            [
                [[0.7, 0.08], [0.08, 0.55]],
                [[0.6, -0.05], [-0.05, 0.75]],
            ],
            dtype=dtype,
        ),
    )


def _identity_data(dtype):
    observations = jnp.asarray(
        [
            [-1.8, 0.0],
            [-1.2, 0.4],
            [-0.9, -0.3],
            [-1.5, 0.2],
            [0.9, -0.4],
            [1.4, 0.1],
            [1.7, -0.2],
            [1.1, 0.35],
        ],
        dtype=dtype,
    )
    one_noise = jnp.asarray([[0.12, 0.02], [0.02, 0.09]], dtype=dtype)
    noise = jnp.broadcast_to(one_noise, (observations.shape[0], 2, 2))
    return observations, noise


def _candidate_at(candidates, index: int) -> Params:
    return Params(
        candidates.weights[index],
        candidates.means[index],
        candidates.covariances[index],
    )


def _with_parameters(
    fit: GroupedGeneralFitInputs, parameters: Params
) -> GroupedGeneralFitInputs:
    return replace(
        fit,
        grouped=replace(fit.grouped, parameters=parameters),
    )


def _grouped_identity_fit(parameters, observations, noise, *, scale=1.0):
    mask = np.ones(tuple(observations.shape), dtype=bool)
    return group_masked_general_fit_inputs(
        parameters,
        np.asarray(observations),
        mask,
        projection=IdentityProjection(2),
        noise=PerItemFullNoise(np.asarray(noise)),
        sample_weight=np.full(observations.shape[0], scale, dtype=np.float64),
        dtype=parameters.means.dtype,
    )


def _manual_best(results) -> int:
    best = 0
    best_value = float(np.asarray(results[0].objective))
    for index, result in enumerate(results[1:], start=1):
        value = float(np.asarray(result.objective))
        if value > best_value:
            best = index
            best_value = value
    return best


def _assert_fit_result_endpoint_equal(actual, expected) -> None:
    for actual_params, expected_params in (
        (actual.parameters, expected.parameters),
        (actual.initial_parameters, expected.initial_parameters),
    ):
        for actual_leaf, expected_leaf in zip(
            actual_params, expected_params, strict=True
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(expected_leaf)
            )
    for field in (
        "objective",
        "objective_valid",
        "history",
        "n_iter",
        "status",
        "mode",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(expected, field)),
        )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_identity_converged_and_fixed_restart_results_match_manual_runs(
    restarts, dtype
):
    observations, noise = _identity_data(dtype)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(dtype, shifted=True), _parameters(dtype, shifted=False)],
        dtype=dtype,
    )

    manual_converged = [
        fit_converged(
            _candidate_at(candidates, index),
            observations,
            noise,
            max_iter=0,
        )
        for index in range(2)
    ]
    converged = restarts.fit_converged_restarts(
        candidates, observations, noise, max_iter=0
    )
    expected_converged = _manual_best(manual_converged)
    assert int(np.asarray(converged.selection.selected_restart)) == (
        expected_converged
    )
    _assert_fit_result_endpoint_equal(
        converged.selected_result, manual_converged[expected_converged]
    )
    assert int(np.asarray(converged.selected_result.status)) == int(
        FitStatus.MAX_ITER
    )

    manual_fixed = [
        fit_fixed_steps(
            _candidate_at(candidates, index),
            observations,
            noise,
            n_steps=1,
        )
        for index in range(2)
    ]
    fixed = restarts.fit_fixed_steps_restarts(
        candidates, observations, noise, n_steps=1
    )
    expected_fixed = _manual_best(manual_fixed)
    assert int(np.asarray(fixed.selection.selected_restart)) == expected_fixed
    _assert_fit_result_endpoint_equal(
        fixed.selected_result, manual_fixed[expected_fixed]
    )
    assert int(np.asarray(fixed.selected_result.status)) == int(
        FitStatus.FIXED_STEPS_COMPLETE
    )


@pytest.mark.parametrize("mode", ["converged", "fixed"])
def test_grouped_restart_result_matches_manual_grouped_runs_and_identity_index(
    restarts, mode
):
    dtype = jnp.float64
    observations, noise = _identity_data(dtype)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(dtype, shifted=True), _parameters(dtype, shifted=False)],
        dtype=dtype,
    )
    grouped_fit = _grouped_identity_fit(
        _candidate_at(candidates, 0), observations, noise
    )

    if mode == "converged":
        manual = [
            fit_converged_grouped(
                _with_parameters(grouped_fit, _candidate_at(candidates, index)),
                max_iter=0,
            )
            for index in range(2)
        ]
        grouped = restarts.fit_converged_grouped_restarts(
            grouped_fit, candidates=candidates, max_iter=0
        )
        identity = restarts.fit_converged_restarts(
            candidates, observations, noise, max_iter=0
        )
    else:
        manual = [
            fit_fixed_steps_grouped(
                _with_parameters(grouped_fit, _candidate_at(candidates, index)),
                n_steps=1,
            )
            for index in range(2)
        ]
        grouped = restarts.fit_fixed_steps_grouped_restarts(
            grouped_fit, candidates=candidates, n_steps=1
        )
        identity = restarts.fit_fixed_steps_restarts(
            candidates, observations, noise, n_steps=1
        )

    expected = _manual_best(manual)
    assert int(np.asarray(grouped.selection.selected_restart)) == expected
    _assert_fit_result_endpoint_equal(grouped.selected_result, manual[expected])
    assert int(np.asarray(grouped.selection.selected_restart)) == int(
        np.asarray(identity.selection.selected_restart)
    )
    np.testing.assert_allclose(
        np.asarray(grouped.diagnostics.objective),
        np.asarray(identity.diagnostics.objective),
        rtol=2e-12,
        atol=2e-12,
    )


def test_grouped_common_weight_scaling_preserves_selected_index(restarts):
    dtype = jnp.float64
    observations, noise = _identity_data(dtype)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(dtype, shifted=True), _parameters(dtype, shifted=False)],
        dtype=dtype,
    )
    ordinary = _grouped_identity_fit(
        _candidate_at(candidates, 0), observations, noise, scale=1.0
    )
    scaled = _grouped_identity_fit(
        _candidate_at(candidates, 0), observations, noise, scale=13.5
    )
    ordinary_result = restarts.fit_converged_grouped_restarts(
        ordinary, candidates=candidates, max_iter=2, tol=0.0
    )
    scaled_result = restarts.fit_converged_grouped_restarts(
        scaled, candidates=candidates, max_iter=2, tol=0.0
    )
    assert int(np.asarray(ordinary_result.selection.selected_restart)) == int(
        np.asarray(scaled_result.selection.selected_restart)
    )
    np.testing.assert_allclose(
        np.asarray(ordinary_result.diagnostics.objective),
        np.asarray(scaled_result.diagnostics.objective),
        rtol=2e-12,
        atol=2e-12,
    )


def test_explicit_warm_start_recomputes_and_resets_fit_state(restarts):
    dtype = jnp.float64
    observations, noise = _identity_data(dtype)
    prior = fit_fixed_steps(
        _parameters(dtype, shifted=True),
        observations,
        noise,
        n_steps=1,
        covariance_ridge=1e-4,
    )
    candidates = restarts.user_supplied_restart_candidates(
        [prior.parameters], dtype=dtype
    )
    restarted = restarts.fit_fixed_steps_restarts(
        candidates,
        observations,
        noise,
        n_steps=0,
        factor_jitter=0.0,
        covariance_ridge=0.0,
    )
    selected = restarted.selected_result
    for actual, expected in zip(
        selected.initial_parameters, prior.parameters, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert int(np.asarray(selected.n_iter)) == 0
    assert selected.history.shape == (1,)
    assert int(np.asarray(selected.attempted_iteration)) == 0
    assert not bool(np.asarray(selected.attempted_objective_valid))
    assert float(np.asarray(selected.factor_jitter)) == 0.0
    assert float(np.asarray(selected.covariance_ridge)) == 0.0


def test_grouped_candidate_shape_mismatch_fails_before_any_controller_call(
    restarts, monkeypatch
):
    observations, noise = _identity_data(jnp.float64)
    fit = _grouped_identity_fit(
        _parameters(jnp.float64, shifted=False), observations, noise
    )
    incompatible = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        means=jnp.asarray([[0.0, 0.0]], dtype=jnp.float64),
        covariances=jnp.asarray([np.eye(2)], dtype=jnp.float64),
    )
    candidates = restarts.user_supplied_restart_candidates(
        [incompatible], dtype=jnp.float64
    )
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("controller must not run")

    monkeypatch.setattr(restarts, "fit_converged_grouped", fail_if_called)
    with pytest.raises(ValueError, match="K|component|shape|compatible"):
        restarts.fit_converged_grouped_restarts(
            fit, candidates=candidates, max_iter=0
        )
    assert calls == 0


def test_common_static_controls_fail_before_candidate_zero_runs(
    restarts, monkeypatch
):
    observations, noise = _identity_data(jnp.float64)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(jnp.float64, shifted=False)], dtype=jnp.float64
    )
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("controller must not run")

    monkeypatch.setattr(restarts, "fit_converged", fail_if_called)
    with pytest.raises((TypeError, ValueError), match="tol|scalar"):
        restarts.fit_converged_restarts(
            candidates,
            observations,
            noise,
            tol=jnp.asarray([1e-6]),
        )
    assert calls == 0


def test_float32_convergence_control_overflow_fails_before_candidate_zero(
    restarts, monkeypatch
):
    observations, noise = _identity_data(jnp.float32)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(jnp.float32, shifted=False)], dtype=jnp.float32
    )
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("controller must not run")

    monkeypatch.setattr(restarts, "fit_converged", fail_if_called)
    with pytest.raises((ValueError, OverflowError), match="tol|finite|float32"):
        restarts.fit_converged_restarts(
            candidates,
            observations,
            noise,
            tol=np.float64(1e300),
        )
    assert calls == 0


@pytest.mark.parametrize("wrapper_kind", ["identity", "grouped"])
@pytest.mark.parametrize("field", ["tol", "decrease_tol"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.float64(1e300), id="finite-source-overflow"),
        pytest.param(np.float64(1e-300), id="positive-source-underflow"),
    ],
)
def test_all_converged_restart_controls_fail_before_candidate_zero(
    restarts, monkeypatch, wrapper_kind, field, value
):
    observations, noise = _identity_data(jnp.float32)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(jnp.float32, shifted=False)], dtype=jnp.float32
    )
    calls = 0

    def fail_if_called(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise AssertionError("controller must not run")

    arguments = {"tol": 1e-6, "decrease_tol": 1e-10, field: value}
    if wrapper_kind == "identity":
        monkeypatch.setattr(restarts, "fit_converged", fail_if_called)
        invoke = lambda: restarts.fit_converged_restarts(
            candidates,
            observations,
            noise,
            **arguments,
        )
    else:
        fit = _grouped_identity_fit(
            _candidate_at(candidates, 0), observations, noise
        )
        monkeypatch.setattr(
            restarts, "fit_converged_grouped", fail_if_called
        )
        invoke = lambda: restarts.fit_converged_grouped_restarts(
            fit,
            candidates=candidates,
            **arguments,
        )

    with pytest.raises(
        (ValueError, OverflowError),
        match=rf"{field}|float32|finite|zero|underflow",
    ):
        invoke()
    assert calls == 0


def test_restart_wrapper_results_are_not_single_fit_serialization_records(
    restarts, tmp_path
):
    observations, noise = _identity_data(jnp.float64)
    candidates = restarts.user_supplied_restart_candidates(
        [_parameters(jnp.float64, shifted=False)], dtype=jnp.float64
    )
    identity = restarts.fit_converged_restarts(
        candidates, observations, noise, max_iter=0
    )
    grouped_fit = _grouped_identity_fit(
        _candidate_at(candidates, 0), observations, noise
    )
    grouped = restarts.fit_converged_grouped_restarts(
        grouped_fit, candidates=candidates, max_iter=0
    )

    with pytest.raises(ArtifactFormatError, match="FitResult"):
        save_identity_fit_result(tmp_path / "identity.artifact", identity)
    with pytest.raises(ArtifactFormatError, match="GroupedGeneralFitResult"):
        save_grouped_general_fit_result(tmp_path / "general.artifact", grouped)
    assert not (tmp_path / "identity.artifact").exists()
    assert not (tmp_path / "general.artifact").exists()
