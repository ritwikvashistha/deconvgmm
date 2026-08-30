"""Red tests for eager grouped general-XD orchestration.

The fixed-``M`` leaf is already tested separately.  This inventory targets the
host-controlled layer that restores variable-``M`` inference, merges every
group before one global M-step, and evaluates the global candidate objective in
a second pass.  It deliberately makes no whole-operation JIT/autodiff claim.
"""

from __future__ import annotations

import importlib
from enum import IntEnum

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import development
from development.general_xd import (
    one_em_step_general,
    posterior_components_general,
    sufficient_statistics_general,
)
from development.identity_xd import EStep, Params
from tests.reference.general_xd import (
    general_e_step,
    general_grouped_m_step,
    general_grouped_objective,
)


DTYPES = (
    pytest.param(jnp.float64, 8e-10, 8e-12, id="float64"),
    pytest.param(jnp.float32, 2e-4, 2e-5, id="float32"),
)


@pytest.fixture
def general_validation():
    return importlib.import_module("development.general_validation")


@pytest.fixture
def general_grouped():
    return importlib.import_module("development.general_grouped")


def _params(dtype, weights, means, covariances) -> Params:
    return Params(
        weights=jnp.asarray(weights, dtype=dtype),
        means=jnp.asarray(means, dtype=dtype),
        covariances=jnp.asarray(covariances, dtype=dtype),
    )


def _ordinary_fixture(dtype):
    params = _params(
        dtype,
        [0.45, 0.55],
        [[-0.7, 0.25], [1.0, -0.35]],
        [
            [[0.8, 0.12], [0.12, 0.65]],
            [[0.55, -0.08], [-0.08, 0.9]],
        ],
    )
    x_full = np.asarray(
        [
            [-1.1, 0.2, 0.5],
            [0.4, -0.7, 0.1],
            [1.3, 0.5, -0.4],
            [-0.2, 0.9, 0.7],
            [0.8, -0.1, 0.35],
            [1.7, 0.4, -0.6],
        ],
        dtype=np.dtype(dtype),
    )
    mask = np.asarray(
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
        [[1.0, 0.2], [-0.3, 0.8], [0.4, -0.5]], dtype=np.dtype(dtype)
    )
    projection = np.stack(
        [base_projection + (sample - 2.5) * 0.015 for sample in range(6)]
    ).astype(np.dtype(dtype))
    noise = np.empty((6, 3, 3), dtype=np.dtype(dtype))
    for sample in range(6):
        diagonal = np.asarray(
            [0.18 + 0.01 * sample, 0.24 + 0.015 * sample, 0.31 + 0.02 * sample],
            dtype=np.dtype(dtype),
        )
        noise[sample] = np.diag(diagonal)
        noise[sample, 0, 1] = noise[sample, 1, 0] = 0.012
        noise[sample, 1, 2] = noise[sample, 2, 1] = -0.009
    sample_weight = np.asarray(
        [0.5, 1.0, 2.0, 0.0, 1.5, 0.75], dtype=np.dtype(dtype)
    )
    return params, x_full, mask, projection, noise, sample_weight


def _group_fit(
    api,
    fixture,
    *,
    dtype,
    factor_jitter=0.0,
    covariance_ridge=0.0,
):
    params, x, mask, projection, noise, sample_weight = fixture
    return api.group_masked_general_fit_inputs(
        params,
        x,
        mask,
        projection=api.PerItemProjection(projection),
        noise=api.PerItemFullNoise(noise),
        sample_weight=sample_weight,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
        dtype=dtype,
    )


def _reference_steps(grouped, params, *, factor_jitter=0.0):
    return [
        general_e_step(
            np.asarray(group.observations),
            np.asarray(group.projection_matrices),
            np.asarray(group.measurement_covariances),
            np.asarray(params.weights),
            np.asarray(params.means),
            np.asarray(params.covariances),
            factor_jitter=factor_jitter,
        )
        for group in grouped.groups
    ]


def _reference_update(fit, *, factor_jitter=0.0, covariance_ridge=0.0):
    grouped = fit.grouped
    old_steps = _reference_steps(
        grouped, grouped.parameters, factor_jitter=factor_jitter
    )
    sample_weights = [np.asarray(group.sample_weight) for group in grouped.groups]
    parameters, statistics = general_grouped_m_step(
        old_steps,
        sample_weights,
        covariance_ridge=covariance_ridge,
    )
    candidate_steps = _reference_steps(
        grouped, parameters, factor_jitter=factor_jitter
    )
    _, _, candidate_objective = general_grouped_objective(
        candidate_steps, sample_weights
    )
    return parameters, statistics, old_steps, candidate_steps, candidate_objective


def _restore(grouped, arrays):
    concatenated = np.concatenate([np.asarray(value) for value in arrays], axis=0)
    return concatenated[np.asarray(grouped.restoration_indices)]


def _assert_exact_params(actual, expected):
    for actual_array, expected_array in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual_array), np.asarray(expected_array)
        )


def test_grouped_general_schema_signatures_and_exports(
    general_validation, general_grouped
):
    required = {
        "GroupedStepStatus",
        "GroupedFailureStage",
        "GroupedPosteriorResult",
        "GroupedGeneralSufficientStatistics",
        "GroupedGeneralEMStepResult",
        "posterior_components_grouped",
        "sufficient_statistics_grouped",
        "one_em_step_grouped",
    }
    assert required <= set(general_grouped.__all__)
    for name in required:
        assert getattr(development, name) is getattr(general_grouped, name)

    assert issubclass(general_grouped.GroupedStepStatus, IntEnum)
    assert {
        name: int(value)
        for name, value in general_grouped.GroupedStepStatus.__members__.items()
    } == {
        "SUCCESS": 0,
        "NUMERICAL_FAILURE": 1,
        "COMPONENT_COLLAPSED": 2,
    }
    assert issubclass(general_grouped.GroupedFailureStage, IntEnum)
    assert {
        name: int(value)
        for name, value in general_grouped.GroupedFailureStage.__members__.items()
    } == {
        "NONE": 0,
        "CURRENT_STATISTICS": 1,
        "M_STEP": 2,
        "CANDIDATE_OBJECTIVE": 3,
    }
    assert general_grouped.GroupedPosteriorResult._fields == (
        "e_step",
        "group_numerical_failure",
    )
    assert general_grouped.GroupedGeneralSufficientStatistics._fields == (
        "mass",
        "first_moment",
        "second_moment",
        "log_mass",
        "component_mean",
        "centered_covariance",
        "weighted_log_likelihood",
        "informative_weight",
        "objective",
        "numerical_failure",
        "failed_pairs",
        "group_numerical_failure",
    )
    assert general_grouped.GroupedGeneralEMStepResult._fields == (
        "parameters",
        "e_step",
        "statistics",
        "objective",
        "previous_objective",
        "attempted_objective",
        "attempted_objective_valid",
        "status",
        "failure_stage",
        "collapsed",
        "collapsed_components",
        "numerical_failure",
        "candidate_group_numerical_failure",
        "candidate_failed_pairs",
    )

    fit = _group_fit(
        general_validation, _ordinary_fixture(jnp.float64), dtype=jnp.float64
    )
    posterior = general_grouped.posterior_components_grouped(fit.grouped)
    statistics = general_grouped.sufficient_statistics_grouped(fit)
    result = general_grouped.one_em_step_grouped(fit)
    assert isinstance(posterior, general_grouped.GroupedPosteriorResult)
    assert isinstance(posterior.e_step, EStep)
    assert isinstance(
        statistics, general_grouped.GroupedGeneralSufficientStatistics
    )
    assert isinstance(result, general_grouped.GroupedGeneralEMStepResult)
    assert isinstance(result.e_step, general_grouped.GroupedPosteriorResult)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_posterior_restores_every_leaf_and_group_status_deterministically(
    general_validation, general_grouped, dtype, rtol, atol
):
    fit = _group_fit(
        general_validation, _ordinary_fixture(dtype), dtype=dtype
    )
    grouped = fit.grouped
    actual = general_grouped.posterior_components_grouped(grouped)
    leaf_results = [
        posterior_components_general(
            grouped.parameters,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
        )
        for group in grouped.groups
    ]

    assert actual.group_numerical_failure.shape == (len(grouped.groups),)
    np.testing.assert_array_equal(
        actual.group_numerical_failure,
        [np.asarray(value.numerical_failure) for value in leaf_results],
    )
    for field in (
        "component_log_density",
        "component_log_joint",
        "score_samples",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual.e_step, field)),
            _restore(grouped, [getattr(value, field) for value in leaf_results]),
            rtol=rtol,
            atol=atol,
        )
    np.testing.assert_array_equal(
        actual.e_step.failed_pairs,
        _restore(grouped, [value.failed_pairs for value in leaf_results]),
    )
    assert not bool(np.asarray(actual.e_step.numerical_failure))


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_ordinary_objective_raw_statistics_and_global_update_match_oracle(
    general_validation, general_grouped, dtype, rtol, atol
):
    fit = _group_fit(
        general_validation, _ordinary_fixture(dtype), dtype=dtype
    )
    expected_params, expected_stats, old_steps, _, candidate_objective = (
        _reference_update(fit)
    )
    actual_stats = general_grouped.sufficient_statistics_grouped(fit)
    actual = general_grouped.one_em_step_grouped(fit)

    assert not bool(np.asarray(actual_stats.numerical_failure))
    assert not bool(np.asarray(actual.numerical_failure))
    assert not bool(np.asarray(actual.collapsed))
    assert int(np.asarray(actual.status)) == int(
        general_grouped.GroupedStepStatus.SUCCESS
    )
    assert int(np.asarray(actual.failure_stage)) == int(
        general_grouped.GroupedFailureStage.NONE
    )
    assert bool(np.asarray(actual.attempted_objective_valid))
    for field in (
        "mass",
        "first_moment",
        "second_moment",
        "log_mass",
        "component_mean",
        "centered_covariance",
        "weighted_log_likelihood",
        "informative_weight",
        "objective",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual_stats, field)),
            np.asarray(getattr(expected_stats, field)),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            np.asarray(getattr(actual.statistics, field)),
            np.asarray(getattr(expected_stats, field)),
            rtol=rtol,
            atol=atol,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual.parameters, field)),
            np.asarray(getattr(expected_params, field)),
            rtol=rtol,
            atol=atol,
        )
    np.testing.assert_allclose(
        np.asarray(actual.previous_objective),
        expected_stats.objective,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.attempted_objective),
        candidate_objective,
        rtol=3 * rtol,
        atol=3 * atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.objective),
        candidate_objective,
        rtol=3 * rtol,
        atol=3 * atol,
    )

    # The restored current E-step is independently checked, not merely compared
    # with another call into the JAX leaf.
    for field in (
        "component_log_density",
        "score_samples",
        "responsibilities",
        "conditional_mean",
        "conditional_covariance",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(actual.e_step.e_step, field)),
            _restore(fit.grouped, [getattr(value, field) for value in old_steps]),
            rtol=3 * rtol,
            atol=3 * atol,
        )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_m_zero_rows_return_prior_and_have_no_fit_effect(
    general_validation, general_grouped, dtype, rtol, atol
):
    params = _params(
        dtype,
        [0.35, 0.65],
        [[-0.5, 0.2], [0.8, -0.3]],
        [
            [[0.9, 0.08], [0.08, 0.7]],
            [[0.6, -0.05], [-0.05, 1.0]],
        ],
    )
    huge = 1e200 if dtype == jnp.float64 else 1e30
    x = np.asarray(
        [[9.0, 8.0], [-0.7, 0.1], [7.0, 6.0], [1.2, -0.4]],
        dtype=np.dtype(dtype),
    )
    mask = np.asarray(
        [[False, False], [True, False], [False, False], [True, False]],
        dtype=bool,
    )
    projection = np.broadcast_to(
        np.asarray([[1.0, 0.2], [-0.3, 0.8]], dtype=np.dtype(dtype)),
        (4, 2, 2),
    ).copy()
    noise = np.broadcast_to(
        np.eye(2, dtype=np.dtype(dtype)) * np.asarray(0.2, dtype=np.dtype(dtype)),
        (4, 2, 2),
    ).copy()
    sample_weight = np.asarray([huge, 0.75, huge, 1.25], dtype=np.dtype(dtype))
    full_fixture = (params, x, mask, projection, noise, sample_weight)
    informative = np.flatnonzero(np.any(mask, axis=1))
    truncated_fixture = (
        params,
        x[informative],
        mask[informative],
        projection[informative],
        noise[informative],
        sample_weight[informative],
    )
    full_fit = _group_fit(general_validation, full_fixture, dtype=dtype)
    truncated_fit = _group_fit(
        general_validation, truncated_fixture, dtype=dtype
    )

    posterior = general_grouped.posterior_components_grouped(full_fit.grouped)
    empty_rows = np.flatnonzero(~np.any(mask, axis=1))
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.component_log_density)[empty_rows],
        np.zeros((2, 2), dtype=np.dtype(dtype)),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.score_samples)[empty_rows],
        np.zeros(2, dtype=np.dtype(dtype)),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.responsibilities)[empty_rows],
        np.broadcast_to(np.asarray(params.weights), (2, 2)),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.conditional_mean)[empty_rows],
        np.broadcast_to(np.asarray(params.means), (2, 2, 2)),
    )
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.conditional_covariance)[empty_rows],
        np.broadcast_to(np.asarray(params.covariances), (2, 2, 2, 2)),
    )

    full_stats = general_grouped.sufficient_statistics_grouped(full_fit)
    truncated_stats = general_grouped.sufficient_statistics_grouped(truncated_fit)
    full_result = general_grouped.one_em_step_grouped(full_fit)
    truncated_result = general_grouped.one_em_step_grouped(truncated_fit)
    for field in (
        "mass",
        "first_moment",
        "second_moment",
        "log_mass",
        "component_mean",
        "centered_covariance",
        "weighted_log_likelihood",
        "informative_weight",
        "objective",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(full_stats, field)),
            np.asarray(getattr(truncated_stats, field)),
            rtol=rtol,
            atol=atol,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(full_result.parameters, field)),
            np.asarray(getattr(truncated_result.parameters, field)),
            rtol=rtol,
            atol=atol,
        )
    np.testing.assert_allclose(
        np.asarray(full_result.objective),
        np.asarray(truncated_result.objective),
        rtol=rtol,
        atol=atol,
    )
    assert float(np.asarray(full_stats.informative_weight)) == pytest.approx(2.0)
    assert not bool(np.asarray(full_result.numerical_failure))


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_positive_weight_controls_factor_failure_semantics_and_mapping(
    general_validation, general_grouped, dtype, rtol, atol
):
    params = _params(
        dtype,
        [0.4, 0.6],
        [[-0.2, 0.3], [0.7, -0.4]],
        [np.eye(2), np.asarray([[0.8, 0.05], [0.05, 1.1]])],
    )
    x = np.asarray([[0.4, 3.0], [5.0, -2.0]], dtype=np.dtype(dtype))
    mask = np.asarray([[True, False], [False, True]], dtype=bool)
    projection = np.asarray(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ],
        dtype=np.dtype(dtype),
    )
    noise = np.asarray(
        [
            [[0.2, 0.0], [0.0, 0.0]],
            [[0.2, 0.0], [0.0, 0.0]],
        ],
        dtype=np.dtype(dtype),
    )
    zero_weight_fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.asarray([1.0, 0.0], dtype=np.dtype(dtype)),
    )
    truncated_fixture = tuple(
        value if index == 0 else value[:1]
        for index, value in enumerate(zero_weight_fixture)
    )
    # Restore the parameter object: it does not carry a leading observation axis.
    truncated_fixture = (params, *truncated_fixture[1:])
    zero_fit = _group_fit(
        general_validation, zero_weight_fixture, dtype=dtype
    )
    truncated_fit = _group_fit(
        general_validation, truncated_fixture, dtype=dtype
    )

    posterior = general_grouped.posterior_components_grouped(zero_fit.grouped)
    assert bool(np.asarray(posterior.e_step.numerical_failure))
    np.testing.assert_array_equal(
        np.asarray(posterior.e_step.failed_pairs),
        [[False, False], [True, True]],
    )
    # Lexicographic masks put (False, True), the failed original row, first.
    np.testing.assert_array_equal(
        np.asarray(posterior.group_numerical_failure), [True, False]
    )

    zero_stats = general_grouped.sufficient_statistics_grouped(zero_fit)
    zero_result = general_grouped.one_em_step_grouped(zero_fit)
    truncated_result = general_grouped.one_em_step_grouped(truncated_fit)
    assert not bool(np.asarray(zero_stats.numerical_failure))
    assert not bool(np.asarray(zero_result.numerical_failure))
    np.testing.assert_array_equal(zero_stats.failed_pairs, np.zeros((2, 2), bool))
    np.testing.assert_array_equal(
        zero_stats.group_numerical_failure, np.zeros(2, bool)
    )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(zero_result.parameters, field)),
            np.asarray(getattr(truncated_result.parameters, field)),
            rtol=rtol,
            atol=atol,
        )

    positive_fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.ones(2, dtype=np.dtype(dtype)),
    )
    positive_fit = _group_fit(
        general_validation, positive_fixture, dtype=dtype
    )
    positive_stats = general_grouped.sufficient_statistics_grouped(positive_fit)
    positive_result = general_grouped.one_em_step_grouped(positive_fit)
    assert bool(np.asarray(positive_stats.numerical_failure))
    assert not np.isfinite(
        float(np.asarray(positive_stats.weighted_log_likelihood))
    )
    assert not np.isfinite(float(np.asarray(positive_stats.objective)))
    np.testing.assert_array_equal(
        positive_stats.failed_pairs, [[False, False], [True, True]]
    )
    np.testing.assert_array_equal(
        positive_stats.group_numerical_failure, [True, False]
    )
    _assert_exact_params(positive_result.parameters, params)
    assert bool(np.asarray(positive_result.numerical_failure))
    assert not bool(np.asarray(positive_result.collapsed))
    assert int(np.asarray(positive_result.status)) == int(
        general_grouped.GroupedStepStatus.NUMERICAL_FAILURE
    )
    assert int(np.asarray(positive_result.failure_stage)) == int(
        general_grouped.GroupedFailureStage.CURRENT_STATISTICS
    )
    assert not np.isfinite(float(np.asarray(positive_result.objective)))
    assert not np.isfinite(
        float(np.asarray(positive_result.previous_objective))
    )
    assert not bool(np.asarray(positive_result.attempted_objective_valid))
    np.testing.assert_array_equal(
        positive_result.candidate_group_numerical_failure, np.zeros(2, bool)
    )
    np.testing.assert_array_equal(
        positive_result.candidate_failed_pairs, np.zeros((2, 2), bool)
    )


@pytest.mark.parametrize(
    "dtype,offset,rtol,atol",
    (
        pytest.param(jnp.float64, 1e12, 8e-10, 8e-12, id="float64"),
        pytest.param(jnp.float32, 1e4, 2e-4, 2e-3, id="float32"),
    ),
)
def test_grouped_centered_merge_preserves_small_spread_at_large_offset(
    general_validation, general_grouped, dtype, offset, rtol, atol
):
    params = _params(dtype, [1.0], [[offset]], [[[10.0]]])
    observed = np.asarray(
        [offset - 2.0, offset - 1.0, offset + 1.0, offset + 2.0],
        dtype=np.dtype(dtype),
    )
    x = np.column_stack(
        [
            np.where(np.asarray([False, True, False, True]), observed, offset),
            np.where(np.asarray([True, False, True, False]), observed, offset),
        ]
    ).astype(np.dtype(dtype))
    mask = np.asarray(
        [
            [False, True],
            [True, False],
            [False, True],
            [True, False],
        ],
        dtype=bool,
    )
    projection = np.ones((4, 2, 1), dtype=np.dtype(dtype))
    noise = np.zeros((4, 2, 2), dtype=np.dtype(dtype))
    fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.ones(4, dtype=np.dtype(dtype)),
    )
    fit = _group_fit(general_validation, fixture, dtype=dtype)

    result = general_grouped.one_em_step_grouped(fit)
    statistics = result.statistics
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_allclose(
        np.asarray(statistics.component_mean), [[offset]], rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        np.asarray(statistics.centered_covariance),
        [[[2.5]]],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.parameters.covariances),
        [[[2.5]]],
        rtol=rtol,
        atol=atol,
    )
    naive = (
        np.asarray(statistics.second_moment)
        / np.asarray(statistics.mass)[:, None, None]
        - np.asarray(statistics.component_mean)[:, :, None]
        * np.asarray(statistics.component_mean)[:, None, :]
    )
    assert not np.allclose(
        naive,
        np.asarray(result.parameters.covariances),
        rtol=0.0,
        atol=1.0,
    )


def test_grouped_log_mass_merge_recovers_component_split_below_one_subnormal(
    general_validation, general_grouped
):
    dtype = jnp.float64
    min_subnormal = np.nextafter(0.0, 1.0)
    params = _params(
        dtype,
        [0.7, 0.3],
        [[0.0], [0.0]],
        [[[1.0]], [[1.0]]],
    )
    x = np.zeros((2, 2), dtype=np.float64)
    mask = np.asarray([[False, True], [True, False]], dtype=bool)
    projection = np.ones((2, 2, 1), dtype=np.float64)
    noise = np.broadcast_to(np.eye(2), (2, 2, 2)).copy()
    fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.asarray([min_subnormal, min_subnormal]),
    )
    fit = _group_fit(general_validation, fixture, dtype=dtype)

    local_minor_mass = []
    for group in fit.grouped.groups:
        local = sufficient_statistics_general(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            group.sample_weight,
        )
        local_minor_mass.append(np.asarray(local.mass[1]))
    np.testing.assert_array_equal(local_minor_mass, np.zeros(2))

    statistics = general_grouped.sufficient_statistics_grouped(fit)
    result = general_grouped.one_em_step_grouped(fit)
    assert np.isfinite(float(np.asarray(statistics.log_mass[1])))
    np.testing.assert_array_equal(
        np.asarray(statistics.mass),
        np.asarray([min_subnormal, min_subnormal]),
    )
    assert not bool(np.asarray(statistics.numerical_failure))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    np.testing.assert_allclose(
        np.asarray(result.parameters.weights),
        np.asarray([0.7, 0.3]),
        rtol=8e-10,
        atol=8e-12,
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_update_is_global_when_every_local_m_step_would_collapse(
    general_validation, general_grouped, dtype, rtol, atol
):
    params = _params(
        dtype,
        [0.5, 0.5],
        [[-25.0], [25.0]],
        [[[1.0]], [[1.0]]],
    )
    observed = np.asarray([-25.1, -24.9, 24.9, 25.1], dtype=np.dtype(dtype))
    mask = np.asarray(
        [[False, True], [False, True], [True, False], [True, False]],
        dtype=bool,
    )
    x = np.zeros((4, 2), dtype=np.dtype(dtype))
    x[:2, 1] = observed[:2]
    x[2:, 0] = observed[2:]
    projection = np.ones((4, 2, 1), dtype=np.dtype(dtype))
    noise = np.broadcast_to(
        np.eye(2, dtype=np.dtype(dtype)) * np.asarray(0.2, dtype=np.dtype(dtype)),
        (4, 2, 2),
    ).copy()
    fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.ones(4, dtype=np.dtype(dtype)),
    )
    fit = _group_fit(general_validation, fixture, dtype=dtype)

    local_results = [
        one_em_step_general(
            params,
            group.observations,
            group.projection_matrices,
            group.measurement_covariances,
            group.sample_weight,
        )
        for group in fit.grouped.groups
    ]
    assert all(bool(np.asarray(result.collapsed)) for result in local_results)

    expected_params, expected_stats, _, _, expected_objective = _reference_update(
        fit
    )
    actual = general_grouped.one_em_step_grouped(fit)
    assert not bool(np.asarray(actual.numerical_failure))
    assert not bool(np.asarray(actual.collapsed))
    assert int(np.asarray(actual.status)) == int(
        general_grouped.GroupedStepStatus.SUCCESS
    )
    np.testing.assert_allclose(
        np.asarray(actual.statistics.mass),
        expected_stats.mass,
        rtol=rtol,
        atol=atol,
    )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            np.asarray(getattr(actual.parameters, field)),
            np.asarray(getattr(expected_params, field)),
            rtol=rtol,
            atol=atol,
        )
    np.testing.assert_allclose(
        np.asarray(actual.objective),
        expected_objective,
        rtol=3 * rtol,
        atol=3 * atol,
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPES)
def test_grouped_controls_use_one_global_ridge_and_one_jitter_policy(
    general_validation, general_grouped, dtype, rtol, atol
):
    factor_jitter = 1e-6 if dtype == jnp.float64 else 1e-4
    covariance_ridge = 2e-3
    fit = _group_fit(
        general_validation,
        _ordinary_fixture(dtype),
        dtype=dtype,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )
    expected_params, expected_stats, _, _, expected_objective = _reference_update(
        fit,
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
    )

    actual = general_grouped.one_em_step_grouped(fit)
    assert not bool(np.asarray(actual.numerical_failure))
    assert not bool(np.asarray(actual.collapsed))
    np.testing.assert_allclose(
        np.asarray(actual.statistics.centered_covariance),
        expected_stats.centered_covariance,
        rtol=rtol,
        atol=atol,
    )
    expected_ridged = expected_stats.centered_covariance + covariance_ridge * np.eye(
        expected_stats.centered_covariance.shape[-1]
    )
    np.testing.assert_allclose(
        np.asarray(actual.parameters.covariances),
        expected_ridged,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.parameters.covariances),
        expected_params.covariances,
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(actual.objective),
        expected_objective,
        rtol=3 * rtol,
        atol=3 * atol,
    )


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32], ids=["float64", "float32"])
def test_grouped_global_component_collapse_rolls_back_before_second_pass(
    general_validation, general_grouped, dtype
):
    params = _params(
        dtype,
        [0.5, 0.5],
        [[0.0], [100.0]],
        [[[0.5]], [[0.5]]],
    )
    x = np.zeros((2, 2), dtype=np.dtype(dtype))
    mask = np.asarray([[False, True], [True, False]], dtype=bool)
    projection = np.ones((2, 2, 1), dtype=np.dtype(dtype))
    noise = np.broadcast_to(
        np.eye(2, dtype=np.dtype(dtype)) * np.asarray(0.2, dtype=np.dtype(dtype)),
        (2, 2, 2),
    ).copy()
    fixture = (
        params,
        x,
        mask,
        projection,
        noise,
        np.ones(2, dtype=np.dtype(dtype)),
    )
    fit = _group_fit(general_validation, fixture, dtype=dtype)

    result = general_grouped.one_em_step_grouped(fit)

    _assert_exact_params(result.parameters, params)
    assert not bool(np.asarray(result.numerical_failure))
    assert bool(np.asarray(result.collapsed))
    np.testing.assert_array_equal(result.collapsed_components, [False, True])
    assert int(np.asarray(result.status)) == int(
        general_grouped.GroupedStepStatus.COMPONENT_COLLAPSED
    )
    assert int(np.asarray(result.failure_stage)) == int(
        general_grouped.GroupedFailureStage.M_STEP
    )
    assert not bool(np.asarray(result.attempted_objective_valid))
    np.testing.assert_array_equal(
        result.candidate_group_numerical_failure, np.zeros(2, bool)
    )
    np.testing.assert_array_equal(
        result.candidate_failed_pairs, np.zeros((2, 2), bool)
    )


def test_grouped_candidate_second_pass_failure_rolls_back_and_maps_later_group(
    general_validation, general_grouped
):
    dtype = jnp.float64
    params = _params(dtype, [1.0], [[0.0]], [[[1e-300]]])
    x = np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
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
    fixture = (params, x, mask, projection, noise, np.ones(2))
    fit = _group_fit(general_validation, fixture, dtype=dtype)

    result = general_grouped.one_em_step_grouped(fit)

    assert not bool(np.asarray(result.statistics.numerical_failure))
    assert np.isfinite(float(np.asarray(result.previous_objective)))
    assert float(np.asarray(result.statistics.centered_covariance[0, 0, 0])) > 0.01
    _assert_exact_params(result.parameters, params)
    assert bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert int(np.asarray(result.status)) == int(
        general_grouped.GroupedStepStatus.NUMERICAL_FAILURE
    )
    assert int(np.asarray(result.failure_stage)) == int(
        general_grouped.GroupedFailureStage.CANDIDATE_OBJECTIVE
    )
    assert not bool(np.asarray(result.attempted_objective_valid))
    assert not np.isfinite(float(np.asarray(result.attempted_objective)))
    np.testing.assert_array_equal(result.objective, result.previous_objective)
    # Lexicographic order is (False, True) then (True, False); only the latter
    # overflows after the global covariance grows from 1e-300 to O(1e-2).
    np.testing.assert_array_equal(
        result.candidate_group_numerical_failure, [False, True]
    )
    np.testing.assert_array_equal(
        result.candidate_failed_pairs, [[True], [False]]
    )


@pytest.mark.parametrize(
    "large_row_mask",
    (
        pytest.param([False, True], id="large-group-first"),
        pytest.param([True, False], id="tiny-group-first"),
    ),
)
def test_grouped_subnormal_fraction_preserves_huge_mean_and_covariance_terms(
    general_validation, general_grouped, large_row_mask
):
    """Chan products use retained log fractions in either group order."""

    dtype = jnp.float64
    min_subnormal = np.nextafter(0.0, 1.0)
    huge = 1e308
    params = _params(dtype, [1.0], [[0.0]], [[[huge]]])
    large_row_mask = np.asarray(large_row_mask, dtype=bool)
    tiny_row_mask = ~large_row_mask
    mask = np.stack([large_row_mask, tiny_row_mask])

    # The ordinary-mass row has posterior mean zero; the min-subnormal row has
    # posterior mean ``huge`` and zero posterior covariance.  Its global mean
    # and between-group Chan term are both representable even though its group
    # fraction is subnormal.  Swapping the masks reverses lexicographic order.
    x = np.asarray([[0.0, 0.0], [huge, huge]])
    projection = np.ones((2, 2, 1))
    noise = np.zeros((2, 2, 2))
    fit = _group_fit(
        general_validation,
        (params, x, mask, projection, noise, [1.0, min_subnormal]),
        dtype=dtype,
    )
    statistics = general_grouped.sufficient_statistics_grouped(fit)
    expected_mean = min_subnormal * huge
    expected_spread = (np.sqrt(min_subnormal) * huge) ** 2
    assert not bool(np.asarray(statistics.numerical_failure))
    np.testing.assert_allclose(
        np.asarray(statistics.component_mean),
        [[expected_mean]],
        rtol=8e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(statistics.centered_covariance),
        [[[expected_spread]]],
        rtol=8e-10,
        atol=0.0,
    )

    # Isolate the within-group covariance term: the ordinary row observes the
    # latent value exactly, while the tiny row has R=0 and retains V=huge.
    # The contracted contribution minsub*huge is representable and must not be
    # flushed merely because the group fraction itself is subnormal.
    covariance_projection = np.ones((2, 2, 1))
    covariance_projection[1] = 0.0
    covariance_noise = np.zeros((2, 2, 2))
    covariance_noise[1] = np.eye(2)
    covariance_fit = _group_fit(
        general_validation,
        (
            params,
            np.zeros((2, 2)),
            mask,
            covariance_projection,
            covariance_noise,
            [1.0, min_subnormal],
        ),
        dtype=dtype,
    )
    covariance_statistics = general_grouped.sufficient_statistics_grouped(
        covariance_fit
    )
    np.testing.assert_allclose(
        np.asarray(covariance_statistics.centered_covariance),
        [[[min_subnormal * huge]]],
        rtol=8e-10,
        atol=0.0,
    )
