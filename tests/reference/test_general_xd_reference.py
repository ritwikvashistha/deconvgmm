"""Analytic and cross-oracle checks for the NumPy general-XD reference."""

from __future__ import annotations

import numpy as np
import pytest

from tests.reference.general_xd import (
    general_e_step,
    general_em_step,
    general_grouped_m_step,
    general_grouped_objective,
    general_m_step,
    general_objective,
    general_sufficient_statistics,
    marginalized_posterior,
)
from tests.reference.identity_xd import (
    identity_e_step,
    identity_em_step,
    marginalized_posterior as identity_marginalized_posterior,
)


def _weighted_fixture():
    observations = np.array([[-1.2], [0.4], [1.1], [2.0]])
    projection_matrices = np.array(
        [
            [[1.0, 0.25]],
            [[0.5, 1.0]],
            [[1.0, -0.5]],
            [[1.5, 0.2]],
        ]
    )
    measurement_covariances = np.array([[[0.2]], [[0.1]], [[0.3]], [[0.15]]])
    weights = np.array([0.4, 0.6])
    means = np.array([[-0.8, 0.3], [1.2, -0.4]])
    covariances = np.array(
        [
            [[0.7, 0.1], [0.1, 0.5]],
            [[0.6, -0.08], [-0.08, 0.9]],
        ]
    )
    return (
        observations,
        projection_matrices,
        measurement_covariances,
        weights,
        means,
        covariances,
    )


def test_one_component_scalar_projection_matches_literal_analytic_result():
    """K=1,D=M=1 with R=2 has a closed-form posterior and density."""

    e_step = general_e_step(
        observations=[[1.75]],
        projection_matrices=[[[2.0]]],
        measurement_covariances=[[[0.5]]],
        weights=[1.0],
        means=[[0.25]],
        covariances=[[[3.0]]],
    )
    posterior_mean, posterior_covariance = marginalized_posterior(e_step)

    expected_log_density = -0.5 * (
        np.log(2.0 * np.pi) + np.log(12.5) + 0.125
    )
    np.testing.assert_allclose(
        e_step.component_log_density,
        [[expected_log_density]],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_array_equal(e_step.responsibilities, [[1.0]])
    np.testing.assert_allclose(e_step.conditional_mean, [[[0.85]]], atol=1e-15)
    np.testing.assert_allclose(
        e_step.conditional_covariance, [[[[0.12]]]], atol=1e-15
    )
    np.testing.assert_allclose(posterior_mean, [[0.85]], atol=1e-15)
    np.testing.assert_allclose(posterior_covariance, [[[0.12]]], atol=1e-15)

    parameters, statistics = general_m_step(e_step)
    np.testing.assert_array_equal(statistics.mass, [1.0])
    np.testing.assert_allclose(statistics.first_moment, [[0.85]], atol=1e-15)
    np.testing.assert_allclose(
        statistics.second_moment,
        [[[0.12 + 0.85**2]]],
        atol=1e-15,
    )
    np.testing.assert_array_equal(parameters.weights, [1.0])
    np.testing.assert_allclose(parameters.means, [[0.85]], atol=1e-15)
    np.testing.assert_allclose(parameters.covariances, [[[0.12]]], atol=1e-15)


def test_identity_projections_match_the_independent_identity_oracle():
    observations = np.array([[-0.8, 0.1], [0.5, -0.2], [1.2, 0.9]])
    measurement_covariances = np.array(
        [
            [[0.2, 0.03], [0.03, 0.1]],
            [[0.15, -0.02], [-0.02, 0.25]],
            [[0.08, 0.01], [0.01, 0.18]],
        ]
    )
    weights = np.array([0.35, 0.65])
    means = np.array([[-0.6, 0.2], [0.9, 0.5]])
    covariances = np.array(
        [
            [[0.8, 0.12], [0.12, 0.6]],
            [[0.5, -0.07], [-0.07, 0.9]],
        ]
    )
    projection_matrices = np.broadcast_to(np.eye(2), (3, 2, 2))

    general = general_e_step(
        observations,
        projection_matrices,
        measurement_covariances,
        weights,
        means,
        covariances,
    )
    identity = identity_e_step(
        observations,
        measurement_covariances,
        weights,
        means,
        covariances,
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
            getattr(general, field),
            getattr(identity, field),
            rtol=2e-14,
            atol=2e-15,
        )

    general_mean, general_covariance = marginalized_posterior(general)
    identity_mean, identity_covariance = identity_marginalized_posterior(identity)
    np.testing.assert_allclose(general_mean, identity_mean, atol=2e-15)
    np.testing.assert_allclose(general_covariance, identity_covariance, atol=2e-15)

    general_parameters, _, general_statistics = general_em_step(
        observations,
        projection_matrices,
        measurement_covariances,
        weights,
        means,
        covariances,
    )
    identity_parameters, _, identity_statistics = identity_em_step(
        observations,
        measurement_covariances,
        weights,
        means,
        covariances,
    )
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            getattr(general_statistics, field),
            getattr(identity_statistics, field),
            rtol=2e-14,
            atol=2e-15,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            getattr(general_parameters, field),
            getattr(identity_parameters, field),
            rtol=2e-14,
            atol=2e-15,
        )


def test_dimension_reducing_selector_preserves_unobserved_prior_coordinate():
    e_step = general_e_step(
        observations=[[3.0, -0.5]],
        projection_matrices=[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]],
        measurement_covariances=[[[1.0, 0.0], [0.0, 0.25]]],
        weights=[1.0],
        means=[[1.0, -2.0, 0.5]],
        covariances=[np.diag([4.0, 9.0, 1.0])],
    )

    expected_log_density = -0.5 * (
        2.0 * np.log(2.0 * np.pi) + np.log(6.25) + 1.6
    )
    np.testing.assert_allclose(
        e_step.component_log_density,
        [[expected_log_density]],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        e_step.conditional_mean,
        [[[2.6, -2.0, -0.3]]],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        e_step.conditional_covariance,
        [[np.diag([0.8, 9.0, 0.2])]],
        rtol=1e-15,
        atol=1e-15,
    )


def test_nonorthogonal_projection_matches_literal_rational_posterior():
    e_step = general_e_step(
        observations=[[3.0, 0.0]],
        projection_matrices=[[[1.0, 1.0], [2.0, -1.0]]],
        measurement_covariances=[[[1.0, 0.0], [0.0, 2.0]]],
        weights=[1.0],
        means=[[1.0, -1.0]],
        covariances=[[[2.0, 0.0], [0.0, 3.0]]],
    )

    expected_log_density = -0.5 * (
        2.0 * np.log(2.0 * np.pi) + np.log(77.0) + 27.0 / 11.0
    )
    np.testing.assert_allclose(
        e_step.component_log_density,
        [[expected_log_density]],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        e_step.conditional_mean,
        [[[1.0, 16.0 / 11.0]]],
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        e_step.conditional_covariance,
        [[[[2.0 / 7.0, 0.0], [0.0, 6.0 / 11.0]]]],
        rtol=2e-15,
        atol=2e-15,
    )


def test_sample_weight_common_scaling_preserves_update_and_normalized_objective():
    fixture = _weighted_fixture()
    sample_weight = np.array([0.5, 2.0, 1.25, 0.75])
    scale = 7.25
    e_step = general_e_step(*fixture)

    parameters, statistics = general_m_step(
        e_step, sample_weight=sample_weight
    )
    scaled_parameters, scaled_statistics = general_m_step(
        e_step, sample_weight=scale * sample_weight
    )

    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            getattr(scaled_parameters, field),
            getattr(parameters, field),
            rtol=2e-15,
            atol=2e-15,
        )
    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            getattr(scaled_statistics, field),
            scale * getattr(statistics, field),
            rtol=2e-15,
            atol=2e-15,
        )
    np.testing.assert_allclose(
        general_objective(e_step, scale * sample_weight),
        scale * general_objective(e_step, sample_weight),
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        general_objective(e_step, scale * sample_weight, normalized=True),
        general_objective(e_step, sample_weight, normalized=True),
        rtol=2e-15,
        atol=2e-15,
    )


def test_zero_weight_row_has_no_effect_on_statistics_update_or_objective():
    fixture = _weighted_fixture()
    full_e_step = general_e_step(*fixture)
    full_sample_weight = np.array([0.5, 2.0, 1.25, 0.0])
    full_parameters, full_statistics = general_m_step(
        full_e_step, sample_weight=full_sample_weight
    )

    truncated_fixture = tuple(value[:-1] for value in fixture[:3]) + fixture[3:]
    truncated_e_step = general_e_step(*truncated_fixture)
    truncated_sample_weight = full_sample_weight[:-1]
    truncated_parameters, truncated_statistics = general_m_step(
        truncated_e_step, sample_weight=truncated_sample_weight
    )

    for field in ("mass", "first_moment", "second_moment"):
        np.testing.assert_allclose(
            getattr(full_statistics, field),
            getattr(truncated_statistics, field),
            rtol=1e-15,
            atol=1e-15,
        )
    for field in ("weights", "means", "covariances"):
        np.testing.assert_allclose(
            getattr(full_parameters, field),
            getattr(truncated_parameters, field),
            rtol=1e-15,
            atol=1e-15,
        )
    np.testing.assert_allclose(
        general_objective(full_e_step, full_sample_weight),
        general_objective(truncated_e_step, truncated_sample_weight),
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        general_objective(full_e_step, full_sample_weight, normalized=True),
        general_objective(
            truncated_e_step, truncated_sample_weight, normalized=True
        ),
        rtol=1e-15,
        atol=1e-15,
    )


def test_zero_observed_dimension_returns_prior_and_zero_fitting_statistics():
    weights = np.array([0.25, 0.75])
    means = np.array([[-1.0, 0.5], [2.0, -0.25]])
    covariances = np.array(
        [
            [[0.8, 0.1], [0.1, 0.6]],
            [[1.2, -0.2], [-0.2, 0.9]],
        ]
    )
    e_step = general_e_step(
        observations=np.empty((3, 0)),
        projection_matrices=np.empty((3, 0, 2)),
        measurement_covariances=np.empty((3, 0, 0)),
        weights=weights,
        means=means,
        covariances=covariances,
    )

    np.testing.assert_array_equal(e_step.component_log_density, np.zeros((3, 2)))
    np.testing.assert_allclose(
        e_step.component_log_joint,
        np.broadcast_to(np.log(weights), (3, 2)),
    )
    np.testing.assert_allclose(e_step.score_samples, np.zeros(3), atol=1e-15)
    np.testing.assert_allclose(
        e_step.responsibilities, np.broadcast_to(weights, (3, 2))
    )
    np.testing.assert_array_equal(
        e_step.conditional_mean, np.broadcast_to(means, (3, 2, 2))
    )
    np.testing.assert_array_equal(
        e_step.conditional_covariance,
        np.broadcast_to(covariances, (3, 2, 2, 2)),
    )

    posterior_mean, posterior_covariance = marginalized_posterior(e_step)
    prior_mean = np.einsum("k,kd->d", weights, means)
    centered = means - prior_mean
    prior_covariance = np.einsum(
        "k,kde->de",
        weights,
        covariances + centered[:, :, None] * centered[:, None, :],
    )
    np.testing.assert_allclose(
        posterior_mean, np.broadcast_to(prior_mean, (3, 2))
    )
    np.testing.assert_allclose(
        posterior_covariance, np.broadcast_to(prior_covariance, (3, 2, 2))
    )

    statistics = general_sufficient_statistics(
        e_step, sample_weight=[1.0, 2.0, 3.0]
    )
    np.testing.assert_array_equal(statistics.mass, np.zeros(2))
    np.testing.assert_array_equal(statistics.first_moment, np.zeros((2, 2)))
    np.testing.assert_array_equal(statistics.second_moment, np.zeros((2, 2, 2)))
    np.testing.assert_equal(
        general_objective(e_step, [1.0, 2.0, 3.0]), 0.0
    )
    with pytest.raises(ValueError, match="positive-weight observed rows"):
        general_objective(
            e_step, [1.0, 2.0, 3.0], normalized=True
        )
    with pytest.raises(FloatingPointError, match="collapsed component"):
        general_m_step(e_step, sample_weight=[1.0, 2.0, 3.0])


def test_grouped_reference_excludes_m_zero_and_uses_centered_global_update():
    offset = 1.0e12
    weights = np.asarray([1.0])
    means = np.asarray([[offset]])
    covariances = np.asarray([[[10.0]]])
    informative_steps = [
        general_e_step(
            observations=np.asarray([[offset - 2.0], [offset + 1.0]]),
            projection_matrices=np.ones((2, 1, 1)),
            measurement_covariances=np.zeros((2, 1, 1)),
            weights=weights,
            means=means,
            covariances=covariances,
        ),
        general_e_step(
            observations=np.asarray([[offset - 1.0], [offset + 2.0]]),
            projection_matrices=np.ones((2, 1, 1)),
            measurement_covariances=np.zeros((2, 1, 1)),
            weights=weights,
            means=means,
            covariances=covariances,
        ),
    ]
    empty_step = general_e_step(
        observations=np.empty((2, 0)),
        projection_matrices=np.empty((2, 0, 1)),
        measurement_covariances=np.empty((2, 0, 0)),
        weights=weights,
        means=means,
        covariances=covariances,
    )
    all_steps = [empty_step, *informative_steps]
    all_weights = [np.asarray([1e200, 1e200]), np.ones(2), np.ones(2)]

    parameters, statistics = general_grouped_m_step(all_steps, all_weights)
    raw, informative_weight, objective = general_grouped_objective(
        all_steps, all_weights
    )

    np.testing.assert_array_equal(parameters.weights, [1.0])
    np.testing.assert_array_equal(parameters.means, [[offset]])
    np.testing.assert_allclose(parameters.covariances, [[[2.5]]], atol=0.0)
    np.testing.assert_allclose(statistics.centered_covariance, [[[2.5]]], atol=0.0)
    assert statistics.informative_weight == 4.0
    assert informative_weight == 4.0
    assert statistics.weighted_log_likelihood == raw
    assert statistics.objective == objective
    # At this offset the raw-moment subtraction no longer contains the small
    # within-group spread; the reference answer comes from the centered pass.
    naive = (
        statistics.second_moment / statistics.mass[:, None, None]
        - statistics.component_mean[:, :, None]
        * statistics.component_mean[:, None, :]
    )
    assert not np.allclose(naive, parameters.covariances, rtol=0.0, atol=1.0)
