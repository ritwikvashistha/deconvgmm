"""Self-checks for the independent NumPy identity-XD oracle."""

from __future__ import annotations

import numpy as np

from tests.reference.identity_xd import (
    identity_e_step,
    identity_em_step,
    marginalized_posterior,
)


def test_one_component_matches_scalar_analytic_posterior():
    e_step = identity_e_step(
        observations=[[1.0]],
        measurement_covariances=[[[0.5]]],
        weights=[1.0],
        means=[[0.0]],
        covariances=[[[2.0]]],
    )
    posterior_mean, posterior_covariance = marginalized_posterior(e_step)

    np.testing.assert_allclose(e_step.responsibilities, [[1.0]], atol=0.0)
    np.testing.assert_allclose(posterior_mean, [[0.8]], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        posterior_covariance, [[[0.4]]], rtol=1e-14, atol=1e-14
    )


def test_tail_responsibilities_remain_normalized():
    dimension = 32
    e_step = identity_e_step(
        observations=np.full((1, dimension), 1_000.0),
        measurement_covariances=np.zeros((1, dimension, dimension)),
        weights=[0.2, 0.3, 0.5],
        means=np.stack(
            [
                np.full(dimension, -2.0),
                np.zeros(dimension),
                np.full(dimension, 2.0),
            ]
        ),
        covariances=np.stack(
            [
                np.eye(dimension) * 0.5,
                np.eye(dimension),
                np.eye(dimension) * 2.0,
            ]
        ),
    )

    assert np.all(np.isfinite(e_step.component_log_density))
    assert np.all(np.isfinite(e_step.responsibilities))
    np.testing.assert_allclose(e_step.responsibilities.sum(axis=1), 1.0)


def test_zero_measurement_error_recovers_observation_exactly():
    observations = np.array([[-0.5, 1.25], [0.2, -0.7]])
    e_step = identity_e_step(
        observations=observations,
        measurement_covariances=np.zeros((2, 2, 2)),
        weights=[0.4, 0.6],
        means=[[-1.0, 0.5], [1.0, -0.5]],
        covariances=[[[1.2, 0.2], [0.2, 0.8]], [[0.7, -0.1], [-0.1, 1.1]]],
    )

    expected_means = np.broadcast_to(observations[:, None, :], (2, 2, 2))
    np.testing.assert_allclose(e_step.conditional_mean, expected_means, atol=1e-14)
    np.testing.assert_allclose(e_step.conditional_covariance, 0.0, atol=1e-14)


def test_reference_em_step_returns_normalized_valid_parameters():
    parameters, _, statistics = identity_em_step(
        observations=[[-1.2, 0.1], [-0.8, -0.2], [0.9, 1.0], [1.3, 0.7]],
        measurement_covariances=np.broadcast_to(np.eye(2) * 0.1, (4, 2, 2)),
        weights=[0.5, 0.5],
        means=[[-1.0, 0.0], [1.0, 0.8]],
        covariances=[[[0.5, 0.05], [0.05, 0.4]], [[0.6, -0.03], [-0.03, 0.5]]],
        covariance_ridge=1e-6,
    )

    np.testing.assert_allclose(parameters.weights.sum(), 1.0)
    assert np.all(statistics.mass > 0.0)
    assert np.all(np.linalg.eigvalsh(parameters.covariances) > 0.0)

