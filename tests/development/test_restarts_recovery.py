"""Phase-4 discriminator cases for ordered restart selection.

The stored recovery workloads remain unchanged.  These tests derive one valid
adverse symmetric candidate from the stored generating mixture moments and pair
it with the existing perturbed candidate.  The evidence is conditioned on that
two-candidate set and makes no global-optimum claim.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitStatus, fit_converged
from development.general_fit_control import fit_converged_grouped
from development.general_validation import (
    PerItemFullNoise,
    PerItemProjection,
    group_masked_general_fit_inputs,
)
from development.identity_xd import Params
from tests.reference.general_xd import (
    general_e_step,
    general_grouped_objective,
)
from tests.reference.identity_xd import identity_e_step
from tests.reference.recovery import mixture_moments, recovery_metrics


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_PATH = FIXTURE_DIRECTORY / "phase4_recovery_001.npz"
METADATA_PATH = FIXTURE_DIRECTORY / "phase4_recovery_001.metadata.json"


@pytest.fixture(scope="module")
def recovery_fixture():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    return arrays, metadata


@pytest.fixture
def restarts():
    return importlib.import_module("development.restarts")


def _stored_initial(arrays, prefix: str) -> Params:
    return Params(
        jnp.asarray(arrays[f"{prefix}_initial_weights"]),
        jnp.asarray(arrays[f"{prefix}_initial_means"]),
        jnp.asarray(arrays[f"{prefix}_initial_covariances"]),
    )


def _symmetric_adverse_candidate(arrays, prefix: str) -> Params:
    weights = arrays[f"{prefix}_true_weights"]
    means = arrays[f"{prefix}_true_means"]
    covariances = arrays[f"{prefix}_true_covariances"]
    mixture_mean, mixture_covariance = mixture_moments(
        weights, means, covariances
    )
    n_components = len(weights)
    return Params(
        jnp.asarray(
            np.full(n_components, 1.0 / n_components, dtype=np.float64)
        ),
        jnp.asarray(np.repeat(mixture_mean[None, :], n_components, axis=0)),
        jnp.asarray(
            np.repeat(mixture_covariance[None, :, :], n_components, axis=0)
        ),
    )


def _metrics(arrays, prefix: str, parameters: Params):
    return recovery_metrics(
        arrays[f"{prefix}_true_weights"],
        arrays[f"{prefix}_true_means"],
        arrays[f"{prefix}_true_covariances"],
        np.asarray(parameters.weights),
        np.asarray(parameters.means),
        np.asarray(parameters.covariances),
        arrays[f"{prefix}_latent_holdout"],
    )


def _candidate_at(candidates, index: int) -> Params:
    return Params(
        candidates.weights[index],
        candidates.means[index],
        candidates.covariances[index],
    )


def _identity_objective(arrays, parameters: Params) -> float:
    e_step = identity_e_step(
        arrays["identity_observations"],
        arrays["identity_measurement_covariances"],
        np.asarray(parameters.weights),
        np.asarray(parameters.means),
        np.asarray(parameters.covariances),
    )
    return float(np.mean(e_step.score_samples))


def _grouped_objective(fit, parameters: Params) -> float:
    e_steps = []
    sample_weights = []
    for group in fit.grouped.groups:
        e_steps.append(
            general_e_step(
                np.asarray(group.observations),
                np.asarray(group.projection_matrices),
                np.asarray(group.measurement_covariances),
                np.asarray(parameters.weights),
                np.asarray(parameters.means),
                np.asarray(parameters.covariances),
            )
        )
        sample_weights.append(np.asarray(group.sample_weight))
    return float(general_grouped_objective(e_steps, sample_weights)[2])


def _assert_selected_recovers_and_adverse_does_not(
    arrays,
    metadata,
    prefix: str,
    selected: Params,
    adverse: Params,
) -> None:
    thresholds = metadata["acceptance_thresholds"][prefix]
    selected_metrics = _metrics(arrays, prefix, selected)
    adverse_metrics = _metrics(arrays, prefix, adverse)

    assert (
        selected_metrics.max_absolute_weight_error
        <= thresholds["max_absolute_weight_error"]
    )
    assert (
        selected_metrics.max_mean_mahalanobis_error
        <= thresholds["max_mean_mahalanobis_error"]
    )
    assert (
        selected_metrics.max_relative_covariance_frobenius_error
        <= thresholds["max_relative_covariance_frobenius_error"]
    )
    assert (
        selected_metrics.latent_log_density_rms_error
        <= thresholds["latent_log_density_rms_error"]
    )
    assert (
        abs(selected_metrics.latent_log_density_mean_gap)
        <= thresholds["absolute_latent_log_density_mean_gap"]
    )
    assert (
        selected_metrics.mixture_mean_l2_error
        <= thresholds["mixture_mean_l2_error"]
    )
    assert (
        selected_metrics.mixture_covariance_relative_frobenius_error
        <= thresholds["mixture_covariance_relative_frobenius_error"]
    )
    assert (
        selected_metrics.alignment.total_symmetric_gaussian_kl
        <= thresholds["total_symmetric_gaussian_kl"]
    )
    assert (
        adverse_metrics.max_mean_mahalanobis_error
        > thresholds["max_mean_mahalanobis_error"]
    )
    assert (
        adverse_metrics.latent_log_density_rms_error
        > thresholds["latent_log_density_rms_error"]
    )


@pytest.mark.slow
def test_identity_restart_selects_recovering_candidate_over_symmetric_basin(
    restarts, recovery_fixture
):
    arrays, metadata = recovery_fixture
    controls = metadata["fit_controls"]
    adverse_initial = _symmetric_adverse_candidate(arrays, "identity")
    recovery_initial = _stored_initial(arrays, "identity")
    candidates = restarts.user_supplied_restart_candidates(
        [adverse_initial, recovery_initial], dtype=jnp.float64
    )
    adverse_canonical = _candidate_at(candidates, 0)

    result = restarts.fit_converged_restarts(
        candidates,
        jnp.asarray(arrays["identity_observations"]),
        jnp.asarray(arrays["identity_measurement_covariances"]),
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
        factor_jitter=controls["factor_jitter"],
        covariance_ridge=controls["covariance_ridge"],
    )
    adverse = fit_converged(
        adverse_canonical,
        jnp.asarray(arrays["identity_observations"]),
        jnp.asarray(arrays["identity_measurement_covariances"]),
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
        factor_jitter=controls["factor_jitter"],
        covariance_ridge=controls["covariance_ridge"],
    )

    assert bool(np.asarray(result.selection.success))
    assert int(np.asarray(result.selection.selected_restart)) == 1
    assert int(np.asarray(result.selected_result.status)) == int(
        FitStatus.CONVERGED
    )
    assert int(np.asarray(adverse.status)) in (
        int(FitStatus.CONVERGED),
        int(FitStatus.MAX_ITER),
    )
    assert float(np.asarray(result.diagnostics.objective[1])) > float(
        np.asarray(result.diagnostics.objective[0])
    )
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.objective),
        np.asarray(
            [
                _identity_objective(arrays, adverse.parameters),
                _identity_objective(arrays, result.selected_result.parameters),
            ]
        ),
        rtol=5e-10,
        atol=5e-10,
    )
    _assert_selected_recovers_and_adverse_does_not(
        arrays,
        metadata,
        "identity",
        result.selected_result.parameters,
        adverse.parameters,
    )


@pytest.mark.slow
def test_grouped_restart_selects_recovering_candidate_over_symmetric_basin(
    restarts, recovery_fixture
):
    arrays, metadata = recovery_fixture
    controls = metadata["fit_controls"]
    adverse_initial = _symmetric_adverse_candidate(arrays, "general")
    recovery_initial = _stored_initial(arrays, "general")
    candidates = restarts.user_supplied_restart_candidates(
        [adverse_initial, recovery_initial], dtype=jnp.float64
    )
    adverse_canonical = _candidate_at(candidates, 0)
    fit = group_masked_general_fit_inputs(
        adverse_canonical,
        arrays["general_observations"],
        arrays["general_observed_mask"],
        projection=PerItemProjection(arrays["general_projection_matrices"]),
        noise=PerItemFullNoise(arrays["general_measurement_covariances"]),
        sample_weight=arrays["general_sample_weight"],
        factor_jitter=controls["factor_jitter"],
        covariance_ridge=controls["covariance_ridge"],
        dtype=jnp.float64,
    )

    result = restarts.fit_converged_grouped_restarts(
        fit,
        candidates=candidates,
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
    )
    adverse = fit_converged_grouped(
        fit,
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
    )

    assert bool(np.asarray(result.selection.success))
    assert int(np.asarray(result.selection.selected_restart)) == 1
    assert int(np.asarray(result.selected_result.status)) == int(
        FitStatus.CONVERGED
    )
    assert int(np.asarray(adverse.status)) in (
        int(FitStatus.CONVERGED),
        int(FitStatus.MAX_ITER),
    )
    assert float(np.asarray(result.diagnostics.objective[1])) > float(
        np.asarray(result.diagnostics.objective[0])
    )
    np.testing.assert_allclose(
        np.asarray(result.diagnostics.objective),
        np.asarray(
            [
                _grouped_objective(fit, adverse.parameters),
                _grouped_objective(fit, result.selected_result.parameters),
            ]
        ),
        rtol=8e-10,
        atol=8e-10,
    )
    _assert_selected_recovers_and_adverse_does_not(
        arrays,
        metadata,
        "general",
        result.selected_result.parameters,
        adverse.parameters,
    )
