"""CPU-only statistical recovery after complete identity and general fits.

These Phase 4 tests are deliberately separate from one-step parity.  They do
not call either NumPy EM oracle and do not compare an implementation trajectory
to itself.  Instead, independently generated noisy observations are fit from a
fixed perturbed initialization, component labels are aligned only after fitting,
and the returned latent mixture is compared with generating truth and a stored
independent latent holdout sample.

The evidence is basin-conditioned: it establishes recovery for these two fixed
initializations, not global optimization, arbitrary-start robustness, or a
multiple-restart policy.  Formal capability-matrix rows remain Pending.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys

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
from scripts.deterministic_npz import npy_bytes
from tests.reference.recovery import (
    align_components,
    mixture_log_density,
    recovery_metrics,
)


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = FIXTURE_DIRECTORY / "phase4_recovery_001.npz"
METADATA_PATH = FIXTURE_DIRECTORY / "phase4_recovery_001.metadata.json"
GENERATOR_PATH = PROJECT_ROOT / "scripts" / "generate_phase4_recovery_fixture.py"
RECOVERY_HELPER_PATH = PROJECT_ROOT / "tests" / "reference" / "recovery.py"
DETERMINISTIC_ARCHIVE_HELPER_PATH = (
    PROJECT_ROOT / "scripts" / "deterministic_npz.py"
)


@pytest.fixture(scope="module")
def recovery_fixture() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        arrays = {name: stored[name].copy() for name in stored.files}
    return arrays, metadata


def _params(arrays: dict[str, np.ndarray], prefix: str) -> Params:
    return Params(
        weights=jnp.asarray(arrays[f"{prefix}_initial_weights"]),
        means=jnp.asarray(arrays[f"{prefix}_initial_means"]),
        covariances=jnp.asarray(arrays[f"{prefix}_initial_covariances"]),
    )


def _metric_arguments(
    arrays: dict[str, np.ndarray], prefix: str
) -> tuple[np.ndarray, ...]:
    return (
        arrays[f"{prefix}_true_weights"],
        arrays[f"{prefix}_true_means"],
        arrays[f"{prefix}_true_covariances"],
    )


def _assert_successful_complete_fit(result, *, max_iter: int) -> None:
    assert int(np.asarray(result.status)) == int(FitStatus.CONVERGED)
    assert bool(np.asarray(result.converged))
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert 1 <= int(np.asarray(result.n_iter)) < max_iter
    assert result.history.shape == (int(np.asarray(result.n_iter)) + 1,)
    assert np.all(np.isfinite(np.asarray(result.history)))
    np.testing.assert_array_equal(result.history[-1], result.objective)
    weights = np.asarray(result.parameters.weights)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0.0, atol=5e-13)
    for covariance in np.asarray(result.parameters.covariances):
        np.testing.assert_allclose(
            covariance, covariance.T, rtol=0.0, atol=2e-13
        )
        np.linalg.cholesky(covariance)


def _assert_recovery_metrics(
    fitted_metrics,
    initial_metrics,
    thresholds: dict[str, float],
) -> None:
    """Apply the recorded absolute and initialization-discrimination bounds."""

    assert (
        fitted_metrics.max_absolute_weight_error
        <= thresholds["max_absolute_weight_error"]
    )
    assert (
        fitted_metrics.max_mean_mahalanobis_error
        <= thresholds["max_mean_mahalanobis_error"]
    )
    assert (
        fitted_metrics.max_relative_covariance_frobenius_error
        <= thresholds["max_relative_covariance_frobenius_error"]
    )
    assert (
        fitted_metrics.latent_log_density_rms_error
        <= thresholds["latent_log_density_rms_error"]
    )
    assert (
        abs(fitted_metrics.latent_log_density_mean_gap)
        <= thresholds["absolute_latent_log_density_mean_gap"]
    )
    assert (
        fitted_metrics.mixture_mean_l2_error
        <= thresholds["mixture_mean_l2_error"]
    )
    assert (
        fitted_metrics.mixture_covariance_relative_frobenius_error
        <= thresholds["mixture_covariance_relative_frobenius_error"]
    )
    assert (
        fitted_metrics.alignment.total_symmetric_gaussian_kl
        <= thresholds["total_symmetric_gaussian_kl"]
    )
    assert (
        fitted_metrics.latent_log_density_rms_error
        <= thresholds["maximum_density_rms_fraction_of_initial"]
        * initial_metrics.latent_log_density_rms_error
    )
    assert (
        fitted_metrics.alignment.total_symmetric_gaussian_kl
        <= thresholds["maximum_alignment_kl_fraction_of_initial"]
        * initial_metrics.alignment.total_symmetric_gaussian_kl
    )

    # Negative control: a no-op controller returning the supplied initialization
    # must not satisfy the final-fit envelope.
    assert (
        initial_metrics.max_mean_mahalanobis_error
        > thresholds["max_mean_mahalanobis_error"]
    )
    assert (
        initial_metrics.latent_log_density_rms_error
        > thresholds["latent_log_density_rms_error"]
    )


def test_phase4_recovery_fixture_is_immutable_auditable_and_pending(
    recovery_fixture,
):
    """The stored statistical evidence has complete deterministic custody."""

    arrays, metadata = recovery_fixture
    assert metadata["fixture_id"] == "xd-phase4-final-fit-recovery-001"
    assert metadata["fixture_version"] == 1
    assert metadata["formal_matrix_status"] == "Pending"
    assert metadata["seed"] == 20260825
    assert metadata["bit_generator"] == "PCG64"
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == metadata[
        "archive_sha256"
    ]
    assert hashlib.sha256(GENERATOR_PATH.read_bytes()).hexdigest() == metadata[
        "generator_sha256"
    ]
    assert metadata["evidence_source_sha256"] == {
        "scripts/deterministic_npz.py": hashlib.sha256(
            DETERMINISTIC_ARCHIVE_HELPER_PATH.read_bytes()
        ).hexdigest(),
        "tests/reference/recovery.py": hashlib.sha256(
            RECOVERY_HELPER_PATH.read_bytes()
        ).hexdigest(),
    }
    assert sorted(arrays) == sorted(metadata["array_schema"])
    for name, value in arrays.items():
        assert list(value.shape) == metadata["array_schema"][name]["shape"]
        assert str(value.dtype) == metadata["array_schema"][name]["dtype"]
        assert hashlib.sha256(npy_bytes(value)).hexdigest() == metadata[
            "npy_payload_sha256"
        ][name]
        assert not value.dtype.hasobject
        assert np.all(np.isfinite(value))

    assert metadata["fit_controls"] == {
        "covariance_ridge": 0.0,
        "decrease_tol": 1e-8,
        "factor_jitter": 0.0,
        "max_iter": 100,
        "tol": 1e-5,
    }
    assert metadata["identity"]["contract_id"] == "xdgmm-jax.identity-xd"
    assert metadata["general"]["contract_id"] == "xdgmm-jax.general-xd"
    assert metadata["identity"]["maximum_true_effective_condition"] < 1e4
    assert metadata["general"]["maximum_true_effective_condition"] < 1e4
    assert min(metadata["identity"]["component_counts"]) >= 500
    assert min(metadata["general"]["component_counts"]) >= 275
    assert min(metadata["general"]["weighted_component_mass"]) >= 275.0
    assert metadata["general"]["observed_dimensions"] == [2, 3]
    assert min(
        metadata["general"]["aggregate_projection_gram_eigenvalues"]
    ) > 0.5
    assert metadata["general"]["aggregate_projection_gram_condition"] < 2.0
    assert all(value > 0 for value in metadata["general"]["mask_counts"].values())
    assert "global-optimum recovery" in metadata["scientific_scope"][
        "not_claimed"
    ]
    assert "one-step or implementation parity" in metadata[
        "scientific_scope"
    ]["not_claimed"]


@pytest.mark.skipif(
    not (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason=(
        "byte-exact .npz regeneration is reproducible only on the macOS/arm64 "
        "host where the fixture was pinned; NumPy/BLAS float rounding differs "
        "across OS/ISA (e.g. Linux x86_64). Numerical recovery is still covered "
        "by the other Phase 4 tests on every platform."
    ),
)
def test_phase4_recovery_fixture_regenerates_byte_identically(tmp_path):
    """The recorded generator reproduces both evidence files from scratch.

    Byte-exact reproduction is host-locked to the macOS/arm64 pinning
    environment; see the skip marker above.
    """

    regenerated_archive = tmp_path / FIXTURE_PATH.name
    regenerated_metadata = tmp_path / METADATA_PATH.name
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR_PATH),
            str(regenerated_archive),
            "--metadata",
            str(regenerated_metadata),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert regenerated_archive.read_bytes() == FIXTURE_PATH.read_bytes()
    assert json.loads(regenerated_metadata.read_text(encoding="utf-8")) == (
        json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    )


def test_recovery_metrics_are_exactly_permutation_invariant():
    """A pure relabeling has zero recovery error and a known alignment."""

    weights = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    means = np.asarray(
        [[-3.0, 0.2], [0.1, 2.0], [2.8, -1.0]], dtype=np.float64
    )
    covariances = np.asarray(
        [
            [[0.7, 0.1], [0.1, 0.5]],
            [[0.9, -0.2], [-0.2, 0.8]],
            [[0.4, 0.05], [0.05, 1.1]],
        ],
        dtype=np.float64,
    )
    candidate_order = np.asarray([2, 0, 1])
    alignment = align_components(
        weights,
        means,
        covariances,
        weights[candidate_order],
        means[candidate_order],
        covariances[candidate_order],
    )
    assert alignment.permutation == (1, 2, 0)
    assert alignment.total_symmetric_gaussian_kl <= 2e-15
    np.testing.assert_array_equal(alignment.weights, weights)
    np.testing.assert_array_equal(alignment.means, means)
    np.testing.assert_array_equal(alignment.covariances, covariances)

    probes = np.asarray(
        [[-3.0, 0.0], [0.0, 2.0], [2.5, -0.5], [0.2, 0.1]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        mixture_log_density(probes, weights, means, covariances),
        mixture_log_density(
            probes,
            weights[candidate_order],
            means[candidate_order],
            covariances[candidate_order],
        ),
        rtol=0.0,
        atol=2e-15,
    )
    metrics = recovery_metrics(
        weights,
        means,
        covariances,
        weights[candidate_order],
        means[candidate_order],
        covariances[candidate_order],
        probes,
    )
    assert metrics.max_absolute_weight_error == 0.0
    assert metrics.max_mean_mahalanobis_error == 0.0
    assert metrics.max_relative_covariance_frobenius_error == 0.0
    assert metrics.latent_log_density_rms_error <= 2e-15
    assert abs(metrics.latent_log_density_mean_gap) <= 2e-15
    assert metrics.mixture_mean_l2_error <= 2e-15
    assert metrics.mixture_covariance_relative_frobenius_error <= 2e-15


@pytest.mark.slow
def test_identity_complete_fit_recovers_generating_latent_mixture(
    recovery_fixture,
):
    """A converged heteroscedastic identity fit recovers the latent density."""

    arrays, metadata = recovery_fixture
    controls = metadata["fit_controls"]
    initial = _params(arrays, "identity")
    result = fit_converged(
        initial,
        jnp.asarray(arrays["identity_observations"]),
        jnp.asarray(arrays["identity_measurement_covariances"]),
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
        factor_jitter=controls["factor_jitter"],
        covariance_ridge=controls["covariance_ridge"],
    )
    _assert_successful_complete_fit(result, max_iter=controls["max_iter"])

    truth = _metric_arguments(arrays, "identity")
    holdout = arrays["identity_latent_holdout"]
    initial_metrics = recovery_metrics(
        *truth,
        arrays["identity_initial_weights"],
        arrays["identity_initial_means"],
        arrays["identity_initial_covariances"],
        holdout,
    )
    fitted_metrics = recovery_metrics(
        *truth,
        np.asarray(result.parameters.weights),
        np.asarray(result.parameters.means),
        np.asarray(result.parameters.covariances),
        holdout,
    )

    _assert_recovery_metrics(
        fitted_metrics,
        initial_metrics,
        metadata["acceptance_thresholds"]["identity"],
    )


@pytest.mark.slow
def test_grouped_general_complete_fit_recovers_generating_latent_mixture(
    recovery_fixture,
):
    """Variable-M dense projections recover one identifiable latent mixture."""

    arrays, metadata = recovery_fixture
    controls = metadata["fit_controls"]
    initial = _params(arrays, "general")
    fit = group_masked_general_fit_inputs(
        initial,
        arrays["general_observations"],
        arrays["general_observed_mask"],
        projection=PerItemProjection(arrays["general_projection_matrices"]),
        noise=PerItemFullNoise(
            arrays["general_measurement_covariances"]
        ),
        sample_weight=arrays["general_sample_weight"],
        factor_jitter=controls["factor_jitter"],
        covariance_ridge=controls["covariance_ridge"],
        dtype=jnp.float64,
    )
    result = fit_converged_grouped(
        fit,
        max_iter=controls["max_iter"],
        tol=controls["tol"],
        decrease_tol=controls["decrease_tol"],
    )
    _assert_successful_complete_fit(result, max_iter=controls["max_iter"])

    truth = _metric_arguments(arrays, "general")
    holdout = arrays["general_latent_holdout"]
    initial_metrics = recovery_metrics(
        *truth,
        arrays["general_initial_weights"],
        arrays["general_initial_means"],
        arrays["general_initial_covariances"],
        holdout,
    )
    fitted_metrics = recovery_metrics(
        *truth,
        np.asarray(result.parameters.weights),
        np.asarray(result.parameters.means),
        np.asarray(result.parameters.covariances),
        holdout,
    )

    _assert_recovery_metrics(
        fitted_metrics,
        initial_metrics,
        metadata["acceptance_thresholds"]["general"],
    )
