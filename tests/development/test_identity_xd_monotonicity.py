"""Stored repeated-EM fixture for the Phase 1 monotonicity contract row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import FitStatus, fit_fixed_steps
from development.identity_xd import Params
from tests.reference.identity_xd import identity_e_step as reference_e_step


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_PATH = FIXTURE_DIRECTORY / "identity_em_002.npz"
METADATA_PATH = FIXTURE_DIRECTORY / "identity_em_002.metadata.json"


DTYPE_CASES = (
    pytest.param(jnp.float64, 1e-10, 5e-10, 5e-10, 5e-13, id="float64"),
    pytest.param(jnp.float32, 2e-5, 2e-4, 2e-5, 2e-5, id="float32"),
)


def _load_fixture(dtype):
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        observations = jnp.asarray(stored["observations"], dtype=dtype)
        noise = jnp.asarray(stored["measurement_covariances"], dtype=dtype)
        parameters = Params(
            weights=jnp.asarray(stored["initial_weights"], dtype=dtype),
            means=jnp.asarray(stored["initial_means"], dtype=dtype),
            covariances=jnp.asarray(
                stored["initial_covariances"], dtype=dtype
            ),
        )
    return parameters, observations, noise


def test_xd_ip_em_002_fixture_archive_matches_recorded_generator_digest():
    """The generated binary fixture remains immutable and attributable."""

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == metadata["archive_sha256"]
    assert metadata["bit_generator"] == "PCG64"
    assert metadata["seed"] == 20260825
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        assert sorted(stored.files) == sorted(metadata["stored_arrays"])
        assert stored["observations"].shape == (128, 3)
        assert stored["measurement_covariances"].shape == (128, 3, 3)


@pytest.mark.parametrize(
    "dtype,decrease_bound,log_rtol,log_atol,weight_atol", DTYPE_CASES
)
def test_xd_ip_em_002_fifteen_updates_are_finite_and_monotone(
    dtype, decrease_bound, log_rtol, log_atol, weight_atol
):
    """XD-IP-EM-002: exact repeated EM satisfies its objective guarantees."""

    initial, observations, noise = _load_fixture(dtype)
    result = fit_fixed_steps(
        initial,
        observations,
        noise,
        n_steps=15,
        factor_jitter=0.0,
        covariance_ridge=0.0,
    )

    assert int(np.asarray(result.status)) == int(FitStatus.FIXED_STEPS_COMPLETE)
    assert int(np.asarray(result.n_iter)) == 15
    assert result.history.shape == (16,)
    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert np.all(np.isfinite(np.asarray(result.history)))

    objective_history = np.asarray(result.history, dtype=np.float64)
    previous_scale = np.maximum(1.0, np.abs(objective_history[:-1]))
    increments = np.diff(objective_history)
    assert np.all(increments >= -decrease_bound * previous_scale)

    for leaf in jax.tree_util.tree_leaves(result.parameters):
        assert np.all(np.isfinite(np.asarray(leaf)))
    weights = np.asarray(result.parameters.weights)
    assert np.all(weights > 0.0)
    np.testing.assert_allclose(
        weights.sum(), 1.0, rtol=0.0, atol=weight_atol
    )
    returned_covariances = np.asarray(result.parameters.covariances)
    np.testing.assert_allclose(
        returned_covariances,
        np.swapaxes(returned_covariances, -1, -2),
        rtol=0.0,
        atol=2e-6 if dtype == jnp.float32 else 2e-13,
    )
    assert np.all(np.isfinite(np.linalg.cholesky(returned_covariances)))

    independent_final = reference_e_step(
        np.asarray(observations, dtype=np.float64),
        np.asarray(noise, dtype=np.float64),
        np.asarray(result.parameters.weights, dtype=np.float64),
        np.asarray(result.parameters.means, dtype=np.float64),
        np.asarray(result.parameters.covariances, dtype=np.float64),
    )
    independent_objective = np.mean(independent_final.score_samples)
    np.testing.assert_allclose(
        np.asarray(result.objective),
        independent_objective,
        rtol=log_rtol,
        atol=log_atol,
    )
    np.testing.assert_allclose(
        np.asarray(result.history[-1]),
        np.asarray(result.objective),
        rtol=0.0,
        atol=0.0,
    )

