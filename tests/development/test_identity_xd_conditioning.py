"""Conditioning-domain evidence for the stored ``XD-IP-COV-001`` fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.identity_xd import Params, em_step, posterior_components
from development.validation import canonicalize_fit_inputs


FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures"
FIXTURE_PATH = FIXTURE_DIRECTORY / "identity_cov_001.npz"
METADATA_PATH = FIXTURE_DIRECTORY / "identity_cov_001.metadata.json"


PROFILES = (
    pytest.param("float64_in", jnp.float64, 1e8, True, id="float64-in"),
    pytest.param("float32_in", jnp.float32, 1e4, True, id="float32-in"),
    pytest.param("float64_out", jnp.float64, 1e12, False, id="float64-out"),
    pytest.param("float32_out", jnp.float32, 1e7, False, id="float32-out"),
)


def _load_profile(label: str, dtype) -> tuple[Params, jax.Array, jax.Array, np.ndarray]:
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        params = Params(
            weights=jnp.asarray(stored[f"{label}_weights"], dtype=dtype),
            means=jnp.asarray(stored[f"{label}_means"], dtype=dtype),
            covariances=jnp.asarray(
                stored[f"{label}_model_covariances"], dtype=dtype
            ),
        )
        observations = jnp.asarray(
            stored[f"{label}_observations"], dtype=dtype
        )
        noise = jnp.asarray(
            stored[f"{label}_measurement_covariances"], dtype=dtype
        )
        effective = np.asarray(
            stored[f"{label}_effective_covariance"], dtype=np.float64
        )
    return params, observations, noise, effective


def _assert_covariance_invariants(values, dtype, *, model: bool) -> None:
    matrices = np.asarray(values, dtype=np.float64)
    transposed = np.swapaxes(matrices, -1, -2)
    scale = np.maximum(
        1.0, np.linalg.norm(matrices, ord=2, axis=(-2, -1))
    )
    residual = (
        np.linalg.norm(matrices - transposed, ord=np.inf, axis=(-2, -1))
        / scale
    )
    symmetry_tolerance = 2e-13 if dtype == jnp.float64 else 2e-6
    assert np.all(residual <= symmetry_tolerance)
    if model:
        assert np.all(np.isfinite(np.linalg.cholesky(matrices)))
    else:
        minimum = np.linalg.eigvalsh(0.5 * (matrices + transposed))[..., 0]
        psd_tolerance = 2e-11 if dtype == jnp.float64 else 5e-5
        assert np.all(minimum >= -psd_tolerance * scale)


def test_xd_ip_cov_001_fixture_digest_and_literal_dct_are_stable():
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == metadata[
        "archive_sha256"
    ]
    with np.load(FIXTURE_PATH, allow_pickle=False) as stored:
        q = stored["dct_orthogonal"]
    np.testing.assert_allclose(q.T @ q, np.eye(5), rtol=0.0, atol=5e-16)


@pytest.mark.parametrize("label,dtype,conceptual_kappa,in_domain", PROFILES)
def test_xd_ip_cov_001_conditioning_domain_never_returns_silent_invalid_state(
    label, dtype, conceptual_kappa, in_domain
):
    params, observations, noise, effective = _load_profile(label, dtype)
    realized_condition = np.linalg.cond(effective)
    assert realized_condition == pytest.approx(conceptual_kappa, rel=0.03)

    validated = canonicalize_fit_inputs(
        params, observations, noise, dtype=dtype
    )
    e_step = posterior_components(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
    )
    update = em_step(
        validated.parameters,
        validated.observations,
        validated.measurement_covariances,
        factor_jitter=0.0,
        covariance_ridge=0.0,
    )

    if in_domain:
        assert not bool(np.asarray(e_step.numerical_failure))
        assert not bool(np.asarray(update.numerical_failure))
        assert not bool(np.asarray(update.collapsed))
        for leaf in jax.tree_util.tree_leaves(e_step):
            assert np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(update.parameters):
            assert np.all(np.isfinite(np.asarray(leaf)))
        _assert_covariance_invariants(
            e_step.conditional_covariance, dtype, model=False
        )
        _assert_covariance_invariants(
            update.parameters.covariances, dtype, model=True
        )
        return

    unsuccessful = bool(np.asarray(update.numerical_failure)) or bool(
        np.asarray(update.collapsed)
    )
    if bool(np.asarray(e_step.numerical_failure)) or unsuccessful:
        for actual, initial in zip(
            update.parameters, validated.parameters, strict=True
        ):
            assert np.all(np.isfinite(np.asarray(actual)))
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(initial))
        return

    for leaf in jax.tree_util.tree_leaves(e_step):
        assert np.all(np.isfinite(np.asarray(leaf)))
    for leaf in jax.tree_util.tree_leaves(update.parameters):
        assert np.all(np.isfinite(np.asarray(leaf)))
    _assert_covariance_invariants(
        e_step.conditional_covariance, dtype, model=False
    )
    _assert_covariance_invariants(
        update.parameters.covariances, dtype, model=True
    )

