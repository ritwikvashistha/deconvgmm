"""Phase 2 fit-result provenance metadata gates."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.fit_control import fit_converged, fit_fixed_steps
from development.identity_xd import Params
from development.metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    ResultMetadata,
    metadata_from_json,
    metadata_to_json,
)


def _stable_problem() -> tuple[Params, jax.Array, jax.Array]:
    params = Params(
        weights=jnp.asarray([1.0], dtype=jnp.float64),
        means=jnp.asarray([[0.0]], dtype=jnp.float64),
        covariances=jnp.asarray([[[1.0]]], dtype=jnp.float64),
    )
    observations = jnp.asarray([[-0.8], [-0.2], [0.3], [1.1]], dtype=jnp.float64)
    noise = jnp.asarray([[[0.1]], [[0.2]], [[0.15]], [[0.05]]], dtype=jnp.float64)
    return params, observations, noise


@pytest.mark.parametrize("mode", ["converged", "fixed"])
def test_xd_ip_meta_001_fit_results_record_versioned_contract_metadata(mode):
    """XD-IP-META-001: every host fit result identifies its contract."""

    params, observations, noise = _stable_problem()
    if mode == "converged":
        result = fit_converged(
            params,
            observations,
            noise,
            max_iter=2,
            tol=1e6,
            decrease_tol=1e-10,
        )
    else:
        result = fit_fixed_steps(
            params, observations, noise, n_steps=1
        )

    assert not bool(np.asarray(result.numerical_failure))
    assert not bool(np.asarray(result.collapsed))
    assert isinstance(result.metadata, ResultMetadata)
    assert result.metadata.contract_id == "xdgmm-jax.identity-xd"
    assert result.metadata.contract_version == "0.1.0-draft.1"
    assert result.metadata.contract_id == CONTRACT_ID
    assert result.metadata.contract_version == CONTRACT_VERSION


def test_xd_ip_meta_001_metadata_json_round_trip_is_exact_and_versioned():
    """XD-IP-META-001: the minimal serialized identity is stable JSON."""

    metadata = ResultMetadata(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )
    encoded = metadata_to_json(metadata)
    decoded = metadata_from_json(encoded)

    assert isinstance(encoded, str)
    assert decoded == metadata
    assert metadata_to_json(decoded) == encoded


@pytest.mark.parametrize(
    "payload",
    [
        "{}",
        '{"contract_id":"wrong","contract_version":"0.1.0-draft.1"}',
        '{"contract_id":"xdgmm-jax.identity-xd","contract_version":"wrong"}',
        '{"contract_id":"xdgmm-jax.identity-xd","contract_version":"0.1.0-draft.1","extra":1}',
        "not json",
    ],
)
def test_xd_ip_meta_001_rejects_unknown_or_malformed_metadata(payload):
    """Metadata readers fail closed instead of guessing schema compatibility."""

    with pytest.raises((TypeError, ValueError)):
        metadata_from_json(payload)
