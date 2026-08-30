"""Versioned host metadata for temporary XD fit results.

This is deliberately a small, strict schema.  Numerical arrays are not
serialized here; the record only identifies the mathematical contract used to
interpret a result and the supported initialization provenance.  It remains
outside the future public package namespace.
"""

from __future__ import annotations

import json
from typing import NamedTuple


CONTRACT_ID = "xdgmm-jax.identity-xd"
CONTRACT_VERSION = "0.1.0-draft.1"
GENERAL_CONTRACT_ID = "xdgmm-jax.general-xd"
GENERAL_CONTRACT_VERSION = "0.2.0-draft.1"


class ResultMetadata(NamedTuple):
    """Minimal identity and version of the result's numerical contract."""

    contract_id: str
    contract_version: str


class GeneralResultMetadata(NamedTuple):
    """Minimal identity and version of the general-projection contract."""

    contract_id: str
    contract_version: str


class InitializationProvenance(NamedTuple):
    """Closed initialization provenance retained by temporary host results."""

    kind: str


def current_result_metadata() -> ResultMetadata:
    """Return the immutable metadata attached to newly created fit results."""

    return ResultMetadata(
        contract_id=CONTRACT_ID,
        contract_version=CONTRACT_VERSION,
    )


def current_general_result_metadata() -> GeneralResultMetadata:
    """Return the metadata attached to grouped general-projection fits."""

    return GeneralResultMetadata(
        contract_id=GENERAL_CONTRACT_ID,
        contract_version=GENERAL_CONTRACT_VERSION,
    )


def user_supplied_initialization() -> InitializationProvenance:
    """Record the only initialization kind supported by this draft."""

    return InitializationProvenance(kind="user_supplied")


def metadata_to_json(metadata: ResultMetadata) -> str:
    """Serialize known result metadata to deterministic compact JSON."""

    if not isinstance(metadata, ResultMetadata):
        raise TypeError("metadata must be a ResultMetadata instance")
    if metadata != current_result_metadata():
        raise ValueError(
            "metadata contract ID/version is not supported by this build"
        )
    return json.dumps(
        {
            "contract_id": metadata.contract_id,
            "contract_version": metadata.contract_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def metadata_from_json(payload: str) -> ResultMetadata:
    """Read exactly the current metadata schema and fail closed on drift."""

    if not isinstance(payload, str):
        raise TypeError("metadata payload must be a JSON string")
    try:
        decoded = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("metadata payload is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata JSON must contain one object")
    expected_fields = {"contract_id", "contract_version"}
    if set(decoded) != expected_fields:
        raise ValueError(
            "metadata JSON must contain exactly contract_id and "
            "contract_version"
        )
    metadata = ResultMetadata(
        contract_id=decoded["contract_id"],
        contract_version=decoded["contract_version"],
    )
    if not all(isinstance(value, str) for value in metadata):
        raise ValueError("metadata contract fields must be strings")
    if metadata != current_result_metadata():
        raise ValueError(
            "metadata contract ID/version is not supported by this build"
        )
    return metadata


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "GENERAL_CONTRACT_ID",
    "GENERAL_CONTRACT_VERSION",
    "GeneralResultMetadata",
    "InitializationProvenance",
    "ResultMetadata",
    "current_general_result_metadata",
    "current_result_metadata",
    "metadata_from_json",
    "metadata_to_json",
    "user_supplied_initialization",
]
