"""Small deterministic writer for evidence-only NumPy archives.

The standard ``numpy.savez`` helpers do not promise byte-identical ZIP
metadata, and compressed ZIP payloads additionally depend on the zlib build.
This helper fixes the NPY version and ZIP metadata and deliberately stores
entries without compression.  It is used by reference-fixture generators, not
by the library runtime.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import numpy as np


ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
NPY_VERSION = (1, 0)


def npy_bytes(array: np.ndarray) -> bytes:
    """Return a canonical NPY 1.0 representation of a non-object array."""

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("deterministic evidence archives cannot store objects")
    buffer = io.BytesIO()
    np.lib.format.write_array(
        buffer,
        value,
        version=NPY_VERSION,
        allow_pickle=False,
    )
    return buffer.getvalue()


def write_deterministic_npz(
    destination: Path, arrays: dict[str, np.ndarray]
) -> dict[str, str]:
    """Write a ZIP_STORED NPZ and return each NPY payload's SHA-256."""

    if not arrays:
        raise ValueError("an evidence archive must contain at least one array")
    invalid_names = sorted(
        name
        for name in arrays
        if not name or name.endswith(".npy") or "/" in name or "\\" in name
    )
    if invalid_names:
        raise ValueError(f"invalid deterministic archive names: {invalid_names}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    payload_hashes: dict[str, str] = {}
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        archive.comment = b""
        for name in sorted(arrays):
            payload = npy_bytes(arrays[name])
            payload_hashes[name] = hashlib.sha256(payload).hexdigest()
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=ZIP_DATE_TIME,
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(
                info,
                payload,
                compress_type=zipfile.ZIP_STORED,
            )
    return payload_hashes
