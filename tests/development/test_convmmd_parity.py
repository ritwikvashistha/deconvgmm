"""convMMD JAX-vs-oracle parity: the correctness gate before any comparison.

Binds the stored ``convmmd_recovery_001`` fixture and requires the pure-JAX
analytic loss, denoiser, posterior components, and parameterization transform to
agree with the independent NumPy oracle at float64 near machine epsilon
(``rtol 5e-8``, ``atol 5e-10``) and within a declared float32 profile. These
gates MUST pass before convMMD is timed, compared to XDGMM, or integrated.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    convmmd_loss_analytic,
    denoise,
    posterior_components,
    to_canonical,
)
from tests.reference import convmmd as oracle


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_recovery_001.npz"
ARCHIVE_SHA256 = "bcc64b0d61da5e3454cec328788d18ab752709af983e661a726cb8c31186ff64"

# float64 parity culture reused from the identity/general parity work.
F64_RTOL = 5e-8
F64_ATOL = 5e-10
# Declared float32 profile (recorded before acceptance; see model contract §13).
F32_RTOL = 1e-4
F32_ATOL = 1e-5


def _fixture() -> dict[str, np.ndarray]:
    with np.load(ARCHIVE) as data:
        return {name: data[name] for name in data.files}


def _params(fixture, dtype) -> ConvMMDParams:
    return ConvMMDParams(
        weights=jnp.asarray(fixture["eval_weights"], dtype=dtype),
        means=jnp.asarray(fixture["eval_means"], dtype=dtype),
        covariances=jnp.asarray(fixture["eval_covariances"], dtype=dtype),
    )


def test_fixture_digest_is_pinned():
    assert ARCHIVE.is_file()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)],
)
def test_analytic_loss_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    loss = convmmd_loss_analytic(
        _params(fixture, dtype),
        jnp.asarray(fixture["observations"], dtype=dtype),
        jnp.asarray(fixture["measurement_covariances"], dtype=dtype),
        jnp.asarray(fixture["bandwidths"], dtype=dtype),
    )
    assert loss.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(loss), float(fixture["oracle_loss"]), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)],
)
def test_per_scale_loss_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    per_scale = fixture["oracle_per_scale_loss"]
    for index, gamma in enumerate(fixture["bandwidths"]):
        single = convmmd_loss_analytic(
            _params(fixture, dtype),
            jnp.asarray(fixture["observations"], dtype=dtype),
            jnp.asarray(fixture["measurement_covariances"], dtype=dtype),
            jnp.asarray([gamma], dtype=dtype),
        )
        np.testing.assert_allclose(
            np.asarray(single), per_scale[index], rtol=rtol, atol=atol
        )


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(jnp.float64, F64_RTOL, F64_ATOL), (jnp.float32, F32_RTOL, F32_ATOL)],
)
def test_denoiser_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    observations = jnp.asarray(fixture["observations"], dtype=dtype)
    noise = jnp.asarray(fixture["measurement_covariances"], dtype=dtype)
    components = posterior_components(_params(fixture, dtype), observations, noise)
    posterior_mean = denoise(_params(fixture, dtype), observations, noise)

    assert posterior_mean.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(components.responsibilities),
        fixture["oracle_responsibilities"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(components.component_means),
        fixture["oracle_component_means"],
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(posterior_mean),
        fixture["oracle_posterior_mean"],
        rtol=rtol,
        atol=atol,
    )


def test_to_canonical_matches_oracle_float64():
    fixture = _fixture()
    canonical = to_canonical(
        ConvMMDUnconstrained(
            jnp.asarray(fixture["eval_alphas"]),
            jnp.asarray(fixture["eval_means"]),
            jnp.asarray(fixture["eval_unconstrained_L"]),
        )
    )
    reference_weights, _, reference_covariances = oracle.unconstrained_to_canonical(
        fixture["eval_alphas"], fixture["eval_means"], fixture["eval_unconstrained_L"]
    )
    np.testing.assert_allclose(
        np.asarray(canonical.weights), reference_weights, rtol=F64_RTOL, atol=F64_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(canonical.covariances),
        reference_covariances,
        rtol=F64_RTOL,
        atol=F64_ATOL,
    )
