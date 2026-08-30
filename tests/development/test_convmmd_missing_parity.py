"""Masked convMMD JAX-vs-oracle parity: the correctness gate before comparison.

Binds the stored ``convmmd_missing_001`` fixture and requires the pure-JAX grouped
masked path (projected analytic loss, projected denoiser/posterior components, and
the deterministic mask grouping in :mod:`development.convmmd_grouped`) to agree with
the independent NumPy oracle at float64 near machine epsilon (``rtol 5e-8``,
``atol 5e-10``) and within a declared float32 profile. Also binds the
fully-observed reduction to the base (§4/§7) operators. These gates MUST pass
before the masked path is compared to XDGMM or integrated.
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
    convmmd_loss_analytic,
    denoise,
)
from development.convmmd_grouped import (
    convmmd_denoise_masked,
    convmmd_loss_analytic_masked,
    convmmd_loss_mc_masked,
    convmmd_posterior_components_masked,
    group_masked_inputs,
    grouped_analytic_loss,
    grouped_denoise,
    grouped_posterior_components,
    median_bandwidths_masked,
)
from development.general_validation import PerItemFullNoise
from tests.reference import convmmd as oracle


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_missing_001.npz"
ARCHIVE_SHA256 = "d70ecc1ca0a1901fccca2bcaae4420516926aa81f1575a5a42f5b455aceb427f"

F64_RTOL = 5e-8
F64_ATOL = 5e-10
F32_RTOL = 1e-4
F32_ATOL = 1e-5

DTYPE_CASES = [
    (jnp.float64, F64_RTOL, F64_ATOL),
    (jnp.float32, F32_RTOL, F32_ATOL),
]


def _fixture() -> dict[str, np.ndarray]:
    with np.load(ARCHIVE) as data:
        return {name: data[name] for name in data.files}


def _params(fixture, dtype) -> ConvMMDParams:
    return ConvMMDParams(
        weights=jnp.asarray(fixture["eval_weights"], dtype=dtype),
        means=jnp.asarray(fixture["eval_means"], dtype=dtype),
        covariances=jnp.asarray(fixture["eval_covariances"], dtype=dtype),
    )


def _grouped(fixture, dtype):
    return group_masked_inputs(
        _params(fixture, dtype),
        jnp.asarray(fixture["observations"], dtype=dtype),
        jnp.asarray(fixture["observed_mask"]),
        noise=PerItemFullNoise(
            jnp.asarray(fixture["measurement_covariances"], dtype=dtype)
        ),
        dtype=dtype,
    )


def test_fixture_digest_is_pinned():
    assert ARCHIVE.is_file()
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    assert digest == ARCHIVE_SHA256


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_masked_analytic_loss_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    loss = grouped_analytic_loss(
        params, grouped, jnp.asarray(fixture["bandwidths"], dtype=dtype)
    )
    assert loss.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(loss), float(fixture["oracle_loss"]), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_masked_per_scale_loss_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    per_scale = fixture["oracle_per_scale_loss"]
    for index, gamma in enumerate(fixture["bandwidths"]):
        single = grouped_analytic_loss(
            params, grouped, jnp.asarray([gamma], dtype=dtype)
        )
        np.testing.assert_allclose(
            np.asarray(single), per_scale[index], rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_masked_denoiser_matches_oracle(dtype, rtol, atol):
    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    components = grouped_posterior_components(params, grouped)
    posterior_mean = grouped_denoise(params, grouped)

    assert posterior_mean.dtype == dtype
    assert posterior_mean.shape == fixture["observations"].shape
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


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_fully_observed_reduces_to_base_operators(dtype, rtol, atol):
    fixture = _fixture()
    params = _params(fixture, dtype)
    observations = jnp.asarray(fixture["observations"], dtype=dtype)
    noise = jnp.asarray(fixture["measurement_covariances"], dtype=dtype)
    bandwidths = jnp.asarray(fixture["bandwidths"], dtype=dtype)
    full_mask = jnp.ones(observations.shape, dtype=bool)

    grouped = group_masked_inputs(
        params,
        observations,
        full_mask,
        noise=PerItemFullNoise(noise),
        dtype=dtype,
    )
    masked_loss = grouped_analytic_loss(params, grouped, bandwidths)
    base_loss = convmmd_loss_analytic(params, observations, noise, bandwidths)
    np.testing.assert_allclose(
        np.asarray(masked_loss), np.asarray(base_loss), rtol=rtol, atol=atol
    )

    # And both equal the independent full-data oracle.
    oracle_full = oracle.convmmd_loss(
        fixture["eval_weights"],
        fixture["eval_means"],
        fixture["eval_covariances"],
        fixture["observations"],
        fixture["measurement_covariances"],
        fixture["bandwidths"],
    ).loss
    np.testing.assert_allclose(
        np.asarray(masked_loss), oracle_full, rtol=rtol, atol=atol
    )

    masked_mean = grouped_denoise(params, grouped)
    base_mean = denoise(params, observations, noise)
    np.testing.assert_allclose(
        np.asarray(masked_mean), np.asarray(base_mean), rtol=rtol, atol=atol
    )


def test_one_shot_masked_operations_match_oracle():
    """The public one-shot ``*_masked`` operations (group + evaluate) agree with
    the independent oracle for loss/denoise/posterior, and the MC one-shot is a
    finite estimate near the analytic value."""

    fixture = _fixture()
    weights = fixture["eval_weights"]
    means = fixture["eval_means"]
    covariances = fixture["eval_covariances"]
    params = _params(fixture, jnp.float64)
    observations = fixture["observations"]
    mask = fixture["observed_mask"]
    noise = fixture["measurement_covariances"]
    bandwidths = fixture["bandwidths"]
    noise_spec = PerItemFullNoise(jnp.asarray(noise))

    loss = convmmd_loss_analytic_masked(
        params, jnp.asarray(observations), jnp.asarray(mask),
        noise=noise_spec, bandwidths=jnp.asarray(bandwidths), dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(loss), float(fixture["oracle_loss"]), rtol=F64_RTOL, atol=F64_ATOL
    )

    posterior_mean = convmmd_denoise_masked(
        params, jnp.asarray(observations), jnp.asarray(mask),
        noise=noise_spec, dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(posterior_mean), fixture["oracle_posterior_mean"],
        rtol=F64_RTOL, atol=F64_ATOL,
    )

    components = convmmd_posterior_components_masked(
        params, jnp.asarray(observations), jnp.asarray(mask),
        noise=noise_spec, dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(components.responsibilities), fixture["oracle_responsibilities"],
        rtol=F64_RTOL, atol=F64_ATOL,
    )

    mc = convmmd_loss_mc_masked(
        params, jnp.asarray(observations), jnp.asarray(mask),
        noise=noise_spec, bandwidths=jnp.asarray(bandwidths),
        key=jax.random.PRNGKey(0), num_samples=4000, dtype=jnp.float64,
    )
    assert np.isfinite(float(mc))
    assert abs(float(mc) - float(fixture["oracle_loss"])) < 0.05


def test_masked_bandwidths_match_oracle_and_fixture():
    fixture = _fixture()
    observations = jnp.asarray(fixture["observations"])
    mask = jnp.asarray(fixture["observed_mask"])
    development = median_bandwidths_masked(observations, mask)
    reference = oracle.median_bandwidths_masked(
        fixture["observations"], fixture["observed_mask"]
    )
    np.testing.assert_allclose(
        np.asarray(development), reference, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(development), fixture["bandwidths"], rtol=1e-13, atol=1e-13
    )


def test_masked_bandwidths_reduce_to_full_on_all_observed():
    fixture = _fixture()
    observations = fixture["observations"]
    full_mask = jnp.ones(observations.shape, dtype=bool)
    development = median_bandwidths_masked(jnp.asarray(observations), full_mask)
    differences = observations[:, None, :] - observations[None, :, :]
    distances = np.sqrt(np.sum(differences * differences, axis=-1))
    upper = distances[np.triu_indices(observations.shape[0], k=1)]
    expected = float(np.median(upper)) * np.logspace(-2.0, 2.0, 9)
    np.testing.assert_allclose(np.asarray(development), expected, rtol=1e-13, atol=1e-13)
