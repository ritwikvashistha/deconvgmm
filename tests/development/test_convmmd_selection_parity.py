"""Selected convMMD JAX-vs-oracle parity: the correctness gate before comparison.

Binds the stored ``convmmd_selection_001`` fixture and requires the pure-JAX
Gaussian-window selection path (the §17.5 parameter transform, the grouped selected
analytic loss, the selection-aware denoiser/posterior components, and the effective
volume) to agree with the independent NumPy oracle at float64 near machine epsilon
(``rtol 5e-8``, ``atol 5e-10``) and within a declared float32 profile. Also binds the
``precision = 0`` (no-selection) reduction to the §16 masked operators. These gates
MUST pass before the selection path is compared to XDGMM/pygmmis or integrated.

Rows: ``CMMD-SEL-XFORM-001``, ``CMMD-SEL-GAUSS-001/002``, ``CMMD-SEL-REDUCE-001``,
``CMMD-SEL-DENOISE-001/002``, ``CMMD-SEL-Z-001`` (analytic part).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import ConvMMDParams
from development.convmmd_grouped import (
    convmmd_loss_analytic_masked,
    convmmd_denoise_masked,
    group_masked_inputs,
    grouped_analytic_loss,
)
from development.convmmd_selection import (
    GaussianSelection,
    convmmd_denoise_selected,
    convmmd_loss_analytic_selected,
    convmmd_posterior_components_selected,
    effective_volume_gaussian,
    grouped_analytic_selected_loss,
    grouped_denoise_selected,
    grouped_posterior_components_selected,
    select_gaussian_params,
)
from development.general_validation import PerItemFullNoise


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = FIXTURE_DIR / "convmmd_selection_001.npz"
ARCHIVE_SHA256 = "0a5205c15e720f9b86c31bb8f2b34d561a777d2310f81c81f25c94c808b0a27c"

F64_RTOL = 5e-8
F64_ATOL = 5e-10
# Declared float32 profile (recorded before acceptance): the selection transform
# adds two 4x4 matrix inversions ahead of the masked path, so the float32 error is
# a little looser than the base masked profile but comfortably within these bounds.
F32_RTOL = 2e-4
F32_ATOL = 2e-5

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


def _selection(fixture, dtype) -> GaussianSelection:
    return GaussianSelection(
        location=jnp.asarray(fixture["selection_location"], dtype=dtype),
        precision=jnp.asarray(fixture["selection_precision"], dtype=dtype),
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
def test_transform_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-XFORM-001: robust precision-native transform == textbook oracle."""

    fixture = _fixture()
    selected = select_gaussian_params(_params(fixture, dtype), _selection(fixture, dtype))
    assert selected.weights.dtype == dtype
    assert selected.means.dtype == dtype
    assert selected.covariances.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(selected.weights), fixture["oracle_selected_weights"],
        rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(selected.means), fixture["oracle_selected_means"],
        rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(selected.covariances), fixture["oracle_selected_covariances"],
        rtol=rtol, atol=atol,
    )
    # SPD transformed covariances.
    for cov in np.asarray(selected.covariances, dtype=np.float64):
        assert np.all(np.linalg.eigvalsh(cov) > 0.0)


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_selected_analytic_loss_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-GAUSS-001."""

    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    loss = grouped_analytic_selected_loss(
        params, grouped, jnp.asarray(fixture["bandwidths"], dtype=dtype),
        _selection(fixture, dtype),
    )
    assert loss.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(loss), float(fixture["oracle_selected_loss"]), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_selected_per_scale_loss_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-GAUSS-002."""

    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    selection = _selection(fixture, dtype)
    per_scale = fixture["oracle_selected_per_scale_loss"]
    for index, gamma in enumerate(fixture["bandwidths"]):
        single = grouped_analytic_selected_loss(
            params, grouped, jnp.asarray([gamma], dtype=dtype), selection
        )
        np.testing.assert_allclose(
            np.asarray(single), per_scale[index], rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_selected_denoiser_matches_oracle(dtype, rtol, atol):
    """CMMD-SEL-DENOISE-001/002."""

    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    selection = _selection(fixture, dtype)
    components = grouped_posterior_components_selected(params, grouped, selection)
    posterior_mean = grouped_denoise_selected(params, grouped, selection)

    assert posterior_mean.dtype == dtype
    assert posterior_mean.shape == fixture["observations"].shape
    np.testing.assert_allclose(
        np.asarray(components.responsibilities),
        fixture["oracle_selected_responsibilities"], rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(components.component_means),
        fixture["oracle_selected_component_means"], rtol=rtol, atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(posterior_mean),
        fixture["oracle_selected_posterior_mean"], rtol=rtol, atol=atol,
    )
    # Responsibilities sum to one over informative rows (M=0 rows take prior pi').
    np.testing.assert_allclose(
        np.asarray(components.responsibilities).sum(axis=1), 1.0,
        rtol=rtol, atol=max(atol, 1e-6),
    )


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_effective_volume_matches_oracle_and_reduces(dtype, rtol, atol):
    """CMMD-SEL-Z-001 (analytic): Z_theta == oracle, and Z_theta -> 1 as Omega -> 1."""

    fixture = _fixture()
    params = _params(fixture, dtype)
    z_theta = effective_volume_gaussian(params, _selection(fixture, dtype))
    assert z_theta.dtype == dtype
    np.testing.assert_allclose(
        np.asarray(z_theta), float(fixture["oracle_effective_volume"]),
        rtol=rtol, atol=atol,
    )
    dimension = fixture["eval_means"].shape[1]
    no_selection = GaussianSelection(
        location=jnp.asarray(fixture["selection_location"], dtype=dtype),
        precision=jnp.zeros((dimension, dimension), dtype=dtype),
    )
    z_one = effective_volume_gaussian(params, no_selection)
    np.testing.assert_allclose(np.asarray(z_one), 1.0, rtol=rtol, atol=max(atol, 1e-6))


@pytest.mark.parametrize("dtype,rtol,atol", DTYPE_CASES)
def test_precision_zero_reduces_to_masked(dtype, rtol, atol):
    """CMMD-SEL-REDUCE-001: precision = 0 reduces loss and denoiser to §16 masked."""

    fixture = _fixture()
    params = _params(fixture, dtype)
    grouped = _grouped(fixture, dtype)
    bandwidths = jnp.asarray(fixture["bandwidths"], dtype=dtype)
    dimension = fixture["eval_means"].shape[1]
    no_selection = GaussianSelection(
        location=jnp.asarray(fixture["selection_location"], dtype=dtype),
        precision=jnp.zeros((dimension, dimension), dtype=dtype),
    )

    selected_loss = grouped_analytic_selected_loss(
        params, grouped, bandwidths, no_selection
    )
    masked_loss = grouped_analytic_loss(params, grouped, bandwidths)
    np.testing.assert_allclose(
        np.asarray(selected_loss), np.asarray(masked_loss), rtol=rtol, atol=atol
    )

    selected_mean = grouped_denoise_selected(params, grouped, no_selection)
    from development.convmmd_grouped import grouped_denoise
    masked_mean = grouped_denoise(params, grouped)
    np.testing.assert_allclose(
        np.asarray(selected_mean), np.asarray(masked_mean), rtol=rtol, atol=atol
    )


def test_one_shot_selected_operations_match_oracle():
    """The public one-shot ``*_selected`` operations agree with the oracle (f64)."""

    fixture = _fixture()
    dtype = jnp.float64
    params = _params(fixture, dtype)
    selection = _selection(fixture, dtype)
    observations = jnp.asarray(fixture["observations"], dtype=dtype)
    mask = jnp.asarray(fixture["observed_mask"])
    noise_spec = PerItemFullNoise(
        jnp.asarray(fixture["measurement_covariances"], dtype=dtype)
    )

    loss = convmmd_loss_analytic_selected(
        params, observations, mask, noise=noise_spec,
        bandwidths=jnp.asarray(fixture["bandwidths"], dtype=dtype),
        selection=selection, dtype=dtype,
    )
    np.testing.assert_allclose(
        np.asarray(loss), float(fixture["oracle_selected_loss"]),
        rtol=F64_RTOL, atol=F64_ATOL,
    )

    posterior_mean = convmmd_denoise_selected(
        params, observations, mask, noise=noise_spec, selection=selection, dtype=dtype
    )
    np.testing.assert_allclose(
        np.asarray(posterior_mean), fixture["oracle_selected_posterior_mean"],
        rtol=F64_RTOL, atol=F64_ATOL,
    )

    components = convmmd_posterior_components_selected(
        params, observations, mask, noise=noise_spec, selection=selection, dtype=dtype
    )
    np.testing.assert_allclose(
        np.asarray(components.responsibilities),
        fixture["oracle_selected_responsibilities"], rtol=F64_RTOL, atol=F64_ATOL,
    )
