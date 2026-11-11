"""Evidence for the public boundary of ``fit_convmmd_recipe`` (contract §18.7).

The recipe is re-exported by ``deconvgmm.convmmd``, so it checks its inputs
eagerly: malformed arrays, counts and rates raise ``ValidationError``, and a
lower-precision float input (or JAX x64 being disabled) raises
``PrecisionError`` instead of being silently upcast, because the recipe's
certificate and curvature thresholds are float64 ones. Valid NumPy inputs are
accepted and give the same fit as JAX arrays.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from development.convmmd import ConvMMDParams
from development.convmmd_recipe import birth_at, fit_convmmd_recipe
from development.validation import PrecisionError, ValidationError


def _data(n=60, d=2, seed=0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.normal(-2.0, 0.5, (n // 2, d)), rng.normal(2.0, 0.5, (n - n // 2, d))])
    noise = np.tile(0.05 * np.eye(d), (n, 1, 1))
    return x, noise


def _quick(x, noise, k=2, **kwargs):
    options = dict(key=jax.random.PRNGKey(0), n_steps=0, witness_moves=0)
    options.update(kwargs)
    return fit_convmmd_recipe(x, noise, k, **options)


def test_numpy_inputs_give_the_same_fit_as_jax_inputs():
    x, noise = _data()
    from_numpy = _quick(x, noise, n_steps=40)
    from_jax = _quick(jnp.asarray(x), jnp.asarray(noise), n_steps=40)
    assert from_numpy.parameters.weights.dtype == jnp.float64
    for a, b in zip(from_numpy.parameters, from_jax.parameters):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(from_numpy.loss) == float(from_jax.loss)


@pytest.mark.parametrize("field", ["observations", "measurement_covariances"])
def test_float32_inputs_are_refused_not_upcast(field):
    x, noise = _data()
    arrays = {"observations": x, "measurement_covariances": noise}
    arrays[field] = arrays[field].astype(np.float32)
    with pytest.raises(PrecisionError, match=field):
        _quick(arrays["observations"], arrays["measurement_covariances"])


def test_float32_bandwidths_are_refused():
    x, noise = _data()
    with pytest.raises(PrecisionError, match="bandwidths"):
        _quick(x, noise, bandwidths=np.asarray([0.5, 1.0], dtype=np.float32))


@pytest.mark.parametrize(
    "case, match",
    [
        ("integer observations", "floating"),
        ("boolean observations", "boolean"),
        ("1-D observations", r"\(N, D\)"),
        ("nonfinite observations", "finite"),
        ("noise shape", "measurement_covariances"),
        ("asymmetric noise", "symmetric"),
        ("indefinite noise", "positive semidefinite"),
        ("too few rows", "more than k"),
    ],
)
def test_malformed_arrays_raise_validation_error(case, match):
    x, noise = _data()
    k = 2
    if case == "integer observations":
        x = np.rint(x).astype(np.int64)
    elif case == "boolean observations":
        x = x > 0
    elif case == "1-D observations":
        x = x[:, 0]
    elif case == "nonfinite observations":
        x = x.copy()
        x[3, 1] = np.nan
    elif case == "noise shape":
        noise = noise[:-1]
    elif case == "asymmetric noise":
        noise = noise.copy()
        noise[5, 0, 1] = 0.02
    elif case == "indefinite noise":
        noise = noise.copy()
        noise[5] = np.diag([0.05, -0.05])
    elif case == "too few rows":
        x, noise, k = x[:2], noise[:2], 2
    with pytest.raises(ValidationError, match=match):
        _quick(x, noise, k=k)


@pytest.mark.parametrize(
    "options, match",
    [
        (dict(k=0), "k must be >= 1"),
        (dict(k=True), "boolean"),
        (dict(k=2.5), "positive integer"),
        (dict(chunk=0), "chunk"),
        (dict(n_steps=-1), "n_steps"),
        (dict(witness_moves=-1), "witness_moves"),
        (dict(max_escapes=-1), "max_escapes"),
        (dict(candidate_rows=0), "candidate_rows"),
        (dict(bandwidth_rows=1), "bandwidth_rows"),
        (dict(lanczos_steps=0), "lanczos_steps"),
        (dict(learning_rate=0.0), "learning_rate"),
        (dict(learning_rate=float("nan")), "learning_rate"),
        (dict(grad_tol=-1e-6), "grad_tol"),
        (dict(floor_fraction=-0.03), "floor_fraction"),
        (dict(covariance_floor=-0.1), "covariance_floor"),
        (dict(bandwidths=np.asarray([0.5, 0.0])), "bandwidths"),
        (dict(bandwidths=np.ones((2, 2))), "bandwidths"),
    ],
)
def test_out_of_range_controls_raise_validation_error(options, match):
    x, noise = _data()
    k = options.pop("k", 2)
    with pytest.raises(ValidationError, match=match):
        _quick(x, noise, k=k, **options)


def test_zero_steps_and_zero_moves_are_valid():
    x, noise = _data()
    result = _quick(x, noise, n_steps=0, witness_moves=0, max_escapes=0)
    assert int(result.n_iter) == 0
    assert result.repairs == ()


def test_birth_at_uses_every_row_below_ten_rows():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(7, 2))
    noise = np.tile(0.01 * np.eye(2), (7, 1, 1))
    params = ConvMMDParams(
        weights=jnp.asarray([0.7, 0.3]),
        means=jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
        covariances=jnp.tile(jnp.eye(2), (2, 1, 1)),
    )
    moved = birth_at(params, jnp.asarray(x), jnp.asarray(noise), 0, 1e-3)
    centred = x - x.mean(axis=0)
    expected = centred.T @ centred / 6 - 0.01 * np.eye(2)
    values, vectors = np.linalg.eigh(expected)
    expected = (vectors * np.maximum(values, 1e-3)) @ vectors.T
    np.testing.assert_allclose(np.asarray(moved.means[1]), x.mean(axis=0), rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(moved.covariances[1]), expected, rtol=1e-12, atol=1e-14)


def test_x64_disabled_raises_precision_error():
    project_root = Path(__file__).resolve().parents[2]
    code = """
import jax
import numpy as np
from development.convmmd_recipe import fit_convmmd_recipe
from development.validation import PrecisionError

assert not jax.config.x64_enabled
x = np.zeros((20, 2)); noise = np.tile(np.eye(2), (20, 1, 1))
try:
    fit_convmmd_recipe(x, noise, 2)
except PrecisionError as error:
    assert "x64" in str(error)
else:
    raise AssertionError("the recipe ran with x64 disabled")
"""
    environment = os.environ.copy()
    environment["JAX_ENABLE_X64"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=project_root, env=environment, capture_output=True, text=True,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stderr
