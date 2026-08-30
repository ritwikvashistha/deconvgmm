"""Temporary pure-JAX prediction and sampling operations for identity XD.

This module assumes canonical, validated :class:`~development.identity_xd.Params`
and full measurement-covariance inputs.  Static Python shape/count checks protect
the array equations, while prediction and random numerical work remain on the
JAX device.  It is not the future public package namespace.
"""

from __future__ import annotations

import operator

import jax
import jax.numpy as jnp

from .identity_xd import (
    Params,
    marginalized_posterior,
    posterior_components,
)


Array = jax.Array


def _canonical_sampling_params(params: Params) -> tuple[Params, int, int]:
    """Coerce one canonical parameter PyTree and validate static shapes."""

    try:
        means = jnp.asarray(params.means)
        weights_input = params.weights
        covariances_input = params.covariances
    except AttributeError as error:
        raise TypeError(
            "params must provide weights, means, and covariances"
        ) from error

    if means.ndim != 2 or means.shape[0] < 1 or means.shape[1] < 1:
        raise ValueError(
            "means must have shape (K, D) with K,D >= 1; "
            f"received {means.shape}"
        )
    if not jnp.issubdtype(means.dtype, jnp.floating):
        raise TypeError(
            "parameter means must have a floating dtype; "
            f"received {means.dtype}"
        )

    n_components, dimension = means.shape
    dtype = means.dtype
    weights = jnp.asarray(weights_input, dtype=dtype)
    covariances = jnp.asarray(covariances_input, dtype=dtype)
    if weights.shape != (n_components,):
        raise ValueError(
            f"weights must have shape {(n_components,)}; "
            f"received {weights.shape}"
        )
    expected_covariance_shape = (n_components, dimension, dimension)
    if covariances.shape != expected_covariance_shape:
        raise ValueError(
            "parameter covariances must have shape "
            f"{expected_covariance_shape}; received {covariances.shape}"
        )
    return Params(weights, means, covariances), n_components, dimension


def _nonnegative_sample_count(value: object) -> int:
    """Return a static nonnegative sample count, rejecting boolean values."""

    if isinstance(value, (bool, jnp.bool_)):
        raise TypeError("n must be an integer sample count, not bool")
    try:
        count = operator.index(value)
    except TypeError as error:
        raise TypeError("n must be a static integer sample count") from error
    if count < 0:
        raise ValueError(f"n must be nonnegative; received {count}")
    return int(count)


def predict_proba(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return component probabilities, or NaNs for a failed observation.

    This numerical-leaf convenience cannot expose the detailed failure mask.
    Call :func:`posterior_components` when status is required.  Failed rows are
    deliberately made nonfinite so fallback responsibilities cannot be mistaken
    for a successful prediction.
    """

    e_step = posterior_components(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    failed = jnp.any(e_step.failed_pairs, axis=-1)
    return jnp.where(
        failed[..., None],
        jnp.asarray(jnp.nan, dtype=e_step.responsibilities.dtype),
        e_step.responsibilities,
    )


def predict(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the most probable component, or ``-1`` for a failed row."""

    probabilities = predict_proba(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    labels = jnp.argmax(probabilities, axis=-1)
    valid = jnp.all(jnp.isfinite(probabilities), axis=-1)
    return jnp.where(valid, labels, jnp.asarray(-1, dtype=labels.dtype))


def _e_step_for_leaf(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array,
):
    """Evaluate one E-step and return its per-observation failure mask."""

    e_step = posterior_components(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return e_step, jnp.any(e_step.failed_pairs, axis=-1)


def score_samples(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return noisy observed-data log density for every canonical item.

    A row with any failed component factorization is returned as NaN. Detailed
    device status remains available through :func:`posterior_components`.
    """

    e_step, failed = _e_step_for_leaf(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    return jnp.where(
        failed,
        jnp.asarray(jnp.nan, dtype=e_step.score_samples.dtype),
        e_step.score_samples,
    )


def log_likelihood(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the sum of per-item observed-data log densities."""

    return jnp.sum(
        score_samples(
            params,
            observations,
            measurement_covariances,
            factor_jitter=factor_jitter,
        )
    )


def score(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return the arithmetic mean observed-data log density."""

    return jnp.mean(
        score_samples(
            params,
            observations,
            measurement_covariances,
            factor_jitter=factor_jitter,
        )
    )


def posterior(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> tuple[Array, Array]:
    """Return marginalized latent posterior mean and covariance.

    Failed observation rows are filled with NaNs. Use
    :func:`posterior_components` for component moments and explicit status.
    """

    e_step, failed = _e_step_for_leaf(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )
    mean, covariance = marginalized_posterior(e_step)
    nan = jnp.asarray(jnp.nan, dtype=mean.dtype)
    return (
        jnp.where(failed[..., None], nan, mean),
        jnp.where(failed[..., None, None], nan, covariance),
    )


def posterior_mean(
    params: Params,
    observations: Array,
    measurement_covariances: Array,
    *,
    factor_jitter: float | Array = 0.0,
) -> Array:
    """Return only the marginalized latent posterior mean."""

    return posterior(
        params,
        observations,
        measurement_covariances,
        factor_jitter=factor_jitter,
    )[0]


def _sample_latent_canonical(
    params: Params,
    key: Array,
    n_samples: int,
    dimension: int,
) -> Array:
    """Draw from already shape-checked parameters without host conversion."""

    dtype = params.means.dtype
    component_key, normal_key = jax.random.split(key, 2)
    if n_samples == 0:
        return jnp.empty((0, dimension), dtype=dtype)
    components = jax.random.categorical(
        component_key,
        jnp.log(params.weights),
        axis=-1,
        shape=(n_samples,),
        mode="high",
    )
    standard_normal = jax.random.normal(
        normal_key, shape=(n_samples, dimension), dtype=dtype
    )
    # Canonical model covariances are already symmetric and positive definite.
    # Disabling implicit symmetrization matches the validated kernel policy and
    # avoids an overflow-prone redundant add near the dtype limit.
    factors = jax.lax.linalg.cholesky(
        params.covariances, symmetrize_input=False
    )
    selected_factors = factors[components]
    centered_draws = jnp.einsum(
        "nij,nj->ni", selected_factors, standard_normal
    )
    return params.means[components] + centered_draws


def sample_latent(params: Params, key: Array, n: int) -> Array:
    """Draw ``n`` latent mixture samples using the required explicit key.

    ``n`` is a static nonnegative integer and the result has shape ``(n, D)``
    in the parameter dtype. Reusing a key intentionally repeats a draw; callers
    split keys when they need independent draws. No internal default key exists.
    """

    n_samples = _nonnegative_sample_count(n)
    canonical_params, _, dimension = _canonical_sampling_params(params)
    return _sample_latent_canonical(
        canonical_params, key, n_samples, dimension
    )


def sample_observed(params: Params, key: Array, S: Array) -> Array:
    """Draw noisy observations for canonical ``S.shape == (n, D, D)``.

    Measurement noise uses the deterministic symmetric eigendecomposition
    square root ``Q @ diag(sqrt(max(lambda, 0)))``.  Canonical ``S`` is already
    symmetric PSD; clipping is therefore only a roundoff guard at zero and lets
    exactly singular covariances sample successfully. A row that is materially
    indefinite under the selected-dtype covariance tolerance returns NaNs;
    actionable rejection still belongs at the eager validation boundary.

    The explicit key is split between the latent and measurement-noise draws.
    Reusing it repeats the result; callers must split keys for independence.
    """

    canonical_params, _, dimension = _canonical_sampling_params(params)
    noise_input = jnp.asarray(S)
    if not jnp.issubdtype(noise_input.dtype, jnp.floating):
        raise TypeError(
            "S must be a canonical floating measurement covariance array; "
            f"received dtype {noise_input.dtype}"
        )
    expected_tail = (dimension, dimension)
    if noise_input.ndim != 3 or noise_input.shape[-2:] != expected_tail:
        raise ValueError(
            "S must have exact canonical shape (n, D, D) with trailing "
            f"shape {expected_tail}; received {noise_input.shape}"
        )

    noise = jnp.asarray(noise_input, dtype=canonical_params.means.dtype)
    n_samples = noise.shape[0]
    latent_key, noise_key = jax.random.split(key, 2)
    if n_samples == 0:
        return jnp.empty((0, dimension), dtype=canonical_params.means.dtype)
    latent_draws = _sample_latent_canonical(
        canonical_params, latent_key, n_samples, dimension
    )
    eigenvalues, eigenvectors = jnp.linalg.eigh(
        noise, symmetrize_input=False
    )
    psd_relative_tolerance = jnp.asarray(
        2e-11 if noise.dtype == jnp.dtype(jnp.float64) else 5e-5,
        dtype=noise.dtype,
    )
    spectral_scale = jnp.maximum(
        jnp.asarray(1.0, dtype=noise.dtype),
        jnp.max(jnp.abs(eigenvalues), axis=-1),
    )
    noise_is_psd = (
        jnp.all(jnp.isfinite(eigenvalues), axis=-1)
        & (
            jnp.min(eigenvalues, axis=-1)
            >= -psd_relative_tolerance * spectral_scale
        )
    )
    square_root = eigenvectors * jnp.sqrt(
        jnp.maximum(eigenvalues, jnp.asarray(0.0, dtype=noise.dtype))
    )[..., None, :]
    standard_normal = jax.random.normal(
        noise_key, shape=(n_samples, dimension), dtype=noise.dtype
    )
    measurement_draws = jnp.einsum(
        "nij,nj->ni", square_root, standard_normal
    )
    observed_draws = latent_draws + measurement_draws
    return jnp.where(
        noise_is_psd[..., None],
        observed_draws,
        jnp.asarray(jnp.nan, dtype=noise.dtype),
    )


__all__ = [
    "log_likelihood",
    "posterior",
    "posterior_mean",
    "predict",
    "predict_proba",
    "sample_latent",
    "sample_observed",
    "score",
    "score_samples",
]
