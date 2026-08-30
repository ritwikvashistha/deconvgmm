"""Temporary general observed-space sampling for fixed ``M``.

The canonical leaf accepts exact per-item ``(n, M, D)`` projection matrices
and ``(n, M, M)`` full measurement covariances.  The eager companion boundary
accepts only the explicit projection/noise tags from
:mod:`development.general_validation`, validates their domains, and constructs
those canonical arrays.  Neither surface implements grouped or ragged
sampling, and neither is a public package API.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .general_validation import (
    IdentityProjection,
    NoiseSpec,
    PerItemDiagonalNoise,
    PerItemFullNoise,
    PerItemIsotropicNoise,
    PerItemProjection,
    ProjectionSpec,
    SharedDiagonalNoise,
    SharedFullNoise,
    SharedIsotropicNoise,
    SharedProjection,
    _canonical_noise_with_raw,
    _canonical_parameters,
    _canonical_projection,
    _shape,
)
from .identity_xd import Params
from .inference import (
    _canonical_sampling_params,
    _nonnegative_sample_count,
    sample_latent,
)
from .validation import ValidationError, _computation_dtype


Array = jax.Array


class ValidatedGeneralSamplingInputs(NamedTuple):
    """Canonical arrays and static dimensions for observed sampling."""

    parameters: Params
    projection_matrices: Array
    measurement_covariances: Array
    n_samples: int
    observed_dimension: int


def _sampling_observed_dimension(
    projection: ProjectionSpec | object,
    *,
    n_samples: int,
    latent_dimension: int,
) -> int:
    """Infer ``M`` only from an explicit projection tag and exact shape."""

    if isinstance(projection, PerItemProjection):
        shape = _shape(projection.values, field="projection matrices")
        if (
            len(shape) != 3
            or shape[0] != n_samples
            or shape[2] != latent_dimension
        ):
            raise ValidationError(
                "projection matrices: received "
                f"{shape}; expected ({n_samples}, M, {latent_dimension})"
            )
        return int(shape[1])

    if isinstance(projection, SharedProjection):
        shape = _shape(projection.matrix, field="shared projection matrix")
        if len(shape) != 2 or shape[1] != latent_dimension:
            raise ValidationError(
                "shared projection matrix: received "
                f"{shape}; expected (M, {latent_dimension})"
            )
        return int(shape[0])

    if isinstance(projection, IdentityProjection):
        if projection.dimension != latent_dimension:
            raise ValidationError(
                "identity projection dimension must equal latent dimension; "
                f"received {projection.dimension} and {latent_dimension}"
            )
        return latent_dimension

    raise ValidationError(
        "projection must use an explicit PerItemProjection, "
        "SharedProjection, or IdentityProjection tag"
    )


def canonicalize_general_sampling_inputs(
    params: Params,
    n: int,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    dtype: object,
) -> ValidatedGeneralSamplingInputs:
    """Validate one tagged fixed-``M`` general observed-sampling call.

    ``n`` is authoritative.  Per-item tags must carry that exact leading axis;
    shared inputs are accepted only through their explicit tags.  This eager
    boundary may inspect and synchronize arrays and is outside JIT/autodiff
    guarantees.
    """

    n_samples = _nonnegative_sample_count(n)
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    canonical_params, _, latent_dimension = _canonical_parameters(
        params,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    observed_dimension = _sampling_observed_dimension(
        projection,
        n_samples=n_samples,
        latent_dimension=latent_dimension,
    )
    canonical_projection = _canonical_projection(
        projection,
        batch_shape=(n_samples,),
        observed_dimension=observed_dimension,
        latent_dimension=latent_dimension,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    canonical_noise, _ = _canonical_noise_with_raw(
        noise,
        batch_shape=(n_samples,),
        observed_dimension=observed_dimension,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    return ValidatedGeneralSamplingInputs(
        parameters=canonical_params,
        projection_matrices=canonical_projection,
        measurement_covariances=canonical_noise,
        n_samples=n_samples,
        observed_dimension=observed_dimension,
    )


def _canonical_sampling_arrays(
    params: Params,
    n_samples: int,
    projection_matrices: Array,
    measurement_covariances: Array,
) -> tuple[Params, Array, Array, int]:
    """Check exact canonical shapes without performing host value validation."""

    canonical_params, _, latent_dimension = _canonical_sampling_params(params)
    try:
        projection_input = jnp.asarray(projection_matrices)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "projection_matrices must be a canonical floating array"
        ) from error
    if not jnp.issubdtype(projection_input.dtype, jnp.floating):
        raise TypeError(
            "projection_matrices must be a canonical floating array; "
            f"received dtype {projection_input.dtype}"
        )
    if projection_input.ndim != 3:
        raise ValueError(
            "projection_matrices must have exact canonical shape "
            f"(n, M, D); received {projection_input.shape}"
        )
    observed_dimension = projection_input.shape[1]
    expected_projection_shape = (
        n_samples,
        observed_dimension,
        latent_dimension,
    )
    if projection_input.shape != expected_projection_shape:
        raise ValueError(
            "projection_matrices must have exact canonical shape "
            f"{expected_projection_shape}; received {projection_input.shape}"
        )

    try:
        noise_input = jnp.asarray(measurement_covariances)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "measurement_covariances must be a canonical floating array"
        ) from error
    if not jnp.issubdtype(noise_input.dtype, jnp.floating):
        raise TypeError(
            "measurement_covariances must be a canonical floating array; "
            f"received dtype {noise_input.dtype}"
        )
    expected_noise_shape = (
        n_samples,
        observed_dimension,
        observed_dimension,
    )
    if noise_input.shape != expected_noise_shape:
        raise ValueError(
            "measurement_covariances must have exact canonical shape "
            f"{expected_noise_shape}; received {noise_input.shape}"
        )

    dtype = canonical_params.means.dtype
    return (
        canonical_params,
        jnp.asarray(projection_input, dtype=dtype),
        jnp.asarray(noise_input, dtype=dtype),
        observed_dimension,
    )


def sample_observed_general(
    params: Params,
    key: Array,
    n: int,
    projection_matrices: Array,
    measurement_covariances: Array,
) -> Array:
    """Draw canonical fixed-``M`` noisy linear observations.

    ``key`` is required and is split before either zero-size return.  The
    existing identity-contract :func:`sample_latent` remains the sole latent
    mixture sampler.  Measurement noise uses the deterministic symmetric
    eigendecomposition square root, so singular PSD covariance rows are valid.
    A materially indefinite raw canonical row produces NaNs; the tagged eager
    boundary rejects that row actionably before this leaf is called.
    """

    n_samples = _nonnegative_sample_count(n)
    (
        canonical_params,
        projection,
        noise,
        observed_dimension,
    ) = _canonical_sampling_arrays(
        params,
        n_samples,
        projection_matrices,
        measurement_covariances,
    )
    latent_key, noise_key = jax.random.split(key, 2)
    dtype = canonical_params.means.dtype
    if n_samples == 0 or observed_dimension == 0:
        return jnp.empty((n_samples, observed_dimension), dtype=dtype)

    latent_draws = sample_latent(canonical_params, latent_key, n_samples)
    projected_draws = jnp.einsum(
        "nmd,nd->nm", projection, latent_draws
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(
        noise, symmetrize_input=False
    )
    psd_relative_tolerance = jnp.asarray(
        2e-11 if dtype == jnp.dtype(jnp.float64) else 5e-5,
        dtype=dtype,
    )
    spectral_scale = jnp.maximum(
        jnp.asarray(1.0, dtype=dtype),
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
        jnp.maximum(eigenvalues, jnp.asarray(0.0, dtype=dtype))
    )[..., None, :]
    standard_normal = jax.random.normal(
        noise_key,
        shape=(n_samples, observed_dimension),
        dtype=dtype,
    )
    measurement_draws = jnp.einsum(
        "nij,nj->ni", square_root, standard_normal
    )
    observed_draws = projected_draws + measurement_draws
    return jnp.where(
        noise_is_psd[..., None],
        observed_draws,
        jnp.asarray(jnp.nan, dtype=dtype),
    )


def sample_observed_general_from_specs(
    params: Params,
    key: Array,
    n: int,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    dtype: object,
) -> Array:
    """Validate explicit tagged modes, then draw canonical observations."""

    canonical = canonicalize_general_sampling_inputs(
        params,
        n,
        projection=projection,
        noise=noise,
        dtype=dtype,
    )
    return sample_observed_general(
        canonical.parameters,
        key,
        canonical.n_samples,
        canonical.projection_matrices,
        canonical.measurement_covariances,
    )


__all__ = [
    "IdentityProjection",
    "PerItemDiagonalNoise",
    "PerItemFullNoise",
    "PerItemIsotropicNoise",
    "PerItemProjection",
    "SharedDiagonalNoise",
    "SharedFullNoise",
    "SharedIsotropicNoise",
    "SharedProjection",
    "ValidatedGeneralSamplingInputs",
    "canonicalize_general_sampling_inputs",
    "sample_observed_general",
    "sample_observed_general_from_specs",
]
