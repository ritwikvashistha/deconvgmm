"""Temporary eager validation and mask grouping for general-projection XD.

The fixed-``M`` kernels in :mod:`development.general_xd` accept canonical JAX
arrays.  This host-controlled module prepares those arrays without making raw
lower-rank inputs implicitly shared.  It also implements the contract's
deterministic boolean-mask grouping.  Array inspection, grouping, restoration,
and exception formatting are intentionally outside JIT and autodiff claims.

This remains development code, not a public package API.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import NamedTuple, Sequence, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from .identity_xd import Params
from .validation import (
    PrecisionError,
    PreparedControls,
    ValidationError,
    _canonical_covariances,
    _computation_dtype,
    _convert_array,
    _host_array,
    _raise_shape,
    _validate_weights,
    validate_controls,
)


Array = jax.Array


class NoInformativeWeightError(ValidationError):
    """A fitting collection has no positive weight on an observed row."""

    code = "no_informative_weight"


class ValidatedGeneralInputs(NamedTuple):
    """Canonical fixed-``M`` arrays accepted by the general numerical leaf."""

    parameters: Params
    observations: Array
    projection_matrices: Array
    measurement_covariances: Array


class ValidatedGeneralFitInputs(NamedTuple):
    """Canonical fixed-``M`` fitting arrays with finite informative weight."""

    parameters: Params
    observations: Array
    projection_matrices: Array
    measurement_covariances: Array
    sample_weight: Array
    informative_weight: Array


@dataclass(frozen=True, slots=True)
class PerItemProjection:
    """Tag one projection matrix for every observation or inference item."""

    values: object


@dataclass(frozen=True, slots=True)
class SharedProjection:
    """Tag one explicitly shared projection matrix."""

    matrix: object


@dataclass(frozen=True, slots=True)
class IdentityProjection:
    """Tag an explicit identity projection of a declared dimension."""

    dimension: int

    def __post_init__(self) -> None:
        if isinstance(self.dimension, (bool, np.bool_)):
            raise TypeError(
                "identity projection dimension must be an integer, not boolean"
            )
        try:
            dimension = operator.index(self.dimension)
        except TypeError as error:
            raise TypeError(
                "identity projection dimension must be an integer"
            ) from error
        if dimension < 1:
            raise ValueError(
                "identity projection dimension must be positive; "
                f"received {dimension}"
            )
        object.__setattr__(self, "dimension", int(dimension))


@dataclass(frozen=True, slots=True)
class PerItemIsotropicNoise:
    """Tag one isotropic measurement variance per item."""

    variances: object


@dataclass(frozen=True, slots=True)
class PerItemDiagonalNoise:
    """Tag one vector of diagonal measurement variances per item."""

    variances: object


@dataclass(frozen=True, slots=True)
class PerItemFullNoise:
    """Tag one full measurement covariance per item."""

    covariances: object


@dataclass(frozen=True, slots=True)
class SharedIsotropicNoise:
    """Tag one explicitly shared isotropic measurement variance."""

    variance: object


@dataclass(frozen=True, slots=True)
class SharedDiagonalNoise:
    """Tag one explicitly shared diagonal measurement variance vector."""

    variances: object


@dataclass(frozen=True, slots=True)
class SharedFullNoise:
    """Tag one explicitly shared full measurement covariance."""

    covariance: object


ProjectionSpec: TypeAlias = (
    PerItemProjection | SharedProjection | IdentityProjection
)
NoiseSpec: TypeAlias = (
    PerItemIsotropicNoise
    | PerItemDiagonalNoise
    | PerItemFullNoise
    | SharedIsotropicNoise
    | SharedDiagonalNoise
    | SharedFullNoise
)


@dataclass(frozen=True, slots=True)
class GeneralMaskGroup:
    """One deterministic fixed-``M`` group produced from a boolean mask."""

    group_index: int
    mask: tuple[bool, ...]
    original_indices: tuple[int, ...]
    coordinate_indices: tuple[int, ...]
    observations: Array
    projection_matrices: Array
    measurement_covariances: Array
    sample_weight: Array


@dataclass(frozen=True, slots=True)
class GroupedGeneralInputs:
    """Eagerly grouped inputs, valid for inference even when every ``M=0``."""

    parameters: Params
    groups: tuple[GeneralMaskGroup, ...]
    grouped_indices: tuple[int, ...]
    restoration_indices: tuple[int, ...]
    n_samples: int
    potential_observed_dimension: int
    latent_dimension: int
    informative_weight: Array


@dataclass(frozen=True, slots=True)
class GroupedGeneralFitInputs:
    """Grouped inputs whose informative weight is finite and positive."""

    grouped: GroupedGeneralInputs
    informative_weight: Array
    controls: PreparedControls


def _shape(value: object, *, field: str) -> tuple[int, ...]:
    return tuple(_host_array(value, field=field).shape)


def _canonical_parameters(
    params: Params,
    *,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> tuple[Params, int, int]:
    """Validate one latent parameter object without using observed dimensions."""

    try:
        weights_source = params.weights
        means_source = params.means
        covariances_source = params.covariances
    except AttributeError as error:
        raise ValidationError(
            "parameters must provide weights, means, and covariances"
        ) from error

    weights_shape = _shape(weights_source, field="weights")
    means_shape = _shape(means_source, field="means")
    covariance_shape = _shape(
        covariances_source, field="parameter covariances"
    )
    if len(means_shape) != 2 or means_shape[0] < 1 or means_shape[1] < 1:
        _raise_shape("means", means_shape, "(K, D) with K,D >= 1")
    n_components, latent_dimension = means_shape
    if weights_shape != (n_components,):
        _raise_shape("weights", weights_shape, (n_components,))
    expected_covariances = (
        n_components,
        latent_dimension,
        latent_dimension,
    )
    if covariance_shape != expected_covariances:
        _raise_shape(
            "parameter covariances", covariance_shape, expected_covariances
        )

    weights = _convert_array(
        weights_source,
        field="weights",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    means = _convert_array(
        means_source,
        field="means",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    covariances = _convert_array(
        covariances_source,
        field="parameter covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
    )
    _validate_weights(weights, dtype=requested_numpy_dtype)
    covariances = _canonical_covariances(
        covariances,
        field="parameter covariances",
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=True,
    )
    return Params(weights, means, covariances), n_components, latent_dimension


def _convert_floating(
    value: object,
    *,
    field: str,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
    nonnegative: bool = False,
) -> Array:
    return _convert_array(
        value,
        field=field,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=True,
        require_nonnegative_source=nonnegative,
    )


def _expect_shape(
    value: object,
    *,
    field: str,
    expected: tuple[int, ...],
) -> None:
    received = _shape(value, field=field)
    if received != expected:
        _raise_shape(field, received, expected)


def _projection_mode_error(
    projection: object,
    *,
    expected_per_item: tuple[int, ...],
) -> ValidationError:
    try:
        received: object = _shape(projection, field="projection")
    except ValidationError:
        received = type(projection).__name__
    return ValidationError(
        "projection: received "
        f"{received}; expected explicit PerItemProjection with shape "
        f"{expected_per_item}, SharedProjection, or IdentityProjection"
    )


def _canonical_projection(
    projection: ProjectionSpec | object,
    *,
    batch_shape: tuple[int, ...],
    observed_dimension: int,
    latent_dimension: int,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> Array:
    per_item_shape = batch_shape + (observed_dimension, latent_dimension)
    if isinstance(projection, PerItemProjection):
        _expect_shape(
            projection.values,
            field="projection matrices",
            expected=per_item_shape,
        )
        return _convert_floating(
            projection.values,
            field="projection matrices",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
        )
    if isinstance(projection, SharedProjection):
        shared_shape = (observed_dimension, latent_dimension)
        _expect_shape(
            projection.matrix,
            field="shared projection matrix",
            expected=shared_shape,
        )
        shared = _convert_floating(
            projection.matrix,
            field="shared projection matrix",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
        )
        return jnp.broadcast_to(shared, per_item_shape)
    if isinstance(projection, IdentityProjection):
        if not (
            projection.dimension == latent_dimension
            and observed_dimension == latent_dimension
        ):
            raise ValidationError(
                "identity projection requires its dimension, M, and D to "
                "agree; received dimension="
                f"{projection.dimension}, M={observed_dimension}, "
                f"D={latent_dimension}"
            )
        identity = jnp.eye(latent_dimension, dtype=requested_jax_dtype)
        return jnp.broadcast_to(identity, per_item_shape)
    raise _projection_mode_error(
        projection, expected_per_item=per_item_shape
    )


def _canonical_covariances_allow_empty(
    covariances: Array,
    *,
    field: str,
    requested_numpy_dtype: np.dtype,
) -> Array:
    """Validate PSD covariances, treating the unique ``0 x 0`` matrix exactly."""

    if covariances.shape[-1] == 0:
        return covariances
    return _canonical_covariances(
        covariances,
        field=field,
        requested_numpy_dtype=requested_numpy_dtype,
        positive_definite=False,
    )


def _noise_mode_error(
    noise: object,
    *,
    expected_per_item: tuple[int, ...],
) -> ValidationError:
    try:
        received: object = _shape(noise, field="measurement noise")
    except ValidationError:
        received = type(noise).__name__
    return ValidationError(
        "measurement noise covariances: received "
        f"{received}; expected explicit PerItemFullNoise with shape "
        f"{expected_per_item} or another explicit noise tag"
    )


def _canonical_noise_with_raw(
    noise: NoiseSpec | object,
    *,
    batch_shape: tuple[int, ...],
    observed_dimension: int,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Return canonical noise and its converted pre-symmetry full form."""

    full_shape = batch_shape + (observed_dimension, observed_dimension)
    identity = jnp.eye(observed_dimension, dtype=requested_jax_dtype)

    if isinstance(noise, PerItemIsotropicNoise):
        _expect_shape(
            noise.variances,
            field="per-item isotropic measurement variances",
            expected=batch_shape,
        )
        values = _convert_floating(
            noise.variances,
            field="per-item isotropic measurement variances",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
            nonnegative=True,
        )
        full = values[..., None, None] * identity
        return full, full

    if isinstance(noise, PerItemDiagonalNoise):
        diagonal_shape = batch_shape + (observed_dimension,)
        _expect_shape(
            noise.variances,
            field="per-item diagonal measurement variances",
            expected=diagonal_shape,
        )
        values = _convert_floating(
            noise.variances,
            field="per-item diagonal measurement variances",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
            nonnegative=True,
        )
        full = values[..., :, None] * identity
        return full, full

    if isinstance(noise, PerItemFullNoise):
        _expect_shape(
            noise.covariances,
            field="measurement noise covariances",
            expected=full_shape,
        )
        full = _convert_floating(
            noise.covariances,
            field="measurement noise covariances",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
        )
        canonical = _canonical_covariances_allow_empty(
            full,
            field="measurement noise covariances",
            requested_numpy_dtype=requested_numpy_dtype,
        )
        return canonical, full

    if isinstance(noise, SharedIsotropicNoise):
        _expect_shape(
            noise.variance,
            field="shared isotropic measurement variance",
            expected=(),
        )
        value = _convert_floating(
            noise.variance,
            field="shared isotropic measurement variance",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
            nonnegative=True,
        )
        full = jnp.broadcast_to(value * identity, full_shape)
        return full, full

    if isinstance(noise, SharedDiagonalNoise):
        diagonal_shape = (observed_dimension,)
        _expect_shape(
            noise.variances,
            field="shared diagonal measurement variances",
            expected=diagonal_shape,
        )
        values = _convert_floating(
            noise.variances,
            field="shared diagonal measurement variances",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
            nonnegative=True,
        )
        full = jnp.broadcast_to(values[:, None] * identity, full_shape)
        return full, full

    if isinstance(noise, SharedFullNoise):
        shared_shape = (observed_dimension, observed_dimension)
        _expect_shape(
            noise.covariance,
            field="shared measurement noise covariance",
            expected=shared_shape,
        )
        full = _convert_floating(
            noise.covariance,
            field="shared measurement noise covariance",
            requested_numpy_dtype=requested_numpy_dtype,
            requested_jax_dtype=requested_jax_dtype,
        )
        canonical = _canonical_covariances_allow_empty(
            full,
            field="shared measurement noise covariance",
            requested_numpy_dtype=requested_numpy_dtype,
        )
        return (
            jnp.broadcast_to(canonical, full_shape),
            jnp.broadcast_to(full, full_shape),
        )

    raise _noise_mode_error(noise, expected_per_item=full_shape)


def _canonicalize_general_inference_inputs_with_raw_noise(
    params: Params,
    observations: object,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    dtype: object,
) -> tuple[ValidatedGeneralInputs, Array]:
    """Validate fixed-``M`` inputs while retaining pre-symmetry full noise."""

    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    canonical_params, _, latent_dimension = _canonical_parameters(
        params,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    observation_shape = _shape(observations, field="observations")
    if len(observation_shape) < 1:
        _raise_shape("observations", observation_shape, "B + (M,)")
    batch_shape = observation_shape[:-1]
    observed_dimension = observation_shape[-1]
    canonical_observations = _convert_array(
        observations,
        field="observations",
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
        require_floating_source=False,
    )
    canonical_projection = _canonical_projection(
        projection,
        batch_shape=batch_shape,
        observed_dimension=observed_dimension,
        latent_dimension=latent_dimension,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    canonical_noise, raw_noise = _canonical_noise_with_raw(
        noise,
        batch_shape=batch_shape,
        observed_dimension=observed_dimension,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    return (
        ValidatedGeneralInputs(
            canonical_params,
            canonical_observations,
            canonical_projection,
            canonical_noise,
        ),
        raw_noise,
    )


def canonicalize_general_inference_inputs(
    params: Params,
    observations: object,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    dtype: object,
) -> ValidatedGeneralInputs:
    """Validate one single or batched fixed-``M`` general inference call."""

    canonical, _ = _canonicalize_general_inference_inputs_with_raw_noise(
        params,
        observations,
        projection=projection,
        noise=noise,
        dtype=dtype,
    )
    return canonical


def _canonical_sample_weight(
    sample_weight: object | None,
    *,
    expected_shape: tuple[int, ...],
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> Array:
    """Validate weights while the pre-conversion host values remain visible."""

    if sample_weight is None:
        return jnp.ones(expected_shape, dtype=requested_jax_dtype)
    source = _host_array(sample_weight, field="sample_weight")
    if source.shape != expected_shape:
        raise ValidationError(
            "sample_weight shape: received "
            f"{tuple(source.shape)}; expected {expected_shape}"
        )
    if np.issubdtype(source.dtype, np.bool_):
        raise ValidationError("sample_weight must not be boolean")
    if np.issubdtype(source.dtype, np.complexfloating):
        raise ValidationError("sample_weight must not be complex")
    try:
        is_real = bool(
            np.issubdtype(source.dtype, np.integer)
            or np.issubdtype(source.dtype, np.floating)
        )
    except TypeError:
        is_real = False
    if not is_real:
        raise ValidationError("sample_weight must contain real numeric values")
    if np.any(~np.isfinite(source)):
        raise ValidationError("sample_weight must be finite")
    if np.any(source < 0):
        raise ValidationError("sample_weight must be nonnegative")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        converted = source.astype(requested_numpy_dtype, copy=False)
    if np.any(~np.isfinite(converted)):
        raise PrecisionError(
            "sample_weight must remain finite after conversion to "
            f"{requested_numpy_dtype.name}"
        )
    if np.any((source > 0) & (converted == 0)):
        raise PrecisionError(
            "sample_weight contains a positive value that underflows to zero "
            f"in {requested_numpy_dtype.name}"
        )
    result = jnp.asarray(converted, dtype=requested_jax_dtype)
    if result.dtype != requested_jax_dtype:
        raise PrecisionError(
            "sample_weight could not be represented in the requested "
            f"{requested_numpy_dtype.name} computation"
        )
    return result


def _informative_weight(
    sample_weight: Array,
    informative_rows: np.ndarray,
    *,
    requested_numpy_dtype: np.dtype,
    requested_jax_dtype: jnp.dtype,
) -> Array:
    host = np.asarray(jax.device_get(sample_weight))
    selected = host[informative_rows]
    with np.errstate(over="ignore", invalid="ignore"):
        total = np.sum(selected, dtype=requested_numpy_dtype)
    if not bool(np.isfinite(total)):
        raise PrecisionError(
            "sample_weight informative total must remain finite in "
            f"{requested_numpy_dtype.name}"
        )
    return jnp.asarray(total, dtype=requested_jax_dtype)


def _require_informative_weight(weight: Array) -> None:
    if not bool(np.asarray(jax.device_get(weight)) > 0):
        raise NoInformativeWeightError(
            "no_informative_weight: fitting requires at least one "
            "positive-weight row with M > 0"
        )


def canonicalize_general_fit_inputs(
    params: Params,
    observations: object,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    sample_weight: object | None = None,
    dtype: object,
) -> ValidatedGeneralFitInputs:
    """Validate one fixed-``M`` fitting group and require informative mass."""

    observation_shape = _shape(observations, field="observations")
    if len(observation_shape) != 2 or observation_shape[0] < 1:
        _raise_shape(
            "observations", observation_shape, "(N, M) with N >= 1"
        )
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)
    canonical = canonicalize_general_inference_inputs(
        params,
        observations,
        projection=projection,
        noise=noise,
        dtype=dtype,
    )
    weights = _canonical_sample_weight(
        sample_weight,
        expected_shape=(observation_shape[0],),
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    informative_rows = np.full(
        observation_shape[0], observation_shape[1] > 0, dtype=bool
    )
    informative = _informative_weight(
        weights,
        informative_rows,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    _require_informative_weight(informative)
    return ValidatedGeneralFitInputs(
        canonical.parameters,
        canonical.observations,
        canonical.projection_matrices,
        canonical.measurement_covariances,
        weights,
        informative,
    )


def _validate_mask(
    observations: object, observed_mask: object
) -> tuple[np.ndarray, tuple[int, int]]:
    mask = _host_array(observed_mask, field="observed_mask")
    if mask.dtype != np.dtype(np.bool_):
        raise ValidationError(
            "observed_mask must have boolean dtype; received "
            f"{mask.dtype}"
        )
    if mask.ndim != 2 or mask.shape[0] < 1:
        _raise_shape(
            "observed_mask", tuple(mask.shape), "(N, P) with N >= 1"
        )
    observation_shape = _shape(observations, field="observations")
    if observation_shape != tuple(mask.shape):
        if observation_shape and observation_shape[0] != mask.shape[0]:
            _raise_shape("observed_mask", tuple(mask.shape), observation_shape)
        _raise_shape("observations", observation_shape, tuple(mask.shape))
    return mask, (int(mask.shape[0]), int(mask.shape[1]))


def _validate_mask_modes(
    projection: object,
    noise: object,
    *,
    potential_dimension: int,
    latent_dimension: int,
) -> None:
    if not isinstance(
        projection, (PerItemProjection, SharedProjection, IdentityProjection)
    ):
        raise ValidationError(
            "projection for the mask adapter must use an explicit projection tag"
        )
    if isinstance(projection, IdentityProjection) and not (
        potential_dimension == latent_dimension == projection.dimension
    ):
        raise ValidationError(
            "identity projection for masked inputs requires P == D and the "
            "declared dimension to equal P; received "
            f"P={potential_dimension}, D={latent_dimension}, "
            f"dimension={projection.dimension}"
        )
    if isinstance(noise, (PerItemIsotropicNoise, PerItemDiagonalNoise)):
        raise ValidationError(
            "mask grouping requires per-item full measurement noise; "
            "per-item isotropic/diagonal modes are not contracted here"
        )
    if not isinstance(
        noise,
        (
            PerItemFullNoise,
            SharedIsotropicNoise,
            SharedDiagonalNoise,
            SharedFullNoise,
        ),
    ):
        raise ValidationError(
            "measurement noise for the mask adapter must use per-item full "
            "or an explicit shared noise tag"
        )


def group_masked_general_inputs(
    params: Params,
    observations: object,
    observed_mask: object,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    sample_weight: object | None = None,
    dtype: object,
) -> GroupedGeneralInputs:
    """Validate and group a finite full-coordinate collection by exact mask."""

    mask, (n_samples, potential_dimension) = _validate_mask(
        observations, observed_mask
    )
    requested_numpy_dtype, requested_jax_dtype = _computation_dtype(dtype)

    # Establish D before checking the mask-specific identity restriction.  The
    # full canonicalizer repeats the complete parameter-domain validation.
    _, _, latent_dimension = _canonical_parameters(
        params,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    _validate_mask_modes(
        projection,
        noise,
        potential_dimension=potential_dimension,
        latent_dimension=latent_dimension,
    )
    canonical, raw_noise = (
        _canonicalize_general_inference_inputs_with_raw_noise(
            params,
            observations,
            projection=projection,
            noise=noise,
            dtype=dtype,
        )
    )
    weights = _canonical_sample_weight(
        sample_weight,
        expected_shape=(n_samples,),
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )
    informative_rows = np.any(mask, axis=1)
    informative = _informative_weight(
        weights,
        informative_rows,
        requested_numpy_dtype=requested_numpy_dtype,
        requested_jax_dtype=requested_jax_dtype,
    )

    mask_keys = tuple(sorted({tuple(bool(value) for value in row) for row in mask}))
    groups: list[GeneralMaskGroup] = []
    grouped_indices_list: list[int] = []
    for group_index, mask_key in enumerate(mask_keys):
        row_indices_array = np.flatnonzero(
            np.all(mask == np.asarray(mask_key, dtype=bool), axis=1)
        )
        coordinate_indices_array = np.flatnonzero(
            np.asarray(mask_key, dtype=bool)
        )
        row_indices = tuple(int(value) for value in row_indices_array)
        coordinate_indices = tuple(
            int(value) for value in coordinate_indices_array
        )
        grouped_indices_list.extend(row_indices)

        group_observations = canonical.observations[row_indices_array]
        group_observations = group_observations[:, coordinate_indices_array]
        group_projection = canonical.projection_matrices[row_indices_array]
        group_projection = group_projection[:, coordinate_indices_array, :]
        group_noise = raw_noise[row_indices_array]
        group_noise = group_noise[:, coordinate_indices_array, :]
        group_noise = group_noise[:, :, coordinate_indices_array]
        # The full-coordinate covariances were validated before slicing.  Slice
        # the retained pre-symmetry values because a residual negligible at the
        # full scale can be material in one selected principal block, then
        # validate and canonicalize that block at its own scale.
        group_noise = _canonical_covariances_allow_empty(
            group_noise,
            field=(
                "selected measurement covariance principal blocks for mask "
                f"group {group_index} at coordinates {coordinate_indices}"
            ),
            requested_numpy_dtype=requested_numpy_dtype,
        )
        group_weight = weights[row_indices_array]
        groups.append(
            GeneralMaskGroup(
                group_index=group_index,
                mask=mask_key,
                original_indices=row_indices,
                coordinate_indices=coordinate_indices,
                observations=group_observations,
                projection_matrices=group_projection,
                measurement_covariances=group_noise,
                sample_weight=group_weight,
            )
        )

    grouped_indices = tuple(grouped_indices_list)
    restoration_indices = tuple(
        int(value) for value in np.argsort(np.asarray(grouped_indices))
    )
    return GroupedGeneralInputs(
        parameters=canonical.parameters,
        groups=tuple(groups),
        grouped_indices=grouped_indices,
        restoration_indices=restoration_indices,
        n_samples=n_samples,
        potential_observed_dimension=potential_dimension,
        latent_dimension=latent_dimension,
        informative_weight=informative,
    )


def group_masked_general_fit_inputs(
    params: Params,
    observations: object,
    observed_mask: object,
    *,
    projection: ProjectionSpec,
    noise: NoiseSpec,
    sample_weight: object | None = None,
    factor_jitter: object = 0.0,
    covariance_ridge: object = 0.0,
    dtype: object,
) -> GroupedGeneralFitInputs:
    """Prepare grouped fitting inputs with control-before-no-info precedence."""

    controls = validate_controls(
        factor_jitter=factor_jitter,
        covariance_ridge=covariance_ridge,
        dtype=dtype,
    )
    grouped = group_masked_general_inputs(
        params,
        observations,
        observed_mask,
        projection=projection,
        noise=noise,
        sample_weight=sample_weight,
        dtype=dtype,
    )
    _require_informative_weight(grouped.informative_weight)
    return GroupedGeneralFitInputs(
        grouped=grouped,
        informative_weight=grouped.informative_weight,
        controls=controls,
    )


def restore_grouped_rows(
    grouped: GroupedGeneralInputs,
    group_values: Sequence[object],
    *,
    field: str,
) -> Array:
    """Concatenate group-leading values and restore original row order."""

    if len(group_values) != len(grouped.groups):
        raise ValidationError(
            f"{field}: received {len(group_values)} group values; expected "
            f"{len(grouped.groups)} groups"
        )
    canonical_values: list[Array] = []
    expected_trailing_shape: tuple[int, ...] | None = None
    expected_dtype: jnp.dtype | None = None
    for group, value in zip(grouped.groups, group_values, strict=True):
        try:
            array = jnp.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"{field}: group {group.group_index} must be an array"
            ) from error
        expected_leading = len(group.original_indices)
        if array.ndim < 1 or array.shape[0] != expected_leading:
            raise ValidationError(
                f"{field}: group {group.group_index} received leading shape "
                f"{array.shape}; expected first axis {expected_leading}"
            )
        trailing_shape = tuple(array.shape[1:])
        if expected_trailing_shape is None:
            expected_trailing_shape = trailing_shape
            expected_dtype = array.dtype
        elif trailing_shape != expected_trailing_shape:
            raise ValidationError(
                f"{field}: group {group.group_index} received trailing shape "
                f"{trailing_shape}; expected {expected_trailing_shape}"
            )
        elif array.dtype != expected_dtype:
            raise ValidationError(
                f"{field}: group {group.group_index} received dtype "
                f"{array.dtype}; expected {expected_dtype}"
            )
        canonical_values.append(array)

    concatenated = jnp.concatenate(canonical_values, axis=0)
    restoration = jnp.asarray(grouped.restoration_indices, dtype=jnp.int32)
    return concatenated[restoration]


__all__ = [
    "GeneralMaskGroup",
    "GroupedGeneralFitInputs",
    "GroupedGeneralInputs",
    "IdentityProjection",
    "NoInformativeWeightError",
    "PerItemDiagonalNoise",
    "PerItemFullNoise",
    "PerItemIsotropicNoise",
    "PerItemProjection",
    "PrecisionError",
    "SharedDiagonalNoise",
    "SharedFullNoise",
    "SharedIsotropicNoise",
    "SharedProjection",
    "ValidatedGeneralFitInputs",
    "ValidatedGeneralInputs",
    "ValidationError",
    "canonicalize_general_fit_inputs",
    "canonicalize_general_inference_inputs",
    "group_masked_general_fit_inputs",
    "group_masked_general_inputs",
    "restore_grouped_rows",
]
