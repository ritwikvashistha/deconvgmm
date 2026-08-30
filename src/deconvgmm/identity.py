"""Curated identity-projection API for the private beta."""

from ._impl.fit_control import (
    FitMode,
    FitResult,
    FitStatus,
    fit_converged,
    fit_fixed_steps,
)
from ._impl.identity_xd import EStep, Params, posterior_components
from ._impl.inference import (
    log_likelihood,
    posterior,
    posterior_mean,
    predict,
    predict_proba,
    sample_latent,
    sample_observed,
    score,
    score_samples,
)
from ._impl.metadata import (
    CONTRACT_ID,
    CONTRACT_VERSION,
    ResultMetadata,
)
from ._impl.validation import (
    PrecisionError,
    ValidatedIdentityInputs,
    ValidationError,
    canonicalize_fit_inputs,
    canonicalize_inference_inputs,
    diagonal_noise,
    fit_isotropic_noise,
    full_noise,
    inference_isotropic_noise,
    shared_full_noise,
)


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_VERSION",
    "EStep",
    "FitMode",
    "FitResult",
    "FitStatus",
    "Params",
    "PrecisionError",
    "ResultMetadata",
    "ValidatedIdentityInputs",
    "ValidationError",
    "canonicalize_fit_inputs",
    "canonicalize_inference_inputs",
    "diagonal_noise",
    "fit_converged",
    "fit_fixed_steps",
    "fit_isotropic_noise",
    "full_noise",
    "inference_isotropic_noise",
    "log_likelihood",
    "posterior",
    "posterior_components",
    "posterior_mean",
    "predict",
    "predict_proba",
    "sample_latent",
    "sample_observed",
    "score",
    "score_samples",
    "shared_full_noise",
]
