"""Curated convMMD API facade.

convMMD (convolutional Maximum Mean Discrepancy) is a likelihood-free density
deconvolution and empirical-Bayes denoiser governed by the ``xdgmm-jax.convmmd``
contract (``docs/convmmd-model-contract.md``; the ``xdgmm-jax`` prefix is a
historical contract identifier that predates the package rename). It is exposed
as ``deconvgmm.convmmd``. Every convMMD capability-matrix row is **Pending** and
``performance_claim`` is ``none`` until qualified; it is validated against an
independent NumPy oracle at float64 near machine epsilon.

Contract revision ``0.2.0-draft.1`` adds **per-coordinate missing-at-random (MAR)**
support (contract §16): a full-width collection plus a boolean ``observed_mask`` is
grouped by missing-pattern and each missing coordinate is **exactly marginalized**
through a per-observation projection. The masked operations (``*_masked``), the
grouped loss/denoiser, the masked fits, ``median_bandwidths_masked``, the grouping
adapters, and the supported measurement-noise tags are re-exported here. These
remain **Pending**; missing-not-at-random selection is out of this revision's scope.
"""

from ._impl.convmmd import (
    ConvMMDParams,
    ConvMMDUnconstrained,
    PosteriorComponents,
    convmmd_loss_analytic,
    convmmd_loss_mc,
    denoise,
    expected_rbf_kernel,
    median_bandwidths,
    posterior_components,
    to_canonical,
)
from ._impl.convmmd_fit import (
    CONVMMD_CONTRACT_ID,
    CONVMMD_CONTRACT_VERSION,
    ConvMMDFitResult,
    ConvMMDFitState,
    ConvMMDFitStatus,
    ConvMMDResultMetadata,
    fit_analytic,
    fit_analytic_state,
    fit_mc,
    fit_mc_state,
)
from ._impl.convmmd_grouped import (
    convmmd_denoise_masked,
    convmmd_loss_analytic_masked,
    convmmd_loss_mc_masked,
    convmmd_posterior_components_masked,
    fit_masked_analytic,
    fit_masked_analytic_state,
    fit_masked_mc,
    fit_masked_mc_state,
    group_masked_fit_inputs,
    group_masked_inputs,
    grouped_analytic_loss,
    grouped_denoise,
    grouped_mc_loss,
    grouped_posterior_components,
    median_bandwidths_masked,
)
from ._impl.general_validation import (
    GroupedGeneralInputs,
    NoInformativeWeightError,
    PerItemFullNoise,
    SharedDiagonalNoise,
    SharedFullNoise,
    SharedIsotropicNoise,
)


__all__ = [
    "CONVMMD_CONTRACT_ID",
    "CONVMMD_CONTRACT_VERSION",
    "ConvMMDFitResult",
    "ConvMMDFitState",
    "ConvMMDFitStatus",
    "ConvMMDParams",
    "ConvMMDResultMetadata",
    "ConvMMDUnconstrained",
    "GroupedGeneralInputs",
    "NoInformativeWeightError",
    "PerItemFullNoise",
    "PosteriorComponents",
    "SharedDiagonalNoise",
    "SharedFullNoise",
    "SharedIsotropicNoise",
    "convmmd_denoise_masked",
    "convmmd_loss_analytic",
    "convmmd_loss_analytic_masked",
    "convmmd_loss_mc",
    "convmmd_loss_mc_masked",
    "convmmd_posterior_components_masked",
    "denoise",
    "expected_rbf_kernel",
    "fit_analytic",
    "fit_analytic_state",
    "fit_masked_analytic",
    "fit_masked_analytic_state",
    "fit_masked_mc",
    "fit_masked_mc_state",
    "fit_mc",
    "fit_mc_state",
    "group_masked_fit_inputs",
    "group_masked_inputs",
    "grouped_analytic_loss",
    "grouped_denoise",
    "grouped_mc_loss",
    "grouped_posterior_components",
    "median_bandwidths",
    "median_bandwidths_masked",
    "posterior_components",
    "to_canonical",
]
