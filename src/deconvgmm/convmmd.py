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
adapters, and the supported measurement-noise tags are re-exported here.

Contract revision ``0.3.0-draft.1`` adds a **known selection function (MNAR)** on the
true value ``Omega(z)`` (contract §17): the ``*_selected`` family. A Gaussian window
(:class:`GaussianSelection`) gives an exact analytic path (the §17.5 parameter
transform reusing the masked operators); a general differentiable completeness
(:class:`CallableSelection`) gives a self-normalized-importance-sampling (SNIS)
Monte-Carlo path with ``Z_theta``/ESS diagnostics. **No machine-epsilon parity is
claimed for a general ``Omega``**; the SNIS estimator is biased at finite ``M`` and
consistent as ``M -> inf``. Selection on the observed value ``Omega(x-tilde)`` is added
under a homoscedastic, fully-observed restriction (the ``*_observed`` operators; §17.12).

Contract revision ``0.4.0-draft.1`` generalizes ``Omega(x-tilde)`` to **heteroscedastic**
per-object noise (the ``*_observed_hetero`` family; §17.13). Because a *detected* object's
own ``S_i`` is known, the §17.5 identity applies per object to ``Sigma_k + S_i``: a
Gaussian window is **machine-epsilon analytic** (a sum of per-object one-sample
discrepancies) and a general ``Omega`` uses a per-object SNIS with the object's own ``S_i``
— **no measurement-noise model is required** (the denoiser is the base §7 MAR posterior at
each ``S_i``). Contract revision ``0.5.0-draft.1`` composes heteroscedastic
``Omega(x-tilde)`` with the §16 projection (the ``*_observed_masked`` family; §17.14): the
Gaussian window is **marginalized onto each observed subspace** (precision
``(P_i Psi P_i^T)^{-1}``, not ``P_i Psi^{-1} P_i^T``) and the per-object transform runs on the
projected inflated covariance ``P_i Sigma_k P_i^T + S_i`` — machine-epsilon analytic for a
Gaussian window, per-object projected SNIS for a general ``Omega``, still with no noise model;
it reduces to §17.13 at ``P_i = I`` and to the base §16 masked loss at ``Psi^{-1} = 0``. Every
convMMD capability remains **Pending**.
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
from ._impl.convmmd_selection import (
    CallableSelection,
    GaussianSelection,
    convmmd_denoise_selected,
    convmmd_denoise_selected_observed,
    convmmd_denoise_selected_observed_hetero,
    convmmd_denoise_selected_observed_masked,
    convmmd_loss_analytic_selected,
    convmmd_loss_analytic_selected_observed,
    convmmd_loss_analytic_selected_observed_hetero,
    convmmd_loss_analytic_selected_observed_masked,
    convmmd_loss_snis_selected,
    convmmd_loss_snis_selected_observed,
    convmmd_loss_snis_selected_observed_hetero,
    convmmd_loss_snis_selected_observed_masked,
    convmmd_posterior_components_selected,
    convmmd_snis_denoise_selected,
    effective_volume_gaussian,
    effective_volume_gaussian_observed,
    fit_selected_analytic,
    fit_selected_analytic_state,
    fit_selected_mc,
    fit_selected_mc_state,
    fit_selected_observed_analytic,
    fit_selected_observed_analytic_state,
    fit_selected_observed_hetero_analytic,
    fit_selected_observed_hetero_analytic_state,
    fit_selected_observed_masked_analytic,
    fit_selected_observed_masked_analytic_state,
    gaussian_selection_omega,
    grouped_analytic_selected_loss,
    grouped_analytic_selected_observed_masked_loss,
    grouped_denoise_selected,
    grouped_posterior_components_selected,
    grouped_snis_denoise,
    grouped_snis_loss,
    grouped_snis_selected_observed_masked_loss,
    marginalize_gaussian_window,
    select_gaussian_params,
    select_gaussian_params_observed,
    select_gaussian_params_observed_hetero,
    select_gaussian_params_observed_masked,
    snis_denoise_projected,
    snis_diagnostics_observed_hetero,
    snis_diagnostics_observed_masked,
    snis_effective_sample_size,
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
    "CallableSelection",
    "ConvMMDFitResult",
    "ConvMMDFitState",
    "ConvMMDFitStatus",
    "ConvMMDParams",
    "ConvMMDResultMetadata",
    "ConvMMDUnconstrained",
    "GaussianSelection",
    "GroupedGeneralInputs",
    "NoInformativeWeightError",
    "PerItemFullNoise",
    "PosteriorComponents",
    "SharedDiagonalNoise",
    "SharedFullNoise",
    "SharedIsotropicNoise",
    "convmmd_denoise_masked",
    "convmmd_denoise_selected",
    "convmmd_denoise_selected_observed",
    "convmmd_denoise_selected_observed_hetero",
    "convmmd_denoise_selected_observed_masked",
    "convmmd_loss_analytic",
    "convmmd_loss_analytic_masked",
    "convmmd_loss_analytic_selected",
    "convmmd_loss_analytic_selected_observed",
    "convmmd_loss_analytic_selected_observed_hetero",
    "convmmd_loss_analytic_selected_observed_masked",
    "convmmd_loss_mc",
    "convmmd_loss_mc_masked",
    "convmmd_loss_snis_selected",
    "convmmd_loss_snis_selected_observed",
    "convmmd_loss_snis_selected_observed_hetero",
    "convmmd_loss_snis_selected_observed_masked",
    "convmmd_posterior_components_masked",
    "convmmd_posterior_components_selected",
    "convmmd_snis_denoise_selected",
    "denoise",
    "effective_volume_gaussian",
    "effective_volume_gaussian_observed",
    "expected_rbf_kernel",
    "fit_analytic",
    "fit_analytic_state",
    "fit_masked_analytic",
    "fit_masked_analytic_state",
    "fit_masked_mc",
    "fit_masked_mc_state",
    "fit_mc",
    "fit_mc_state",
    "fit_selected_analytic",
    "fit_selected_analytic_state",
    "fit_selected_mc",
    "fit_selected_mc_state",
    "fit_selected_observed_analytic",
    "fit_selected_observed_analytic_state",
    "fit_selected_observed_hetero_analytic",
    "fit_selected_observed_hetero_analytic_state",
    "fit_selected_observed_masked_analytic",
    "fit_selected_observed_masked_analytic_state",
    "gaussian_selection_omega",
    "group_masked_fit_inputs",
    "group_masked_inputs",
    "grouped_analytic_loss",
    "grouped_analytic_selected_loss",
    "grouped_analytic_selected_observed_masked_loss",
    "grouped_denoise",
    "grouped_denoise_selected",
    "grouped_mc_loss",
    "grouped_posterior_components",
    "grouped_posterior_components_selected",
    "grouped_snis_denoise",
    "grouped_snis_loss",
    "grouped_snis_selected_observed_masked_loss",
    "marginalize_gaussian_window",
    "median_bandwidths",
    "median_bandwidths_masked",
    "posterior_components",
    "select_gaussian_params",
    "select_gaussian_params_observed",
    "select_gaussian_params_observed_hetero",
    "select_gaussian_params_observed_masked",
    "snis_denoise_projected",
    "snis_diagnostics_observed_hetero",
    "snis_diagnostics_observed_masked",
    "snis_effective_sample_size",
    "to_canonical",
]
