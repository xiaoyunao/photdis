# Figure index

All plates use the locked v3.8/Pal5 palette, typography and export settings.
The two original v3.8 regression PNGs are checked pixel by pixel. No
manuscript edits are included. File links below point to PDFs; paired PNGs
have identical stems. Tables are in the corresponding manifest directory.

## Full-footprint maps

- [Main five distance slices](../../figures/halo_response_20261118/atlas/cmd_filter_aitoff_distance_slices.pdf),
  [annotated](../../figures/halo_response_20261118/atlas/cmd_filter_aitoff_distance_slices_annotated.pdf):
  observed grz<23; normalized average template-node density in6.3–10,
  10–15,15–26,26–50,50–100kpc. Native geometry and depth20 area shared.
- [Conservative bright comparison](../../figures/halo_response_20261118/atlas/cmd_filter_aitoff_distance_slices_bright.pdf):
  observed grz<20. Each version has a common scale across distance panels;
  numerical scales are provided rather than treating different panels as
  independent source memberships.
- [Eight fixed mu nodes](../../figures/halo_response_20261118/atlas/cmd_filter_aitoff_fixed_mu_deep.pdf):
  mu14.5–18.0 in0.5mag steps. A conservative `_bright` counterpart is supplied.
- [Coverage, depth, dust and native](../../figures/halo_response_20261118/atlas/cmd_filter_coverage_depth.pdf):
  independent random area, 0.1mag grz sensitivity, meanE(B−V), northern area fraction.
- [Signed background residual](../../figures/halo_response_20261118/atlas/cmd_filter_background_residual_deep.pdf)
  and [background](../../figures/halo_response_20261118/atlas/cmd_filter_background_deep.pdf):
  0.3degree count/area density minus5degree count/area background. No significance label.
- [Same-source QRF versus CMD](../../figures/halo_response_20261118/atlas/photdis_vs_cmd_same_sources_deep.pdf):
  same CMD-union objects with PDFs, same area/depth/mask. Mean CMD count and
  exclusive PDF mass are deliberately retained as different statistics.
- [Additional score cut](../../figures/halo_response_20261118/atlas/cmd_filter_score_comparison_deep.pdf):
  imaging/CMD parent versus score≥0.8 on exactly the same parent. Bright versions
  of background/residual/QRF/score maps are supplied.
- [Actual M13 geometry/CMD control](../../figures/halo_response_20261118/known/cmd_filter_known_system_check.pdf):
  on/control raw Hess diagrams and real positions; independent areas0.508/2.4444deg².
  Solid isochrone is at the external reference center; dotted curves show the
  intrinsic color/modulus envelope. Actual source acceptance also includes
  twice the empirical g-error curve. M5 has zero eligible on-area here.

## Distance-scale and selection controls

- [Identical-member reference/QRF CMD](../../figures/halo_response_20261118/sequence/known_sequence_same_source_cmd.pdf).
- [Three fixed distance relations on known systems](../../figures/halo_response_20261118/sequence/known_sequence_scale_comparison.pdf).
  Small-system/member uncertainties and all N, including sparse M92, are tabulated.
- [Independent training label distribution and old sequence](../../figures/halo_response_20261118/sequence/training_luminosity_old_sequence.pdf).
- [M5 conditional luminosity](../../figures/halo_response_20261118/sequence_views/sequence_luminosity_M5.pdf)
  with matching M13/M30/GD1_thin plates: same real reference positions,
  training distributions, conditional medians/16–84 intervals and residuals.
- [Continuous observable balance](../../figures/halo_response_20261118/balance/continuous_observable_balance.pdf).
  Actual balance failures are retained. Same-ID estimand contrasts, support,
  ESS, CDF gaps and leave-one sensitivities accompany the plot.
- [Native-template transfer](../../figures/halo_response_20261118/native/native_template_transfer.pdf).
- [Frozen seam decomposition](../../figures/halo_response_20261118/seam/native_seam_frozen_decomposition.pdf).

## Statistical corrections and foreground diagnostics

- [Corrected conditional scale/branch power](../../figures/halo_response_20261118/power/corrected_scale_branch_power.pdf).
  Wilson bands describe128-replication Monte Carlo uncertainty.
- [Whole-imaging donor noise control](../../figures/halo_response_20261118/noise_imaging/whole_imaging_noise_control.pdf).
  Dotted curves are unperturbed predictions on the same surviving source IDs.
- [Actual on/off Hess contributions, field1](../../figures/halo_response_20261118/hess_views/foreground_on_off_segment_1.pdf),
  with fields2 and5: on versus scaled off; template and numerator/denominator
  at observed peak and external reference. Original `hess/` plates are also retained.
  Alpha, visibility, all contributions and dominant-cell sums are numerical products.

All axes-outside, color-clipped and zero-area counts are in figure-side CSVs.
Colors in the density maps remain quantitative and have explicit colorbars;
no Gaia fill, additive sky correction, hole closing or artificial structures are added.
