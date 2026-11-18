# Science-validation figure index

Ten new plates use the shared Pal5 / v3.8 palette, axes and export functions,
with the established compact paper typography. PDFs are intended for a
7.1-inch two-column width. Every plate has a 300 dpi PNG and numerical
sidecars. The [physical-width preview](aas_width_preview.png) compares
7.1 and 3.5 inches at 120 ppi. Multi-panel labels need the two-column width.
The [pixel-regression receipt](../../manifests/science_validation_20261118/style_regression/complete.json)
records exact matches to the original v3.8 F02/F03 PNG pixels.

| Scientific question | PDF | Numerical source |
|---|---|---|
| What does each native training set represent, and where do the same evaluation stars land in reference and inferred CMDs? | [Training / reference CMD](figures/native_training_reference_cmd.pdf) | [Counts and bins](../../manifests/science_validation_20261118/figures/native_training_reference_cmd.csv) |
| How do representative ordinary, metal-poor and intermediate-gravity stars receive distance probability? | [Training and full PDFs](figures/training_and_branch_pdfs.pdf) | [Curve coordinates](../../manifests/science_validation_20261118/figures/representative_curves.csv) |
| How much improvement remains with exactly the same training identities and labels? | [Same-training residual distributions](benchmark/same_training_baseline.pdf) | [Histogram](../../manifests/science_validation_20261118/benchmark/same_training_baseline.csv) |
| How does working selection change population composition? | [Selection chemistry](definitions/selection_chemistry.pdf) | [Block intervals](../../manifests/science_validation_20261118/definitions/selection_block_intervals.csv) |
| Is the metal-poor residual offset stable to rematching and a Gaia scale? | [Matched populations](populations/matched_metallicity.pdf) | [Values and intervals](../../manifests/science_validation_20261118/populations/matched_metallicity.csv) |
| Does that offset persist after excluding additional calibration and selection-fit identities? | [Excluded-calibration comparison](populations_clean/matched_metallicity.pdf) | [Values and intervals](../../manifests/science_validation_20261118/populations_clean/matched_metallicity.csv) |
| How do common-distance systems change along the sequence and by magnitude? | [Known-system sequences](systems/known_system_sequences.pdf) | [Magnitude / phase statistics](../../manifests/science_validation_20261118/systems/known_scale_members.csv) |
| What happens when a fixed bright population receives actual faint-star errors? | [Fixed-population noise](noise/fixed_population_noise.pdf) | [Realizations and summary](../../manifests/science_validation_20261118/noise/noise_summary.csv) |
| Does predictive probability agree with held-out measurements? | [Observation-space calibration](predictive/measurement_space_calibration.pdf) | [Coverage and scores](../../manifests/science_validation_20261118/predictive/observation_space_metrics.csv) |
| Does correcting the template's observable support recover the known stream distance? | [Filter selection and injections](filter/filter_selection_recovery.pdf) | [Per-distance scans](../../manifests/science_validation_20261118/filter/distance_scans.csv) |

## Plot interpretation

The CMD evaluation columns use identical qualified SpecDis stars within each
native. Training panels use the original admitted labels and their own
selection. Counts share a logarithmic scale. The much tighter inferred
sequence reflects the learned color–luminosity mapping.

The baseline residual plot uses the original 32-draw empirical control.
The [128-draw refinement](../../manifests/science_validation_20261118/numerics/baseline_integration.csv)
bounds that integration effect and supplies the headline numerical result.
Histograms record objects outside the displayed residual interval.

Metallicity error bars are sky-rematched group-contrast intervals. GD-1
error bars are stellar-bootstrap intervals for its median conditional on
one external distance track. Noise summaries use the same stars repeatedly.
The adjacent statistical meanings are preserved in the source tables.

Known-system sequence points use the archived low-crowding members. The
unmasked and all-member comparisons remain separate table rows. M68's
binned transfer result is supplied in a table because its archived input
has no matching individual full-PDF member sample. Sequence axes show
−4 to +4 mag; [display counts](../../manifests/science_validation_20261118/systems/sequence_display_counts.csv)
record points beyond that frame.

The filter image shows the first registered local segment/window/width,
with its full selected template. Injection histograms pool all registered
segments, backgrounds, windows and widths and normalize over the displayed
residual range. Full outliers and summary statistics stay in the injection
table. The filter gate fails, so this stage's structural result is its
recovery experiment and local distance scan.

The observation-space comparison uses 18–20 mag historical evaluation PDFs.
Gaia includes the signed measurement and the S/N-conditioned selection.
Spectroscopic curves use an explicit Gaussian VAC-summary approximation.
The [power table](../../manifests/science_validation_20261118/predictive/scale_branch_power.csv)
gives the scale and branch errors these samples can resolve.

The existing [direct distance, CMD, density and migration atlas](../paper_v2/README.md)
provides the companion catalogue-wide figures. The scientific choices and
remaining limits are in the [decision record](../../manifests/science_validation_20261118/science_decisions.md).

Recreate the preview locally with `python3 scripts/sv18_preview.py`.
