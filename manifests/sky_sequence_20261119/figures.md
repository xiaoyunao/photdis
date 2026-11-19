# Figure index — full footprint, controls and sequence science

All scientific plates have matching PNG/PDF under `figures/sky_sequence_20261119`.
The Pal5/v3.8 base, V5 RGB geometry and original brightness are retained. Galactic
coordinates, b=0/±20/±30, full actual Sagittarius/Pal5 tracks and known systems are
explicit; both tracks are dashed.

## Main display: all available gr footprint

- **`selected_channels/rgb_selected_fixed_gain`**: final main overview,
  B5–10/G10–20/R20–100kpc. This is the user's final selection.
- `selected_channels/rgb_selected_balanced`: identical channels/data with V5
  percentile balance learned on the same fixed old-domain reference.
- `selected_channels/physical_selected`: individual B/G/R density and signed
  residual with physical colorbars.
- `rgb/rgb_overview`: original10–15/15–26/26–100 channels, same expanded footprint.
- `rgb/rgb_bin_*`: five broad intervals each with its own near/mid/far RGB.
- `rgb/physical_overview`: scalar density and signed-residual companions.
- `coverage/sky_coverage`: original native PSF / usable-gr density, depth and EBV.

These images use all locally available imaging that satisfies the new gr entrance,
about29,387deg2. Gray means missing/quality-masked geometry; black is observed zero
or a value below the display floor. Positive residual means the0.3deg-smoothed
CMD density minus its5deg-smoothed background, with only the positive part shown.
Its black pixels do not mean there are no stars. Signed physical arrays are retained.

## Deliberately restricted controls

- **`old_domain/rgb_*`** and `physical_overview`: new source admission inside the
  old high-latitude, low-dust, common-depth domain, about14,156deg2. Extra blank
  sky is the explicit control mask, not an incomplete full-footprint run.
- **`prior_display/rgb_overview`** and `physical_overview`: exact previous-version
  RGB cells/stretch with the new coordinate/structure overlays only.
- `no_extinction/rgb_overview` and `physical_overview`: same full-footprint parent
  and corrected-overview stretch, with observed magnitudes in the input CMD.

## Channel trials

`channel_trials/rgb_near5_10_100_*`, `rgb_near3_15_100_*`,
`rgb_near10_30_100_*` retain the three initial user-requested alternatives.
Each has `fixed_gain` and `balanced` displays. The finite near edge is0.01kpc,
shown in their legends. These are display comparisons, not new stellar distances.
`comparisons/channel_layout_comparison` composes density-panel thumbnails, with
the final user-selected layout in the last column. Full density/residual plates
remain separate and unchanged.

## Native-sequence science

- `ridge/native_ridge_M13_M5`: all primary native members, observed two-color
  ridge/scatter, original/same-version/matched models and binary reference.
- `ridge/native_ridge_M30`: lower-metallicity comparison in the southern system.
- `ridge/sequence_shape_residuals`: color and reference-selected gray projections
  along the observed sequence; their branch-choice interpretation is in the caption
  records and science decisions, not a new distance estimator.
- `width/sequence_center_width`: same v5 common members, original and diagnostic
  group-centered residuals against interval half-width. Exact asymmetric intervals,
  scores and common-reference/zero-point results are in `width/center_width.csv`.

Use `scripts/sky6_figures.py` for all scientific plates from compact public arrays;
`scripts/sky6_compare.py` composes the comparison from the resulting PNGs.

## Preview only: background smoothing

`smoothing_trials/rgb_background_2deg`, `_5deg`, `_10deg` compare Gaussian
background scales2/5/10deg with the same0.3deg density, same source counts,
coverage and exact main color stretch. `comparisons/smoothing_comparison`
places their positive-residual panels side by side. These previews do not
replace the main overview; `smoothing_trials/main_preserved.json` proves its
bytes are unchanged. This varies the subtracted background, not the density blur.
