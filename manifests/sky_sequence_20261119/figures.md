# Display A: completed figure index

All plates have PNG/PDF counterparts under `figures/sky_sequence_20261119`.
The original v3.8/Pal5 base and v5 RGB layout remain. Pal5 and Sagittarius are
thin dashed tracks. Exact previous v5 brightness limits are restored for every
corresponding bin; physical data are unchanged (`brightness/unchanged_maps.json`).

- `rgb/rgb_overview`: all available native gr sky, original10–15/15–26/26–100 BGR.
- `rgb/rgb_bin_*`: five wide intervals, each with true equal-modulus near/mid/far RGB.
- `old_domain/rgb_*`: identical display admission inside old geography/dust/depth.
- `no_extinction/rgb_overview`: exact same admitted parent with observed input CMD,
  shown with exactly the corrected-overview stretch.
- `prior_display/rgb_overview`: exact previous v5 RGB cells/stretch, new references.
- `coverage/sky_coverage`: native PSF/valid-gr density, shallower gr depth and EBV.
- `rgb/physical_overview`, `old_domain/physical_overview`,
  `no_extinction/physical_overview`, `prior_display/physical_overview`:
  individual B/G/R scalar density and signed-residual panels with physical colorbars.

The user-requested nearer-channel comparisons are a separate registered display
extension (`configs/sky_sequence_channels_20261119.json`); original images remain.
The sequence-science figures and decisions follow in their own stages.
