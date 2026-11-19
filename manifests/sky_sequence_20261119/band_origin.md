# Wide-band origin: direct observational checks

The exact previous v5 image contains **zero displayed cells at abs(b)<29 deg**.
Its high-latitude bright band is therefore not the Galactic plane. The old RGB
cells and stretch are unchanged in `prior_display`; only the requested Galactic
parallels, actual galstreams tracks and named-system annotations were added.

The Sagittarius track is `Sagittarius.antoja2020`, transformed from its actual
RA/Dec samples to Galactic coordinates. The actual `Pal5.ibata2024` track is
also shown, both dashed. Longitude wrap segments are split. No planar great
circle is substituted for the Sagittarius track.

We compare the CMD-weighted density within5deg of Sgr to its5–10deg neighbors,
outside abs(b)>=20deg. Within each2deg Galactic-latitude band, both strips use
the same min(on area,off area) weight, then their weighted densities are compared.
The unchanged previous v5 map gives on/neighbor ratios **1.155,1.248,1.334** in
B/G/R (48 paired bands). The expanded map gives **1.436,1.691,1.811** (62 bands).
Thus there is an Sgr-aligned CMD-density enhancement after standardizing latitude.
These are descriptive angular responses, not stream membership or a purity estimate;
the full broad foreground/halo gradient is not attributed entirely to Sagittarius.
See `band/Sgr_by_Galactic_latitude.csv` and `latitude_standardized_Sgr.csv`.

The new low-latitude bright population has another substantial dependence. On the
**same admitted sources**, removing the full-column correction from the input CMD
reduces B-channel acceptance: corrected/uncorrected is **2.19** at abs(b)<10deg,
**1.67** at abs(b)<20deg, **3.77** where the pixel-mean E(B-V)>=0.5, but **0.976**
at abs(b)>=30deg. Both source crowding/composition and full-column correction
therefore matter for the new near-plane appearance. This control keeps parent,
area, CMD prescription and displayed stretch fixed. It does not determine a 3D
extinction correction or split the broad band into physical disk/halo/stream stars.

The available gr geometry covers only83.20deg2 at abs(b)<10deg. Remaining gray
sky includes genuine missing/quality-masked imaging, not a reintroduced latitude
cut. Source-coordinate admission cutflows and pixel-center strip summaries have
slightly different boundary counts; both exact definitions are retained.

All figures retain Galactic coordinates. The b=0 line, b=±20 dashed and b=±30
dotted parallels directly locate the plane and old high-latitude display boundary.
