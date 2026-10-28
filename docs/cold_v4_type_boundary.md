# Cold-v4 DESI type-boundary audit

This audit uses unique, quality-qualified DESI spectra in the four existing
development/final fields. It applies the frozen cold-v4 optical color box,
support domain and WISE20 route. The individual records are in
`manifests/cold_v4_type_boundary.json`; the color and Gaia-CMD view is in
`figures/cold_v4/05_type_boundary.png`.

| Field (RA, degrees) | Accepted cold giants / qualified | Tier of accepted giants | Metal-poor cold dwarfs rejected by optical box / qualified |
|---|---:|---|---:|
| 205–215 (development) | 12/32 | 11 A, 1 C | 25/40 |
| 0–10 | 0/0 | — | 0/0 |
| 150–160 | 0/10 | — | 5/8 |
| 220–230 | 1/12 | 1 C | 14/14 |

The 12 accepted development-field giants have median DESI `logg=2.39` and
median calibrated `[Fe/H]=-0.86` among the 10 objects where that calibration
is defined. All 12 pass `grz_clean`. The one accepted RA220–230 giant has
`logg=3.23` and raw `[Fe/H]=-2.13`; its metallicity lies outside the
calibration domain. None of the 13 accepted giants has Gaia parallax S/N≥10,
so the right-hand CMD panel cannot confirm their absolute magnitudes.

Every box-rejected metal-poor dwarf lies blueward of the dereddened `g-r`
lower edge, 0.909. The `r-z` lower edge is 0.401.
Among these, 17/25, 2/5 and 8/14 also lie below the `r-z` lower edge in the
three populated fields. The rejected dwarfs have median DESI `logg` of
4.67, 4.62 and 4.69 respectively. Only four have Gaia parallax S/N≥10 and
thus a reliable plotted Legacy `M_g`. In the development field, 20/25
rejected dwarfs nevertheless pass `grz_clean`; in the RA220–230 field,
10/14 pass. Image quality alone does not explain their rejection.

**Decision.** Keep the frozen color box for the existing cold-dwarf distance
baseline. Report its metal-poor dwarf incompleteness and A-layer giant
contamination explicitly. A future probability or multiple-stellar-branch
selection needs a label-selection audit and independent class-prior evidence.
These DESI counts are conditional on spectroscopic targeting and are not
population contamination fractions for the photometric catalog.
