# Bounded DESI–Gaia–Legacy same-star calibration inventory

The first joint-label feasibility study reads seven existing 10°×10° DR11
field diagnostic tables on allocated NERSC CPU (job 60484207). The fields and
their development/reserved-validation roles were registered in
[`configs/joint_calibration_fields.json`](../configs/joint_calibration_fields.json)
before the count. They cover selected sky positions, not a random or complete
DESI/Legacy footprint. The full 37,214-row same-star FITS remains in ignored
NERSC `data/joint_calibration_v1/same_star.fits`; the
[small manifest](../manifests/joint_calibration_v1.json) records each field,
role, denominator and color–metallicity cross-tabulation.
The [color–metallicity–luminosity figure](../figures/joint_calibration/01_color_metallicity_luminosity.png)
shows the clean RVS core and its high-S/N geometric subset on the same axes.
The plotted abundance subset has 2,314 objects; two other clean core objects
do not have a valid calibrated abundance.

## Selection and labels

Start with unambiguous DESI matches already joined to DR11 measurements and
Gaia identities. Retain one Legacy row per DESI object and, where present,
one best-S/N spectrum per Gaia source. Keep raw `g0,r,z0`, color and formal
errors. `clean_low_dust` requires measured and clean `grz` and `E(B−V)<0.03`.
The high-confidence RVS main-sequence core requires a qualified stellar
spectrum, `4200≤T_eff<5000 K` and `logg−σ_logg>4.2`. The cool-giant control
uses `logg+σ_logg<3.2` in the same temperature range; transitional/other
stars, nonstellar Redrock evidence and unknown quality remain separate.
An optical Gaia CMD white-dwarf region is flagged as a *candidate* check,
not assigned a pure white-dwarf class. These new labels do not alter the
frozen cold-v4 route or its broader earlier `logg≥3.5` audit.

The calibrated RVS `[Fe/H]` is retained as a continuous value alongside its
formal error only inside the published correction domain and for qualified
stellar spectra. Formal error does not absorb calibration systematics.
Parallax layers require usable Gaia astrometry, valid zero-point correction,
clean low-dust DR11 photometry and corrected S/N respectively ≥20, 10–20,
0–10, or negative. Raw parallax and its error remain in *every* matched row.
Only the ≥20 layer receives a directly inverted `M_g` diagnostic. Weak and
negative parallaxes remain for future forward-likelihood inference.

## Measured overlap

| Same-star quantity | Count |
|---|---:|
| Unique unambiguous DESI–Legacy matches | 37,214 |
| Conservative 4200–5000 K main-sequence core | 3,213 |
| Core with clean, low-dust `grz` | 2,316 |
| Core with clean photometry and Gaia corrected S/N≥20 | 224 |
| Core high-S/N with calibrated `[Fe/H]<−1.5` | **1** |
| Core high-S/N with `0.6≤(g−r)₀<0.909` | 82 |
| Core high-S/N, blue of v4 edge **and** `[Fe/H]<−1.5` | **0** |
| Conservative cool giants / clean low-dust / high-S/N | 59 / 32 / **0** |

The 224 high-S/N core stars are all at `r<18`: 103 at `14≤r<16` and
121 at `16≤r<18`. They occupy `0.6≤(g−r)₀<1.6`, with just seven at
`1.2≤(g−r)₀<1.6`. In metallicity bins they number 0 below −2, 1 at
−2 to −1.5, 9 at −1.5 to −1, 72 at −1 to −0.5, 124 at −0.5 to 0 and
18 at 0 to 1. The lone star below −1.5 lies at
`0.909≤(g−r)₀<1.2`; the blue-side metal-poor high-S/N intersection is zero.

Development fields contain 72 high-S/N core objects, including the sole
`[Fe/H]<−1.5` object. Three reserved fields contain 152 high-S/N core
objects and zero below −1.5. The available high-S/N objects could support
descriptive local color–luminosity checks near ordinary abundance, but they
cannot separately constrain a blue metal-poor zero point, a slope across
the blue sequence and metallicity dependence with an independent validation
sample. No new flexible blue distance model is fit or released.

Across all classes in the clean low-dust domain, 1,534 objects have Gaia
corrected S/N≥20, 2,662 have 10–20, 15,726 have 0–10 and 3,076 have
negative corrected parallax. These layers are not interchangeable distance
labels. The spectroscopic sample is selected by DESI target priorities, so
its class counts do not estimate photometric-catalog contamination fractions.
One development field (RA330–340) has zero unambiguous matched DESI records
in its prepared diagnostic input; this describes current label coverage.

## K giant and cool RGB branch

The same sample contains 59 conservative 4200–5000 K cool giants, 32 with
clean low-dust DR11 photometry, and none with Gaia corrected S/N≥20. Their
RVS `T_eff`, `logg` and calibrated abundance can support class audits, but
these fields alone do not supply a direct geometric luminosity scale for
them. The NERSC source register includes [DESI SpecDis](data_sources.md) and
SPDist for later spectral-distance comparisons. SpecDis is Gaia-parallax
trained and reports external cluster/stream validation; SPDist uses Gaia
photometry and DESI parameters. Neither is independent Gaia geometric truth
for a Gaia-based calibration, and both need an identity-verified join and
quality audit before use.

[Conroy et al. (2018)](https://arxiv.org/abs/1805.05954) demonstrate useful
PS1 `grz` + WISE `W1W2` K-giant selection without `u`, with about 30%
contamination in their selected sample. Its PS1 color cuts need validation
in Legacy/DECam photometry and do not set a per-star distance precision.
The next scientific branch should seek a targeted, quality-controlled giant
luminosity calibration (Gaia high-S/N nearby giants, cluster membership or
independently checked spectral distances) while retaining class/metallicity
uncertainty for no-spectrum application targets. DESI BHB distances are a
different stellar-type resource, not a cool-RGB calibration.

## DESI-first targeted expansion

An allocated scan of the PRIMARY DESI MWS projected pool ranks complete
DR11 10°×10° cells using qualified 4200–5000 K stars with
`logg−σ_logg>4.2` and calibrated `[Fe/H]<−1.5`. The full ranking and
selection details are in
[`manifests/desi_target_field_ranking.json`](../manifests/desi_target_field_ranking.json).
There are 5,173 such DESI targets with Gaia IDs and −40≤Dec<35; 1,949
also meet |b|≥30 and MWS `EBV<0.03`, including 137 with *raw MWS Gaia*
parallax S/N≥20. These are DESI catalog counts, rather than DR11
clean-photometry joint anchors. Four available sweep cells were registered in
[`configs/joint_targeted_fields.json`](../configs/joint_targeted_fields.json):

| Field RA / Dec | Role | Low-dust, high-lat DESI poor-core targets | Raw Gaia S/N≥20 among them |
|---|---|---:|---:|
| 180–190 / 0–10 | Development | 115 | 8 |
| 180–190 / −10–0 | Development | 89 | 8 |
| 190–200 / 20–30 | Reserved validation | 120 | 6 |
| 20–30 / 0–10 | Reserved validation | 23 | 4 |

Jobs 60485082/60485094 read the actual DR11 sweep photometry, matched it to
DESI and Gaia, and applied the same clean-photometry, low-dust, class and
corrected-parallax definitions as the seven-field study. The roles were
fixed before those outcomes were inspected. The
[targeted manifest](../manifests/joint_calibration_targeted_v1.json) contains
the four field and role cross-tabulations; the full 132,530-row FITS stays
at NERSC `data/joint_calibration_targeted_v1/same_star.fits`.
The [targeted figure](../figures/joint_calibration/02_targeted_color_metallicity_luminosity.png)
shows the color–abundance and color–luminosity overlap. It plots the 9,702
clean core and 1,179 high-S/N core objects with valid calibrated abundance;
the table below includes seven and one additional objects, respectively,
without such an abundance.

| Same-star quantity in four targeted fields | Count |
|---|---:|
| Unambiguous DESI–Legacy matches | 132,530 |
| Conservative 4200–5000 K main-sequence core | 12,943 |
| Core with clean low-dust `grz` | 9,709 |
| Core with clean photometry and corrected Gaia S/N≥20 | 1,180 |
| High-S/N core with calibrated `[Fe/H]<−1.5` | **18** |
| Above, blue of frozen `(g−r)₀=0.909` edge | **5** |
| Conservative cool giants / clean low-dust / high-S/N | 230 / 103 / **0** |

The 18 metal-poor high-S/N anchors split 11 development / 7 reserved
validation; the five blue-side anchors split **4 / 1**. One development
blue-side object at `(g−r)₀=0.351` has `(r−z)₀=0.923`, Gaia
`BP−RP=1.303` and DESI `T_eff=4331 K`. Its optical colors are discordant
and its photometry merits individual review before any fit. The other four
blue-side anchors span only `(g−r)₀=0.728–0.832`.

The [individual-anchor audit](../manifests/joint_targeted_poor_anchors.json)
also records the low-metallicity clean core's parallax layers:

| Color domain | S/N≥20 | S/N 10–20 | S/N 0–10 | Negative | Unusable |
|---|---:|---:|---:|---:|---:|
| All colors | 18 | 42 | 228 | 3 | 8 |
| `(g−r)₀<0.909` | 5 | 17 | 127 | 1 | 2 |

These weaker and negative measurements retain raw parallax and uncertainty
for a later forward-likelihood analysis. Their inverse parallax is not used
as a direct luminosity label. The high-S/N overlap now supports an initial
descriptive color–luminosity check, but four development and one reserved
blue metal-poor anchors cannot separately validate a blue-branch zero point,
color slope and metallicity term. No blue distance model is fit or released.

Across these targeted fields, 230 conservative cool giants have DESI class
evidence, 103 have clean low-dust optical photometry, and none has corrected
Gaia S/N≥20. The next branch is therefore a **giant classification and
luminosity-label study**: target bright nearby Gaia giants, cluster members
and independently checked spectroscopic-distance resources, with an identity
and quality audit before any calibration. The DESI-targeted cool-main-sequence
fields alone do not provide that geometric giant anchor set.

## Decision

Retain frozen v4 as the conditional cold-dwarf distance baseline and the
XD result as a research comparison. The targeted sample improves low-metallicity
same-star overlap from one to 18 high-S/N anchors, while preserving 42
intermediate, 228 weak and three negative-parallax low-metallicity core
objects. The five blue-side anchors include one discordant-color object and
only one reserved-validation object, so the current data do not support a
new calibrated blue metal-poor distance relation. Continue with a targeted
giant label and luminosity resource audit and a forward-parallax method for
the metal-poor weak-parallax population. Keep their physical branches and
validation identities separate from the frozen v4 result.
