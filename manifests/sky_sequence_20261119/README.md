# All available-sky CMD display and native sequence science — 2026-11-19

This package executes the archived v6 prompt and review in `docs/v6_inputs`.
The frozen production distance catalog remains `87069408`. No manuscript,
production forest, source distance PDF or selection has been replaced.

## Display A

All **2,126 locally available native south/north sweeps** were read. The
geographic, extinction-ceiling and common-23-mag-depth restrictions were removed
from file enumeration, source admission and independent random geometry.
This is the available imaging footprint, not complete all-sky coverage.

| Stage | Sources |
|---|---:|
| Raw sweep records | 3,967,618,454 |
| TYPE=PSF | 2,221,816,974 |
| Native-owned unique PSFs | 2,198,481,902 |
| Valid gr, before positive-Gaia alias handling | 493,319,577 |
| Valid gr, final unique display parent | 493,314,135 |
| Accepted by at least one of the 61 fixed CMD nodes | 144,211,360 |

Non-PSF exclusions total 1,745,801,480; missing TYPE is zero. Exact native-ID
checks pass before any CMD selection. Repeated positive Gaia identities remove
5,442 rows using the frozen minimum-native-ID rule; unlinked sources retain
native identities. Node acceptances total 2,244,930,193: one source may contribute
to several overlapping CMD nodes. They are not independent stellar counts.

Source admission requires native ownership, verified PSF, observed gr,
finite positive flux/ivar/transmission, both S/N>=5, gr pixel bits
2/3/5/6/10 clear, FRACMASKED<0.4, FRACIN>0.7 and g0/r0<23.
The z band is not an admission condition. FRACFLUXgr<0.05 and ANYMASKgr=0
are a strict companion only. No distance support, dwarf score or spectral
classification enters this display. Catalog MW_TRANSMISSION supplies a **2D
full-column extinction correction**; its proximity/crowding limitations remain.

Independent native gr randoms give **29,386.5984 deg2** geometric area.
Random geometry uses real gr observations, finite positive PSFDEPTH and gr pixel
quality, without a common depth, latitude or extinction ceiling. This geometric
area does not correct source detection, morphology or deblending completeness.
Observed source evidence retains rare random-empty edge pixels with smoothed
area support. Missing sky is gray; observed zero CMD acceptance is zero.

| Overlapping domain | Valid gr | CMD union | Area (deg2) |
|---|---:|---:|---:|
| Low latitude, abs(b)<=30 deg | 344,211,981 | 95,510,304 | 9,128.0716 |
| High extinction, E(B-V)>=0.10 | 256,348,803 | 72,415,254 | 5,575.8248 |
| At least one formal dereddened gr depth below23 | 228,742,689 | 64,983,903 | 9,512.2932 |
| Old geographic/dust/depth domain, new display admission | 99,225,105 | 34,423,948 | 14,155.7048 |

These domains overlap and must not be added. The exact unchanged v5 display is
also shown separately; the old-domain control above has the new S/N>=5 entrance.

The fixed original MS/TO template, native north transfer, common error-width
curve and 61 nodes (mu14–20 by0.1) remain frozen. Overview channels are
B=10–15, G=15–26, R=26–100 kpc. Each of five wide intervals
6.3096–10,10–15,15–26,26–50,50–100 kpc has three equal-modulus subchannels,
with exact cell-overlap weights normalized within each channel. Thus physical
units are node-averaged CMD-compatible sources per square degree, not a histogram
of inferred single-star distances. Positive residuals subtract the 5-degree
background from the 0.3-degree smoothed density; signed arrays are retained.

RGBs retain the v5 layout/colors and Pal5/v3.8 base settings. The exact previous v5 per-channel brightness limits and asinh10 transform
are restored by the separately bound display settings; saturation is saved
separately from physical arrays. The earlier darker full-footprint percentile
display remains in private execution history. Scalar overview
companions retain physical colorbars. Galactic coordinates, longitude increasing
left, b=0/±20/±30, full actual galstreams Sagittarius/Pal5 tracks and named systems
are explicit. The unchanged previous RGB preserves its exact RGB cells/stretch.

`band/` compares native PSF/valid-gr/CMD rates, dust, latitude and actual Sgr
strips. The no-extinction overview uses the same admitted sources, geometry,
fixed CMD/error-curve prescription and corrected-overview stretch, changing only
the input CMD coordinates to observed magnitudes. It tests extinction sensitivity;
it is not a three-dimensional dust solution or a membership decomposition.

Large individual catalogs, node cube, full HEALPix physical/signed channels and
source caches remain at NERSC under
`/pscratch/sd/y/yunao/photdis_pipeline/data/sky_sequence_20261119`.
Compact redraw grids and small numerical tables are the public evidence.

## Reproduce

From the repository root, with numpy/scipy/astropy/healpy/matplotlib:

```bash
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/sky6_figures.py \
  --root manifests/sky_sequence_20261119 --output /tmp/photdis_v6_redraw
```

This command reads only public compact arrays and tracked reference/model files.
For the full raw extraction use `scripts/sky6_run.py` **inside an allocated NERSC
compute job**. Source code/configuration, exact input hashes, source-bound stage
receipts and execution logs preserve the actual scientific choices and attempts.

## Sequence science B

The independent native-ridge and same-member center/width results will be recorded
in `science_decisions.md` with the completed tables and actual figure audit.
No new empirical distance correction is fitted in this round.
