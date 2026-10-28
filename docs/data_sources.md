# NERSC data sources

Verified 2026-10-28 through representative headers and small record inspections.
This is a source register, not a constructed sample. Row counts and field
availability do not establish usable overlap or scientific quality.

| Source | NERSC path | Format / use |
|---|---|---|
| Legacy DR11 south 11.0 | `/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0/` | FITS SWEEP; grzW1W2 flux/ivar/transmission, EBV, quality and reference IDs |
| Gaia DR3 full | `/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix/` | `healpix-{pixel:05d}.fits`, GAIAHPX, NSIDE=32/NESTED; parallax/errors/quality/zero-point inputs |
| DESI DR1 MWS | `/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws/iron/v1.0/mwsall-pix-iron.fits` | RVTAB/SPTAB/FIBERMAP/SCORES/GAIA; parameters, errors, quality and identities |
| APOGEE DR17 | `/global/cfs/cdirs/sdss/data/sdss/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits` | HDU 1: TEFF/LOGG/FE_H, errors, quality, Gaia EDR3 ID |
| SEGUE/SSPP | `/global/cfs/cdirs/sdss/data/sdss/dr17/sdss/sspp/ssppOut-dr12.fits` | Alias to DR9 SSPP, not a new DR17 release; ADOP parameters/errors, SNR/flags, coordinates/observation IDs |
| DESI BHB | `/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-bhb/v1.0/MWS_BHB.fits` | RVTAB/FIBERMAP/GAIA; DIST in kpc |
| SpecDis v2.1 | `/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-specdis/v2.1/iron-yr1-v2.1.fits` | SPECDIS; DIST/DISTERR in kpc, IDs and binary diagnostics |
| SPDist v1.0 | `/global/cfs/cdirs/desi/public/dr1/vac/dr1/mws-spdist/v1.0/mws-iron-SPDist-240613.fits` | SPDIST; Gaia G luminosity quantiles and quality flags |

MWS embedded photometry includes Legacy RELEASE=9010. Join to DR11 for the new
photometric experiment. Check catalog-specific missing/sentinel values,
calibration objects and duplicate spectra. Do not join catalogs by row number.
Full Gaia is required; the lightweight Gaia product lacks needed label fields.

## Spectroscopic-distance VAC assessment

The likely catalog recalled as Wenting Wang's work is **SpecDis**:
Songting Li et al. (2025), Wenting Wang corresponding author,
[AJ 170:171 / arXiv:2503.02291](https://arxiv.org/abs/2503.02291),
DOI 10.3847/1538-3881/adf1a0. It predicts luminosity from PCA-compressed spectra
using Gaia-parallax supervision. Use **v2.1**; the
[official release](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-specdis/)
says it fixes a v2.0 crossmatch error affecting about 1% of objects.

Register it for later auxiliary distant-star/giant comparisons and possible
calibration studies. Do not automatically use it as precise training truth:
its Gaia calibration makes it correlated with Gaia-based labels, and individual
train/test membership is not provided. It is not independent validation of Gaia.
The catalog lacks LOGG/FEH/SNR/RVS_WARN, so join MWS by verified identities and
apply appropriate quality filters before scientific use.

The paper's giant analysis reports typical quoted errors of 58%, 38%, 16% and 8%
for spectral S/N bins <20, 20–60, 60–100 and >=100 respectively
([section IV.4](https://arxiv.org/html/2503.02291v3#S4.SS4)).
These are published sample-dependent results, not our project's performance.
`MG_NN` is **Gaia G**, not **Legacy g**. A later Legacy absolute-magnitude label
must combine the distance with matched, corrected Legacy photometry.

**SPDist is distinct**: spectral parameters plus Gaia photometry, with contact
Guillaume Thomas. The
[official documentation](https://data.desi.lbl.gov/doc/releases/dr1/vac/mws-spdist/)
recommends MG_50, FLAG_GOOD/QFLAG_INPUT and MWS quality checks, adding 0.167 mag
intrinsic scatter to the quantile uncertainty. Its current 6,272,510 rows differ
from the MWS file's 6,372,607; verify identity joins rather than positional rows.

Neither catalog has been copied, scanned into a sample or used for training.
LAMOST and independent cluster/wide-binary test resources remain unverified.
