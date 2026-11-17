# Photdis v1 evidence and staged data access

The scientific release remains `3-final.1`. Its production FITS and native
PDF partitions use the original schema and SHA256 checks. The new
`core_observable_support.fits` sidecar is a keyed diagnostic. It adds one
row per core LS_ID, the original science tier, an observable cell ID, the
count of qualifying paired F15 dwarfs for each reference, and flags for
at least 20 pairs. The fields indicate observable overlap on the chosen
bins. Each source keeps its original distance and qualification. A source
with both flags false can have other independent evidence, as is common
in the separately validated nearby tier.

| NERSC path relative to checkout | Product |
|---|---|
| `data/v3_final/release/photdis-dr11-core-v3-final.1.fits` | 2,281,023 core PSFs with tier, status and zero-based PDF row |
| `data/v3_final/release/pdf_partitions.csv` | Full 43,347,612 owned posteriors by original partition |
| `data/v3_final/release/ownership_overrides.npz` | Verified seam owner keys |
| `data/paper_v1_evidence/benchmark/core_observable_support.fits` | Coverage sidecar keyed by LS_ID |
| `data/paper_v1_evidence/benchmark/support_schema.json` | Decoding of the spectral-overlap cells and their axes |
| `data/v3_9/existing/pdf/` | Calibrated near-width companion with original medians and mass |

The [portable reader](../scripts/paper_v1_read.py) requires Python 3.11+,
NumPy and Astropy. It can read summaries from the local pair of FITS
files. Supplying an archive root enables exact PDF retrieval and integrity
checks. Keep the tree paths from `pdf_partitions.csv`.

```sh
cd /pscratch/sd/y/yunao/photdis_pipeline
source /global/common/software/desi/desi_environment.sh main
python scripts/paper_v1_read.py \
  --package data/v3_final/release \
  --support data/paper_v1_evidence/benchmark \
  --archive "$PWD" \
  --ls-id 12105994119152426
```

For bulk release and calibrated companion rules, use the frozen
[release instructions](v3_final_release.md) and schema. The same LS_ID
joins the products. The bright-red calibration companion has its own
admission and a width transform. The new coverage sidecar annotates the
original FITS. It does not substitute calibrated widths.

Large products are staged on NERSC. Public hosting, a persistent
identifier, and external download checks follow the author-authorized
release. A publication should link the versioned core FITS, support
sidecar, schema, index, original PDFs and calibrated companion together.
