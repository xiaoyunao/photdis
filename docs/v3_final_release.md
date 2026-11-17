# Photdis DR11 catalogue release staging

The v3 science release uses the frozen native DR11 parents and QRF predictions
audited at `c530007c`. The release version is `3-final.1`. The data are staged
in `/pscratch/sd/y/yunao/photdis_pipeline/data/v3_final/release`. The core FITS is 251 MiB. The index lists 28,917 partitions and 28,890
native PDF files. [Release receipt](../manifests/v3_final/release/complete.json) and
[reference lookups](../manifests/v3_final/release/pdf_access_examples.csv)
record the staged files and verification. Public hosting, identifier
registration and a public access test remain release operations.

## Files

| File | Contents |
|---|---|
| `photdis-dr11-core-v3-final.1.fits` | The 2,281,023 core sources, sorted by native LS_ID, with FITS HDU checksums |
| `pdf_partitions.csv` | Native partition identifiers, survey, component, original relative paths, SHA-256 digests and owned row counts |
| `ownership_overrides.npz` | Frozen north/south preference for verified seam identities |
| `schema.json` | Machine-readable source and product definitions |
| `pdf_access_examples.csv` | Tested examples from the four native components |
| `SHA256SUMS` | Digests for package files |

The core FITS contains coordinates, the native Gaia ID and association state,
observed `r`, dereddened `g-r`, reddening, distance quantiles and shell
probability integrals. The two science tiers describe the bright red nearby
sample and the 2–5 kpc dwarf-enhanced sample. Their classification scores use
different frozen calibrations. A preferred use keeps those tier codes during
analysis. The W1 state and nominal numerical flag describe the bright red
selection. For the extended tier, W1 state is −1 and the nominal SED numerical
flag is inapplicable. The forced grz state code on the frozen southern rows
identifies the original production sample. See the schema for definitions and
the release-readiness report for the optional calibrated bright-red width
companion. The FITS distance summary uses the original QRF posterior.

## Access the PDF of a core source

Read the native LS_ID from the FITS file. Its `PARTITION_ID` points to a row
in `pdf_partitions.csv`. Download the file in that row's `pdf_file` column,
preserving its relative path under the downloaded archive root. Verify its
`pdf_sha256`. Use the source's zero-based `PDF_ROW` in that NPZ and check
that `ls_id` and `TYPE` agree with the FITS row. The row stores interval edges
in distance modulus and their integrated probabilities.

```sh
PYTHONPATH=src:scripts:.deps python scripts/v3_final_release.py \
  --package /path/to/package --archive-root /path/to/pdf-archive --ls-id ID
```

For bulk retrieval of the full catalogue, iterate through partitions with
positive `owned_pdf_rows` and apply the native ownership rules with the
accompanying seam overrides. Native `photo_file` and, where present,
`state_file` are required for the existing verified reader
`release_readiness_access.get_partition`. File-internal rows can contain both
retained and displaced seam sources. The unfiltered NPZ row count can differ
from the owned count. Every retained source has `TYPE=PSF`. The full archive
contains 43,347,612 owned posteriors. Use the provided core IDs when joining an
analysis on the two science tiers. The higher-latitude faint extensions and
other supported sources span the research regime described in the scientific
report.

## Publication plan

Reserve a persistent record for `3-final.1` through the institutional data
repository at publication. Archive the FITS, schema, checksums, native index
and access examples under that record. Serve the original PDF, photo and state
partitions at an institutional download endpoint with the same relative paths.
Group optional bulk links by native component and sweep. Link the endpoint to
the record and pin the software commit, calibrated companion receipt, area
definition and selection criteria there. Confirm storage and bandwidth with
the host, test fresh downloads in each of the four components and verify the
checksums before listing public URLs in the publication.
