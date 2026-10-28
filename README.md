# photdis

Photometric stellar distances from Legacy Survey DR11 `grzW1W2`, following the
GPT Pro restart plan in conversation
`6ae164f5-1a74-83e9-a2c0-768b28e09d1b`.

Current state: the 10°×10° pilot is complete. See the
[per-step counts and figures](docs/pilot_10deg.md). Waiting for user review.

## Locations

- Local repository: `/Users/yunaoxiao/Desktop/photdis`
- GitHub: https://github.com/xiaoyunao/photdis (public)
- NERSC working checkout: `/pscratch/sd/y/yunao/photdis_pipeline`
- NERSC `/pscratch/sd/y/yunao/photodis` is intentionally empty.
- Source catalogs remain in their shared NERSC locations; see
  [data sources](docs/data_sources.md).

## Environment and operation

Synchronization helpers use Python 3 standard library, Git and SSH.
NERSC login: `ssh yunao@perlmutter.nersc.gov`; renew expired authentication with
`sshproxy -u yunao`. Small metadata/few-row reads may run on login nodes.
Full scans, crossmatches and training must use allocated compute resources.

Stage 1 uses the NERSC DESI environment (`source
/global/common/software/desi/desi_environment.sh main`) for NumPy, fitsio,
Astropy, healpy, SciPy and Matplotlib. `gaiadr3-zeropoint==0.1.0` is installed
without dependencies into ignored `.deps/` for the published Gaia parallax
zero-point correction. Export `PYTHONPATH=$PWD/src:$PWD/.deps` when running.

See [synchronization](docs/sync.md) for exact commands and recovery procedures.
Programs, configuration, documentation and selected figures are versioned.
Catalogs, generated sample tables, models, caches and logs stay out of Git.

## Continuity

Read Git history, [WORKLOG.md](WORKLOG.md), then [PLAN.md](PLAN.md) when resuming.
The previous project is preserved outside the active directories; see
[archive locations](docs/archives.json). Its last commit was
`f846aee9ce0ce39a5b28f48801b77b29506a0ff1`; the new repository has a fresh history.

## Current 10°×10° pilot

Field: RA [0,10) deg, Dec [-10,0) deg; four explicit sweep files in
[configs/pilot_10deg.json](configs/pilot_10deg.json). In an allocated NERSC step:

```bash
python scripts/pilot_field.py --config configs/pilot_10deg.json
```

Outputs: `data/pilot_10deg_v1/field_diagnostics.fits`, the per-step
`summary.json`, `manifests/pilot_10deg.json` and `figures/pilot_10deg/`.
The Gaia CMD compares the astrometric sample, measured-grz subset and final
anchors using Gaia BP-RP and Gaia absolute G magnitude.

## Stage 1 operation

Read [sample definitions](docs/stage1_selection.md) before interpreting outputs.
The fixed configuration is [stage1.json](configs/stage1.json); data products live
under `data/stage1_v1/` on NERSC. `parent` is the retained audit pool, not every
optical object. Roles overlap; always select a `partition` before future fitting.

Inside an allocated CPU environment on NERSC, from the checkout:

```bash
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/src:$PWD/.deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# For a fresh run only. Published reference files listed in the reference
# manifest must already be downloaded to data/reference under their exact names.
python -m photdis.holdouts prepare --reference-dir data/reference
python -m photdis.holdouts validate --reference-dir data/reference
python scripts/prepare_stage1.py

# Scan resumes only matching code/config/reference/input signatures.
python scripts/build_stage1.py scan --workers 32
python scripts/build_stage1.py finalize
python scripts/validate_stage1.py --run-dir data/stage1_v1
python scripts/plot_stage1.py --run-dir data/stage1_v1 \
  --output-dir figures/stage1 --manifest-path manifests/stage1_plot_manifest.json
```

Spectroscopy preparation and table finalization refuse to overwrite existing products. A changed
scientific configuration or processing module requires a new output directory.
`--limit` on scan is only a recorded I/O smoke test, never the final coverage run.
Normalized references and their source hashes are listed in
[the reference manifest](manifests/holdout_reference_manifest.json).

Local regression checks use `python3 -m unittest discover -s tests -v` with
NumPy, fitsio, healpy and SciPy available. They exercise synthetic measurements
and identities; only the allocated NERSC run produces scientific sample counts.
