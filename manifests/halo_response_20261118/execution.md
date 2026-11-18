# Execution and reproducibility

- Baseline: `7f9fcc787e06b5472797173ab1f4015d1f3e5dd3`.
- Frozen production: `8706940843975e9f9af657ea298aa5b7c94f64bc`.
- Registered config and template metadata are under `configs/halo_*`.
- NERSC checkout: `/pscratch/sd/y/yunao/photdis_pipeline`.
- Private outputs: `data/halo_response_20261118/`.
- Actual CPU allocations:61518416(metadata),61518711(cache/areas/native),
  61519139(64CPUs,192GiB, nid200023: counts and scientific controls).
- Python: DESI desiconda20260910-3.1.0; single BLAS/OpenMP thread per
  independent subprocess. Cache/count work uses8 parallel subprocesses;
  harmonic transforms use6 independent workers on allocated CPUs.

Full rebuild stages, each inside a CPU allocation using
`PYTHONPATH=scripts:src:.deps:.deps/v39_dust`:

```bash
python scripts/halo_cache.py prepare
python scripts/halo_driver.py
python scripts/halo_recover.py
python scripts/halo_native.py
python scripts/halo_dedup.py
# independent shards, numbers0..31, max8 concurrent:
python scripts/halo_count.py 0
python scripts/halo_atlas.py
python scripts/halo_stages.py roles power noise hess seam sequence --workers 4
python scripts/halo_stages.py noise_imaging known balance --workers 3
python scripts/halo_stages.py balance_compare population sequence_views hess_views --workers 4
python scripts/halo_pack.py
python scripts/halo_verify.py
```

`roles` and `noise` precede `balance` and `noise_imaging`; `sequence`
precedes `sequence_views`. The stage dispatcher only groups independent
stages when their input markers already exist. For an entirely new
rebuild respect these dependencies. Batch wrapper `scripts/run_halo.sh`
accepts a stage name and stage arguments; request CPUs/memory explicitly
with `sbatch -c64 --mem=192G -t03:00:00 scripts/run_halo.sh <stage>`.
The login node is used only for version transfer and small metadata reads.

Each `complete.json` binds source receipts, config, actual code hashes,
style lock, output checksums and actual Slurm job ID. Source/PSF and original
native ownership are checked at admission and export. Admission tables log
missingTYPE, nonPSF, discarded photometry/low-latitude rows and recovered
QRF-unsupported PSFs. Original weak-flux/failure history remains frozen.

Recovery records:

- NumPy private array-header API had changed; public readers were used.
- Shared interactive64CPU/256GiB exceeded its half-node memory equivalent;
  allocation was requested with192GiB instead.
- An initial cache smoke detected a task-order bright/faint flag problem;
  cache shards were rebuilt with version/config-aware resume checks.
- Continuous polynomial propensity columns needed scaling; the ill-conditioned
  first optimizer was stopped, its log preserved, and all final fits use
  standardized columns with a positive ridge and convergence assertion.
- Coordinate validation initially compared equivalent negative/wrappedRA;
  normalizedRA is used in the Galactic-center test. The plot-grid rotation
  also required flattened coordinate arrays. Its targeted regression now
  checks the full1440×720 projection grid. The source counts were preserved.
- Full raw FITS cube was checksum-verified and count/PDF conservation rechecked
  before resuming the plot stage. Final visual polish uses `--render-only`
  on the saved physical maps, avoiding another source scan or new smoothing.
- Primary sequence results separate unmasked rows; combined low-crowding
  first tables are archived in `sequence/revision_01`.
- Scientific tests exercise declared noise variance, preselection branch
  conditioning, overlapping CMD nodes, common-support refits and spherical
  coordinates. Original v3.8F02/F03 pixel regression is independent of
  whether a scientific model passes its scale criteria.

The exact raw-cube location/checksum and array-axis/units index are in
`redraw/private_cube.json`. `redraw/compression.csv` verifies lossless
compression of every plotted-grid array. Figure-side CSVs retain color
scales, physical units and displayed/clipped counts. The repository's
large task inventory is compressed losslessly; receipts reference its
uncompressed private original. No production or manuscript file was rewritten.
