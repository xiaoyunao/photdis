# Execution and provenance

## Fixed sources

- Scientific prompt read in full from `Codex_science_improvement_prompt_20261118.md`.
- Complete review read from thread 6afd1691-5264-83e9-8e2e-a4c4f1c91ec8,
  reply a05e3c54-5d9a-5dc5-bd91-f038018fe74f.
- Evidence commit f4dc28f57aec954644dcb4f0e76275292d01afc3.
- Production commit 8706940843975e9f9af657ea298aa5b7c94f64bc.
- Registered protocol `configs/science_validation_20261118.json`.
- Class control `configs/science_validation_20261118_oracle.json`.
- Integration refinement `configs/science_validation_20261118_numerical.json`.
- Canonical NERSC checkout `/pscratch/sd/y/yunao/photdis_pipeline`.
- Experiment data `data/science_validation_20261118` in that checkout.

Each stage's complete.json records input hashes, its configuration hash,
all sv18 source hashes as seen by that process, the figure-style lock and
output hashes. Source changes after a completed stage have distinct receipts.
The transfer receipt lists every locally selected output with a SHA-256.
The delivery receipt binds all final remote stage receipts.

## Reproduction

Run after SSH login to NERSC with a valid proxy certificate. This sequence
uses an allocated CPU node and the canonical checkout. The existing archived
input catalogues, models and caches are dependencies.

```bash
cd /pscratch/sd/y/yunao/photdis_pipeline
salloc -N1 -n1 -c32 -C cpu -q shared_interactive --mem=128G -t 01:00:00
source /global/common/software/desi/desi_environment.sh main
export PYTHONPATH="$PWD/scripts:$PWD/src:$PWD/.deps:$PWD/.deps/v39_dust"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for stage in definitions observable filter benchmark noise systems predictive provenance cohorts numerics populations oracle figures style
do
    srun -n1 -c32 python "scripts/sv18_${stage}.py" || exit 1
done
srun -n1 -c32 python scripts/sv18_populations.py --exclude-calibration
srun -n1 -c32 python scripts/sv18_verify.py
exit
```

Individual experiments can also use `sbatch --cpus-per-task=32 --mem=128G
--time=00:30:00 scripts/run_sv18.sh STAGE`. `filter_complete` builds the
observable response before filtering. `sv18_closeout.py` runs filter,
integration and oracle work concurrently before final figures, pixel
regression and delivery checks. It expects the earlier P1 stage outputs.
The full sequential recipe above shows all dependencies explicitly.

A reproduction writes into the experiment output root. Preserve an existing
experiment root under a new sibling name before a deliberate full replay.
The original production catalogue and prior experiment roots are input paths.

## Successful runs and recoveries

| Job | Work |
|---|---|
| 61514293 | Exact Gaia-definition and selected-count replay after cached bin optimization |
| 61515493 | Independent benchmark, noise, systems and observation-space stages; initial staged-wrapper and rematch failures retained in execution logs |
| 61516156 | Full-role provenance, geographic cohort differences and calibration-excluded rematching |
| 61516277 | Canonical observable response, initial canonical filter, 128-draw numerical refinement, class oracle and training/PDF figures |
| 61516620 | Final filter with finite-support profile normalization and complete numerical delivery checks |
| 61516715 | Same-star reference/training CMD addition, known-system figure cleanup and checks; a direct regression call omitted the required STYLE context and failed |
| 61516786 | Dedicated original-context pixel regression and delivery verification |
| 61516883 | Fixed system color identity, exterior-point counts and final verification |

The first definitions job repeated large-array binning unnecessarily. The
replacement computes bins once and reproduces all old aggregate rows.
A new rematching helper reused a sort-index name inside a cell, corrupting
subsequent cells. Separate row and residual orders now reproduce both old
metallicity anchors and pass a multi-cell regression.
Historical script imports prepend their own source path. An isolated staged
wrapper consequently imported canonical older siblings. Version-bound sibling
imports were repaired, then the committed code was fast-forwarded to the
canonical checkout for the final executions.
The filter's equal-grid CDF needed half-bin edges. Nodes with no observable
template then exposed a NaN profile normalization. They now carry zero profile
mass, with finite empirical bounds verified. All earlier filter versions are
archived in sibling directories on NERSC.

## Figure and test checks

PDFs are inspected at their physical two-column width. PNGs are 300 dpi.
The original v3.8 F02/F03 regression compares the PNG pixels under STYLE.
New science plates use the existing Pal5 palette, axes and export functions
with the established compact paper-font context. The style lock is
`configs/v3_10_figure_style.json`.

Local validation runs syntax checks, the targeted numerical regressions,
and the existing test suite with plugin auto-loading disabled.

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 PYTHONPATH=src:scripts:.deps python3 -m pytest -q
```

The final local receipt records the test count, inspected plates and export
hashes. Failed scientific criteria are recorded as measured outcomes in the
filter receipt. Numerical delivery checks accept a correctly measured failure.
