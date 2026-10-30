# photdis

Photometric stellar distances from Legacy Survey DR11 `grzW1W2`, following the
GPT Pro restart plan in conversation
`6ae164f5-1a74-83e9-a2c0-768b28e09d1b`.

Latest RGB result: [independent NGC 4147 test](docs/cluster_rgb_ngc4147_result_v1.md)
of the frozen lower-RGB candidate is complete. The 42-star same-band test
has median modulus residual −0.205 mag and NMAD 0.187 mag. It retains
color information over a uniform luminosity prior, while shared reference
scales and abundance priors substantially affect the distance center.
The report includes fixed-model controls and matched whole-cluster simulations.

The bounded [four-cluster external-scale experiment](docs/cluster_rgb_scales_protocol_v1.md)
uses the existing members and frozen curves. Its optional passband integration
uses project-local `speclite==0.20` (install with `python -m pip install --no-deps --target .deps speclite==0.20` in the allocated DESI environment); core likelihood tests need only
NumPy/SciPy. Submit from the NERSC checkout:

```bash
sbatch scripts/run_cluster_rgb_scales_v1.sh response
sbatch scripts/run_cluster_rgb_scales_v1.sh joint --cluster ngc4147
# Repeat joint for m3, m53, ngc5466; all large outputs stay in data/cluster_rgb_scales_v1.
# Within an allocated environment after all four finish:
python scripts/cluster_rgb_scales_summarize.py
```

Current state: the 10°×10° pilot, [photometry audit](docs/pilot_audit.md),
[cold-main-sequence baseline](docs/cold_baseline.md),
[external/depth follow-up](docs/followup_validation.md), and
[second-round final-field validation](docs/pro_followup_v2.md) are complete.
The separately trained [cold-v4 route](docs/cold_v4.md) has expanded strict
Gaia anchors and three registered strict-anchor final fields. Its frozen
forests were also evaluated under an independently chosen clean WISE20 route
and [measured-r deep calibration/final test](docs/cold_v4_application_decision.md).
In the independent deep field the calibrated 95% final-distance-modulus
interval covers 92.6%/92.9%/95.1%/92.3% at measured `r=18–19/19–20/20–21/21–22`;
the 68% interval covers 62.6%–65.2%. These intervals are not released as
nominal-coverage intervals. Median distances are cold-dwarf-hypothesis
estimates, with type and support flags; type probabilities are uncalibrated.
The current diagnostic round includes the [extinction audit](docs/extinction_audit.md),
[selection-aware weak-parallax check](docs/cold_v4_weak_parallax_v2.md),
[WISE20 wide-binary check](docs/cold_v4_system_validation_v3.md),
[DESI type-boundary audit](docs/cold_v4_type_boundary.md), and
[template/noise-field crossover](docs/cold_v4_crossover.md), and its
[Gaia-template census](docs/cold_v4_native_population.md). The
[registered native-transfer test](docs/cold_v4_native_transfer_final.md)
finds spatially varying center residuals under the frozen route; no global
center correction has been fitted. These diagnostics have not changed the
frozen distance route. DR11 `MW_TRANSMISSION` is applied to all five input
bands and the final `g0` distance-modulus term; see the
[extinction audit](docs/extinction_audit.md). A bounded
[dust-sensitivity check](docs/cold_v4_dust_sensitivity.md) shows that
uniformly scaling down this correction does not remove the native
north–south center contrast. A registered
[PS1 relative optical audit](docs/cold_v4_ps1_relative_audit.md) finds a
field-dependent g-band difference on a restricted matched subset and
quantifies how the frozen Legacy-trained route responds to replacing grz.
The [latest Pro mechanism review](docs/pro_round_v5.md) corrects the
crossover and weak-parallax statistics, compares A/B/C on identical stars,
decomposes PS1 optical-band response, and audits WISE fallback reasons.
The current choice is to retain the frozen v4 distance and expose a
quality-qualified infrared support-conflict type-risk flag. It is not a
calibrated class probability or a hard exclusion rule.
The bounded [cold-v5 XD comparison](docs/cold_v5_xd_result.md) fitted
error-deconvolved Gaussian mixtures on the frozen training identities;
the registered same-star development gate rejected the selected K=32 model
because its g/r response was larger than v4. The v4 route remains the
distance baseline. A separate [blue-side inventory](docs/cold_v5_blue_branch.md)
defines the metal-poor branch's label and anchor requirements; no blue
distance product has been released. A subsequent
[metal-poor forward-parallax experiment](docs/metal_poor_forward_v1.md)
found abundance information beyond the two optical colors in a bounded
spectroscopic sample; it is not a purely photometric blue-star model. The
[candidate export report](docs/cold_v5_candidate_export.md)
summarizes three field products; the export
(`scripts/cold_v4_candidate_export.py`) writes per-object v4 tier, fallback
reason, support ratio, conflict and depth state to ignored NERSC FITS files.
Versioned v2 files add existing DESI class evidence while preserving all
frozen distance columns. A bounded
[DESI–Gaia–Legacy same-star inventory](docs/joint_calibration_v1.md) finds
only one calibrated `[Fe/H]<−1.5` star among 224 conservative high-S/N
cool-main-sequence anchors and none in the blue-side metal-poor intersection.
Four DESI-ranked DR11 fields raise the low-metallicity high-S/N overlap to
18, with only five blue of the old color edge and one in a reserved
validation field. The targeted sample retains weak and negative parallaxes
for a future forward-likelihood analysis. K giant/cool RGB luminosity
calibration now uses independent globular-cluster distances and whole-member
Legacy photometry. The first pooled RGB model was rejected; see the
[native-system audit](docs/cluster_rgb_native_system_v2.md). A registered
[DECam conditional RGB experiment](docs/cluster_rgb_conditional_result_v1.md)
tests a frozen forward color model on NGC 5466. Its grz lower-RGB transfer
improves over gr on the same stars, but strong magnitude-dependent residuals
and broad intervals prevent adoption as a field giant-distance product.
The [lower-RGB controls](docs/cluster_rgb_lower_result_v2.md) subsequently
corrected training-domain leakage: restricting to M3/M53 lower support
removes quadratic reversals, and same-star grz scatter improves substantially.
NGC 5466 is now an exposed diagnostic; a new independently registered system
is the next validation step.

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
Astropy, healpy, SciPy, Matplotlib, scikit-learn and joblib.
`gaiadr3-zeropoint==0.1.0` is installed
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
anchors using validated Gaia BP/RP measurements and Gaia absolute G magnitude.

The audited baseline reads `data/pilot_audit_v1/reference.fits` on NERSC:

```bash
python scripts/cold_baseline.py
```

It writes models and per-star test predictions to `data/cold_baseline_v3/`,
aggregate metrics to `manifests/cold_baseline.json`, and plots to
`figures/cold_baseline/`. The pilot-only spatial test map is frozen in the
manifest. The bounded model uses only photometric colors as predictors.

The second-round validation uses `scripts/noise_injection_v2.py`,
`scripts/conditional_calibration.py` and `scripts/type_diagnostics.py` on
allocated NERSC CPUs. Their exact field roles and small result manifests are
linked from [the report](docs/pro_followup_v2.md). The separate
`scripts/cold_candidate_demo.py` exports up to 2,000 sampled rows by default from
independent-field color-box candidates to an ignored NERSC `data/` FITS file.
It reports frozen-model median distances under the cold-dwarf hypothesis and
support, optical-quality, depth and unverified-type flags; it assigns no type
probability or calibrated interval. The support flag covers the two optical
colors only. Run in an allocated CPU environment after
loading the DESI Python environment:

```bash
python scripts/cold_candidate_demo.py --max-rows 2000
```

`scripts/selected_tier_validation.py` checks the fixed color-box, two-color
support and highest-available A/B/C route on all four registered final fields.
Its metrics are in `manifests/selected_tier_validation_v1.json`; the original
forest intervals do not yet achieve 95% coverage under this complete route.

The separately versioned cold-v4 workflow is registered in
`configs/cold_v4_fields.json`. It first fits expanded strict-anchor A/B/C
forests and calibrates the complete C→B→A route in the RA205–215 field, then
checks three new final fields. Run these inside an allocated NERSC CPU step
with the DESI environment and `PYTHONPATH` above:

```bash
python scripts/cold_v4.py prepare
python scripts/cold_v4.py final
python scripts/cold_v4_noise.py
```

The preparation, final-field and shared-noise small manifests are linked in
the [cold-v4 report](docs/cold_v4.md). Forests, route state and per-star
products stay under ignored `data/cold_v4/` on NERSC. Each final/noise run
checks the published preparation and model hashes before reading its inputs.

The later application audit and measured-depth study use
`configs/cold_v4_deep_fields.json` and the locked policy in
`configs/cold_v4_deep_policy.json`. On allocated NERSC CPUs, the bounded
calibration/final runner is `scripts/run_cold_v4_deep.sh` with `calibrate`
or `final`; it refuses to overwrite existing outputs and the final run checks
the tracked calibration hash. See the [application decision](docs/cold_v4_application_decision.md)
before interpreting candidate distances.

The latest native-transfer diagnostics use the registered field and role
configs under `configs/cold_v4_native_transfer_*`. On allocated NERSC CPUs,
`scripts/cold_v4_native_transfer.py` applies the unchanged trained route;
`scripts/run_cold_v4_sky_census_fields.sh` builds six additional diagnostic
fields, and `scripts/run_cold_v4_sky_census_score.sh` scores them after all
field jobs succeed. Their source catalogs, per-object diagnostics and models
remain under ignored NERSC `data/`; only small aggregate results and selected
figures are synchronized. The sky census is for cause analysis, not tuning or
new independent validation of a correction.

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

Local regression checks use `python -m pytest -q` with
NumPy, fitsio, healpy and SciPy available. They exercise synthetic measurements
and identities; only the allocated NERSC run produces scientific sample counts.
