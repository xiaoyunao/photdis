# photdis

Active development: **v3.1 scientific calibration and use domains**, authorized
2026-11-13 from the [latest Pro review](docs/pro_review_31f1277.md).
[Four-package protocol](docs/v3_1_protocol.md) ·
[Integrated methods and estimands](docs/v3_1_methods.md) ·
[Integrated results — delivery in progress](docs/v3_1_result.md) ·
[Evidence and remaining-work ledger](docs/v3_1_requirements_result.md) ·
[Registered radial comparison](docs/v3_1_radial_protocol.md) ·
[Current evidence and status](docs/v3_1_status.md).
Implementation is isolated under `src/photdis/v3_1` and `scripts/v3_1_*.py`;
NERSC launch wrappers are `scripts/run_v3_1_*.sh`. Scientific execution requires
allocated compute resources. Inspect the current `PLAN.md`, execution manifests
and live SLURM jobs before submitting, to avoid duplicate writers. Final reference
association requires the complete authenticated prediction freeze; numerical
failures stay explicit and do not authorize retuning on final outcomes. The main
candidate is response12/calibration12 with the fixed-kernel, candidate-cohort
population and evidence-weighted luminosity bootstrap; the structure application
and 64-response probe have separately documented identities.
The frozen final validation does not support adopting the correction as a
calibrated distance product. Complete transfer, structure inference and final
domain adjudication remain required; numerical or artifact checks alone do not
establish scientific calibration. The integrated report explicitly marks pending
results, with the live execution state in `PLAN.md` and `WORKLOG.md`.

Latest completed research delivery: **Photdis v3.0 continuous SED and population system**.
All 1,183,990 qualified existing-field candidates, their full joint posterior links,
shared counts/covariance and 42 English scientific figure sets are delivered.
Final allocated audit 61163478 passes 1,309 actual-array/artifact checks; 73 focused
mathematical/unit tests and 336 figure format/hash/text checks pass. All plates
were visually inspected. Six original reference-quadrature flags and one Monte
Carlo flag remain, including within high-near-probability selections.

[Results and limitations](docs/v3_0_result.md) · [42-plate atlas](docs/v3_0_atlas.md) ·
[Method](docs/v3_0_methods.md) · [Requirement evidence](docs/v3_0_requirements_result.md) ·
[Candidate interface](docs/v3_0_catalog.md) · [Actual final audit](manifests/v3_0/delivery.json).
Large products remain at NERSC `/pscratch/sd/y/yunao/photdis_pipeline/data/v3_0/`.
The candidate posterior transfers the shared fit from 26,000 sampled targets;
it is not a full-catalog population refit or a calibrated Galactic census.
Poor-metal scales, evolved branches, interval undercoverage and narrow-structure
recovery remain scientific limitations. The deployed v1 catalog is unchanged.
The completed v3.0 scope has no remaining jobs. Authorized v3.1 research is
tracked separately above; no full-catalog rerun, monitoring or deployment is scheduled.
Use `python scripts/v3_0.py status` for small-manifest status; actual production
requires allocated NERSC compute resources through `run_v3_0_stage.sh`.
Historical statements below about closed prospective outcomes describe earlier stages.

Previous completed revision: **Photdis v2.6 observed-catalog posterior inference**.
[Results](docs/v2_6_result.md) · [Nine English plates](docs/v2_6_atlas.md) ·
[Requirement ledger](docs/v2_6_requirements_result.md).
Per-star joint probabilities and shared-parameter counts reuse192mock catalogs.
Restricted controls identify template-response transfer failure. Nested2048-source
response marginalization improves counts, but uncertainty remains uncalibrated.
Source-conditioned signed-Gaia luminosity calibration has mixed transfer results;
no deployment change.1648actual artifact checks,16focused tests and104compact
delivery checks pass; all nine plates visually inspected.
Large arrays and initial/refined calibration remain on NERSC;3250prospective
outcomes stay closed. Verify compact delivery with `python3 scripts/v2_6_delivery.py`.

Previous completed prototype: **Photdis v2.5 target-population inference**.
[Results](docs/v2_5_result.md) · [Eight English plates](docs/v2_5_atlas.md) ·
[Requirements and limits](docs/v2_5_requirements_result.md).
Outcome-blind red controls and signed-Gaia metal pairs are complete. A four-
parameter, selection-normalized empirical response improves distance-distribution
means in192independent mock catalogs, but composition and interval coverage fail
parts of acceptance. No deployment or physical-population calibration claim.
1966actual compute checks,4focused tests and80delivery checks pass. Verify with
`python3 scripts/v2_5_delivery.py`. Source-separated256cal/224test templates,
large posterior caches and initial numerical archive remain on NERSC.
3250prospective nativePSF identities stay unopened pending suitable validation.

Previous completed audit: **Photdis v2.4 frozen-posterior scientific use**.
[Results](docs/v2_4_result.md) · [11 English figure sets](docs/v2_4_atlas.md) ·
[Requirements and scientific limits](docs/v2_4_requirements_result.md).
Fixed-identity Gaia/SPDist anomalies, source-separated metal-poor scales, joint
application counts, matched-band information and independent-template population
recovery are complete. New3250PSF identities remain reserved without opening
reference outcomes; final independent validation is not yet achieved.
Frozen models and deployment remain unchanged.1898compute checks,4focused tests
and167compact delivery checks pass. Local verification:
`python3 scripts/v2_4_delivery.py`. Large products stay on NERSC; production programs require Slurm allocations.

Previous completed correction: **Photdis v2.3.1 frozen-model inference and selection audit**.
[Results](docs/v2_3_1_result.md) · [16 English plates](docs/v2_3_1_atlas.md) ·
[Requirements and limits](docs/v2_3_1_requirements_result.md).
Corrected the shared-g prediction covariance without refitting models; exact
historical selection replay, forward checks and paired evidence are complete.
The corrected XD32 still fails the original calibration guards; deployment is
unchanged. Compact-artifact verification: `python3 scripts/v2_3_1_delivery.py`.
Large predictions and immutable pre-correction archives remain on NERSC.

Previous completed audit: **Photdis v2.2 selected-distance and actual-application evidence**.
[Results](docs/v2_2_result.md) · [Seven English plates](docs/v2_2_atlas.md) ·
[Requirement evidence and limits](docs/v2_2_requirements_result.md) ·
[Delivery manifest](manifests/v2_2/delivery.json).
Reliability endpoints are repaired; two frozen full-PDF models are compared on
26000 weighted PSF application objects, with selected-distance, repeated-noise
and far-branch audits. Original deployment remains unchanged. Application
coverage and independent-validation gaps are quantified, not assumed solved.
Local v2.2 figure/format checks use the existing Matplotlib, Pillow and pypdf
environment plus Poppler `pdfinfo`; `python3 scripts/v2_2_validate_delivery.py`
checks the committed compact artifacts. Large-cache programs require allocated
NERSC resources and the established DESI environment.

Previous completed audit: **Photdis v2.1 near-selection and joint-reference probability**.
[Scientific results](docs/v2_1_result.md) · [Six-group/eight-plate atlas](docs/v2_1_atlas.md) ·
[Review requirement ledger](docs/v2_1_requirements_result.md) · [Delivery checks](manifests/v2_1/delivery.json).
Full frozen PDFs and one same-PSF-identity partition candidate were evaluated.
The candidate fails cal guards; deployed baseline_v1 remains unchanged.
Independent final validation is not established; evidence gaps are explicit.

Current stage report: **Photdis v2.0**. [Scope](docs/v2_0_scope.md) · [Methods](docs/v2_0_methods.md) · [Status](docs/v2_0_status.md) · [55-figure atlas](docs/v2_0_figure_atlas.md) · [Requirement checklist](docs/v2_0_requirements.md). Original model identifiers and application baseline_v1 are retained in the report.

Latest revision: [1050-figure complete atlas](docs/v2_0_revision_atlas.md) · [Gaia anomaly audit and results](docs/v2_0_revision_result.md). All figure text is English; PNG300dpi, PDF, CSV and metadata are linked per figure.

Photometric stellar distances from Legacy Survey DR11 `grzW1W2`, following the
GPT Pro restart plan in conversation
`6ae164f5-1a74-83e9-a2c0-768b28e09d1b`.

Latest development experiment: [fixed weak WISE flux contrast](docs/baseline_v1_6_wise_result.md). The report includes signed Gaia checks, joint measurement coverage and three Pal 5 New-style figures.

Previous experiment: [expanded A apparent-magnitude contrast and revised coverage](docs/baseline_v1_5_r0_result.md). Five Pal 5 New-style figures accompany the matched reference, population and Gaia comparisons.

Latest scientific diagnostics: [interpretation and key figures](docs/baseline_v1_4_science_result.md)
and [complete 29-figure atlas](docs/baseline_v1_4_science_atlas.md), with
[reproducibility audit](manifests/baseline_v1_4_science_delivery.json).
Frozen per-star analysis finds evolved-star branch failures inside the support
region and across populated sky cells, plus metal-poor dwarf distance bias.
Independent WISE cuts expose overlapping depth/quality loss.1943 actual checks
and4 focused tests pass; references and deployed v1 remain unchanged.

Latest completed experiment: **v1.4 — broader DESI references, unseen sky and optional i**.
[Results and five inspected figures](docs/baseline_v1_4_result.md) ·
[protocol](docs/baseline_v1_4_protocol.md) ·
[actual results](manifests/baseline_v1_4_result.json) ·
[delivery audit](manifests/baseline_v1_4_delivery.json).
All 1600 sweeps are assembled with independent eligibility and new held-out sky.
New train has 138,080 class labels and 153,403 distance-training objects; new test
has 42,964 class labels. Fixed data controls give modest, mixed improvements.
Matched grz/griz AP is 0.2729/0.2823 and severe distance tails 9.71%/9.57%, but
same-star evolved tails remain 97.46%/96.52%. Preserve deployed v1.
3331 actual artifact checks, 26 focused tests and five inspected figures complete.
Large references, models and predictions remain on NERSC under data/baseline_v1_4.
No full-parent prediction rerun or automatic deployment change; broad22–25
validation remains absent. Further experiments require a new authorized scope.

Previous completed experiment: **v1.3 — independent label eligibility and fixed candidate**.
[Results and three figures](docs/baseline_v1_3_result.md) ·
[protocol](docs/baseline_v1_3_protocol.md) · [actual evidence](manifests/baseline_v1_3_result.json).
Independent class training releases6287 A labels (395 evolved,508 without W1).
Fixed forest data controls show small, mixed changes; faint class coverage remains
insufficient. The fixed predicted-evolved B/forest group has59.18% severe chosen
reference error versus49.66% for wide on the same147 objects; retain v1 deployment.
93 actual checks and18 focused tests pass. Large reference/model/prediction files
remain under NERSC data/baseline_v1_3. No full-catalog prediction rerun.

Previous completed experiment: **v1.2 — color separability and distance decisions**.
[Results and ten figures](docs/baseline_v1_2_result.md) ·
[protocol](docs/baseline_v1_2_protocol.md) ·
[delivery audit](manifests/baseline_v1_2_delivery.json).
2364 actual checks and16 focused tests pass. At the cal2% dwarf budget, C evolved
recall is35–46% and known-reference precision39–42%. Complete Gaia likelihood
favors the old wide model over new mixtures; hard branches worsen tails.
Cal-passing single groups can still fail minority populations. Retain deployed v1,
full PDFs and all historical artifacts. No full-catalog rerun; no Gaia r>=22 validation.

Previous completed experiment: **v1.1 — fallback decomposition and learned branch weights**.
[Results and eight figures](docs/baseline_v1_1_result.md) ·
[protocol](docs/baseline_v1_1_protocol.md) ·
[actual validation](manifests/baseline_v1_1_validation.json).
All six jointly supervised gates fail the cal-only adoption criteria; **deployed
baseline remains v1**. On the same266 evolved C-test references, full fallback
changes oracle severe-error rate from13.91% to15.04%; the new gate reaches98.87%
versus80.08% for wide QRF. This localizes the remaining branch-weight problem
without establishing a physical color-information limit.
470 actual checks and11 focused tests pass. The fixed26,000-object parent sample
and7,672 Gaia objects (including1,800 negative parallaxes) quantify coverage and
observed-parallax behavior. Gaia validation has no objects at r>=22; reference
risk is not whole-parent purity. No full catalog was rerun.

Latest completed baseline: **v1 — population branches, numerical propagation and depth**.
[Results and eight figures](docs/baseline_v1_result.md) ·
[protocol](docs/baseline_v1_protocol.md) · [catalog and replay](docs/baseline_v1_catalog.md).
The full14-field internal catalog is complete:9,390,518 rows,321 artifact checks.
Cal retains wide QRF with128-point Sobol/CDF interpolation; explicit population
mixing fails the class-specific adoption guards. Both numerical error layers
retain convergence-failure flags, and no all-class quality cut passes.
This improves propagation and diagnoses population failures; field-giant
accuracy and real faint/no-WISE populations remain inadequately validated.

Completed baseline: **unified multi-population baseline v0**.
[Results and nine figures](docs/baseline_v0_result.md) ·
[catalog and runbook](docs/baseline_v0_catalog.md) ·
[full review](docs/pro_review_60f65d5.md) · [protocol](docs/baseline_v0_protocol.md).
Application uses all clean optical objects; Gaia/spectroscopy are reference
labels, WISE is optional. Internal uncalibrated field predictions are allowed.

| Stage | v0 scope | State |
|---|---|---|
| Parent and references | Existing14 fields, DECam main; Gaia-first/SPECDIS supplementary | Complete:9,390,518 unique objects |
| Training | Broad QRF A/B/C plus same-training controls | Complete: six fixed models |
| Development validation | Whole fields, complete references, raw/calibrated intervals | Complete:27,212 preferred test labels |
| Internal application | Every unique optical parent, all available bands and risk flags | Complete:8,899,334 DECam predictions;491,184 northern status rows |
| Unified results | Coverage, distance density, CMD, ambiguity and population scorecard | Complete:9 figures,206 artifact checks,3 mathematical tests |

W1 and W1W2 cover31.7% and17.8% of the DECam optical parent. On matched
Gaia references with identical training identities, A/B/C NMAD is
0.266/0.242/0.218mag. The recommended workflow's Gaia relative NMAD is10.4%,
but this is dwarf-dominated:81.9% of298 low-gravity SPECDIS test objects have
|delta-mu|>1mag, and faint/low-precision objects lack adequate validation.
The internal catalog preserves these failures and support flags; it is not
a calibrated all-population science release. Next global work concerns
population degeneracy/reference composition and depth validation.

Cluster and64-warm-member follow-ups are retained evidence, not prerequisites
for this baseline. Historical models/results below remain frozen.

Previous completed follow-up: [Pal14 parameter sources and fixed giant coverage](docs/giant_parameter_audit_result_v1.md).
120 reliable spectra are120 unique members. The64 additional quality-qualified
DECam warm members do not overlap the current K-target color ranges; the four
warm northern color/luminosity overlaps are all metal-poor NGC2419. Pal14's strong
identity association coexists with unresolved literature/DESI parameter conflict.
All22 historical predictions are unchanged. The evaluated DESI channel covers
18/159 linked systems, not the entire reference resource. Current K absolute
calibration remains evidence-insufficient; phase/support definition for the existing warm-evolved branch is retained as
a deferred local direction; unified baseline v0 now takes priority.
Four figures, the full member evidence and five actual checks are preserved.

Previous completed follow-up: [fixed multi-system reference batch](docs/kgiant_reference_batch_result_v1.md).
BV21's162-system distance table and170 member archives were linked before a
member-first full DESI audit. Eight new K members yield one eligible DECam Pal14
star; five are northern-system objects and two fail frozen optical quality.
Pal14 grz distances55.68/48.22kpc versus73.579kpc reference expose transfer offsets.
Including historical Terzan7, this is4 stars /2 systems /22 configurations,
not22 independent validations. Six focused tests and11 artifact checks pass.
[Scope, reference and local support addendum](docs/kgiant_external_addendum_v1.md)
and seven figures preserve limitations. Fixed parent is complete; no new W1
pair or calibrated field-giant product. No further automatic cluster search.

Previous completed follow-up: [fixed six-target external K-giant check](docs/kgiant_external_result_v1.md).
Three Terzan7 members support18 fixed teacher predictions; Pal12 lacks a secure
external temperature linkage,1665 fails temperature/photometry,1708 has ambiguous
matching. All scored stars lie outside the old low-dust scope. Distributions,
support flags and common-scale sensitivities are preserved; no calibrated field
product. Teacher coverage and empirical interval interpretation are clarified.

Previous completed experiment: [actual K-giant conditional distances](docs/kgiant_distance_result_v1.md)
uses all 289 W1-qualified K giants with available teacher labels (77 SPDist,
258 SPECDIS), with whole-field paired grz/grzW1/no-color predictions. On the
frozen selected subset, colors improve teacher-scale predictive scores; the W1
increment is small and does not establish stable scatter improvement. The 39
common selected teachers differ by a median -0.274 mag; the actual 189 SPECDIS
fractional errors have median 29.2%. Full mixtures preserve uncertainty, but are
not calibrated true-distance posteriors or a released field-giant product.
[Protocol](docs/kgiant_distance_protocol_v1.md),
[paired evidence](manifests/kgiant_selected190_evidence_v1.json), and
[independent-reference feasibility](docs/kgiant_pal12_feasibility_v1.md) define
what is supported. Large mixtures remain on NERSC in data/kgiant_distance_v1.

The preceding [controlled broad-dwarf comparison](docs/giant_wide_result_v1.md)
remains frozen: B-wide retains 190/289 K giants versus 28/289 for B-K at the
calibration-side 1% FPR working point; held-out dwarf FPR is 1010/98536.
It is a scoped DECam enrichment/ranking recipe, not a high-purity catalog or a
calibrated global class probability. The old poor-metal lower-RGB relation does
not cover the selected population. Next progress requires independent distances
and phase evidence in the brighter, more metal-rich K population, rather than
another classifier or kernel-bandwidth search.

Requested distance density plots reproduce frozen held-out predictions:
[cold phot-d / Gaia-d](figures/distance_comparison_v1/01_cold_phot_vs_gaia.png),
[warm phot-d / SPDist and SPECDIS](figures/distance_comparison_v1/02_warm_phot_vs_spectral.png),
and [all warm reserved objects](figures/distance_comparison_v1/03_warm_all_reserved.png).
[Matched-source CMDs](docs/giant_wide_result_v1.md#同一批源的-cmd-补充) show reference-distance
and photometric-distance absolute magnitudes for these exact samples.
Spectral comparisons test teacher/catalog agreement, not independent geometric
accuracy; the warm 5000–7000 K sample differs from the selected K population.

The [preceding information experiment](docs/giant_information_decision_v1.md)
remains frozen. DR11 includes north RELEASE=11011 products; its native-system
cross-field validation is still insufficient. No release migration is needed.

The completed [local conditional-color RGB experiment](docs/cluster_rgb_local_color_result_v1.md)
finds color information after controlling brightness boundaries. Its joint-cluster
widths are not single-star errors. Low gravity does not establish lower-RGB phase;
no field-giant distance product is released.

The bounded [four-cluster external-scale experiment](docs/cluster_rgb_scales_protocol_v1.md)
uses the existing members and frozen curves. Its optional passband integration
uses project-local `speclite==0.20` (install with `python -m pip install --no-deps --target .deps speclite==0.20` in the allocated DESI environment); core likelihood tests need only
NumPy/SciPy. Submit from the NERSC checkout:

```bash
sbatch scripts/run_cluster_rgb_scales_v1.sh response
sbatch scripts/run_cluster_rgb_scales_v1.sh joint --cluster ngc4147
# Repeat joint for m3, m53, ngc5466; all large outputs stay in data/cluster_rgb_scales_v1.
sbatch scripts/run_cluster_rgb_scales_v1.sh joint --cluster ngc4147 --qmc-power 11 --only-model robust_grz
sbatch scripts/run_cluster_rgb_shared_sample.sh --cluster ngc4147
# Numerical refinement of failed frozen subsets, preserving initial outputs:
sbatch scripts/run_cluster_rgb_shared_sample.sh --cluster ngc4147 --counts 20 42 --draws 48000 --tag mcmc48000
# Within an allocated environment after all four finish:
python scripts/cluster_rgb_scales_summarize.py
```

The [scale and shared-distance report](docs/cluster_rgb_scales_result_v1.md)
records the completed scientific comparisons and numerical checks. Shared
posterior inference uses the same frozen likelihood and priors, with exact
Gaussian collapse and four chains; the no-color/support control quantifies
distance information already supplied by the selected luminosity range.

The [local conditional-color experiment](docs/cluster_rgb_local_color_protocol_v1.md)
uses the fixed interior members and delta-mu window, and conditions on observed
g after shared nuisance integration. From NERSC, submit one allocated job per
cluster; outputs stay under `data/cluster_rgb_local_color_v1`:

```bash
sbatch scripts/run_cluster_rgb_local_color.sh --cluster ngc4147
# Repeat for m3, m53, ngc5466. After all complete, inside an allocation:
python scripts/cluster_rgb_local_color_summarize.py
```

The [local information report](docs/cluster_rgb_local_color_result_v1.md) separates
brightness-boundary cancellation, posterior edge behavior, and abundance-prior
sensitivity. Aggregation uses checked adaptive brightness integration and retains
original quadrature and sampling diagnostics.

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
This historical stage was followed by the independent NGC 4147 test, the
external-scale comparison, and the completed local conditional-color experiment.
NGC 5466 remains an exposed diagnostic. The selection and abundance-information experiment has completed. The current
negative-coverage comparison is described above.

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

### Completed v2.3 experiment

See [results](docs/v2_3_result.md), [nine-plate atlas](docs/v2_3_atlas.md),
[requirements and evidence limits](docs/v2_3_requirements_result.md) and
[frozen protocol](docs/v2_3_protocol.md). Same-PSF QRF/XD32 comparison completed;
the candidate fails full-Gaia and near-contamination guards. Cache noise priors
and reserved-reference eligibility are corrected. Existing deployment retained.
All8123production checks and7focused tests pass; actual PDFs remain on NERSC.
No whole-catalog expansion or automatic further modeling.
