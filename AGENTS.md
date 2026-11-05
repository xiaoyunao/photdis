# Project instructions

- Latest completed stage report: Photdis v2.0. Read docs/v2_0_scope.md,
  docs/v2_0_methods.md, docs/v2_0_status.md and docs/v2_0_figure_atlas.md.
  10 main figure groups (19 plates),36 appendix plates;55 sets of PNG300dpi,
  PDF,CSV,JSON.19 normalized model/route/experiment views on NERSC.391 checks.
  Main diagnostic expanded_A; actual application baseline_v1. Full-reference
  morphology audit and PSF subset complete (47740/47756 chosen test PSF).
  Same-selection DR11 denominator and effective masked area remain null.
  No automatic modeling or application expansion; next scope follows review.

- Latest completed experiment: docs/baseline_v1_6_wise_result.md. Shared
  joint templates and raw signed WISE flux; real weak W1/W2 do not improve
  priority 18–20 development strata. Weak-W1 evolved labels absent; weak-W2
  with valid W1 has 28/28 severe failures. Preserve full PDF chunks and v1.
  No automatic continuation or additional experiment without scope.

- Latest fixed development contrast: docs/baseline_v1_5_r0_result.md.
  Same expanded A identities and QRF settings with added r0; evolved tail
  96.56% to 89.85%, dwarf 1.95% to 2.16%, overall Gaia PIT tails worsen.
  Coverage erratum and five Pal 5 New-style figures delivered. Keep experiment
  outputs isolated from deployment. Independent validation needs new scope.

- Latest completed scientific diagnostics2026-11-05: eight groups from
  docs/pro_review_46cb68a.md; results docs/baseline_v1_4_science_result.md and
  full29-figure atlas docs/baseline_v1_4_science_atlas.md.1943 actual checks and
  4 focused tests pass. Evolved failures persist inside support and across sky;
  metal-poor dwarf median distance ratio1.194 with68% coverage49.93%.
  W1 total low-SNR fraction56.92%, not the old sequential7.49%.
  Only9 chosen-distance test cells; preserve reference-selection limits and
  all negative findings. Frozen models/primary outputs and deployed v1 retained.
  Further model experiments require new user-authorized scope. No automatic fit,
  full-catalog rerun, monitoring or subagents.

- Latest completed experiment: v1.4, docs/baseline_v1_4_result.md and
  manifests/baseline_v1_4_delivery.json. Full1600-sweep/MWS/VAC assembly,
  independent eligibility, new sky, frozen ABC, fixed new-data ABC and matched
  grz/griz classification+distance are complete.3331 actual artifact checks,
  26 focused tests and5 inspected figures. Controls60799511 COMPLETED0:0.
  New train138080 class labels/153403 distance objects; new test42964 classes.
  i covers87.01% of weighted parent. Matched AP0.2729/0.2823 and distance
  tails9.71%/9.57%, but evolved tails97.46%/96.52% on1064 same stars.
  Priority18–20/no-W1 AP worsens0.2247 to0.2148;20–22 has no evolved
  cal/test labels;22–25 remains broadly unvalidated. Preserve deployed v1,
  immutable new references, raw weak fluxes and all negative findings.
  No active automatic continuation: both timers were deleted at user request.
  Do not recreate monitoring, rerun completed stages, search model families,
  retune candidates, rerun full catalogs or deploy replacements automatically.
  No subagents. Further scientific work requires new user-authorized scope.

- Previous completed follow-up: v1.3, docs/baseline_v1_3_result.md. Independent
  class entry releases6287 A training labels incl395 evolved/508 no-W1.
  Six fixed forest data comparisons show small mixed changes;93 actual checks,
  18 tests and3 inspected figures complete. Fixed predicted-evolved candidate
  has59.18% chosen distance tails vs wide49.66% on147 same sources: do not adopt.
  Faint classification validation remains absent at22–25; only2 train labels.
  Next evidence acquisition must target faint/no-WISE labels with new held-out
  sky; no automatic model search or full-catalog rerun. Preserve v1 deployment.

- Previous completed experiment: baseline v1.2; docs/baseline_v1_2_result.md.
  2364 actual checks,16 focused tests and10 inspected figures. Fixed classifiers
  and cal workpoints complete; retain v1 deployment. C evolved recall35–46%,
  known-reference precision39–42% at cal2% dwarf budget. Full Gaia scores favor
  wide; C main-branch high PIT tail9.75% vs wide5.69%. Cal-passing groups can
  still fail minority evolved stars. No Gaia r>=22 validation. Preserve all
  frozen artifacts and negative findings; no new models/full-catalog rerun or
  subagents. Next work requires explicit scope on reference composition/faint data.

- Previous completed experiment: baseline v1.1; docs/baseline_v1_1_result.md.
  Six gates reject on cal; retain deployed v1. Fixed C evolved oracle/full
  fallback tails13.91%/15.04%, new gate98.87% versus wide80.08%.
  470 actual checks,11 focused tests and8 inspected figures complete; fixed
  samples26,000 parent/7,672 Gaia incl1,800 negative. No Gaia r>=22 validation.
  Preserve all v0/v1 products and v1.1 negative findings. No automatic full-parent
  rerun, cluster expansion, model-family search or subagents. Next global issues
  are branch weights/reference composition and real faint/no-WISE validation.

- Latest completed checkpoint: baseline v1; docs/baseline_v1_result.md and
  docs/baseline_v1_catalog.md. Fixed branch/oracle, numerical and depth experiments
  plus one full9,390,518-row application are complete;321 actual checks and8
  focused tests pass. Cal retains wide QRF with Sobol128/CDF interpolation.
  Numerical layer tolerances and all-class quality cuts fail; preserve flags.
  Pure photometric gates do not pass dwarf/evolved tradeoff guards. No subagents.
  Next work concerns calibrated branch weights and real faint/no-WISE validation;
  no automatic cluster expansion, model-family search or full-catalog rerun.
- Current checkpoint: unified multi-population baseline v0 is complete.
  Read docs/baseline_v0_result.md, docs/baseline_v0_catalog.md, WORKLOG/PLAN.
  All14 fields have9,390,518 unique catalog rows;8,899,334 DECam predictions,
  north491,184 no-native-model rows. Six fixed models,9 figures,206 artifact
  checks and3 mathematical tests complete. Large products remain on NERSC.
  Preserve this internal uncalibrated baseline and all status flags. WISE improves
  matched-reference scatter, but evolved-star luminosity solutions often fail
  and faint/low-precision reference coverage is inadequate. Next priorities are
  population degeneracy/reference composition and depth validation at global scope.
  Do not make cluster checks or64-warm-member phase evidence prerequisites.
  Preserve all old cold-v4/RGB/classifier/K-giant models and historical predictions.
  Distinguish teacher agreement, development validation and absolute accuracy.
- NERSC: `yunao@perlmutter.nersc.gov`; canonical checkout:
  `/pscratch/sd/y/yunao/photdis_pipeline`.
- Small header/column/few-row inspections may use login nodes. Full scans,
  crossmatches and training require allocated compute resources.
- Source catalogs and large products stay on NERSC. Track programs, docs,
  configuration and selected small figures.
- Before work: status, branch, fetch, recent history, WORKLOG, PLAN. Sync safely;
  never overwrite remote edits, force-push, or use cleanup commands on data.
- Update WORKLOG in reverse chronological order and maintain PLAN at milestones.
