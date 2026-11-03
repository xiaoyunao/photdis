# Project instructions

- Latest completed experiment: baseline v1.1; docs/baseline_v1_1_result.md.
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
