# Project instructions

- Current task: bounded baseline v1; docs/pro_review_494b12c.md and
  docs/baseline_v1_protocol.md define convergence, population branches and depth
  experiments. Use existing identities/splits; cal-only selection; v1 outputs
  separately. One full application after selecting the version. No subagents.
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
