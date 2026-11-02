# Project instructions

- Current checkpoint: unified multi-population baseline v0 is the active main task.
  Read docs/pro_review_60f65d5.md and docs/baseline_v0_protocol.md, WORKLOG/PLAN.
  Deliver broad optical-parent training/test/application, optional WISE, Gaia-first
  and SPECDIS supplementary labels, same-star band tests and internal catalog.
  Internal field predictions with uncalibrated/support flags are authorized.
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
