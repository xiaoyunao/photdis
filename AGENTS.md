# Project instructions

- Current checkpoint: user approved Pro's next same-field photometry audit.
  Use RA 0–10 deg, Dec -10–0 deg. Audit joint saturation/mask failures, Gaia
  BP/RP validity, external photometry and three input-color planes. Preserve
  the current strict sample as the comparison; APOGEE stays auxiliary.
  Present this audit for review before proceeding to model training.
  Keep progress and explanations direct and factual.
- NERSC: `yunao@perlmutter.nersc.gov`; canonical checkout:
  `/pscratch/sd/y/yunao/photdis_pipeline`.
- Small header/column/few-row inspections may use login nodes. Full scans,
  crossmatches and training require allocated compute resources.
- Source catalogs and large products stay on NERSC. Track programs, docs,
  configuration and selected small figures.
- Before work: status, branch, fetch, recent history, WORKLOG, PLAN. Sync safely;
  never overwrite remote edits, force-push, or use cleanup commands on data.
- Update WORKLOG in reverse chronological order and maintain PLAN at milestones.
