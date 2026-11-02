# Project instructions

- Current checkpoint: recover current status from WORKLOG.md and PLAN.md.
  The broad-dwarf classifier comparison and matched distance/CMD plots are
  complete at 1689ec2. Preserve cold-v4, prior RGB and classifier results.
  Latest Pro review: docs/pro_review_68eb2f3.md. Active bounded task:
  docs/giant_parameter_audit_protocol_v1.md, Pal14 same-star parameter evidence
  and exact120 reliable DESI spectra observational coverage. Preserve22 predictions;
  no new temperature window, no scoring outside it, no141-system expansion.
  Previous completed task is the registered
  fixed BV21 multi-system reference batch: docs/kgiant_reference_batch_protocol_v1.md.
  Prior six-target result remains frozen; new scope/reference/image addendum:
  docs/kgiant_external_addendum_v1.md. Batch uses162 fixed distance systems,
  170 membership systems (159 linked), member-first full DESI type audit and
  directed DR11 measurements. Inventory frozen at9c023fa before scoring, grz main/W1 paired.
  Result: docs/kgiant_reference_batch_result_v1.md. One new Pal14 grz-only
  star plus3 historical Terzan7:22 predictions,2 systems,11 checks pass.
  Pal14 centers55.68/48.22kpc vs73.579kpc reference; absolute calibration fails
  to gain sufficient coverage. Fixed batch closed; do not restart searches.
  Coverage must be judged jointly in colors, luminosity, abundance, phase and
  quality, not by source-dependent temperature labels alone. No field product.
  Do not restart the original cold-main-sequence follow-up. Maintain the
  project's multi-population scope and distinguish teacher agreement from
  independently calibrated distance accuracy.
- NERSC: `yunao@perlmutter.nersc.gov`; canonical checkout:
  `/pscratch/sd/y/yunao/photdis_pipeline`.
- Small header/column/few-row inspections may use login nodes. Full scans,
  crossmatches and training require allocated compute resources.
- Source catalogs and large products stay on NERSC. Track programs, docs,
  configuration and selected small figures.
- Before work: status, branch, fetch, recent history, WORKLOG, PLAN. Sync safely;
  never overwrite remote edits, force-push, or use cleanup commands on data.
- Update WORKLOG in reverse chronological order and maintain PLAN at milestones.
