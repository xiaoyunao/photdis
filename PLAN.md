# PLAN

## Current Objective

Build and audit Pro step 1 on DR11 south: Gaia high-quality luminosity anchors,
weak/negative-parallax extended samples, spectroscopy auxiliaries, and locked
external/spatial tests. The user authorized this step and public GitHub visibility.
Stop before model training and report actual coverage/selection losses.

## Milestones

1. Verify current source schemas, source-backed selection rules and reference
   catalogs. Public repository confirmed; selection modules in progress.
2. Project DESI/APOGEE auxiliary catalogs and prepare external cluster/binary
   identifiers on a compute node.
3. Scan all DR11 south sweep tiles that can intersect |b|>30 degrees; apply the
   object-level low-extinction domain and measured photometry selection. Build
   Gaia associations without an observed parallax S/N/sign cut on the extended pool.
4. Freeze whole-group and HEALPix partitions, assemble role tables and audit
   duplicate/overlap/quality/contamination/coverage statistics.
5. Generate source-backed coverage figures, validate tables and counts, commit
   programs/docs/figures and synchronize all three locations. Stop for review.

## Outstanding Issues

- Actual star counts, overlap and coverage are not yet established.
- Gaia flux substitutions and saturation limit the brightest measurable Legacy
  stars; preserve rejection counts and relaxed/strict quality flags.
- Point-source morphology and colors alone do not prove stellar nature. Keep
  known spectroscopic contamination and Gaia candidate risk flags explicit.
- External cluster/binary membership catalogs use Gaia and are leakage controls;
  independent absolute-distance truth requires separate provenance checks.

## Validation Criteria

- All eligible tiles complete or any missing coverage is explicitly reported.
- No global W1/W2 requirement; raw flux/ivar and signed parallax survive in outputs.
- Anchors use measured independent Legacy fluxes, corrected high-S/N parallax and
  explicit photometric/astrometric quality flags.
- Each row retains role, provenance, quality, group and partition metadata.
- Whole external groups are reserved; no shared Gaia/Legacy IDs cross partitions.
- Counts reconcile, weak/negative parallaxes are present, and coverage plots use
  labelled denominators. No model is trained.
- Data stay on NERSC; programs, manifests and selected figures match across local,
  GitHub and NERSC after the final milestone commit.

## Next Recommended Step

Complete sample construction and validation, then stop for user confirmation.
