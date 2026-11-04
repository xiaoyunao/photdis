# Baseline v1.4: broader evidence, unseen sky and optional i

Registered 2026-11-04 before any new performance inspection. Authorization:
latest user request to implement `docs/pro_review_5cf8f6d.md`, with 30-minute
follow-up for long NERSC jobs. This supersedes previous no-new-experiment scope
only for the fixed experiments below. Preserve deployed v1 and every v0–v1.3
artifact. No subagents, family search, candidate threshold retuning or full
9.39-million-row prediction rerun.

## Delivery and execution order

1. Inventory full registered DESI MWS RVTAB/FIBERMAP/GAIA, SpecDis v2.1 and
   SPDist v1.0. Retain raw rows, PRIMARY, quality, SURVEY, PROGRAM, all available
   `*_TARGET` flags, exact target/Gaia keys and distance provenance/errors.
   No temperature, gravity or old ranked-field entrance cut. Preserve unknown
   and nonstellar evidence; do not mislabel it as qualified stellar truth.
2. Extract native south DR11 photometry for the full positive-Gaia-ID union
   and the fixed 26,000-row v1.1 parent probability sample. A full sweep read is
   an input extraction, not a new all-catalog prediction. Keep raw griz/W1/W2
   flux/ivar/transmission/nobs/quality/mask fields, including negative WISE flux.
   Exact `REF_CAT=G3, REF_ID=SOURCE_ID` matching prevents epoch-dependent spatial
   false associations. Preserve zero-ID source rows separately; quantify and
   resolve their positional/epoch associations before claiming complete reference
   coverage. Audit VAC-only and duplicate target/source pairs, DR11 duplicate
   detections, ID conflicts and absent north coverage before assembly.
3. Freeze new geometry before scoring: hash seed20261104 assigns whole10x10
   degree cells70/15/15 train/cal/test. Any cell intersecting ANY existing
   `configs/field_*.json` bounds is quarantined (split3), even when the overlap is
   partial. Invalid coordinates and multi-cell identities are quarantined too.
   Persist per-source split and file hashes before reading residuals. Verify
   identity disjointness against all previous training/evaluation artifacts;
   legacy identities cannot enter the new cal/test via coordinate differences.
   New validation is geographically held out, not an unbiased sky sample.
4. Assemble independent class, distance and weak-parallax eligibility. Reuse
   v1.3 class definitions; class training must not require distance/FeH/precise
   optical errors. Reuse qualified Gaia astrometry/zero point and exact-pair
   SpecDis/SPDist quality recipes, with provenance-specific score tables.
   Weak/negative parallaxes remain likelihood evidence, never inverted labels.
5. Evaluate frozen v1 distance models and v1.3 fixed classifiers on new sky
   FIRST. Freeze those predictions before the fixed new-data comparison.
   Train the same algorithms/hyperparameters once on the registered new train
   set, optionally augmented by legacy training only after identity exclusion.
   Explicitly record that composition and preserve the same evaluation IDs.
6. Measure i coverage on the fixed26,000 parent identities, with sample weights
   retained from original sampling design; tabulate count and weighted fraction
   by r, optical error, color and W1 state. Missing native i is unavailable,
   never zero-filled. Check actual reference support before fitting.
7. Matched grz/griz: identical i-usable train/cal/test IDs, weights, labels and
   hyperparameters. Features are (g-r,r-z) vs (g-r,r-i,i-z). New explicit
   `griz_v1`, bands[G,R,I,Z]; old BANDS and5D Sobol remain unchanged. Use a
   shared4D Sobol128 realization and identity-derived randomness for both
   controls, selecting the same g/r/z deviations; include shared g error in
   effective absolute-magnitude/distance propagation. No tuning on new test.
8. Deliver source/coverage audit, frozen-model new-sky results, one fixed
   new-data comparison, same-star i classification AND distance results,
   inspected figures, actual validation checks, WORKLOG/PLAN and milestone git
   commits. Raw extraction alone does not complete this objective.

## Fixed quality and evaluation

Optional i requires nobs>0, finite positive flux/ivar/transmission,SNR>=5,
fracmasked<.4,fracin>.7,fracflux<.5, no SATUR_I(bit14),ALLMASK_I(bit15) or
BAILOUT(bit10). Existing grz requirements still apply. Reference:
https://www.legacysurvey.org/dr11/catalogs/ and
https://www.legacysurvey.org/dr11/bitmasks/ (checked2026-11-04).
WISE states distinguish no observation, invalid measurement, quality failure,
low SNR (including negative flux), and usable SNR>=5 color. Raw low-SNR flux is
retained but not fed into the Gaussian-magnitude experiment.

Fixed RF classifier256trees,minleaf32,maxdepth20,maxfeatures1.0; fixed QRF
128trees,minleaf32,maxdepth20,.1mag bins. Reuse prior class-weight recipe.
Evaluate AP, evolved recall and known-reference contamination, with any
workpoints selected ONLY on cal using previous dwarf2% budget. Report counts
and unknown fractions. Distance endpoints: delta_mu bias/scatter, |error|>1mag
severe tails,68/95% coverage,width, PIT and full Gaia likelihood including
negative parallaxes. Same-star comparisons at every endpoint, also by class,
reference source,r(<18,18–20,20–22,22–25),optical error,W1/W2 states and sky.
Unknown labels cannot be counted as negative. Zero-count cells report absent
validation. Prioritize18–20 noWISE and available20–22.22–25 remain internal and
unvalidated where evidence is absent; missing full faint truth does not block
this registered phase. No automatic deployment change.

## Resumption and long jobs

`python scripts/sync.py push`, then on NERSC canonical checkout:
`sbatch scripts/run_baseline_v1_4_extract.sh`. Four I/O workers, one allocated
CPU node,4h wall limit. `data/baseline_v1_4/freeze.json` pins protocol/code/config;
per-sweep NPZ+JSON uses atomic rename and content hashes. `flock` prevents
concurrent writers; only resubmit after terminal state. Code changes after a
freeze require an explicit migration/new run directory, never silent mixing.

Every30minutes inspect squeue/sacct and a bounded log tail. Unchanged running
state stays quiet. On completion audit and continue the steps above; on failure
read logs and fix within scope. Record each job ID and continuation milestone.
If NERSC credentials expire, ask for password/OTP renewal; never infer job
failure from SSH authentication loss. Stop the heartbeat after final delivery.
