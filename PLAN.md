# PLAN

## Current Objective

The 10°×10° pilot is complete. Present its results and wait for user review.

## Completed

- RA [0,10) deg, Dec [-10,0) deg; four files in configs/pilot_10deg.json.
- Every optical, Gaia-anchor and spectral-quality cut has a remaining/removal
  count in docs/pilot_10deg.md and manifests/pilot_10deg.json.
- Four figures: selection counts, Gaia CMD, spectroscopy, parallax/brightness.
- Saved-table checks passed; allocation 60427335 released.

## Results to Review

- 312,975 measured grz records; 1,802 anchors; 49,906 extended records.
- 6,683 negative parallaxes; 4,586 remain in the extended quality subset.
- Optical quality keeps 1,804 of 6,942 Gaia high-S/N references (26.0%).
- The bright main sequence, turnoff and giant branch lose many objects.
- APOGEE: 242 quality grz matches, 164 with Gaia anchor astrometry, zero with
  optical high-quality photometry. DESI: 55 quality spectral anchors.

## Checkpoint

Wait for the user to review the pilot counts, diagrams and optical cuts.
The earlier full-footprint processing remains stopped; its fragment and partial
output locations are recorded in WORKLOG.md. Training has not started.
