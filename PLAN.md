# PLAN

## Current Objective

Complete the Pro-recommended photometry/coverage audit in the existing 10°×10°
field, use its findings to define a cold-main-sequence baseline, fit and test
the grz, grzW1 and grzW1W2 models, then present the result for user review.

## Milestones

1. Reconcile Gaia BP/RP measurements and the zero-color column; use one canonical
   Legacy counterpart per Gaia source for consistent diagnostic counts.
2. Audit joint SATUR/FRACMASKED/ALLMASK/BAILOUT and real-measurement failures,
   stratified by band, brightness, reliable color and CMD position.
3. Compare Legacy to PS1 stellar standards with published DECam color terms,
   propagate Gaia positions to PS1 epochs and measure offsets/scatter by group.
4. Compare target points, fixed strict anchors, quality DESI stars and classified
   nonstars in the grz, grzW1 and grzW1W2 color planes; count anchors per layer.
5. Store raw/corrected DESI RVS metallicities and published applicability flags.
6. Define a cold-main-sequence sample and adapt spatial partitioning from actual
   counts while retaining every originally reserved test object.
7. Fit a simple color relation and quantile forests for three input tiers.
   Compare all tiers on identical stars and measure each tier's full coverage.
8. Validate residuals, interval coverage and trends, plot, synchronize
   programs/figures, then stop for user review.

## Scope

RA [0,10) deg and Dec [-10,0) deg. Current strict anchors: 1,802. Existing
APOGEE matching is diagnostic. The prior full-footprint run stays stopped.

## Next Checkpoint

Show the optical audit and the tested cold-main-sequence baseline together.
