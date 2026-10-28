# PLAN

## Current Objective

Present the completed Pro-recommended photometry audit and cold-main-sequence
baseline in the existing 10°×10° field for user review.

## Milestones

1. Done: reconcile Gaia BP/RP measurements and the zero-color column; use one canonical
   Legacy counterpart per Gaia source for consistent diagnostic counts.
2. Done: audit joint SATUR/FRACMASKED/ALLMASK/BAILOUT and real-measurement failures,
   stratified by band, brightness, reliable color and CMD position.
3. Done: compare Legacy to PS1 stellar standards with published DECam color terms,
   propagate Gaia positions to PS1 epochs and measure offsets/scatter by group.
4. Done: compare target points, fixed strict anchors, quality DESI stars and classified
   nonstars in the grz, grzW1 and grzW1W2 color planes; count anchors per layer.
5. Done: store raw/corrected DESI RVS metallicities and published applicability flags.
6. Done: define a cold-main-sequence sample and adapt spatial partitioning from actual
   counts while retaining every originally reserved test object.
7. Done: fit a simple color relation and quantile forests for three input tiers.
   Compare all tiers on identical stars and measure each tier's full coverage.
8. Done: validate residuals, interval coverage and trends, and plot.
9. Synchronize final report and project logs; stop for user review.

## Scope

RA [0,10) deg and Dec [-10,0) deg. Current strict anchors: 1,802. Existing
APOGEE matching is diagnostic. The prior full-footprint run stays stopped.

## Next Checkpoint

Review [the optical audit](docs/pilot_audit.md) and
[the cold-main-sequence baseline](docs/cold_baseline.md) together. The next
scientific decision is whether to study the recovered bright-star offset,
recalibrate the prediction intervals, or add an independent sky field.
