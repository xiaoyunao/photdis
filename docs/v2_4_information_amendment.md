# Fixed information-prior support correction

The initial fine10degree sky-cell×r baseline falls back to global for every
development object because cal/dev sky cells are disjoint by design. Keep its
original result as a provenance record; it does not measure r/sky information.

Add an r-only cal prior and a fixed coarse60degreeRA×north/south-declination×r
prior, using30object shrinkage toward the r-only prior. Empty coarse cells fall
back to r-only, then global; record these support fractions. This is a structural
support correction, not a search for sky resolution with the best dev scores.
All color probabilities, identities and existing calibration remain unchanged.

For uniform-class population stress, retain the original raw macro result but
also retarget EVERY method and baseline by the same prescribed label-shift
factor (1/3)/global-cal-prior, then normalize. This is a conditional label-shift
experiment, not a claim that real targets satisfy label shift. The corresponding
global no-color baseline is exactly uniform. Natural-reference results remain
unadjusted. No new classifier/distance model or threshold fitting.
