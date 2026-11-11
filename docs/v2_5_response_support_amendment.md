# Response support clarification

The fixed1e-4 conditional contamination floor must be confined to cells allowed
by the hard observed-r selection. Applying it to impossible cells would give
positive conditional probability outside S, despite conserving alpha. Selected
response smoothing is therefore uniform only across14<=r<20cells; complement
may occupy all cells because error rejection can occur at any r. No change to
QRF, templates, generated noise, parameterization or science selection.

This correction is prepared while61126775is in the empirical diagnostic phase;
production response source hashes must match the corrected implementation. If
a pre-correction population freeze exists, preserve it and require an explicit
inference-only migration with catalogs/QRF bytes retained.
