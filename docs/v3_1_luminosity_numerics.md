# Numerical sidecar: piecewise-linear likelihood shift

Initial fit61216288 failed after75s at bootstrap draw19, evolved class. Both
registered starts reached delta≈.14mag and sigma=0; L-BFGS-B returned ABNORMAL
with a nonzero one-sided numerical gradient. The shift operation interpolates
a.02mag native likelihood grid, so at zero extra scatter the objective has
grid-aligned derivative kinks. This is a numerical optimizer failure, not an
accepted scientific result or evidence that the class has no scatter.

Keep the original driver, all completed drawJSON files and failure log unchanged.
The sidecar verifies and reuses every previously successful draw. For the failed
draw it repeats the original two starts and keeps the L-BFGS-B records; a failed
start is followed by bounded Powell optimization from its returned point,
max400iterations, xtol1e-8, ftol1e-11. Bounds, priors, objective, data/roles and
bootstrap weights remain exactly the registered ones. Select only a converged
finite objective as before. No new model, scientific parameter or outcome gate
is introduced. Report all fallback records and preserve a failure at the bound.

This sidecar does not replace the required.01native-grid audit. A coarse-grid
optimizer success alone cannot certify the correction's distance probabilities.

Held-out calibration guards report the point fit and the equal-weight mixture
of32shared sky-bootstrap conditional predictions separately; the latter is the
candidate guard column. This is a cut propagation of reference-calibration
variation, not a joint target-update of calibration draws and not an exact
Bayesian population posterior. Any target hierarchy must name that distinction.
The guard implementation uses the log-space equivalent of the same linear
convolution/interpolation, preserving arbitrarily small likelihood tails. On the
fixed32audit objects also compare proper reference scores between grids and
between8/12sigma convolution truncation, tolerance.02nat. These extra numerical
checks can reject a candidate; they cannot relax any original scientific gate.
