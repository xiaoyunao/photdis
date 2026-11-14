# Registered giant branch probability adjustment

The opened 2×2 study at `25df7b0` found far giant modes in all 142
new-sky reference giants alongside low fitted giant probabilities.
Restoring tilt recovers most of the Gaia loss but reduces the Gaia score
conditional on known giant status. The mechanism experiment targets the
posterior class allocation and holds the fitted class-conditional distances
fixed. [Factorial result](v3_2_factorial_result.md)

Start from the conditional candidate/free posterior. For each object find
its giant log odds and those implied by the same response12/calibration12
photometry under the fixed common Gaussian distance prior and the frozen
SED mixture class frequencies. Use the midpoint of the two log odds.
Retain the candidate/free odds between dwarf and other conditional on
being nongiant. Rescale each class curve to its resulting mass and retain
its normalized class-conditional distance curve. The rule is a posterior
adjustment, with no second likelihood or classifier factor. The midpoint
weight is fixed at one half before scoring. Reference label proportions
and prospective outcomes do not enter the adjustment.

Evaluate one adjustment on both opened roles, separately for the full
photometric cohorts and candidate admissions. Compare to the same-object
candidate/free fit and the frozen nominal, old target, historical
six-parameter and QRF comparators where available. Report signed Gaia
score and PIT, class probabilities and confusion, teacher distance
scores and residuals, near probability and joint expected counts.
Numerically verify the class-conditional invariance and the existing
0.003 joint-probability, 0.03 mag quantile, 0.02 nat score and 0.003 PIT
tolerances. Keep nonfinite values and adverse paired changes. Opened
results diagnose the change. Prospective acceptance requires separate
frozen forecasts and evidence access. No parameter tuning follows the
opened evaluation. The adjustment defines finite-cohort conditional
probabilities, and supplies no generative population fit or evidence
normalizer of its own.
