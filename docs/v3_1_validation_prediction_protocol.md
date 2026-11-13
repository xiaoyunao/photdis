# Outcome-unopened reference-prior predictions

Use every identity in the frozen `validation_roles` native PSF field/common
photometry files, with their already frozen role labels. No new selection,
reference-distance filtering or residual-dependent subsampling is permitted.
This stage reads native photometry only. It hashes the reference manifest but
does not load reference values or score outcomes. Final outcome opening still
requires the complete product/method freeze, including the target-prior arm.

Compute original nominal SED + Normal(15,5) reference prior, response12 with its
matching global calibration + the same prior, and response12 plus the registered
32-shared-bootstrap luminosity correction + that prior. Fixed binary proportions
and class frequencies remain matched to each model. Production prediction never
conditions on known class or FeH. The accepted correction guards are a prerequisite,
not an independent-validation claim. A point-correction arm is retained separately.

For each model calculate native class-conditional log likelihood on [-20,50]mag
with .02 spacing; sum over all kernels and binary states before class-conditional
luminosity convolution. Store those likelihood curves so a future class-distance
prior can be withdrawn/replaced explicitly. They do not suffice for arbitrary
within-class SED tilts; such target priors require the full component likelihood.
All global bootstrap correction indices are shared across objects, not redrawn
per object. They are a cut calibration sensitivity ensemble, not joint target-
updated Bayesian uncertainty or independently measured absolute-scale scatter.

Choose64 fine-grid audit identities per cohort by SHA256 rank with salt319501,
before outcomes, using photometric identities only. Compare .01 grids, joint-bin
probability <.003 and mu-quantile difference <.03mag for all four views; keep all
original failures. This is sampled numerical evidence, not proof about every
object. Any subsequent source-score instability requires a separate retained
refinement. Export native TYPE, role, cell, model/calibration/prior hashes,
normalized log PDFs and shared draw class-distance probabilities. No score,
physical count, contamination rate or structure detection is produced here.
