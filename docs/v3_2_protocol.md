# Photdis v3.2 targeted mechanism study

The review of commit 7699299 sets the scope. See [Pro review](pro_review_7699299.md).
The v3.1 outcomes are available for development diagnostics. A formal evaluation
needs a separately frozen set of outcomes on unused sky or reference roles.

## Opening comparison

Use the hash-verified v3.1 field score records and original candidate admission
for both external roles. The primary comparison uses strict new sky. Retain the
source-new seen-sky role separately. Keep all eligible rows with their original
numerical flags. The path is nominal reference, response12 reference, old
target, candidate zero, candidate point and candidate bootstrap evidence.
Score each edge directly with the same objects and whole sky-block bootstrap
(1,024 replicates and seed 320511, inherited from the underlying report
statistic). Compare log scores, argmax accuracy and qnear, overall and by
reference class. Retain empty or nonfinite results explicitly. Recover the
original absolute class and confusion tables with their recorded hashes.

Reliability uses equal-width probability bins from zero to one and one-vs-rest
dwarf and giant labels. Report occupancy and sky-block intervals. Reference
availability enriches the giant fraction, so reliability describes the evaluated
cohort. Add a retrospective frozen QRF comparison with matching band and
admission support and record exclusions by cause.

## Population-by-tilt experiment

Use response12/calibration12, zero luminosity correction, existing double-star
and local dust treatment, five radial components, and existing field hierarchy
scales and mean-one design weights. Cross broad (~26,000) and candidate (3,540)
with free and zero tilt. Each admission uses its own selection probability
calculated with the same kernel weights as the numerator. Compare the tilt
contrast in each population, the domain contrast at each tilt setting, and their
interaction using the same held-out development objects. Retain the nominal
response and old target as anchors. Estimate any gap from those historical
anchors explicitly. Only reuse a fit when its source arrays and statistical
identity match. First run a single zero-correction state for each arm.

The v3.1 matrices contain class-collapsed response and selection, so they can
support zero tilt. A free tilt requires separate component likelihoods and
selection integrals (or exact sufficient statistics for that tilt). A fit that
tilts only the numerator is ineligible. Before full production compare the
kernel-resolved zero collapse against v3.1 native and entry values on a fixed
pilot, including independent seed and quadrature checks. Confirm the stored
broad/free fit's selection before deciding whether it can serve as an arm.
Freeze implementation details and tolerances before evaluating the new arms.
A broad random sample of 3,540 objects is a follow-up control if population
and sample-size effects remain confounded.

Report signed Gaia log scores, absolute class confusion and giant probability,
class-conditioned distance scoring and far-branch support, qnear and class by
distance counts. Under a fixed common reference prior save per-class evidence
and compare to its target counterpart. Diagnose branch availability, suppression
and remaining class-conditional bias in color, magnitude, WISE, FeH and
reference strata. Select one targeted mechanism change after this diagnosis.
Keep the six-parameter correction available as a control.

## Auxiliary work and validation

Separate absolute evidence, relative weights, per-object probabilities, counts
and external scores in numerical acceptance. Target the remaining conditional
count sensitivity and the two-dimensional NGC288 integration, with both
original branches and failures retained. Test prior shrinkage using model
matched and empirical SED simulations with 10%, 30%, 60%, and 90% structure
fractions, several widths and backgrounds, and uncertainty on coverage.
Keep prior predictive coverage distinct from fixed-truth recovery.

Freeze candidate, new test roles and thresholds before outcome access. Seek
precise 2–5 and 5–10 kpc references, giant and metal-poor giant scales,
multiple common systems and a parent-sample contamination design. Retain the
10–50 kpc objective with evidence-dependent claims. Compare with QRF, nominal
reference and old target. All new samples require verified native DR11
TYPE=PSF at admission and export and explicit exclusions. Use allocated NERSC
compute for new scientific processing. The original v3.1 release remains a
frozen comparison.

### Pilot decision

Job 61259079 established native numerator identity on eight objects but failed
all 12 entry checks at the original precision. Rare kernels had raw entry
estimates as high as 1.0188. The refined job 61259164 normalized radial
importance weights within each kernel and compared 2^20 with 2^21 draws and
two independent scrambles. All 12 contexts passed. The largest absolute
zero-collapse discrepancy against v3.1 was 0.000362 and the largest tested
tilted seed discrepancy was 0.000353. The normalized estimate has finite
sample ratio bias, so require further transfer checks in the production
contexts and at the fitted tilt. Preserve original and refined pilot records.

The historical broad/free response replaced the numerator at fixed original
selection. Its denominator comes from the original profile-interpolated
selection table. This makes it an anchor outside the matched factorial design.
Both stored zero-state fits bind the same v3.1 target matrices and pass their
sampler and bridge checks. Additional conditional count flags remain distinct.
Compare matching zero-state fits on the common production integration rule if
the denominator change affects the diagnostic beyond its registered tolerances.


### First production pass

Partition the 26,000 frozen application rows by original row index modulo 32.
Each shard writes every row and both applicable admissions. Keep three entry
arrays per context (2^20 at seed 319711 and 2^21 at seeds 319711 and 319712).
Compare entry under each of the five pilot tilts across seed and level with
absolute tolerance 0.0005 and stabilized relative tolerance 0.01. Match the
zero collapse to the frozen v3.1 selection with the same tolerances. Retain
all context flags and arrays for targeted refinement and later fitted-tilt
checks. Both zero and free fits must use the same production denominator
within each cohort for the factorial contrast.

Refine the flagged contexts with 2^22/2^23 draws and the same two scrambles.
Keep the initial files and use hash-bound refinement sidecars. Fit both tilt
settings against the same merged input. For each arm compare MAP results at
all three entry levels and retain the weighted fixed-parameter log-entry drift
separately from per-object and count changes. Use four 512-draw HMC chains with
512 warmup draws, inherited regularization and a 0.9 target acceptance.
A fit with unresolved entry checks can run explicitly as a flagged sensitivity
diagnostic. Before interpreting the factorial, check fitted-tilt sensitivity.
Opened-field predictions use each posterior draw normalized separately and
the existing regional shrinkage for unfit fields, with common random seeds.
