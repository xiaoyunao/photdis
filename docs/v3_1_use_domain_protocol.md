# v3.1 scientific use-domain and evidence-availability map

Freeze before new reference outcomes. Only the existing26000application sample
and frozen526field/3032common objects enter; no fullcatalog expansion or new
source search. All admission/export objects must have native DR11 TYPE=PSF.
The original3540candidate membership is reproduced, not reselected by outcomes.

Fixed75observable cells: observed r edges16/18/20/22; extinction-corrected g-r
edges0/.3/.6/.9; W1 unavailable, legal signed weak SNR<5, or significant SNR>=5.
Use lower-inclusive bins. These are display/availability cells, not newly fitted
selection cuts. Candidate membership still uses the exact QRF support, observed
r and grz-error gate. The domain preparation stage reads photometry only and
retains all identifiers, candidate flags and empty cells before final freeze.

The map has separate axes; none may substitute for another:
1. Computational availability: outside candidate; missing prediction; retained
   numerical flags; or conditional-model output available under the actual
   ensemble/count/transfer gates. Native fine-audit coverage is an explicit
   separate flag. A258-row audit is not a full26000fine-grid calculation.
2. Empirical evidence availability: counts per source and role in the same
   observable cell and reference-distance bins0–2/2–5/5–10/10–20/20–50/>50kpc.
   Keep new-sky versus new-source/seen-sky separate. AstroNN, SpecDis and SPDist
   qualification/precision remain separate; never sum correlated channels into
   an independent sample size. Precision uses the registered20percent threshold
   (VAC sigma_mu converted through its lognormal fractional-error convention).
   Signed Gaia remains a forward-observation channel with no inverted distance.
   Known class/FeH inventories and score failures link to the full finalreport.
3. Common-system evidence: count members and unique scale systems separately.
   NGC1261 contributes ONE HST scale irrespective of its1732candidate members.
   Its scale/member comparisons do not populate independent field-validation
   cells or certify a fullapplication contamination rate.

For each candidate object export the sum of its marginal distance probability
in bins with NO precise new-sky field reference in the same observable cell.
This is an evidence-availability mismatch diagnostic. One reference makes a bin
nonempty but does not establish calibration, representativeness or reliable
coverage. Do not call the complementary mass a validated probability. Preserve
the source-specific count cube and all sparse/zero cells alongside it.

Model class-distance cell sums are conditional expected numbers within3540,
not measured populations or physical densities. Do not add independent-star
Poisson error bars; shared count covariance is supplied by the separate complete
cohort/highq products. Use the exact maincandidate bootstrap_evidence view and
retain all four controls. Export actual correction-state ESS/maximumweight and
every state/variant's numerical importance ESS, HMC ESS/Rhat, bridge and gate
flags. These finite bootstrap weights are not the64-SED probe evidence, nor a
full target-population response marginalization. Keep those scopes separate.

Scientific certification is not automatic: the map cannot promote a cell to
absolute-distance or population calibration merely because numerics pass or a
reference exists. Computational availability only means conditional model
exploration is possible with documented assumptions; it is not a scientifically
usable or calibrated-domain label. Extrapolated/untested cells and failed numerical results
remain explicit. The final interpretation must use the actual source-qualified
scores, sky/shared-scale uncertainty, radial recovery and finite-support limits.
If no broadly calibrated domain is established, state that outcome directly.
The availability map alone does not complete WP4: final scientific adjudication
must separately assign calibrated/only-ambiguous-or-multimodal/unvalidated or
failed domains, citing the actual product-matched scores and their correlated
uncertainty. Apply the already frozen calibration-guard definitions, including
the fewer-than20-references insufficiency rule. Do not infer absolute calibration
from teacher-center agreement or a failure to reject a noisy predictive test.

Final English figures show candidate abundance, precise new-sky reference
availability and missing-reference probability mass on these SAME fixed cells,
with W1 panels and explicit empty cells. A separate source/role table documents
Gaia, classes, FeH and one-system evidence. No plot merges those roles into truth.
