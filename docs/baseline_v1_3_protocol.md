# v1.3: independent label eligibility and fixed evolved candidate

2026-11-04; baseline ebf37f8; source:pro_review_ebf37f8.md.
Preserve all frozen products and v1 deployment. No full-catalog inference,
new distance expert, hyperparameter search, cluster expansion or subagents.

1. Build a versioned reference table from existing v0 references (which include
all matched DESI candidates before distance eligibility). Preserve identities,
owner fields and spatial splits. Class eligibility requires qualified class,
no identity conflict/nonstellar flag and finite optical colors; it does not
require distance or Mg. Training retains optical_error<0.03 to isolate the
removed distance requirement. Classification evaluation includes all eligible
cal/test objects without that training-error cut. Distance eligibility and
source-specific distance evaluation retain the original definitions.
2. Count old/new class training and validation, distance and weak/negative Gaia
labels by split/class/r (<18,18–20,20–22,22–25,>=25), optical error
(<.03,.03–.1,>=.1), WISE (none,W1 only,W1W2), and g-r (<.5,.5–1,>=1).
Scan existing owned DECam parents only for denominators; no new predictions.
Save zero-evidence cells and class2 Teff/logg/Mg summaries. This is a finite
inventory of existing fields; describe concrete remaining acquisition gaps.
3. Where added training identities exist and all three classes remain present,
fit exactly the fixed v1.2 balanced forest once per A/B/C and same-training
A_on_B/A_on_C/B_on_C configuration. Compare old/new forests on the SAME expanded
cal/test IDs and also the original evaluation subset. Fix cal dwarf FPR budgets
1/2/5%; test never chooses parameters. No new luminosity mixtures are needed
for this classification-only data change control. If no new labels, skip fit
explicitly. Save identities, weights, models and hashes.
4. Reuse frozen B/forest predictions; candidate = single_mask(confidence=.9,
margin=.4) AND main_index==1, with support conditions unchanged. Use all
selected sources, including contaminants and unknowns. Save candidate IDs,
source-specific wide/mixture/main distance metrics and per-class absolute risk
with Wilson intervals, cal/test counts, fixed-parent design-weighted coverage.
Use existing row-wise full Gaia checks for the SAME prediction-defined group.
An empty group is a result, not justification for threshold adjustment.
5. Output explicit relative_comparison_pass, absolute per-class risks, and
candidate_only status (scientific absolute acceptance remains undefined).
Call class2 'other qualified stars / transition', not a pure subgiant class.
Correct the W2 wording: small incremental recall at2% does not imply negligible
AP or1% budget benefit. Publish actual outputs, figures and limitations, update
project memory, run focused tests and audit saved artifacts, commit/sync.
