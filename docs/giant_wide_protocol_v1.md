# Fixed negative-coverage comparison (registered 2026-11-02)

Source: [complete Pro reviews](pro_review_20261102.md). This is existing-field
**development validation**, not a fresh untouched external test. Cold-v4, RGB,
previous classifier/FeH results and quality definitions remain frozen.

## Sample, roles and controls

Use the existing identity-deduplicated `giant_information_v1/labels.npz`, already
clean grz, |b|>30 and EBV<.03. Require RELEASE11010 and clean W1 S/N>=5;
no W2 requirement. Positives are reliable K giants (4200<=Teff<5000).
Negatives are reliable dwarfs: K-only for B-K, all temperatures for A-wide and
B-wide. Transition/unknown/raw nonstellar/other giants are evaluation strata,
never automatic negative labels. Application uses photometry only.

Sort the available DECam field indices. Each field is tested once; the next two
cyclic fields calibrate, all others train. Assert disjoint identities and fields
for all three roles. No field selection based on scores or giant yield.

Fit exactly B-K(grzW1), A-wide(grz), B-wide(grzW1): quadratic features, C=1,
16 antithetic band realizations, unchanged solver. Generate the same realization
for every shared training star. In each split Nref is K-only training count;
each realization of class c has weight Nref/(2*n_c*16). Class totals and total
weight are identical for all three models. Fit polynomial standardization on
K-only training rows; B-K/B-wide share it exactly, A-wide uses its optical
projection. No new sky weighting, feature, regularization or threshold search.

Primary paired B comparison uses the same K-only training color box. A uses
its optical projection. Also report each model's own train-color box as a
separate end-to-end diagnostic. Neither box sees calibration or test data.
For each box and model, freeze its serialized hash, calibrate 1% and .1%
thresholds on all reliable calibration dwarfs (box outsiders rejected), then
score the test field without refitting. Include .5 as a fixed diagnostic.
Calibration counts/resolution and binomial intervals are retained.

## Metrics and stopping

Report all-reference and inside-box denominators, cold/K/warm dwarf rates,
K giant counts and recall, r16–18 primary and r18–20 secondary, other classes,
W2 availability, selected K-giant abundance/gravity. No r>=20 giant promise.
Aggregate count intervals are descriptive; additionally resample whole fields
for paired recall/FPR differences. Fields overlap in training, so bootstrap is
not an independent replication significance claim. Score is a ranking, not
population probability. TPR/FPR and a hypothetical 1% two-class purity are
planning scenarios, not actual candidate purity.

Freeze a scoped selector only with useful recall and low leakage across fields;
otherwise report enrichment/ranking or close this fixed family. Do not tune
thresholds again based on test scores. Assess phase/FeH calibration gaps for
selected K giants; existing poor-metal lower RGB is not a general field model.
FeH mechanism study is closed.

## Requested distance figures

Reproduce frozen cold-v4 predictions on registered final fields, with strict
Gaia anchors and WISE20 routing; show positive corrected high-S/N Gaia inverse
parallax distances in kpc. Reproduce the fixed warm-giant kernel on its reserved
fields and compare with SPDist and matched SPECDIS where available. Preserve
source identities, support flags and rejection counts. Spectroscopic teachers
are noisy, possibly correlated references, not independent geometric truths.
Use log-log hexbin count density, equality line, relative residual panels and
sample-specific bias/NMAD. Sparse panels use individual points. Do not invent
K-giant photometric distances from the new classifier.
