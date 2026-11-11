# v2.4 frozen-model scientific-use audit

Authorized by the latest review of0f9c412, archived in pro_review_0f9c412.md.
No fit, hyperparameter/threshold selection, deployment, or full-catalog prediction.
All compute runs on allocated NERSC CPUs. Verified native DR11 PSF is mandatory;
record missing/nonPSF exclusions. All figures English. Preserve v2.3.1 bytes.

## 1. Concrete anomalous identities and reference scales
- Freeze QRF reserved near_scope plus supported Gaia:711objects; mark the147
  existing conditional PIT<.05 cases. Keep the other564 as the full denominator.
- Join the399QRF near SPDist references by exact identity. Compare corrected XD
  on the SAME711, regardless of XD selection. Retain signed Gaia observations.
- Report color, observed r, optical/Gaia errors, W1, sky and astrometric-quality
  strata; missing diagnostics stay missing. SPDist/Gaia agreement is not
  independent truth. No inverse parallax distance for weak/negative observations.
- Full PDF diagnostics: probability within reference mu±1mag; main-mode offset;
  reference-near local peak after fixed0.1mag Gaussian smoothing, prominence1%
  of maximum density. These operational descriptions do not identify a cause.
- Preserve2055metal-poor pairs. Evaluate Gaia anchor, SpecDis, SPDist separately,
  retaining both members only if both qualify; distances0–2,2–5,5–10,all.
  Bootstrap1000replicates by repeated ordinary-control identity and separately
  by sky-cell pair. Report both intervals and unique counts, not IID pair errors.

## 2. Photometric target, reference coverage, joint counts and reservation
- Freeze observable parent: existing13DECam fields, verified PSF, r<20,
  maximum grz error<.1 and common frozen QRF color/neighbor support. qnear is
  a subsequent model selection, never part of reference availability.
- Existing26000weighted photometric-parent sample is diagnostic, already viewed.
  Join references only after target admission; no reference-based sample cuts.
  Preserve missing evidence, weak/negative Gaia and source provenance.
- Reconstruct original QRF leaf class×Mg and source×Mg with same training IDs,
  trees, shared-g noise, Sobol128. Marginals must reproduce cached QRF PDFs and
  class contributions (absolute tolerance2e-6); compute joint distance-bin counts
  for0–2,2–5,5–10,10–20,20–50,50+ and near_scope. Missing-label contribution is
  not a physical class. Explicitly compare product-of-marginals as a diagnostic.
- Coverage means descriptive reference support in fixed observable cells, not
  representative validation; report none/1–19/20+ and photometric sample weights.
- Reserve up to250new identities per application field using identity-only
  ranking after observable admission, exclude all historical reference/training/
  inspected application identities. Store native PSF and photometry only; no
  reference association, prediction or outcome inspection for these IDs.
  This is same-footprint prospective identity reservation, not fresh sky or an
  independently validated final sample. System memberships remain unverified.

## 3. Matched-population information audit
- Reuse frozen v2.1 identical-ID grz/griz/W1/W2 probe/leaf probabilities.
- Use calibration-role class frequencies, and a cal-only r×sky prior with
  30object shrinkage to the global cal prior. Compare on identical dev objects.
  Use both natural reference composition and fixed uniform class weights.
- Report log score/gain, Brier, reliability and cell-cluster uncertainty;
  color probes versus r/sky-only baseline separate the available information.
  Existing partitions/probes are not information upper bounds; selection,
  error treatment and narrower matched-band coverage remain limitations.

## 4. Independent-template posterior population recovery
- Templates from development split2, no overlap with frozen model fitting.
  Require verified PSF, valid chosen teacher distance and grz errors<.02.
  Identity-only selection capped64ordinary dwarfs,64metal-poor dwarfs,
  48evolved and48other; same template never split across roles/replicas.
- Estimate absolute grz from template photometry/chosen distance; translate to
  fixed1,3.5,7.5,15,30kpc. Draw one shared band noise vector per replica from
  observed parent errors matched by r and colors (fixed scales1,.3,.3).
  Three noise replicas plus clean measurement case; no sampling from model PDF.
- Four predeclared design populations: ordinary dominated(.80dwarf,.05evolved,
  .15other;10%poor within dwarf), giant-rich(.50,.35,.15;10%poor), metal-poor
  rich(.80,.05,.15;60%poor), and balanced(1/3each;30%poor). Distance probabilities
  fixed(.25,.30,.25,.12,.08). These are stress designs, not Galactic priors.
- Compare point-distance histogram, integrated PDF and joint class-distance
  expectation against known injected counts, before/after observable selection
  and near_scope. Report stochastic predictive count intervals and their
  coverage across identity-cluster bootstrap pseudo-populations; distinguish
  conditional-on-template posterior variance from template/teacher uncertainty.
- Template teacher errors remain a systematic limit; repeat count truth under
  fixed±1teacher-sigma common distance-scale sensitivity without changing input.
  This tests conditional empirical SED populations, not absolute real-sky truth.

## Completion
Deliver actual identity joins, denominators, outcomes, English figures with
machine-readable sources, tests and unchanged-model checks. Assess whether the
data distinguish weight/branch/scale/observation explanations; allow unresolved
identifiability instead of claiming a unique cause. No automatic next model.
