# Fixed published-reference batch protocol

Registered 2026-11-02 after review of35009d2, before new-member photometry or residuals.
Source review: pro_review_35009d2.md. Existing predictions remain frozen.

## Parent and stopping rule

- Freeze BV21 published Table2 via versioned arXiv2105.09526v2, cross-check
  Terzan7 with the published values in the review; preserve all seven method
  columns, asymmetric errors, publication identifier and byte hash.
- Deduplicate only identical repeated table rows, fail on conflicting duplicates.
  Require162 unique systems. Join all170 existing membership systems by explicit
  name aliases; retain every unmatched system, never substitute live web distances.
- Use existing complete VB21 member archive (P>=0.5) and complete DESI DR1 MWS
  catalog, not the selected190 list. Reject cross-system ambiguous identities.
  Register P>0.9 as primary member evidence and P=0.5..0.9 as secondary only.
- Apply existing reliable MWS giant quality rules and4200<=Teff<5000K;
  preserve temperature source, abundance and unknown RGB/RC/HB phase. Include
  the previously registered Terzan7 literature stars and unresolved Pal12 record
  as an already-observed diagnostic layer; do not add new literature targets.
- Complete this parent-table audit, directed photometry and all eligible scores;
  no iterative additions. Three comparable systems is a design aim, not a cut.

## Observation and pre-score freeze

- Fetch only DR11 sweeps intersecting typed member coordinates; exact Gaia IDs
  with G3, unique primary measurement per native release. Never choose by residual.
- Preserve original grz_clean plus PSF. No RCHISQ cut, uncertainty rescaling or
  new floor; report raw quantities and limitations. A does not require W1.
  B additionally uses existing w1_clean and S/N>=5; compare A/B only on same stars.
- DECam11010 is the scored system; BASS/MzLS11011 is a separately counted gap.
  Keep old |b|>30 and0<=EBV<.03 as independent scope flags, not quality overrides.
- Freeze identities, quality, support boxes, effective weight diagnostics and
  coverage in(gr,rz,zw1,r,Mg_ref), FeH and phase before inspecting residuals.
  No new color/support/absolute-magnitude rejection based on test outcomes.

## Frozen scoring and uncertainty

- Use original77/258 teacher pools, existing w1-qualified training eligibility,
  smoothing and shared-g likelihood. Explicitly exclude all reference-system
  member identities and record counts; no new classifier or model search.
- grz main baseline; grzW1 incremental comparison; unconditional prior control.
  Save per-star full mixtures remotely and public quantiles/PDF/CDF/support.
- BV21 mean distance is primary batch reference; retain each method component.
  Do not call every combined value geometric or independent of Gaia/teachers.
  Missing displayed parallax distance does not prove zero Gaia contribution:
  BV21 uses weak parallax information in the combination too.
- Shared distance errors shift all residuals within a system together. Report
  systematic bounds without sqrt(N); fixed +/-0.01 catalog-EBV perturbations
  are sensitivity scenarios, not corrections. No product of marginal star PDFs.
- Existing Terzan7 comparison retains registered17.05; BV21 is a separate
  sensitivity branch, not a replacement chosen to improve fit.

## Small image check and deliverables

- Check exactly1272/1282/1515 using available DR11 image/model/residual cutouts
  or independent measured photometry; record unavailable evidence explicitly.
  Same-image residuals cannot establish independent absolute photometric accuracy.
- Deliver reference mother table/method provenance, full cutflow/unmatched list,
  frozen pre-score coverage, per-system distributions and paired A/B results,
  teacher/local-support/latitude/reference addendum, plots and validation.
- Close with measured coverage limits; no calibrated field-giant product, no
  uniform offset inferred from the three historical stars, preserve all branches.
