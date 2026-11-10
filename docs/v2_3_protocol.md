# v2.3 frozen protocol — 2026-11-10

Authority: complete Pro review of83e1fe8 in pro_review_83e1fe8.md and user's
request to implement. No deployment, catalog expansion, agent or timer.

1. Re-score cached v2.2 noise with uniform and realized-design class priors,
per-class/macro scores; retain source-paired noisy-minus-clean changes. No new
noise simulations. Rejoin42076reserved identities to assembled references;
separate fitting, distance evaluation, source validity, actual parallax SNR,
negative/weak measurements and errors. Do not read their prediction outcomes.

2. Prospective near scope: verified native PSF, existing clean native grz
admission (DECam DR11 release11010, |b|>30deg,0<=EBV<.03), finite positive
grz errors, maximum grz error<.1mag, observed r<20, qnear>.95. Require colors
within min/max of the common admitted PSF training pool and QRF support flag
computed with that pool; apply the SAME support rule to both models. WISE is
optional, report no-W1/W1/W1W2 separately. Sky restricted to the existing13
DECam application fields for weighted scope counts, existing cal/dev/reserved
reference sky for evaluation. No inference of new-sky coverage. Evidence-cell
counts0/1–19/20+ are diagnostic only, never a reliability cut. Also report the
broader r<20,q>.95 selection to make the effect of quality/support explicit.

3. One fixed candidate:32-component extreme-deconvolution Gaussian mixture in
(g-r,r-z,Mg), full covariance, seed20261110. Heteroscedastic observation
covariance includes shared g/r color covariance and Cov(g-r,Mg)=sigma_g²;
Var(Mg)=sigma_g²+sigma_mu_reference². Reference errors use the existing chosen
Gaia/SpecDis modulus errors (Gaussian approximation; shared teacher/scale and
non-Gaussian distance-error limitations remain). EM on all admitted training
identities; fixed initialization from32component sklearn GMM, single start,
no hyperparameter search. Maximum256EM iterations, relative mean-loglikelihood
change<1e-5 after>=10iterations; covariance eigenfloor1e-6mag², fixed.
Training pool is frozen expanded_A membership with verified PSF and valid
reference+grz uncertainties, NO reliable-class requirement and no luminosity
cut. Auxiliary component class frequencies retain a separate unlabeled channel;
class labels never gate admission or application, and no global giant boost.
Analytic noisy-color conditioning returns full Gaussian mixture distance PDF,
including covariance of shared g with colors, no hard classification.

Method reference: Bovy,Hogg,Roweis2011, https://arxiv.org/abs/0905.2979 .
This estimates a reference-selection-dependent density, NOT a prior-free
likelihood. It does not infer a Galactic population prior or dust covariance.

4. Same-sample control: original HistogramQRF grz configuration128trees,
leaf32,depth20,seed20261102, shared-band Sobol128. Both train on exactly the
same identities/labels/bands. Historical expanded_A retained separately for
admission-effect context. No tuning on cal/dev or reserve. Freeze models and
training/program/protocol hashes BEFORE any reserved prediction. All42076
reserved IDs are evaluation-only; no reserved calibration subset required.
Unconverged or numerically invalid candidate is not opened on reserve.

5. Evaluate full PDFs, quantiles, interval scores68/95, teacher-convolved
scores, full signed Gaia likelihood/PIT with actual errors and negatives.
Report qnear reliability/risk/completeness and selected distance quality
separately; all, dwarf,evolved,other,unknown;0–2,2–5,5–10,10–20,20–50,
>50kpc; magnitude,color,error,WISE,sky and corrected metal-poor dwarf strata.
Track low-luminosity and unlabeled support, conditional far shapes and
SpecDis/SPDist disagreement. No clipping scientific metrics. Descriptive
source/spatial bootstrap for key paired differences; do not treat repeated
noise versions as independent stars.

Predeclared cal decision: no worsening all/evolved severe-error rate or mean
full-Gaia log score; dwarf tail degradation<=0.5percentage point; near far-
contamination degradation<=0.5point and completeness loss<=2points; require
improved absolute metal-poor bias or interval score without erasing unlabeled
low-luminosity support. Any failed/missing guard rejects adoption. New-ID
scores assess transfer after freeze and cannot reopen tuning. No automatic
adoption even if all guards pass. Posterior truncation/numerical error must be
logged and validated; retain full mixture parameters and QRF PDF chunks.

6. Poor-metal mechanism diagnostic: same objects across models and selections,
match ordinary dwarf controls in colors,temperature,reference source and
reference/photometric errors; report retained fraction and match quality,
not a causal attribution. No true-FeH application cut. Evidence priorities:
18–22noWISE, independent poor-metal scale, PSF nonstellar controls, independent
20–50giants and10–20teacher differences. Existing references cannot satisfy
independent-scale or membership audit by relabeling; list uncovered claims.

7. Deliver English figures in existing Pal5 style, compactCSV/JSON and audit,
requirements ledger including unresolved evidence acquisition. Preserve all
frozen historical artifacts, exact identities and negative findings. Milestone
WORKLOG/PLAN/git and safe three-endpoint sync. Never rerun completed stages.
