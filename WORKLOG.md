# WORKLOG

## 2026-10-28

- Gaia-template census job 60457418 completed in 4m43s on NERSC CPU with
  exact crossover source hashes. Corrected parallax medians are 2.542/
  2.634 mas, S/N 29.65/29.32, Gaia G 17.26/16.92 and RUWE 1.041/1.017
  for RA35–45/RA170–180. Native residual medians agree near −0.01 mag
  for p=1–2 mas but differ −0.009 versus +0.075 for p>=2 mas (342/351
  sources). BP-RP bins 1.5–2.3 and 2.3–3.0 also show the field shift.
  Added aggregate manifest and binned figure; prepared an outcome-blind
  multivariate match including Gaia parallax/color/G. Next: run matching
  and registered native-transfer fields; do not infer a center correction
  from these two inspected fields.

- Registered four additional disjoint 10° DR11 south fields for frozen
  cold-v4 native transfer before reading their source rows: RA60–70 and
  RA235–245 are development diagnostics, RA330–340 and RA20–30 are reserved
  final tests. All 16 specified sweep filenames exist; only file existence
  and sky coordinates informed selection. Added an allocated CPU array
  runner and corrected the pilot cutflow's hardcoded original sky label.
  Config geometry and Python syntax checks pass. Next: construct fields
  after the exact-template Gaia population census, then score with a
  preregistered native-transfer script.

- NERSC outcome-blind native-template match job 60457203 completed in 19 s.
  One-to-one matching on g-r and r-z retains 460 pairs with final-minus-cal
  median residual +0.073 mag (bootstrap95 +0.038 to +0.105). Adding source
  A_g retains 434 pairs and +0.065 mag (+0.024 to +0.100). The offset remains
  after these measured-property matches. Added the small match manifest and
  prepared an allocated-node Gaia-template census for corrected parallax,
  zero-point, BP-RP, RUWE and native-residual strata. Next: run census and
  decide fresh registered field roles from the evidence.

- NERSC crossover job 60456650 completed in 10m23s; dependent strata
  job 60456926 completed in 8 s. Reused the same 500 Gaia source IDs in
  each prior deep field and matched target r0 across noise fields to better
  than 0.0001 mag (95th percentile). Native median residual is −0.009
  versus +0.050 mag. Same-template noise-field swaps change the paired
  median by at most 0.011 mag but deepen NMAD by ~0.09–0.10 mag at r=21–22
  and reduce acceptance 10–13 points. Selected used-color sigma>0.10 mag
  is almost absent; the red source stratum has much larger deep scatter.
  Added manifest, strata, plot, report. Prepared outcome-blind source
  matching on g-r/r-z and A_g; next run it on allocated CPU, then decide
  whether to register more fields.

- DESI boundary rerun 60456349 completed on allocated CPU after fixing
  on-demand Gaia `M_g`. Added per-source manifest and color/CMD figure.
  Development RA205–215 has 12/32 accepted cool giants (11 in A tier),
  while RA150–160 and RA220–230 have 5/8 and 14/14 metal-poor cool
  dwarfs rejected by the optical box. Every rejected dwarf crosses the
  blue `g-r` edge; many retain clean grz photometry. Documented the
  conditional DESI counts in `docs/cold_v4_type_boundary.md`. Changed the
  pending crossover from 1h regular allocation 60455711 to a 30m debug
  allocation 60456650; the job is running. Added an allocated-node
  postprocessor for paired residual strata by used sigma, support, source
  color and source extinction. Next: inspect crossover cells and strata.

- Type-boundary job 60456142 failed immediately because the pilot field
  diagnostics omit a stored `mg` column; the cold-v4 pipeline calculates
  labels on demand. Fixed the boundary audit to read `g0`, corrected
  parallax and formal error and calculate diagnostic `M_g` only for
  parallax S/N>=10; added an explicit CMD validity label. System job
  60456143 succeeded in 36 s: WISE20 retains the same 456 wide-binary
  pairs and 201 Blanco 1 members. Pair fainter-minus-brighter median is
  +0.0441 mag (component bootstrap95 +0.0141 to +0.0774), versus old
  +0.0455; matched pair change median is zero. Blanco 1 within-cluster
  NMAD remains 0.244 mag, not an absolute zero-point test. Next: rerun
  boundary audit, then inspect crossover when regular allocation starts.

- Prepared two independent NERSC audits while crossover job 60455711 waits
  for regular-queue priority. The DESI boundary script records every
  selected cold giant and optical-box-rejected metal-poor dwarf with quality,
  color-edge and support-radius evidence; a plot script uses the small
  manifest. The system v3 script compares old clean and locked WISE20
  predictions on the same wide-binary pairs and cluster members. Focused
  object-boundary and matched-pair tests pass. Temporarily held the pending
  crossover job to synchronize committed scripts safely, then release it.

- Added the registered-field crossover diagnostic using the same 500 Gaia
  source identities as each frozen v2 field. For each template and target
  brightness it pairs RA35–45/RA170–180 grz-base noise environments by
  intrinsic target `r0`, then evaluates all four source×environment cells
  under unchanged WISE20 routing. It records per-source native residual,
  degradation increment, fixed-tier noise effect and tier-switch effect,
  plus full acceptance and matched-environment differences. The exact
  decomposition and nearest-target matching pass focused tests. Next: run
  one allocated NERSC CPU job and inspect error/population strata before
  choosing any correction.

- NERSC weak-parallax v2 job 60454762 completed in 37 s, exit 0. The
  WISE20 route retains the same 607/202/222 weak-parallax DESI dwarfs as
  old clean routing, with C→B fallback changes. Forest raw low PIT in the
  two DESI-covered original final fields is 9.4%/9.9%; the explicit
  truncated-sample expectation is 9.0%/7.4%, and conditional low PIT is
  5.9%/7.2%. Model-minus-observed parallax median remains +0.040/+0.041
  mas. Copied the small manifest and documented that CQR interval endpoints
  do not define the tested CDF. Next: paired native/degraded and 2×2
  template/noise-environment experiment.

- Implemented weak-parallax v2 diagnostic without changing frozen forests.
  `QuantileForest.predict_weights` exposes the exact training-label weights;
  the new forward model convolves this discrete Mg distribution with Gaia
  formal errors and divides PIT by F(10 sigma) for the explicit selected
  parallax threshold. It reports raw/conditional PIT, the raw-tail null
  expectation, historical five-CQR-point heuristic on same objects, and
  old/WISE20 routing on common objects. Added unit tests for weights and
  analytic truncated Gaussian; 11 focused tests pass. Next: run on NERSC
  allocated CPU, inspect results and document inference limits.

- Started the latest Pro follow-up as a detailed active goal. Recovered
  `dc51856` on clean local/GitHub/NERSC main (all refs identical) and read
  current WORKLOG/PLAN. Audited extinction flow: DR11 SFD98-derived
  `MW_TRANSMISSION` enters all five AB magnitudes, Gaia `M_g` uses `g0`, and
  cross-field simulation scales flux by target/source transmission before
  recomputing colors. Added a numerical cross-field transmission regression;
  8 focused tests pass. Documented full-column and dust-uncertainty limits in
  `docs/extinction_audit.md`. Found the old weak-parallax PIT uses CQR endpoints
  as a full CDF and compares a parallax-truncated sample to uniform PIT;
  next: implement forest-weighted distribution and selected PIT under WISE20.

- Final deep manifest SHA-256 matches the NERSC compute output
  (`0f9a907aacf02757503a703c2d543985115d9a8aa4694983f13ad36ccccee13c`).
  Verified role/source separation, four observed-r bins, tier and tail-count
  partitions, and model/policy/registration hashes. All 52 local tests pass.
  `2ed06ce` synchronized final program, manifest, report and figure to local,
  public GitHub and NERSC. This checkpoint is ready for user review; next
  research round requires newly registered development/calibration/final
  fields because the v2 final result was inspected and remains frozen.

- Independent deep final job 60445991 completed on NERSC CPU in 8m55s,
  exit 0, after calibration v2 was hash locked in `a0de76f`. RA170–180 has
  500 disjoint Gaia templates ×4 draws per target noise depth. In measured
  r18–19/19–20/20–21/21–22 the selected observation/source counts are
  1707/493, 1694/493, 1607/489, 1296/484. Calibrated 95% final-modulus
  coverage is 92.6%/92.9%/95.1%/92.3%; 68% coverage is 63.5%/62.6%/
  65.2%/63.3%, with positive center bias +0.046 to +0.066 mag. The
  deepest A tier has 92.1% 95% coverage and 0.405 mag NMAD. Model,
  registration and policy hashes match. Rendered/visually inspected the
  measured-r coverage figure; prepared the bounded application decision,
  updated README and PLAN. Next: complete local validation, commit and sync
  the final manifest/report/figure, then stop for user review.

- Measured-r v2 calibration job 60444980 completed on NERSC CPU in 13m32s,
  exit 0. Copied and checked the small manifest against the locked policy and
  frozen preparation hashes. Observed r=18–19/19–20/20–21/21–22 have
  467/453/444/396 independent source identities in offset fitting; full
  route yields 467/453/447/398 observations. Calibration-field 95% coverage
  is 95.3%/96.0%/96.2%/96.0% after CQR. Added a focused report and final
  observed-r plotting program. Next: lock this exact manifest and run the
  independent RA170–180 final field.

- Found that the first deep CQR grouped by target-noise r bin while the
  application only sees measured r. Canceled unpublished final job 60444633.
  Changed the calibration key to the measured r=18–22 bin and deduplicated
  source identities within each bin across target-noise experiments. Added
  measured-bin aggregate metrics and a regression for boundary crossings;
  focused tests pass. The v1 calibration is superseded as an application
  interval rule. Next: rerun independent calibration v2 and final test, then
  update the application boundary report and synchronize.

- Locked-policy application job 60444634 (34 s) and DESI job 60444635
  (24 s) completed on NERSC CPU. Verified policy/preparation SHA. Three
  final fields retain 395,629 clean candidates versus 443,280 baseline,
  with A/B/C 249,654/79,418/66,557; 116,736 have r22–26 outside deep
  calibration. Strict-Gaia tier identities stay unchanged; direct model
  intervals remain weak in A/B. The two DESI-covered finals retain
  573/984 and 630/1,187 cool dwarfs, plus 13/1,419 and 10/1,687 known
  nonstars. Added bounded application report. Next: finish the independent
  RA170–180 deep final before stating distance interval use boundaries.

- Deep calibration job 60443692 completed on NERSC CPU (14m58s, exit 0),
  after the WISE20 policy lock. Verified registration, policy and model SHA;
  copied the small manifest. In RA35–45, 500 independent templates per depth
  yield accepted A/B/C counts 59/78/330, 100/192/161, 259/159/28,
  337/63/0 from r18–19 through 21–22. The deepest raw 95% modulus
  coverage 92.5% becomes 96.0% in the calibration field, widening median
  width 1.517→1.708 mag. Sparse deep B uses depth-global fallback;
  C has no selected calibration source there. Added report and final-mode
  hash gate against this tracked calibration. Next: sync, run independent
  RA170–180 final and re-audit locked policy on real candidates/DESI.

- Rendered and visually checked `figures/cold_v4/03_policy_depth.png` from
  the registered development manifest, comparing acceptance and final-
  modulus NMAD for clean highest-supported, clean WISE20 and strict-quality
  routing. The figure is linked from the policy report. Next: publish it
  with the calibration result and continue independent final testing.

- Added a DESI class-conditional re-audit for the newly locked clean WISE20
  route, reusing the same unique-spectrum labels and calibrated-domain poor
  subset as the original v4 audit. It records retained cool dwarfs,
  cool giants/nonstars, tier transitions and lost baseline selections.
  Local parse and focused tests pass. Next: sync once the calibration job
  completes, then run this audit alongside the real candidate composition.

- Extended the locked-policy application audit to report both real candidate
  tier/clean/brightness transitions and direct, un-recalibrated distance
  metrics on independently selected strict Gaia anchors in the same three
  fields. Source, route and forest hashes are checked. This keeps the new
  routing assessment separate from its forthcoming noise-specific CQR.
  Local parse and route-policy test pass. Next: sync after calibration job
  ends and run this bounded application audit on CPU.

- Center-diagnostic job 60443694 completed (25 s, exit 0), with frozen
  preparation SHA confirmed. Published manifest/report. RA150–160 strict
  anchors show +0.005 mag bias at r14–17 versus +0.084 at r17–19;
  redder and support-edge groups also shift positive. S/N15–25 versus
  >=25 gives +0.043/+0.045, four sky quadrants +0.030 to +0.062, and
  Gaia IPD multi-peak 0 versus >0 gives +0.044/+0.046. Valid DESI Fe/H
  groups are sparse; no center correction was fitted. Next: compare the
  independent deep-calibration/final fields and system brightness trend.

- Policy-development job 60442286 completed on NERSC CPU (16m06s, exit 0).
  Verified registration/prepared SHA and published the same-draw 500-source,
  2-repeat manifest. At target r=21–22, clean highest-available versus
  clean+WISE S/N>=20 has acceptance 83.7%/82.7%, modulus NMAD
  0.342/0.318 mag and direct 95% coverage 93.5%/94.0%; strict-quality
  acceptance is 56.8%. Locked clean+WISE20 before reading the separate
  calibration/final simulation results. Next: sync the lock and run the
  independent source/sky calibration, then final field.

- Rendered `figures/cold_v4/02_application_tiers.png` from the checked
  application manifest to show candidate tier×optical-quality composition
  beside strict-Gaia 95% coverage; visually inspected the output. Switched
  the registered deep calibration/final runner to a 30-minute debug CPU
  allocation for each bounded field job. Next: finish the policy-development
  job, choose and hash-lock one route, then run independent calibration/final.

- Added a no-fit v4 center-offset diagnostic for the three final fields.
  It groups the unchanged strict-Gaia residuals by selected tier, original
  six route groups, r/r-z, used-band error, support radius, parallax S/N,
  E(B-V), RUWE, Gaia binary/image indicators, sky quadrant and DESI-valid
  Fe/H. Local parse and focused group smoke check pass. Next: run it on
  NERSC CPU after current policy simulation releases the checkout.

- Weak-parallax forward job 60442287 completed (35 s, exit 0); verified
  frozen-preparation SHA and published its small manifest/report. Qualified
  DESI cool dwarfs with clean v4 selection and corrected parallax S/N<10:
  607/202/222 in the calibration/two DESI-covered final fields; corrected
  negative counts 5/1/0. The two final fields show approximate low PIT
  tails 9.4%/9.0% and model-minus-observed parallax medians +0.039/+0.040
  mas. Next: investigate selection and zero-point contributions; no center
  correction is applied from this diagnostic.

- System-validation v2 job 60442060 completed (33 s, exit 0). Verified its
  frozen-preparation SHA. Published 456 independent wide-binary pairs and
  201 Blanco 1 routed members in a small manifest and report. The fainter-
  minus-brighter pair modulus difference median is +0.046 mag with component
  bootstrap 95% interval +0.015 to +0.079; pair NMAD is 0.353 mag. Next:
  investigate this brightness dependence with photometric-error, color and
  binary diagnostics before applying any center correction.

- Policy-development job 60442059 failed before data simulation: the
  frozen `configured_field` helper reads only fit columns, so the new
  template selector tried to access absent `full_high_quality`. Fixed the
  runner to retrieve that flag, Legacy ID and `r0` from the same anchor-row
  order as a sidecar, checking Gaia IDs for exact alignment. Next: parse,
  sync and resubmit once.

- Added a DESI-qualified cool-dwarf weak/negative-Gaia-parallax forward
  check. It retains corrected parallaxes below S/N 10, maps frozen v4 Mg
  quantiles to a positive parallax distribution and convolves Gaia formal
  errors for an approximate PIT and standardized residual audit. It never
  inverts low-S/N measurements. Focused two-test and parse checks pass.
  Next: run on allocated CPU and report actual counts before interpreting
  the population diagnostic.

- Held-out system job 60441842 completed in 23 s. The three final fields
  contain 456 clean-routed published wide-binary pairs and one cluster with
  >=5 members (Blanco 1, 201 stars). Pair modulus differences have NMAD
  0.353 mag; fainter-minus-brighter median is +0.046 mag. Added component-
  bootstrap median intervals and the cluster name to a v2 result; preserved
  the original NERSC result and local ignored archival copy. Regular-queue
  policy job 60441428 stayed pending, so canceled it before execution and
  reduced development comparison to 500 independent templates × 2 draws
  per depth for a debug allocation. Next: sync and run both bounded jobs.

- Added independent deep split-CQR runner with one calibration draw per
  Gaia source and depth, hierarchical depth/tier/color/error/support groups,
  and a final field with clustered repeated-draw intervals. Added a test
  that rejects repeated source draws in calibration and checks sparse-group
  fallback. Added held-out wide-binary/cluster relative-distance runner
  using the already reserved published reference IDs and unchanged v4 route.
  Parse, shell syntax and focused tests pass. Next: run the system check on
  allocated CPU while the development policy job waits for resources.

- Registered three preexisting but cold-v4-independent fields for deep-noise
  policy development (RA50–60), source-independent calibration (RA35–45),
  and final evaluation (RA170–180) before reading their v4 simulation
  residuals. Added a frozen-support policy selector for available, optical
  clean, strict quality and WISE S/N 5/10/20, plus a same-draw development
  script using a WISE-unselected grz noise mother. Local parse and focused
  six-test suite pass. Next: execute policy comparison on allocated CPU,
  freeze the decision using development evidence only, and calibrate/test
  with the two separate fields.

- DESI type audit job 60440796 completed on NERSC CPU (22 s, exit 0).
  Published manifest and report after verifying frozen preparation SHA,
  source-deduplication and class/tier count identities. The RA0–10 field has
  no DESI matches; two other final fields retain 778/984 and 870/1,187
  qualified cool dwarfs, with 2/10 and 3/12 cool giants and 14/1,419 and
  12/1,687 known QSO/galaxies routed. Calibrated-domain metal-poor cool
  dwarfs are retained at 3/8 and 0/14. Next: quantify tier-specific deep
  behavior and compare limited routing rules on shared observations.

- Corrected application-audit job 60440314 completed (exit 0); validated
  prepared hashes and per-field count identities. Three new fields contain
  443,280 routed candidates: A/B/C = 251,212/85,717/106,351;
  398,387 pass `grz_clean`. Strict Gaia final A/B 95% interval coverage
  ranges 91.3–92.3%/85.3–91.8%, versus C 95.0–96.4%; nonclean A/B are
  lower. Added the small manifest and `docs/cold_v4_application.md`. The
  clean-but-low-precision candidate layer has 173,331 rows but no matching
  high-quality Gaia references in these fields. Next: finish DESI type audit,
  then design independent deep calibration for A/B and this quality layer.

- V4 DESI audit job 60440313 computed all four fields but failed while
  serializing a `+inf` upper error-bin edge to strict JSON. Replaced that
  displayed upper edge with `null`, retaining infinity only in the numerical
  comparison. Initial field read shows the RA0–10 field has no matched DESI
  labels; RA150–160 and RA220–230 do. No output manifest was published.
  Next: validate JSON serialization, sync and rerun once.

- Added `scripts/cold_v4_type_audit.py` and its allocated NERSC runner for
  DESI source-deduplicated v4 route diagnostics. Labels distinguish qualified
  3000–5000 K cool dwarfs/giants, other stars, known QSO/galaxy and unknown;
  output records class-conditional routed/clean rates with Wilson intervals,
  chosen tier, higher-tier support fallback, color/brightness/error strata,
  and calibrated-domain metal-poor cool dwarfs. A synthetic duplicate-spectrum
  check selected the higher-S/N counterpart and the expected labels. Next:
  sync and run on NERSC CPU after the application-count correction.

- NERSC job 60439866 finished the first v4 application audit (exit 0).
  Its conditional distance metrics correctly used the cold-Gaia mask, but
  `cold_Gaia_distribution` counted all Gaia reference rows because the
  shared counter had no parent mask. Added an explicit parent mask, with
  shape checking, to all candidate/quality counts. Preserve the original
  `data/cold_v4_application/audit.json` on NERSC as a diagnostic and write
  corrected `audit_v2.json`. Preliminary strict-independent Gaia results
  show much lower A/B than C interval coverage. Next: rerun, verify all
  count identities, then publish corrected per-tier results.

- NERSC application-audit job 60439716 stopped during environment setup after
  5 s: DESI's environment script references an unset `DESI_ROOT`, while the
  new batch wrapper enabled Bash `nounset` before sourcing it. Moved `set -u`
  after the DESI source command; no scientific code or data ran in that job.
  Next: validate the wrapper, sync, and resubmit once.

- Latest Pro review restart: fetched local/GitHub/NERSC Git history and confirmed
  clean identical `4a695af`. Updated PLAN for application-range audit, v4
  spectral labels, routing and independent deep calibration. Fixed
  `simulate_shared_flux` to record the effective inverse variance when the
  scaled template is noisier than the target; added a focused regression test.
  Corrected the cold-v4 report's mask-source description while preserving its
  published old-noise result. Added `scripts/cold_v4_application_audit.py` to
  score the unchanged forest/route by tier, optical clean/precision state,
  used-band error and support position in calibration and three new final
  fields. Its additional Gaia cold diagnostic uses Gaia absolute G, avoiding
  a Legacy-g-based dwarf decision on nonclean photometry. Parse, focused tests
  and import pass. Next: commit/sync this diagnostic, run on NERSC CPU,
  inspect application-candidate/strict-Gaia tier mismatch and clean fraction.

- Cold-v4 shared-noise job 60436242 completed on NERSC CPU in 16m02s with
  exit 0. Copied small `manifests/cold_v4_noise.json` and figure; verified
  prepared SHA, 223 templates × 8 repeats, tier-count identities, coverage
  tail sums and figure SHA. Full-route acceptance across r=18–19/19–20/
  20–21/21–22 is 95.5%/94.7%/93.2%/85.8%; strict-quality fraction is
  90.5%/83.6%/67.7%/32.3%. Final-modulus 95% coverage is
  92.5%/91.8%/92.4%/91.0%, all below nominal; 21–22 source-cluster 95%
  coverage interval is 87.8%–93.7%. Added results and deep-use boundary to
  `docs/cold_v4.md` and updated PLAN/README. All 44 local tests pass. Three
  new-field manifests and 12 plots were SHA-verified against NERSC and copied
  locally; remote untracked copies were preserved under ignored data/ for
  clean Git synchronization. Next: review diff, commit and sync all three
  endpoints, then report the deep undercoverage explicitly.

- Cold-v4 new final fields: jobs 60435341/342/345 processed all three
  preregistered DR11 fields; job 60436283 evaluated the published forest and
  complete route after the last source job succeeded (exit 0). Copied the
  small `manifests/cold_v4_final.json` to local. Selected strict cold anchors
  are 4,991/4,784/2,681; final distance-modulus 95% coverage is
  96.0%/94.7%/95.0% and relative-distance NMAD 9.8%/10.2%/10.0%.
  Color-brightness groups still show center offsets up to 0.153 mag and
  coverage as low as 92.8%. Added `docs/cold_v4.md` and a reproducible
  final-field plot; Python parse and plot execution pass. The shared-noise
  NERSC job 60436242 is running. Next: inspect its final-distance and
  acceptance results, finish report and sync the milestone.

- Cold-v4 preparation job 60435825 completed on NERSC CPU in 63 s with exit
  0. Strict fit anchors A/B/C: 7,782/7,351/7,341; matched-common fit uses
  7,341 identities for every tier and 223 frozen original test identities.
  Four development fields contribute 780 Gaia BP-RP 2.8–3.5 red anchors
  versus 140 in the original non-test pilot. Independent RA205–215 route
  calibration retains 1,257/1,317 cold strict anchors, with six group sizes
  148–240 and no fallback. Prepared forest, route and calibration hashes are
  in manifests/cold_v4_prepared.json; large joblib files remain on NERSC.
  Added exact published-manifest hash gate to final and noise runners. No new
  final-field catalog rows or residuals inspected. Next: commit/sync the
  preparation manifest, then evaluate every registered final field.

- Cold-v4 depth test prepared: added `scripts/cold_v4_noise.py` to reuse the
  same 223 frozen templates, 8 repeats and WISE-unconditioned grz target
  mother from the prior shared-noise experiment. Each simulated five-band
  observation now passes the expanded model's color box, per-tier support,
  C→B→A fallback and route CQR; it reports Mg, final modulus/distance,
  acceptance, strict quality and source-cluster uncertainty. Python parse and
  all 44 tests pass. NERSC jobs 60435341/342 completed source processing;
  60435345 and preparation 60435825 were pending. No new final-field rows or
  residuals inspected. Next: sync this program, then run it after preparation.

- Cold-v4 implementation: added `src/photdis/route.py` and
  `scripts/cold_v4.py`. Preparation uses the original non-test nside64 pilot
  cells and three former development fields for tier-specific strict-anchor
  training, checks frozen model hashes, and uses RA205–215 only for full-route
  split-CQR. The route tests its own 2/3/4-color tenth-neighbor support and
  falls back C→B→A. A separate `final` mode checks prepared hashes before
  reading the three new fields and reports full-route selection, Mg/final
  distance metrics, color/depth bins and sky-block uncertainty. Added two
  synthetic route tests; all 44 tests and Python parse pass. Submitted three
  new-field processing jobs 60435345, 60435342 and 60435341; they were
  pending at last check. No new-field results inspected. Next: commit/sync
  method, run preparation on allocated CPU, inspect training/calibration only,
  then freeze before final evaluation.

- Cold-v4 registration: local, origin and NERSC were clean and identical at
  `9d2dbc0` after fetch. Added configs/cold_v4_fields.json and three runner
  configs before reading any new final-field catalog rows. Training sources
  are original non-test pilot plus the three former development fields;
  RA205–215 is route calibration; new final fields are RA0–10/Dec−35–−25,
  RA150–160/Dec20–30, RA220–230/Dec20–30. Their centers have Galactic
  latitudes about −82.5°, +56.1°, +61.1°, and all 12 DR11 south sweep
  filenames exist. Fixed tier forests, feature-matched support, fallback
  routing and six-group split-CQR rules in registration. No new source row
  or model residual was inspected. Next: validate configs, commit/sync, then
  process fields on allocated NERSC CPUs.

- Final v2 checkpoint: committed selected-tier metrics, candidate-demo manifest,
  report, figures, source and tests. Commit `fb9c09f` matched local, public
  GitHub and NERSC after SHA-256-confirmed generated manifests were moved to
  ignored NERSC data backups before Git synchronization. All 42 tests passed;
  the checkout is clean. The original frozen model remains unchanged. Next
  method iteration: expand strict blue/red training anchors, calibrate the
  complete selected route, and reserve new final fields for its evaluation.

- Selected-tier independent validation: NERSC job 60434623 completed with
  exit 0. Fixed route (color box, 2-D tenth-neighbor support, highest available
  A/B/C) retained 23/25 cold anchors in the sparse final field and
  8,138/8,644, 2,817/3,032, 7,911/8,442 in the three large fields. Their
  direct forest 95% coverage was 91.3%, 91.5%, 92.0%, 92.3%; Gaia BP-RP
  2.8–3.5 coverage in large fields was 87.4%, 88.8%, 89.3%. Stored counts,
  spatial-block intervals and color diagnostics in
  manifests/selected_tier_validation_v1.json. Direct intervals need
  calibration of the complete routing flow; the previous CQR offsets were
  fitted to tier-specific corrected predictions. Next: update final report,
  sync and review.

- Selected-tier job 60434418 stopped before writing its manifest. Root cause:
  `load_field` returns a dict of columns; `len(table)` counted 44 keys rather
  than 47 first-field rows. Changed prediction allocation to
  `len(table["source_id"])`. Next: parse/test, sync and rerun.

- Whole-route validation preparation: added
  `scripts/selected_tier_validation.py` to score the frozen A/B/C forest with
  the fixed color box, corrected tenth-neighbor support and highest-available
  tier on all four registered final fields. It uses only photometry for routing;
  Gaia color is a diagnostic label. Python parse and all 42 tests pass. Next:
  run on allocated NERSC CPU and publish the small metrics manifest.

- Candidate demo and final report: NERSC job 60433940 stopped at import because
  the DESI Python environment was not loaded. Re-submitted with the environment
  as job 60433959; it completed in 30 s with exit 0. In the independent
  RA50–60 final field, 290,075 color-box rows were found and 2,000 sampled.
  The frozen A/B/C tiers supplied 1,033/309/658 medians; 1,363 passed color
  support, 835 strict optical quality, and 1,568 have observed r>=20. The
  per-star FITS remains in NERSC data/, while its hash and counts are in
  manifests/cold_candidate_demo_v1.json. Added the result and release
  decision to docs/pro_followup_v2.md and updated README/PLAN. Validation:
  42 local unit tests, Python parse and diff check pass. Next: final commit,
  three-end sync, and user review.

- Final Pro v2 evaluation: added `docs/pro_followup_v2.md`, final-field
  plots with Wilson intervals and n labels, and a bounded candidate-export
  utility. Original RA230–240 field yielded 25 cold anchors; registered
  supplemental RA170–180 and RA35–45 fields before source inspection.
  All four final fields were scored without changing frozen model, center
  coefficients, color edges or CQR offsets. The three large fields give C
  overall 95% coverage 94.4%, 95.9%, 94.9%, while Gaia red-end coverage is
  90.6%, 92.5%, 90.0%. The center correction makes C overall bias more
  negative in every large field; reject it for release. New output keeps
  cold-dwarf type unverified and support/quality/depth flags visible.
  Validation: local candidate/support unit tests and Python parse pass;
  NERSC candidate demonstration remains to run. Next: sync code, execute
  bounded demo on allocated CPU, then commit its small manifest and finish
  three-end synchronization.

- Supplemental evaluation code: added a separate `supplemental` mode to the
  frozen conditional model runner and a matching DESI label-audit mode. Both
  read the prepared calibration.joblib and its published SHA-256; no model
  coefficients, color edges, CQR offsets, class definitions or candidate
  thresholds change. Original final products remain immutable.

- Final-field sample-size diagnosis: RA230–240/Dec5–15 retains only 11,492
  of 4,647,352 PSF rows after the preexisting E(B−V)<0.03 cut, leaving 47
  strict Gaia anchors and 25 cold strict anchors. The result remains a final
  field. Before reading new source rows, registered two supplemental high-
  latitude final fields RA170–180/Dec15–25 and RA35–45/Dec−35–−25, with a
  minimum useful count of 200 and a rule to report every field. All eight
  DR11 sweep files exist. Next: sync registration, then process on compute.

- Type audit extension: added `--final` mode using the same frozen color and
  support rule and identical DESI class definitions in the two preregistered
  final fields. It writes a separate final manifest and figure after those
  source catalogs finish; no class threshold is tuned on their labels.

- Figure review: revised the target-support chart to show both color-box and
  box-plus-support rates. Revised DESI type chart to show disconnected field
  estimates, Wilson 95% intervals and label counts; the previous line plot
  visually implied continuity across fields and hid the 2-object giant bin.
  Both figures regenerated locally from the frozen small manifests and viewed.

- Calibration diagnosis: first prepared version used a global observed-r
  median across three r-z bins. New calibration field had 1,317/1,276/1,274
  A/B/C anchors, but blue-faint and red-bright groups had only 8–17 objects.
  Revised the registered method before final-field processing: each color bin
  now uses its own development-field r median. Outputs move to conditional_v2;
  v1 products remain on NERSC as a diagnostic archive. Added direct/center/
  calibrated Gaia-color diagnostics to measure red and blue bias explicitly.
- Baseline code audit: scripts/cold_baseline.py already trains A/B/C on the
  identical 1,230 common fit anchors and evaluates the identical 223 test
  anchors at fixed leaf 20; its manifest includes block-bootstrap ablation.
  No additional common-training experiment is required.

- Debugging: first conditional-preparation job 60432776 stopped while building
  Gaia color flags. Root cause: stripped six rather than five characters from
  the `gaia_` column prefix, producing `aia_match`; corrected in
  scripts/conditional_calibration.py. No prepared calibration or final-field
  outputs were written. Next: rerun preparation on allocated CPU.

- Task: add a DESI label audit in scripts/type_diagnostics.py across four
  development/calibration fields, observed r and r-z bins. It separates cool
  dwarfs, cool giants, other parameterized stars, known nonstars and DESI-
  unverified objects; rates are conditional on label with Wilson intervals.
  DESI target selection is recorded as a limitation. Next: run on CPU and
  inspect sparse giant/nonstar strata before considering type probabilities.

- Task: implement a two-stage conditional calibration. Prior north/south
  fields fit regularized residual-center corrections from Legacy/WISE colors,
  observed r and errors; the new RA205–215 field calibrates six fixed
  color-by-depth CQR groups. Final-field mode checks the prepared model hash
  and preregistered roles before one-time evaluation, including same-star
  A/B/C, tails, widths, distance modulus and sky-block resampling.
- Files: conditional.py, conditional_calibration.py, two focused tests.
  Validation: parse and full 41-test suite passed. The new final-field
  catalogs have not been processed.
- Next: run `prepare` on a compute node, review development/calibration
  diagnostics, commit the prepared manifest, then process final fields.

- Task: implement shared-noise final-distance experiment without changing the
  frozen v1 experiment. Added noise.py and noise_injection_v2.py: same 223
  A/B/C test identities, one target row and five-band realization per draw,
  WISE-agnostic grz mother, raw flux IVAR and target quality flags, extinction
  transfer, template-noise accounting, source-cluster uncertainty, Mg/mu/pc
  interval diagnostics and same-accepted-star comparison.
- Validation: focused noise tests, Python parse and full 39-test suite passed.
  Original v1 outputs are unchanged. Next: run v2 on allocated NERSC CPU,
  inspect metrics and correct any data-dependent issues.

- Task: read the complete latest Pro review (11,113 characters) and create a
  detailed active goal covering statistical corrections, same-noise final
  distance evaluation, conditional calibration, type labels and three-end
  synchronization. Baseline c3637cb and follow-up d6371ce remain frozen.
- Files: corrected PS1 median wording and noise scope in
  docs/followup_validation.md; added support.py, its two unit tests and
  target_coverage_v2 output; preregistered one new development-calibration
  and two untouched final fields in configs/next_fields.json and three runner
  configs; updated PLAN/WORKLOG.
- Checks: clean branch synced with origin/NERSC before edits; all 12 selected
  DR11 sweep filenames exist on NERSC. Python parse and two focused support
  tests pass. No new field rows or residuals have been read.
- Next: commit and sync the preregistration, rerun support on compute,
  implement and validate the shared-noise distance experiment, then develop
  conditional calibration without touching final-field results.

- Task: implement the latest Pro follow-up with frozen baseline models and
  preregistered field roles. Results in docs/followup_validation.md.
- Runs: NERSC allocation 60430171, nid200038. Three 10° fields processed;
  same-field recovery, external CQR, matched-band, flux-noise and target
  coverage scripts ran on allocated CPU. Allocation released with scancel.
- Recovery: 324 prior PS1<0.05 candidates all pass common nonoptical cuts;
  303 pass <0.03. Frozen A residual +0.109 mag. The PS1-replaced residual
  median is +0.12332 versus Legacy +0.10852 mag (difference of medians
  +0.01480); the median within-star Legacy-minus-PS1 difference is −0.00062.
  Matched strict-test difference
  +0.063 mag; all out-of-fit strict-control difference -0.020 mag.
- External: cold anchors 2077 calibration, 4145 north final, 271 south final.
  CQR 95% offsets A/B/C 0.071/0.081/0.086 mag. North C coverage 93.5%,
  south 94.3%; north C red-end 90.3%. Identical-star north C-A NMAD -0.048
  mag, sky-block 95% interval [-0.060,-0.040]; south -0.058 interval
  [-0.103,+0.0003].
- Noise/target: C input acceptance at r=21-22 falls to 32.2%, calibrated
  95% coverage 87.1%. Full-PSF denominators corrected in target coverage.
  Color-only cold box selects 4323/4782, 9067/9755, 686/786 Gaia cold
  stars across three fields; known DESI nonstars 65/2039 and 36/1330 also
  pass in the two fields with matches. North cool DESI giants 7/23 pass.
- Files: new followup, external validation/common, noise and coverage scripts;
  manifest/figure products; follow-up report, README, PLAN and WORKLOG.
  Validation: Python compile, 35-test suite, source/model/test hash
  checks, same-star identity checks, figures viewed; local/GitHub/NERSC commits
  synchronized after each analysis. Remaining: conditional interval
  calibration and probabilistic type selection after user review.

- Task: follow the latest Pro review after baseline c3637cb. Recovered clean
  local/main and matching GitHub/NERSC history; read complete latest review.
- Predeclared: one external-calibration field RA 140–150/Dec 5–15 and two
  final-test fields RA 190–200/Dec 15–25 and RA 320–330/Dec -25–-15.
  Confirmed all 12 DR11 south sweep files exist; fixed baseline manifest,
  model and original test-prediction checksums before processing new fields.
- Plan: correct recovery/spectroscopy conditions, analyze paired PS1 input
  replacement, direct field transfer, CQR interval calibration, flux-noise
  sensitivity and photometric target eligibility. No baseline model retraining
  or released-test refitting is authorized by the evidence yet.
- Sources: Romano et al. 2019 CQR original paper and official DR11 docs checked.
- Next: commit/sync field-role registration, then inspect count changes and
  implement the bounded analyses on allocated NERSC compute.

- Milestone: completed the same-field photometry audit and bounded cold-main-
  sequence baseline. The final forest run used job 60429237 on nid200038;
  model evaluation completed in 13.1 s. Final source commit 6ad4da9 and
  generated-product commit f3fb179.
- Sample: 1,802 strict anchors; exclude 82 external-reference anchors; verified
  Gaia color and cold CMD branch leave 1,527. NESTED nside=64 cells freeze
  1,039/250/238 train/validation/test; all original nside=8 selected stars
  were in train. Full A/B/C tier counts: 1,527/1,456/1,453.
- Results: empirical r-z cubic test NMAD 0.362 mag; QRF full-tier NMAD
  0.294/0.282/0.240 mag, actual 95% coverage 0.895/0.911/0.915.
  Same 223 test stars at leaf 20: 0.283/0.283/0.251 mag. Spatial-block
  bootstrap C-minus-A 95% interval [-0.095,+0.0004] mag. PS1-agreeing
  excluded cold group (324 stars) has +0.109 mag A-model median residual.
- Method fix: replace interpolated weighted forest CDF with empirical weighted
  order-statistic quantiles, excluding zero-weight labels. Added a regression
  assertion that quantiles are observed response values; three targeted tests
  and full 32-test suite passed. This changed final figures and metrics.
- Validation: test_predictions.fits has 238 unique Gaia IDs and 35 columns;
  all sky cells map to frozen test partition. Recomputed A/B/C test NMAD from
  saved per-star quantiles and matched the published manifest within 1e-4 mag.
  Viewed final audit/CMD/baseline figures. Models and per-star tables remain
  under ignored NERSC data/cold_baseline_v3.
- Files: docs/pilot_audit.md, docs/cold_baseline.md, corrected pilot docs,
  README/PLAN/WORKLOG; scripts/cold_baseline.py, src/photdis/quantile.py,
  three baseline figures and manifests. Remaining science checks: 95% interval
  undercoverage, bright-star recovery offset and few spectral test stars.
- Sync: final programs, manifests, documentation and figures match on local,
  NERSC and public GitHub at 79edb89; allocation 60429237 was released.
- Next: user review of the photometric selection and baseline before further
  training or expansion.

- Task: user extended the Pro review goal through a bounded cold-main-sequence
  baseline. The same-field audit completed on nid200059 in 441.8 s and synced.
- Findings: canonical Gaia reference 6,942; strict anchors 1,802; 22 catalog
  BP-RP=0 entries lack valid BP/RP. PS1 three-band comparison yields 617
  SATUR/FRACMASKED recovery candidates, 610 within 0.05 mag in all bands.
  ALLMASK/BAILOUT removes the known APOGEE giants. Input-tier strict anchors:
  1,802/1,699/1,688 for grz/grzW1/grzW1W2.
- Split correction: stage1 uses HEALPix NESTED. The nside=8 map assigns all
  1,609 initially selected cold anchors to train in this particular field;
  an earlier exploratory RING count was invalid and corrected to the user.
  Of the 1,802 strict anchors, 82 are in external reference groups. A finer
  cell split will be frozen before fitting.
- Files changed: pilot/audit plot fixes, quantile-forest implementation,
  cold-baseline runner, targeted tests and current plan. Commands: full
  32-test suite passed; NERSC reference/PS1/FITS counts inspected on compute.
- Next: sync code; run the cold baseline under a fresh CPU allocation,
  inspect metrics and figures, document the audit and model scope.

- Task: user approved continuation using Pro's latest review. Read the complete
  review (9,396 characters) from the referenced conversation.
- Current step: same-field photometry/coverage audit before a cold-dwarf baseline.
- Recovered: clean main at 139a440, all three remotes synchronized; prior pilot
  and scope instructions read. NERSC SSH authentication is valid.
- Sources verified: DR11 flag/flux semantics; PS1-to-DECam polynomial and color
  range; DESI RVS metallicity correction and validity range from author code.
- Located: /global/cfs/cdirs/cosmo/work/ps1/cats/chunks-qz-star-v3; median
  grizy, NMAG_OK, STDEV and per-object mean epochs. Gaia BP/RP flux/NOBS exist.
- Next: implement the bounded audit and run it on allocated compute.

- Task completed: 10°×10° pilot, RA [0,10), Dec [-10,0); all four named sweeps
  read on nid200006 under allocation 60427335 in 285.8 seconds.
- Results: 13,036,998 input rows → 5,574,170 PSF → 1,302,324 low-extinction PSF
  → 312,975 measured grz → 60,633 optical high-quality → 1,802 Gaia anchors.
  Extended: 49,906 records / 49,903 Gaia IDs; 6,683 negative parallaxes;
  extended_quality has 33,243 records including 4,586 negative parallaxes.
- Spectroscopy in grz: DESI 2,590 matches / 2,589 targets, 1,065 quality records;
  APOGEE 272 matches / 270 targets, 242 quality records. DESI classes are
  STAR 1,321, QSO 1,117, GALAXY 152. Quality giant records (logg<3.2): 19/41.
- Finding: 6,942 Gaia high-S/N references → 1,804 with optical high-quality
  → 1,802 final anchors. 5,079 reference stars carry z saturation flags.
  All 242 quality APOGEE grz matches fail the optical high-quality conditions;
  164 satisfy Gaia anchor astrometry. Final quality spectral anchors: DESI55,
  APOGEE0. The CMD shows strong removal of the bright main sequence and giants.
- Files: docs/pilot_10deg.md; four PNGs; result and inspection manifests;
  scripts/pilot_field.py; configs/pilot_10deg.json. Catalog stays on NERSC at
  data/pilot_10deg_v1/field_diagnostics.fits (313,064 diagnostic rows).
- Validation: exact cumulative photometry/anchor predicates; Legacy IDs and
  anchor Gaia IDs unique; saved-table bounds and counts checked; four figures
  viewed. Refreshed plot axes include the full measured separation/temperature
  range. Both original build and plot-only modes ran successfully.
- Sync: programs/figures/manifests returned via NERSC Git commits and sync pull.
  Allocation 60427335 released. Next: user review of this field and its cuts.

- Pilot diagnostics: add per-condition optical attrition within the high-S/N
  Gaia reference, including individual DR11 mask-bit counts; document pilot CLI.
- Compute allocation: 60427335, nid200006, 20-minute pilot allocation.
- Next: execute the four configured files and inspect the resulting figures.

- Task: User authorized a 10°×10° pilot with every selection/count, Gaia CMD
  and spectroscopy matching results; stop after presentation.
- Field: RA [0,10) deg, Dec [-10,0) deg; four named sweeps in pilot_10deg.json.
- Files: pilot config and script; current PLAN updated.
- Method: inspect raw field rows, PSF/sky/photometry sequential cuts; match
  prepared DESI/APOGEE pools and Gaia IDs; compare Gaia CMD before/after Legacy
  photometry cuts; tabulate separate spectroscopy quality cuts.
- Validation: Python syntax checks passed. Next: allocated pilot execution.

- User correction: first test one sky field and show it for review. I expanded
  two I/O smoke files into a 1,074-tile scan too early.
- Status: full scan finished; serial full-table merge stopped; 14,629,273,920
  bytes of partial tables are in
  data/stage1_v1/tables_partial_serial_20261028T060107Z. No final summary exists.
- Actions: stopped bulk-writer development; moved its unfinished local script to
  ../photdis_archives/20261028_stage1_scope_reset/; released both allocations
  60424566 and 60426781; updated the project scope and next checkpoint.
- Next: one-field sample/plot review. Full-footprint processing remains stopped.
- Communication: use direct factual explanations; omit defensive caveats.

- Task: Complete the full scan and compare input-feature coverage explicitly.
- Result: all 1,074 eligible tiles completed in 1,511 seconds with zero missing
  Gaia files; full pre-dedup grz base contains 101,087,401 point-source records.
- Files: plotting script adds a sixth figure comparing the deterministic optical
  reference, anchors and quality spectroscopy in color and apparent magnitude.
  Color-cell support is reported separately for all and train-partition anchors;
  it describes coverage, not distance accuracy or confirmed stellar purity.
- Validation: six synthetic figure renders and empty/support-denominator checks
  passed. Production finalization is still running; FITS read benchmarks did not
  support the initial indexed-read bottleneck hypothesis. Investigating writes.
- Next: finish finalization, run independent audits and generate actual figures.

- Task: Make the frozen test partition independently identifiable.
- Files: independent validator and its tests.
- Change: stream SHA256 of each role's ordered (Legacy ID, Gaia ID, partition)
  using documented little-endian 18-byte rows; also hash the scan manifest.
- Validation: five targeted validator tests passed, including independent byte
  encoding, empty tables and chunk-size invariance. No science module changed.
- Next: run these audits on the completed production tables.

- Task: Start production scan after corrected real-data smoke.
- Commands/results: 20 regression tests passed; corrected two-tile smoke on
  nid200018 produced 585 anchors, 18,516 extended objects including 2,622 negative
  parallaxes. Full scan launched with 32 workers over all 1,074 eligible tiles
  (1,165,426,139,520 source bytes). These smoke counts are not final results.
- Files: preparation/validation CLI, plotting denominator fixes, README runbook,
  source-backed selection notes, independent validator regression tests.
- Provenance: science code 78db981; production signature
  cd69dcbb11232cc40c526f8ed91a4938cafa67cde00b101fc77c90a6f2c7529f.
- Next: complete scan, finalize roles, independently validate and plot actual data.

- Task: Correct actual Gaia FITS semantics before the full run.
- Root cause: Gaia QSO/galaxy candidate fields are 5A strings `False`/`True`,
  whereas DUPLICATED_SOURCE is logical. Generic bool coercion made every nonempty
  `False` true; the initial two-tile smoke correctly exposed zero anchors.
- Fix/validation: explicit fail-closed boolean decoding; regression fixtures now
  match real FITS strings. Extended duplicate/spectrum-link tests also pass.
  Failed smoke products remain isolated in ignored data/stage1_smoke; they are
  not scientific results. Production uses the corrected, version-locked code.
- Next: repeat real two-tile smoke, then complete all eligible tiles.

- Task: Implement sample construction and leakage checks before full scan.
- Files: Gaia/holdout/stage1 modules, build/plot scripts, reference manifests, tests.
- Commands: allocated DESI/APOGEE projection; seven unit tests and two synthetic
  FITS integration tests; actual zero-point dependency smoke.
- Findings: 5,930,124 DESI primary records and 656,962 deduplicated APOGEE records
  projected; quality subsets 3,673,970 / 544,624 before Legacy/domain matching.
  Reference memberships/binary groups locked; physical cluster-region guards
  use max(rt,rJ), not circles spanning full tidal tails. References are leakage
  controls, not independent Gaia distance truth.
- Fixes: recover consistent spectroscopy Gaia identities before holdouts; reserve
  shared spectrum/ambiguous and duplicate identities; handle FITS Unicode and
  empty HDUs explicitly; normalize single-point Galactic-coordinate output.
- Validation: nine tests passed; initial real sweep smoke caught and fixed
  singleton-coordinate behavior before any fragments were written.
- Remaining/next: real sweep smoke, full eligible-sky scan, final coverage audit.

- Task: User approved Pro step 1 sample/coverage work and public GitHub visibility;
  training remains excluded.
- Files changed: current scope/plan, stage1 config and source-selection modules.
- Commands run: status/branch/fetch/history recovery; gh visibility update;
  representative DR11/Gaia/catalog header reads; primary-source literature checks.
- Key findings: GitHub is public; actual DR11 Gaia REF_CAT is G3; measured optical
  IVAR must be positive (some saturated Gaia stars use transformed flux with IVAR0).
  A 1,600-file/1.80-TB DR11 south source inventory supports a full eligible-sky
  scan, not the old tiny-field selection. Global ANYMASK/RCHISQ or color-box cuts
  would alter label coverage; retain diagnostics with explicit clean subsets.
- Validation result: schemas and provisional thresholds reviewed; compute job
  60424566 allocated on nid200018 for catalog processing.
- Remaining issues: build actual tables, freeze test groups and quantify coverage.
- Next step: finish modules, project spectroscopy/references and scan eligible tiles.


- Task: Complete archival/reset and the program/figure synchronization checkpoint.
- Files changed: `.gitignore`, `.gitattributes`, `AGENTS.md`, `README.md`,
  `PLAN.md`, `WORKLOG.md`, archive/source/sync docs, `scripts/sync.py`,
  `scripts/make_sync_check.py`, `figures/sync_check.svg`.
- Commands run: archive rename/verification scripts; `gh repo create --private`;
  remote Git init with `receive.denyCurrentBranch=updateInstead`;
  `python3 -m py_compile scripts/sync.py scripts/make_sync_check.py`;
  `git check-ignore` on data/model/log paths; `python3 scripts/sync.py push`;
  NERSC SVG generation/commit; `python3 scripts/sync.py pull`; GitHub connector
  metadata/file reads; SHA-256 and Git-blob comparisons.
- Key findings:
  - Private GitHub repository: https://github.com/xiaoyunao/photdis.
  - Fresh local root: `/Users/yunaoxiao/Desktop/photdis`; sole active NERSC
    checkout: `/pscratch/sd/y/yunao/photdis_pipeline`; `photodis` is empty.
  - Archives are independent of active roots; exact locations and old HEAD are
    in `docs/archives.json`. No old implementation or history was imported.
  - Push sends committed programs/docs/selected figures to GitHub and NERSC.
    Pull audits NERSC before fetching, fast-forwards locally, then publishes
    GitHub. No GitHub credentials were installed on NERSC.
  - Source/history guards reject data/model paths, unsupported/oversized files,
    and symlinks. Normal Git fast-forward and clean-worktree rules protect edits.
  - The likely Wenting Wang VAC is SpecDis (Songting Li et al. 2025, Wenting Wang
    corresponding author). Verified SpecDis v2.1 and SPDist headers and sources;
    registered their paths, version/matching issues and Gaia dependence in
    `docs/data_sources.md`. They are future auxiliary comparisons, not adopted
    independent truth or current training labels.
- Validation result:
  - Local archive: 149 file checksums verified; both remote archived directory
    inode/device identities preserved by rename.
  - Actual deployment and NERSC-to-local-to-GitHub figure round trip passed.
    All three main refs matched `0935e44706df73943a9f4dbc8f9102fb7c921eaf`
    before this final documentation commit.
  - Figure SHA-256: `d704a8c753456b8b3284669d8e2c0f5f0cf3159447bc0db8188cd3c1d8f2fd0b`;
    GitHub blob `e3941fbcd06f57aef5958f83f479ad7487b521a0` matches local Git.
  - Disposable/mocked guard tests passed for current/historical FITS rejection,
    locally-ahead pull rejection, failed remote preflight preventing fetch,
    deterministic SVG output and preservation of a conflicting existing SVG.
  - No catalog data were synchronized, and no scientific sample or model was
    built. Only source metadata and the synthetic synchronization SVG were used.
- Remaining issues: user review; LAMOST and independent cluster/wide-binary test
  resources remain unverified.
- Next step: synchronize this final record and stop for user confirmation.

- Task: User authorized archival/reset, a new GitHub repository and program/figure
  synchronization; explicitly stop before building samples or training.
- Files changed: fresh README, PLAN, WORKLOG, AGENTS, ignore rules and source/
  archive/synchronization documentation; sync helpers under development.
- Commands run: recovered old Git status/branch/fetch/history and project logs;
  verified NERSC authentication and absence of running jobs; local archival via
  Python rename plus SHA-256 verification; remote archival via directory rename.
- Key findings:
  - Old local HEAD: `f846aee9ce0ce39a5b28f48801b77b29506a0ff1`; no old GitHub repository existed.
  - Local archive: `/Users/yunaoxiao/Desktop/photdis_archives/20261028T123010+0800/project`;
    149 files checksum-verified.
  - NERSC archive: `/pscratch/sd/y/yunao/photdis_archives/20261028T123010+0800`;
    both directory inode/device identities verified after rename; active
    directories were confirmed empty before reinitialization.
  - New local Git root is `/Users/yunaoxiao/Desktop/photdis`; canonical NERSC
    checkout will be `/pscratch/sd/y/yunao/photdis_pipeline`.
  - Shared source catalogs remain outside reset targets. Previous worklogs and
    old Git history remain in the archive; old low-S/N inverted-parallax giant
    results are not valid scientific labels.
  - Small header/column/few-row reads may use login nodes per user clarification;
    full scans/crossmatches/training require compute allocations.
- Validation result: local and remote archives verified; private GitHub repository
  created using authenticated gh (connector has no repository-create operation)
  and verified through the GitHub connector; sync validation in progress.
- Remaining issues: complete GitHub/NERSC setup and figure round trip.
- Next step: finish this checkpoint, report exact archive/repository locations,
  then wait for user confirmation without building samples or training.
