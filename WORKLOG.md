# WORKLOG

## 2026-10-29

- Registered NGC 2419 as an independent cool-giant system before inspecting
  its DR11 rows: exact Gaia member IDs, RA 110–120 / Dec 35–45 sweeps,
  published RR Lyrae modulus 19.71±0.08 mag. The targeted allocated-CPU
  script reports individual measured grz quality, E(B−V), RVS type and
  SpecDis/SPDist distance overlap. Files: cool-system script/Slurm wrapper,
  giant photo protocol. Command: Python compilation passed. Next: run the
  system audit and compare with M3 and the reserved-field warm baseline.

- Corrected v3 giant VAC job 60488080 completed on allocated CPU: among
  2,487 clean low-dust RVS giants, unique exact catalog joins yield 2,479
  SpecDis and 2,341 SPDist matches; 2,217/2,044 pass published quality
  and finite-error criteria. The warm 5000–7000 K group has 2,002 SPDist
  labels; 4200–5000 K has only 42, and <4200 K has none. Registered a
  grz-only conditional-luminosity kernel-mixture experiment on the warm
  group with the eleven fields' existing roles and independent M3 system.
  Files: `giant_vac_v3` manifest, giant kernel and runner, protocol and
  focused tests. Commands: Slurm 60488080, Python compilation, two kernel
  tests and one summary smoke test. Validation: identity counts and tests
  pass. Next: obtain quality/profile manifests, execute the registered
  model on allocated NERSC CPU, then report reserved-field and M3 results.

- Root cause of the apparent giant VAC shortage: after source-ID deduplication,
  `giant_vac_inventory.py` reordered giant rows and TARGETIDs but joined to
  the pre-reorder Gaia source-ID array. Direct ID audits found 3,636 exact
  SpecDis and 3,687 exact SPDist pairs among 3,694 giants, while v1/v2
  mislabeled only 1/0. The new `deduplicate_giant_identities` returns aligned
  rows and both keys, with an end-to-end regression test. Files: giant VAC
  inventory, profile, quality audit, test; v3 results pending allocated job
  60488080. Validation: four focused tests and Python compilation passed.
  Next: check v3 matched/valid labels before fitting any giant baseline.

- Completed the bounded metal-poor forward-parallax comparison on allocated
  NERSC CPU (jobs 60487276 and 60487474). Five development fields contain
  6,672 stars and five reserved fields 4,947, including 81/52 negative
  corrected parallaxes. A continuous iron coefficient improves reserved mean
  log density by 0.114/star; β=−0.6655 mag/dex, and leave-one-development-field
  fits give −0.679 to −0.654. Expected negative counts are compatible with
  observed counts. Files: forward experiment and diagnostic manifests,
  scientific report, held-out comparison script/figure. Validation: four
  forward-model tests, Python compilation and manifest arithmetic. Next:
  giant quality and whole-system tests; no photo-only blue model adopted.

- Completed Pro P0 class/evidence correction and independent blue-anchor audit.
  Versioned joint v3 excludes one DR11 r-band discordant source from the
  trusted blue core while retaining its row; the candidate v3 export preserves
  all 395,629 source identities and all 20 frozen non-class columns. DR11
  corrected magnitudes already use MW_TRANSMISSION; the targeted CMD now
  displays M_g>10. Files: class audit report, v2/v3 joint manifests, candidate
  manifests, outlier manifest and expanded CMD figure. Commands: allocated
  Slurm evidence/export/validation jobs, identity and PS1 audit, focused tests.
  Validation: candidate equality check and source identity checks passed.
  Next: giant VAC identity diagnosis and external whole-system scale test.

- Corrected giant VAC join 60487258 produced 3,694 qualified RVS giants,
  2,487 with clean low-dust DR11 photometry and 13 published HSC_2310
  cluster members, but only one SpecDis and no SPDist strict valid label.
  A small-row inspection confirmed target/Gaia IDs are integer exact; the
  VACs may cover different observations or target programs. Added an
  allocated exact-pair versus target-only versus unique-Gaia-ID audit before
  changing the join policy. Files: VAC identity audit script/wrapper/test.
  Validation: one targeted test and syntax/shell checks pass. Next: inspect
  audit counts, then decide whether a cross-target same-Gaia join is justified.

- Giant VAC job 60486865 finished but its summary was all zeros: FITS string
  columns read back as Unicode and a byte-string giant comparison selected
  no rows. The same type error was found in pending metal-poor role/parallax
  predicates and MWS cluster STAR selection; those jobs were cancelled
  before execution. Normalized all three predicates to stripped Unicode and
  added FITS round-trip tests for giant selection and development/reserved
  deduplication. Files: three experiment scripts and two tests. Validation:
  six focused tests, Python compilation and diff check pass. Next: archive
  the empty giant artifact, sync repair and rerun allocated jobs.

- Added a bounded profile of matched giant VAC labels: per-temperature and
  low-gravity counts, per-star label uncertainties, SpecDis–SPDist same-star
  disagreements, and published cluster-member groups with reference
  distances. Files: `scripts/profile_giant_vac.py` and Slurm wrapper.
  Validation: syntax and shell checks pass. Next: run after the exact-ID VAC
  join and decide the giant fit and whole-system validation domain.

- Found and fixed a numerical integration failure in the preliminary
  metal-poor forward likelihood before running it on science data: fixed
  Gauss–Hermite nodes centered on the prior missed narrow high-S/N Gaia
  likelihood peaks. The queued job 60486996 was cancelled before starting.
  Recentered and rescaled each quadrature at its likelihood posterior mode,
  found by bracketed derivative search. Files: forward model, test and
  experiment description. Validation: direct dense-grid integrations at
  positive and negative observed parallax agree to <0.001 log-likelihood;
  three focused tests, syntax and diff checks pass. Next: sync and resubmit
  after versioned joint photometry audit jobs 60486994/60486995 finish.

- Prepared an independent-system target ranking from published Vasiliev/
  Baumgardt globular membership and distances, joined to DESI MWS qualified
  giant spectra by exact Gaia source ID on allocated CPU. This will identify
  an actual whole-cluster holdout before any giant photo-only fit. Files:
  `scripts/rank_giant_cluster_systems.py` and Slurm wrapper. Validation:
  Python compilation and shell syntax passed. Next: rank systems, select
  DR11-covered members and evaluate actual Legacy photometry overlap.

- Blue-anchor audit job 60486734 verified exact Gaia/DR11 identity for
  source 3601173023628559616 and no DR11 neighbor within 10 arcsec.
  DR11 minus transformed PS1 is −0.011/+0.463/+0.022 mag in g/r/z;
  DR11 r-band RCHISQ is 2097 with three exposures. The established
  `grz_clean` cut did not inspect RCHISQ. Added versioned joint evidence
  retaining every row but marking this star `r_band_discordant` and removing
  it from the trusted blue training core based on independent photometry.
  Pointed the forward experiment at v3. The queued pre-audit forward job
  60486919 was cancelled before execution. Files: independent audit
  manifest, joint-v3 script/wrapper and test, forward script. Validation:
  focused tests (3 passed), syntax and shell checks. Next: run v3 on allocated
  CPU, then the forward model after both v3 products exist.

- Implemented a registered low-degree color–absolute-magnitude comparison
  with zero versus one continuous calibrated iron term. The likelihood
  integrates positive latent parallax over photometric, abundance and
  intrinsic-scatter uncertainty, then convolves with measured Gaia parallax;
  negative measurements are retained. Development and reserved fields stay
  separate after Gaia-ID deduplication. Files: `metal_poor_forward.py`,
  experiment/Slurm scripts and two focused tests. Commands: `py_compile`,
  targeted pytest, shell syntax and a 180-star synthetic fit. Validation:
  finite negative-parallax likelihood, shared-band covariance and successful
  synthetic recovery. Next: run on allocated NERSC CPU and evaluate reserved
  parallax-space likelihood and prior sensitivity.

- Allocated joint evidence jobs 60486731/60486732 completed, preserving
  37,214/132,530 rows. The broader qualified RVS rule finds 609/3,085
  giants, of which 383/2,104 have clean low-dust DR11 photometry. The
  candidate v3 preservation validator 60486733 completed. Registered a
  SpecDis v2.1 / SPDist v1.0 exact TARGETID+Gaia-ID giant join with
  catalog-specific quality, Legacy-g luminosities and uncertainties, and
  published cluster membership. Files: `scripts/giant_vac_inventory.py`,
  Slurm wrapper, focused pair-join test, joint v2 manifests. Validation:
  `py_compile`, one focused pytest and `bash -n` passed. Next: run allocated
  VAC inventory and inspect label quality and independent-system overlap.

- Candidate v3 jobs 60486601/60486604/60486605 completed with empty stderr.
  Joint v2 jobs 60486608/60486609 stopped before writing because the old
  same-star FITS omitted `desi_spectype`; the new class function requires
  the original Redrock type. Repaired by joining each source field's
  diagnostic type back by packed Legacy ID with Gaia ID assertion, not row
  order. Added a focused reversed-row/identity-mismatch test. Prepared a
  single-star DR11/Gaia/DESI/PS1 check for the discordant blue anchor and
  expanded the targeted CMD plot to `M_g=10.8`. Files: joint v2 and star
  audit scripts, test, plot/wrapper. Commands: `sacct`, error-log read,
  `python3 -m py_compile`, `bash -n`, focused `pytest` (3 passed).
  Next: sync repaired code and rerun allocated jobs.

- Started Pro review of `caeb218`: read the full referenced reply via
  `read_thread`, fetched both remotes, confirmed clean and identical heads,
  and created a detailed active goal. Header inspection on the NERSC login
  node found MWS RVTAB `RR_SPECTYPE`/`RR_Z` but no Redrock `ZWARN` or
  `DELTACHI2`; SpecDis and SPDist registered files are present. Added a
  versioned general class-evidence definition separating broad-temperature
  RVS dwarf/giant status, blue training eligibility, abundance validity and
  raw Redrock nonstellar type. Prepared versioned candidate and joint-table
  reconciliation scripts while preserving prior products. Files: PLAN,
  `src/photdis/class_evidence.py`, new v3/v2 scripts and class test.
  Validation: focused class tests, Python compilation, shell syntax and
  diff check passed. Next: allocated re-export and cross-count audit, then
  anomalous-star and giant-label work.

- Completed the DESI-first DR11 follow-up and scarce-anchor audit on allocated
  NERSC CPUs (array 60485082, inventory 60485094, plot 60485687, anchor
  audit 60485813). Four registered 10°×10° fields yielded 132,530 unique
  DESI–Gaia–Legacy matches, 9,709 clean low-dust conservative cool-MS
  labels, 1,180 high-S/N geometric anchors, 18 with calibrated
  `[Fe/H]<−1.5`, and five blue of the old color edge (4 development,
  1 reserved validation). Among the low-metallicity clean core, 42 have
  corrected S/N 10–20, 228 have S/N 0–10 and three negative parallaxes.
  One blue high-S/N object's `g-r` disagrees with its `r-z`, Gaia color and
  DESI temperature; record for individual photometric review. The sample
  has 230 conservative cool giants, 103 clean low-dust, zero high-S/N
  geometric labels. Files: targeted field/inventory/anchor manifests,
  targeted joint figure, audit scripts, `docs/joint_calibration_v1.md`,
  PLAN. Commands: Slurm `sbatch` for field/inventory/plot/audit jobs,
  `scp` for small manifests and figure, `python3 -m py_compile`, `bash -n`,
  `git diff --check` and `python3 -m pytest -q`. Validation: all four field
  predicate checks pass; plot visually inspected; 83 local Python tests pass.
  Next science step: build giant luminosity labels and a forward-parallax method for the
  metal-poor weak-parallax population; no new distance model this round.

- Seven-field joint inventory 60484207 produced 37,214 unique DESI matches;
  among 3,213 conservative cool-main-sequence labels, 224 have clean,
  low-dust DR11 photometry and corrected Gaia S/N≥20, one has calibrated
  `[Fe/H]<−1.5`, and zero jointly satisfy that metallicity and the blue
  color region. Fifty-nine conservative cool giants include zero high-S/N
  geometric labels. Three v2 candidate jobs 60484208/209/210 exported
  395,629 rows: 1,211 carry conservative class evidence, four indicate
  nonstellar conflict, and 394,418 are unknown. Allocated job 60484704
  verified 14 frozen v1/v2 columns on every row. Fixed FITS Unicode-versus-
  bytes comparison in the plot/validation readers; rerendered and inspected
  the informative joint-overlap plot. Files: joint and candidate reports,
  manifests, figure, validator and plot script. Validation: manifest count
  identities pass; 83 Python tests pass with BLAS threads fixed to one (the
  local Anaconda BLAS teardown exited 139 without that setting). Next:
  finish the four DESI-ranked DR11 fields and reassess branch feasibility.

- Repaired ranking job 60484858 completed on allocated CPU in 68 s with
  2.01 GiB peak RSS. DESI-first selection yields 5,173 qualified
  4200–5000 K `logg−σ_logg>4.2`, calibrated `[Fe/H]<−1.5` targets with
  Gaia source IDs at −40≤Dec<35; 1,949 also have |b|≥30 and MWS EBV<0.03,
  including 137 raw Gaia parallax S/N≥20. Registered four available DR11
  10°×10° sweep fields from that ranking: two development and two reserved
  validation, with expected 8/8/6/4 low-dust high-lat raw-S/N20 DESI targets.
  Files: ranking script/manifest, four field configs, targeted field and
  joint-inventory Slurm wrappers; next: run DR11 measured-photometry joins on
  allocated CPUs and quantify the actual overlap.

- DESI-targeted field-ranking job 60484807 caught a row-alignment error before
  writing output: the projected MWS pool contains only PRIMARY RVTAB rows,
  while the Gaia/FIBERMAP HDUs retain every source row. Reconstructed the
  original PRIMARY indices before sampling those HDUs, and retained an
  explicit source-ID alignment assertion. Files: `scripts/select_desi_target_fields.py`.
  Validation: syntax check pending; next: sync and rerun on allocated CPU.

- Started Pro `1acf946` follow-up: read the full referenced review via
  `read_thread`, checked local/GitHub/NERSC heads and NERSC authentication,
  and created a detailed goal. Corrected XD coverage/derivative wording and
  blue base/clean counts. Added conservative DESI class-evidence fields to a
  versioned candidate export and prepared a seven-field DESI–Gaia–Legacy
  same-star inventory with reserved validation fields. Files: XD/blue docs,
  candidate/class-evidence and joint-inventory code, config, tests, PLAN.
  Validation so far: Python compilation and three focused candidate/evidence
  tests passed; joint inventory still needs allocated-run verification.
  Next: sync this code milestone, submit allocated jobs, inspect results,
  report branch feasibility and update candidate documentation.

- Color-response job 60483193 completed on allocated CPU in 43 s, exit 0,
  empty stderr and 1.45 GiB peak RSS. Its small manifest matches the
  frozen-comparison SHA256 and reproduces 1,257/6,594 development stars,
  with color-bin counts summing exactly to each field. In both occupied
  `(g−r)₀` bins and both fields, XD's median absolute +0.01 mag g/r
  distance-modulus response exceeds frozen v4's. Added the table to
  `docs/cold_v5_xd_result.md` and rendered/inspected
  `figures/cold_v5/03_xd_color_response.png`. The registered v4-retain
  decision remains unchanged. Validation: 81 tests passed, Python files
  compiled, Slurm wrappers passed `bash -n`, and `git diff --check` passed.
  Next: commit and synchronize the last report, manifest and figure, then
  close the active goal after verifying all three Git heads.

- Color-response diagnostic job 60483193 remains live in Slurm pending
  shared-partition resources. The earlier identical-model comparison used
  548,752 KiB peak RSS, so reduced this pending job from four CPUs/8 GiB
  to one CPU/2 GiB with `scontrol update`; its code already sets numerical
  threads to one. Slurm accepted the new request and the job remains
  pending, without resubmission. Next: inspect its terminal output and
  color-bin manifest when resources become available.

- Requirement-by-requirement audit found that the frozen XD comparison
  reported residuals by color but g/r perturbation response only by whole
  field. Extended `scripts/cold_v5_xd_compare.py` with color-bin responses
  and a separate diagnostic output linked to the frozen comparison SHA256.
  The original model selection/gate and candidate products remain unchanged.
  Three focused comparison tests and Python syntax validation pass. Next:
  sync the extension, run it on allocated NERSC CPU, then document the
  color-dependent response and finish the goal audit.

- Candidate export jobs 60482677/60482678 completed on allocated NERSC CPU
  nodes with exit code zero and empty stderr. RA0/RA220 contain
  197,165/72,866 selected rows in one FITS table each; header, columns and
  first/last rows were checked on the login node. Their small manifests
  match remote SHA256, and the three valid field manifests total 395,629
  selected rows (A/B/C: 249,654/79,418/66,557) with 47,372 infrared
  fallback-conflict flags. The invalid original RA150 product stays
  archived. Files: two new candidate manifests,
  `docs/cold_v5_candidate_export.md`, README/PLAN and the expanded XD
  error-tertile report. Next: run final validation, commit and sync.

- Follow-up completion audit: local/origin/NERSC all at `58e376a` with a
  clean checkout. Candidate jobs 60482677/60482678 are still pending in
  shared QOS for resources (8 GiB, four requested CPUs each). Reviewed the
  XD fit, likelihood, posterior and same-star comparison code against the
  registered covariance/selection logic; no new implementation fault found.
  Added measured-error-tertile interval coverage to
  `docs/cold_v5_xd_result.md` so the raw probability diagnostics are visible
  alongside the failed optical-response gate. Next: inspect both allocated
  export products when Slurm starts them, then commit this documentation
  and synchronize the three copies.

- Allocated comparison job 60482510 completed for preregistered XD K=32 on
  1,257 and 6,594 identical development stars. XD reduced distance-modulus
  NMAD by 0.0263/0.0255 mag, but raw 95% coverage fell from 94.59%/94.34%
  to 89.58%/91.64%. Its fixed-tier +0.01 mag g/r response was larger in
  both fields. Ran `scripts/cold_v5_xd_decide.py` against the unchanged
  configuration: center/NMAD/tail checks passed, both optical response
  checks failed in both fields; retain frozen v4 and close this model
  comparison. Files: comparison and decision manifests, XD result report,
  same-star figure and its plot script. Visually inspected the figure.
  Next: validate, sync, and finish candidate route exports.

- Corrected candidate export job 60482344 completed on allocated CPU. Its
  RA150 FITS has one table extension and 125,598 rows (A/B/C:
  71,472/28,991/25,135); 14,547 carry infrared fallback conflict.
  Header, columns and few rows were inspected on login, and its small
  manifest copied locally. Two more allocated exports 60482677/60482678
  were submitted for RA0/RA220 and remained pending at last check. Full
  catalogs stay in ignored NERSC `data/cold_v4_candidates/`.

- Blue inventory 60482142 completed on allocated CPU. Four fields contain
  4,207 strict blue Gaia luminosity anchors (404/1,037/1,624/1,142);
  142 have 18≤r<20 and none in the recorded 20≤r<26 bins. The 44 unique
  quality DESI metal-poor
  cool dwarfs blue of the v4 box have zero overlap with these stricter
  S/N≥20 anchors; four retain Gaia M_g at S/N≥10. The same blue region has
  DESI giant and known-nonstar controls, while its Gaia CMD includes a
  distinct faint sequence. Updated the independent blue-branch training
  definition, corrected the figure's distinct S/N thresholds, verified the
  plot-data SHA256 and visually inspected the rerendered figure. Files:
  `docs/cold_v5_blue_branch.md`, `manifests/cold_v5_blue_inventory.json`,
  `figures/cold_v5/01_blue_inventory.png` and its plot script. Next: keep
  blue branch separate and finish the selected K=32 comparison.

- XD refits 60482139/140/141 completed with converged A/B/C models and
  exact frozen training hashes/counts; minimum latent covariance eigenvalues
  are at or above the registered 1e-4 mag² floor. Copied only small manifests
  locally; models remain on NERSC. The preregistered equal-field held-out
  conditional NLL is 0.02515/0.00626/0.03742 for K=16/32/64, selecting
  K=32. Its per-field strict-anchor counts are 1,257 and 6,594. Next:
  commit the selection, sync three ends and submit the one same-star
  development comparison; use its locked gate before any new final field.

- Candidate export 60481110 finished but wrote seven FITS table extensions
  (18,017/17,905/18,343/17,735/18,061/18,434/17,103 rows) because
  `fitsio.write(..., append=True)` ignores `append`. The manifest total was
  125,598, so first-extension readers would see only 18,017. Archived the
  invalid product and manifest under ignored NERSC data. Replaced the writer
  with `FITS(..., "rw")[1].append` and a single-extension/row-count check;
  the two-chunk targeted test passes. Field FITS sizes 154–534 MiB support
  8 GiB requests; updated four pending jobs and future wrappers. Revised
  K=16 fit 60482139 completed with all three tiers converged in 64/30/51
  iterations. Next: sync, rerun candidate export, inspect remaining fits.

- Validated the optimizer/blue-side repair with seven focused tests and
  Python compilation. Commit `2fb80a4` synchronized to all three copies.
  Resubmitted allocated shared-CPU jobs 60482139/140/141 for K=16/32/64
  and 60482142 for the blue inventory; the old pending candidate export is
  60481110. Updated the thread heartbeat to follow the new job IDs. Jobs
  were pending for priority/resources at the last check. Next: inspect
  terminal outputs and continue with registered selection and comparison.

- NERSC job 60481106 finished K=16 in 66 s with the exact frozen A/B/C
  identities (7,782/7,351/7,341) and positive covariance eigenvalues.
  The 40-iteration cap was active for A and C; their last log-likelihood
  gains were 5.26 and 7.06. Archived its manifest and ignored NERSC models,
  cancelled still-pending K=32/64 jobs, raised the cap to 200 for all sizes,
  and required `converged` before model selection. No held-out scores were
  compared before this optimizer repair. Blue inventory job 60481185 failed:
  `training_blue_coverage` requested `ebv` from the compact v4 anchor
  sidecar. The source field already enforces the low-dust domain, so the
  count now uses its selected/clean flags; added a test without an `ebv`
  column. Next: validate, sync, resubmit fit and blue inventory jobs.

- Added `scripts/cold_v5_xd_decide.py` and one focused test to apply the
  registered center/NMAD/tail/g-and-r response gate on both held-out
  development fields. `docs/cold_v5_xd_protocol.md` now labels both compared
  interval sets as raw diagnostics. The focused test, Python compilation and
  `git diff --check` pass. Next: score actual comparison results, apply this
  gate once, and either retain v4 or lock one candidate for new validation.

- Added `scripts/plot_cold_v5_xd_compare.py` for a same-star development
  figure of center, NMAD, raw interval coverage and fixed-tier g/r response.
  A temporary two-field manifest rendered a nonempty 129 kB PNG; script
  compiled. Next: render and inspect the real held-out results after NERSC
  jobs complete.

- The five allocated jobs remained pending in shared QOS. `scontrol show job`
  revealed that the 48 GiB request reserved 26 CPUs per job despite only
  four requested compute threads; reduced pending jobs to 16 GiB (nine
  scheduler CPUs) and updated four Slurm wrappers for future runs. The
  new blue inventory initially referenced a nonexistent field `mg` column;
  commit `225b47c` derives it from extinction-corrected g and positive
  high-SNR corrected Gaia parallax instead. Added the missing fixed-tier
  +0.01 mag r response to `docs/pro_round_v5.md`: medians
  +0.0173/+0.0209/+0.0105 mag, NMAD 0.048/0.048/0.046 mag. Commands:
  field FITS header inspection, `scontrol update`, bounded manifest queries,
  Python syntax and targeted tests. Next: inspect allocated job logs and
  results, then choose one XD size on held-out development fields.

- Four-core, 48 GiB NERSC jobs were pending for priority under exclusive
  regular QOS. Consulted current NERSC queue policy and changed the five
  still-pending jobs 60481106/108/109/110/185 to CPU shared QOS, which
  fits their small resource requests; matching run wrappers now default
  to shared. Added `docs/cold_v5_blue_branch.md` with the 44 spectroscopic
  box-rejected metal-poor dwarfs, their sparse Gaia overlap and a separate
  label/anchor/holdout contract. Added a Gaussian simulation check of
  correlated distance-posterior coverage; four targeted XD tests pass.
  Next: wait for allocated jobs, inspect outputs and then update this
  branch definition with new field/brightness/contaminant counts.

- Added the pre-result method document `docs/cold_v5_xd_protocol.md` and
  comparison/select scripts. The comparison reads the same Gaia strict
  anchors and frozen WISE20 tier, scores v4 and XD distance-modulus center,
  NMAD, tails, coverage and width by field/tier/error/color/conflict, and
  applies identical fixed-tier +0.01 mag g/r perturbations. `cold_v5_xd_select`
  chooses one 16/32/64 fit by equal-field held-out conditional log density.
  Focused comparison tests (5) and Python compilation pass. All five NERSC
  jobs remain pending for priority; no results inspected. Next: synchronize
  comparison code, collect fit manifests when available, then choose one
  configuration and score it once on the development fields.

- Submitted allocated NERSC jobs 60481106/108/109 for the three XD sizes
  and 60481110 for one frozen-v4 candidate export; all were pending at the
  last queue check. Added a bounded blue-side inventory program and plot:
  it counts unique quality DESI metal-poor dwarfs, giants and nonstars
  against the v4 optical boundary, with strict Gaia anchor availability
  by field and brightness. The full sampled plot rows remain in ignored NERSC
  data; only aggregate counts and a small figure will be tracked. Python
  compilation, shell syntax and `git diff --check` pass. Next: sync and run
  this inventory on an allocated CPU, then inspect the fitting/export jobs.

- Read the complete newest Pro review (a18ef9b assessment) through
  `read_thread`; opened a new detailed goal. `git status`, branch, fetch and
  15-commit history confirm clean three-end `a18ef9b` before edits. Registered
  `configs/cold_v5_xd_experiment.json` with frozen v4 identities, 16/32/64
  components, development fields, fixed modulus prior and a decision gate.
  Added per-candidate frozen-route export and heteroscedastic full-covariance
  XD mixture prototype, including correlated band/parallax error propagation
  and analytic one-dimensional distance-modulus posterior. Focused local
  tests: 4 passed; Python compilation and shell syntax passed. The complete
  72-test suite passed with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1`; without these limits, all assertions pass but the local
  process exits 139 after pytest teardown when the new sklearn fit test is
  included. This points to a local threaded-library teardown interaction;
  NERSC job wrappers already set the same thread limits. Next: run bounded
  exports and fits on allocated CPUs, compare on identical development
  stars, then define the metal-poor blue branch.

- Final v5 review: checked the fallback manifest counts and selected plot,
  corrected report wording, and verified the route decision against labeled
  development objects. `PYTHONPATH=src:.deps python3 -m pytest -q` passed
  68 tests; `python3 -m py_compile scripts/plot_pro_round_v5.py` and
  `git diff --check` passed. Files changed: `PLAN.md`, `README.md`, this
  worklog, `docs/pro_round_v5.md`, `scripts/plot_pro_round_v5.py`, the
  fallback manifest and tradeoff figure. Next: commit, synchronize all
  three checkouts, then obtain independent validation before any route change.

- Fallback audit 60476326 completed on allocated CPU in 24m30s. Development
  clean candidates with reliable higher-tier support conflict number
  7,808/71,460 (RA205), 17,724/147,122 (RA60) and 50/363 (RA235). A hard
  hold removes 10/12 accepted DESI cold giants in RA205, but also 149/1,116
  DESI cold dwarfs and 44/1,257 strict Gaia cold anchors; RA60 losses are
  21/143 and 272/6,594. Ten A-tier giants with conflict have WISE masks zero
  and contamination fraction <1. Only about half the sampled conflicts remain
  outside support in >=90% of 64 photometric-error draws. Decision: retain
  frozen v4 distances and add a type-risk conflict marker; no hard hold or
  center correction. Copied aggregate manifest, made/inspected tradeoff
  figure, updated report/PLAN/README. Next: final tests and three-end sync.

- Optical-response job 60476003 completed (3m44s); same-star rerun 60476083
  completed (1m12s). Copied aggregate manifests and plotted/inspected both.
  The north/south center sign appears in A/B/C on identical stars; g-only
  PS1 replacement drives most center response, with worse RA20 scatter.
  Fallback 60476051 exposed missing z0/W10/W20 input columns for magnitude
  perturbations after raw WISE reads; added those saved columns and changed
  raw reads to identity scan plus selected rows. Compile passes. Next:
  rerun fallback audit, interpret label losses, finish method decision.

- Same-star job 60476002 stopped at its algebra invariant: float32 Gaia Mg
  subtraction lost precision relative to the float64 model terms. Promoted
  g0 and reference Mg to float64 before pair decomposition; focused test
  passes. Optical response 60476003 and fallback 60476051 are running.
  Replaced the old crossover center claim with the correct paired estimand
  and linked v3 weak-parallax null results. Generated and inspected the
  selected statistics figure. Full local suite: 68 passed. Next: rerun the
  same-star job and collect all mechanism results.

- Added two frozen-model cause diagnostics: same-star forced A/B/C and
  complete-input/measurement-error cross-field matching; fixed-tier PS1
  g/r/z/grz substitutions with separate rerouting and symmetric small
  perturbations. New code leaves v4 route/model untouched. Python parse,
  shell syntax and focused decomposition/color-propagation tests pass.
  Fallback audit 60475742 is queued on NERSC. Next: run the new jobs on
  allocated CPUs, inspect result manifests, and decide if a candidate rule
  has enough evidence for separate validation.

- Crossover common-sample audit job 60475661 completed in 14 s; weak-parallax
  v3 job 60475664 completed in 53 s, both on allocated NERSC CPUs. Copied
  small aggregate manifests. Difference-of-medians reaches +0.031 mag in one
  crossover cell although its paired-difference median is −0.005 mag; all
  eight direct-bootstrap center intervals include zero. At r21–22 the
  common-source NMAD rises 0.391→0.460 and 0.324→0.429 mag across the two
  templates. Selected-null center tests put the two +0.04 mas weak-parallax
  statistics at simulation p=0.63 and 0.81. Added report and diagnostic
  fallback code; next run the fallback audit on allocated compute.

- Recovered the complete latest Pro review after the ChatGPT source login was
  refreshed. Registered the three detailed diagnostic groups as the active
  goal and updated PLAN. Added a saved-row crossover common-acceptance audit
  and weak-parallax v3: subgroup PIT now conditions on each observed-SNR
  interval, and the center statistic has a selected-observation simulation
  null. Commands: status/branch/fetch/history, NERSC login-node metadata
  check, Python compile and 7 focused tests (all passed). Frozen v4 products
  were not changed. Next: run both audits on allocated NERSC CPUs, inspect
  results, then implement fallback and same-star cause diagnostics.

- Final local validation for this diagnostic round: `python -m pytest -q`
  passed 62 tests; PS1 manifest field identity, match/quality counts and
  common-route invariants passed; `git diff --check` and both selected
  figures passed inspection. Updated README/PLAN/integrated Pro report.
  Next: milestone commit and exact three-end sync.

- PS1 v2 job 60463638 completed in 2m03s on allocated CPU. Added
  `docs/cold_v4_ps1_relative_audit.md`, selected three-panel figure and
  aggregate manifest. On 1,613/2,712/1,485 same accepted sources,
  Legacy residual medians −0.0135/−0.0480/+0.0324 mag become
  +0.0331/−0.0144/+0.0121 under simultaneous PS1 grz substitution;
  nearby-star results and nside64 intervals are reported separately.
  PS1-quality anchors are bluer/farther than the full sample, and the
  unchanged forest was trained on Legacy colors. Scientific decision:
  record relative optical sensitivity, keep the frozen route, require an
  independent calibrated reference/new registered fields for a correction.
  Figure inspection passed. Next: full validation, commit and three-end sync.

- PS1 job 60463252 completed in 2m01s on allocated CPU. Quality-qualified
  matches among frozen selected anchors number 1,623/2,734/1,495 in south
  RA330/south RA20/north RA130. Median Legacy-minus-transformed-PS1 g is
  +0.0089/+0.0046/−0.0039 mag; r and z are within ~0.003 mag of zero.
  Same-source optical replacement moves median predicted modulus by
  +0.0109/+0.0056/−0.0069 mag, while the distribution of residuals is
  nonlinear. Added exploratory per-field residual sky-block intervals and
  color/model-component summaries for a bounded rerun before interpretation.

- PS1 audit job 60463039 failed after 36 s because new field diagnostics
  retain the observed `g/r/z` magnitudes but only the g-band transmission;
  it requested nonexistent `mw_transmission_r/z` columns. The audit now
  uses the stored observed `g/r/z` directly for same-source PS1 differences
  and applies those differences to the already dereddened colors and g0.
  Updated targeted propagation test. Next: rerun after three-machine sync.

- Registered the three-field PS1 relative-optical diagnostic before
  inspecting any new PS1 source rows or results. Added NERSC allocated-node
  crossmatch, frozen-route replacement, fixed quality domain and a targeted
  optical-color propagation test. The comparison is explanatory because
  these native fields have already been scored. Next: run and record match
  counts, per-band offsets and same-source distance response.

- Dust-sensitivity Slurm job 60462368 completed on allocated CPU in 70 s;
  copied aggregate result, added `docs/cold_v4_dust_sensitivity.md` and
  figure. At 75% of catalog extinction, north-minus-south overall residual
  contrasts move by only 0.003–0.004 mag; even zero correction leaves
  +0.066/+0.098 mag. `A_g` from transmission matches 3.214 EBV within
  4.3e-8 mag. Python parse, focused dust tests and figure inspection pass.
  Next: compare same-star Legacy/PS1 optical measurements for a separate
  relative-photometry diagnostic.

- Prepared a bounded dust-response experiment on the three already
  scored native final fields, keeping Gaia anchors, Gaia reference
  parallax, forests and WISE20 policy fixed. It compares assumed fractions
  1, 0.75, 0.5 and 0 of the DR11 foreground correction in all five bands,
  reroutes the same stars and reports paired distance-modulus shifts and
  acceptance, including p>=2 mas. The perturbation uses the measured
  catalog A_g and official band-coefficient ratios; a source-hash check
  reproduces published baseline identities. Python parse, shell syntax
  and nine focused photometry/noise/dust tests pass. Next: run on allocated
  NERSC CPU, interpret as sensitivity rather than a fitted dust map.

- Coordinate-sign crossover arrays 60461738/60461825 completed all four
  fields, and frozen score 60461961 completed in 31 s. Registered
  b>0/β<0 fields RA160/RA180 yielded 16/0 selected strict cold anchors;
  b<0/β>0 RA340/RA0 yielded 0/680. RA0 median Δμ is −0.0023 mag with
  sky-block CI [−0.0263,+0.0124]; 16-source RA160 has no block interval.
  The unchanged EBV<0.03 cut is responsible for the zero/low yields.
  Updated sky-census and integrated Pro reports with all outcomes.
  Next: assess how much a bounded dust-correction perturbation can move
  the frozen predictions on the already inspected native final fields.

- Six-field sky census scans completed in debug arrays 60460898 and
  60461294; frozen score 60461354 completed in 41 s on allocated CPU.
  Three positive-b fields select 5,331/5,395/2,218 cold anchors and have
  native median Δμ +0.039/+0.050/+0.078 mag; their >=2 mas medians are
  +0.066/+0.076/+0.171. Negative-b RA10 selects 5,656, median −0.009;
  RA340 selects 216, median +0.007 with wide CI; RA310 has zero because
  all 4.91 million |b|>30 PSF rows fail E(B−V)<0.03. Added aggregate
  manifest, selected sky plot and report. The positive-b sign repeats,
  but b and ecliptic latitude signs coincide in these six fields.
  Cross-sign arrays are now running. Next: score their fixed coordinate
  comparison and decide the next cause test.

- Before the six-field sky-census scores were read, noticed all six
  diagnostic centers have matching signs of Galactic and ecliptic
  latitude. Registered four disjoint 10-degree coordinate-sign crossover
  fields (two b>0/β<0, two b<0/β>0) in
  `configs/cold_v4_native_transfer_cross_sign.json`; all 16 DR11 south
  sweep files exist. Fixed the comparison of native median residuals
  overall and at corrected parallax>=2 mas, with no model correction.
  Registration SHA links to the six-field census registration, geometry
  and disjointness checks pass. Added compute-array and frozen-score
  runners. Next: finish six-field jobs, then run four crossover fields.

- Cancelled still-pending regular sky-census array 60460329 and dependent
  score 60460382 after scheduler returned no start estimate. Changed the
  identical six-field array to 30-minute debug QoS; previous analogous
  fields each built in under five minutes. No source output had been
  generated by the cancelled jobs. Next: resubmit array and dependent
  score without changing frozen science configs.

- Gaia-motion diagnostic 60460729 completed on allocated CPU in 47 s.
  In corrected-parallax>=2 mas and tangential speed<40 km/s subsets,
  south RA330/south RA20/north RA130 residual medians remain
  −0.031/−0.083/+0.121 mag (n=1,793/2,152/1,152). The 40–80 km/s
  strata show the same sign contrast. Northern |z|<200 pc stars have
  +0.170 mag median. Recorded source-cluster/sky-block intervals and
  integrated the result into the native-transfer and Pro-round reports.
  Next: six-field sky census; no kinematic population assignment or
  distance correction.

- Registered sky-census field array 60460329 and dependent unchanged-model
  score 60460382 on NERSC; the array is waiting on scheduler priority.
  Prepared a fixed b-versus-residual plot, integrated Pro-round report and
  README update. Added a separate Gaia proper-motion diagnostic for the
  already inspected final fields: tangential speed and absolute vertical
  height strata, with no assigned population membership. Python parse,
  shell syntax and mas/yr-to-km/s numerical smoke check pass. Next: run
  kinematic diagnostic on allocated CPU while the six-field census queues.

- Gaia/DESI diagnostic job 60460082 completed in 41 s on allocated CPU.
  Median applied Gaia zero points are −0.0345/−0.0336/−0.0370 mas
  (south RA330/south RA20/north RA130), and the native residual contrast
  persists with RUWE<=1.1: −0.009/−0.046/+0.073 mag. Only 0/13/35
  selected anchors have unambiguous qualified DESI dwarf metallicities in
  calibration domain; those spectra cannot characterize the full-field
  populations. Added aggregate manifest and report. Before reading any
  new source rows, registered six disjoint 10-degree diagnostic fields
  spanning Galactic b≈−87 to +79 degrees; all 24 DR11 south sweep files
  exist. Added allocated-node field-array and frozen scorer runners.
  Next: run sky census as a diagnostic, keep corrections unfitted.

- Outcome-blind final-field match job 60459873 completed in 44 s on
  allocated NERSC CPU. Six-feature unique matching on Legacy colors,
  source A_g, corrected parallax, Gaia BP-RP and r0 retains 2,016
  north/RA330 and 1,470 north/RA20 pairs; north-minus-south median
  residual remains +0.046 and +0.076 mag (northern nside64 block CIs
  [+0.033,+0.062] and [+0.060,+0.092]). In >=2 mas pairs these become
  +0.081 and +0.104 mag. Added manifest/report and prepared an allocated-
  node audit of Gaia RUWE, formal parallax error, applied zero point and
  quality-qualified DESI metallicity. Next: run that diagnostic and decide
  which cause tests are supported by label counts; keep final fields out
  of any fitted correction.

- The three preregistered native-transfer final score jobs completed on
  allocated NERSC CPUs (60459490/60459491). With the frozen WISE20 route,
  south RA330 and RA20 select 5,776/7,097 cold anchors and have native
  median Δμ −0.0178/−0.0520 mag; north RA130 selects 2,941 and has
  +0.0687 mag. Nside64 median bootstrap95 intervals do not overlap.
  The contrast is strongest for corrected parallax >=2 mas: −0.0221,
  −0.0662, +0.1121 mag. Recorded all final results, source/registration
  hashes, marginal color/r/dust strata, and a summary plot in the native
  transfer final report. The preregistered directional hypothesis holds
  in these three fields; no center correction or new deep interval was
  fitted. Added an outcome-blind final-field matching diagnostic on
  allocated CPU for cause analysis only; Python parse and small synthetic
  match pass. Next: run matching, assess cause limits, sync all artifacts.

- Before either newly registered final-field residual was read or scored,
  cancelled pending score jobs 60459346/60459360 and added nside64
  sky-block median/NMAD uncertainty for the preregistered corrected-
  parallax 1–2 and >=2 mas groups. Python parse and a focused block-
  bootstrap smoke check pass. Next: sync the fixed score implementation,
  resubmit final jobs; development scores can be reproduced with the same
  CI addition.

- Supplemental development field RA230–240 built in 3m27s (job
  60459005) and scored in 32 s (60459011) under frozen WISE20. It has
  7,128 strict anchors, 4,746 cold anchors and 4,554 selected; overall
  median Δμ +0.0415 mag, NMAD 0.2146, nside64 block-bootstrap95 median
  [+0.0335,+0.0509]. By corrected parallax, 1–2 mas is −0.0028 mag
  (1,928) while >=2 mas is +0.0845 (2,606), matching the older positive
  latitude RA170–180 pattern. RA60–70 at b≈−45° is near zero. Registered
  one disjoint positive-latitude final field RA130–140/Dec25–35 before
  reading its source rows; fixed prediction is a more positive native
  median than two reserved southern finals, especially p>=2 mas. Added
  configs/runners and updated development report. Next: build northern
  final field, then score all three fixed final fields once.

- NERSC anchor-cutflow job 60458837 completed in 19 s. The supplemental
  RA180–190 field has 6,815 strict anchors, 5,401 with Gaia BP-RP in the
  cold range and 6,744 with Legacy M_g>5; all 6,815 fail the frozen
  `not_external_reserved` condition. Root cause is reference-region
  quarantine, not a missing cold stellar population. Updated the development
  report, retained this field as a recorded zero, and preregistered a
  disjoint high-latitude RA230–240/Dec25–35 development field before source
  inspection. Added config and allocated-node runners; next build and score
  this replacement while keeping the two final fields reserved.

## 2026-10-28

- Native development scoring jobs 60458064 and 60458386 completed on
  allocated CPUs (27 s and 24 s). RA60–70 selects 6,594/6,821 strict
  cold anchors, median Δμ −0.0059 mag, NMAD 0.2056, nside64 bootstrap95
  median [−0.0139,+0.0023]. RA235–245 selects 9/10 cold anchors from
  only 17 strict anchors; all counts are retained. Supplemental RA180–190
  unexpectedly has 6,815 strict anchors but zero cold anchors. Its Gaia
  CMD contains 6,813 finite anchor points, so this needs a frozen-rule
  cutflow audit; added allocated-node diagnostic script. The two reserved
  new final fields have been constructed but their residuals remain unread.
  Next: identify the supplemental failure before any final scoring.

- Four-field array 60458047 entered NERSC debug compute. The two
  preregistered development scans completed in 3m52s/3m45s: RA60–70 has
  9,431 strict anchors, RA235–245 only 17 after the unchanged E(B-V)<0.03
  cut. Both remain in the result. Before reading any supplemental source
  rows, registered RA180–190/Dec25–35 as an extra high-latitude development
  field in a separate amendment; all four sweep filenames exist. The two
  original independent final fields remain unchanged and are still being
  constructed. Next: run supplemental scan and score all roles with the
  locked native-transfer method.

- Outcome-blind Gaia-property match job 60457843 completed in 17 s.
  Adding corrected parallax to Legacy colors/A_g leaves +0.054 mag
  final-minus-cal native residual (460 pairs; bootstrap95 +0.019 to
  +0.089). Adding Gaia BP-RP too leaves +0.062 mag (458 pairs; +0.024
  to +0.101); matching Gaia G leaves +0.063 mag (465 pairs; +0.028 to
  +0.096). No single measured distribution matched here explains the
  native field shift. Updated report. For newly registered fields, added
  exact DR11 g transmission to pilot diagnostics and a frozen native
  transfer scorer with spatial-block uncertainty. Next: run four field
  preparations on allocated CPUs, then score development and final roles.

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
