# PLAN

## Current Objective

Implement the 2026-10-29 Pro review on the three-end synchronized `a18ef9b`
checkpoint. Keep the cold-v4 route and all locked evaluations frozen. Export
per-candidate route diagnostics, compare one measurement-error-aware smooth
probabilistic distance candidate with v4 on identical training and development
identities, and define a separate metal-poor blue branch. The active goal
records the full scientific and validation criteria.

## New Round Milestones

1. Reconcile the prior review's interpretation and publish an explicit
   development/selection protocol and candidate-output schema.
2. Export v4 per-candidate A/B/C choice, higher-tier fallback reasons, support
   ratios, infrared-conflict marker, cold-dwarf-hypothesis status and validated
   depth state without changing distances or deleting conflicts.
3. Build a full-covariance heteroscedastic deconvolved Gaussian mixture
   `p(M_g, colors)` on exactly the v4 A/B/C training identities. Use 16/32/64
   components and propagate shared-band and Gaia label uncertainty.
4. Compare configurations by held-out development sky blocks. On identical
   accepted objects, compare chosen candidate against frozen v4 in bias,
   NMAD, tails, probability quality, g/r perturbation response and error/color
   strata. Decide to reject or lock the candidate before any new final test.
5. Define the metal-poor blue branch from reliable Gaia/DESI labels: coverage,
   color-boundary failures, CMD position and giant/nonstar confusion. Keep it
   distinct from the frozen cold-v4 box.
6. Validate, document, commit and synchronize programs, small manifests and
   figures across local, GitHub and NERSC. Keep catalogs and model files on
   NERSC and use allocated CPUs for scans/training.

## Completed Previous Round Milestones

1. Complete: recompute common-acceptance crossover centers, paired differences,
   directly bootstrapped median differences and common-sample NMAD from the
   existing per-source arrays. Keep full-flow acceptance and NMAD visible.
2. Complete: correct the weak-parallax subgroup PIT for each observed-SNR truncation
   interval; simulate the same selected statistic to calibrate its center.
3. Complete: record each tier fallback reason and complete WISE quality evidence.
   Compare a conflict marker and a conservative hold policy on development
   dwarf, giant, metal-poor and unlabeled candidate samples. Assess boundary
   stability under reported measurement errors.
4. Complete: force A/B/C on identical supported stars in the three native fields;
   report Gaia residuals and within-star tier differences. Match same-tier
   complete inputs and photometric quality across fields; separate label and
   model-prediction contributions, with parallax/brightness strata.
5. Complete: measure fixed-tier PS1 g/r/z/grz substitutions and ±0.005/0.01/0.02 mag
   band responses on the same qualified stars, plus optional rerouting.
   Report center, scatter, tails, support-edge and color dependence.
6. Decision: retain v4 distance routing and add a diagnostic infrared support
   conflict/type-risk flag. A hard rejection loses 3.5%–4.1% of strict Gaia
   cold anchors and 13.4%–14.7% of quality-qualified DESI cold dwarfs in the
   two useful development fields. A future hard-hold or fitted correction
   needs separate calibration and independent final validation.
7. Complete: validate code/results, update reports/worklog, and sync committed
   programs, manifests and selected figures to local, GitHub and NERSC.

## Previous Diagnostic Round (through 88f7330)

The preceding Pro-review requirements are complete: the frozen model and route
were held fixed; extinction algebra, selection-aware weak-parallax PIT,
WISE20 wide-binary and DESI boundary checks, and the same-source 2×2
noise/template crossover are documented in `docs/pro_latest_round.md`.

Three preregistered independent native-transfer fields have been scored once.
South RA330 and RA20 medians are −0.0178 and −0.0520 mag; north RA130 is
+0.0687 mag. The contrast persists at corrected parallax ≥2 mas, after
outcome-blind source matching, in RUWE≤1.1 and ordinary tangential-speed
strata. Six preregistered sky-census fields repeat positive medians in
three informative positive-b regions; four opposite-sign Galactic/ecliptic
fields yield too few low-extinction anchors to isolate coordinates.

The bounded foreground-dust sensitivity job 60462368 is complete. Scaling
the catalog extinction correction to 75% leaves north-minus-south contrasts
within 0.003–0.004 mag of baseline; the difference persists even with no
correction. See `docs/cold_v4_dust_sensitivity.md`. This is explanatory use
of already inspected fields, not a fitted dust map.

The PS1 relative grz audit was registered before reading PS1 rows and is
complete. On the restricted PS1-quality samples, Legacy-minus-transformed-
PS1 g medians are +0.0089/+0.0046/−0.0039 mag in south RA330/south RA20/
north RA130. Same-source three-band substitution changes the frozen route's
residual centers, but the restricted color and footprint coverage plus
Legacy/PS1 calibration ancestry make this a sensitivity diagnostic. See
`docs/cold_v4_ps1_relative_audit.md`.

## Outstanding Issues and Validation Criteria

- A causal optical-system test needs an independent reference with a
  validated cool-star bandpass transformation or a matched cross-field
  training/calibration design. The existing PS1 comparison constrains
  relative measurement response on only its qualified subset.
- Preserve the v4 forests, color box, support and WISE20 policy. The current
  median distance remains conditional on the cold-dwarf hypothesis. Deep
  nominal intervals are still withheld because the independent deep test
  undercovers in multiple measured-r bins.
- If PS1 or a later independent source identifies a plausible cause, register
  new calibration and final sky regions before fitting a correction. Existing
  final regions stay explanatory and cannot validate that correction.
- Run targeted tests, verify frozen hashes and source identities, inspect
  figures, update documentation and sync committed programs, reports, small
  manifests and figures across local, GitHub and NERSC. Catalogs and models
  remain on NERSC.

## Milestones

1. Complete: freeze baseline commit, model checksums, original test predictions and
   nside=64 test cells. Preassign one external calibration and two external
   final-test fields before examining model residuals.
2. Complete: reconcile strict and recovery nonoptical conditions; require DESI quality
   and calibration domain in spectral residual bins. Record impact on counts.
3. Complete: decompose the recovered bright-star offset with matched strict controls,
   out-of-fit strict predictions and paired Legacy/PS1 photometry replacement.
4. Complete: run the three external fields with unchanged selection and first measure
   direct transfer of the frozen baseline models.
5. Complete: calibrate 68%/95% intervals on the preassigned external calibration field;
   score them once in the two final-test fields, including red-color bins.
6. Complete: run raw-flux noise injection at observed target-depth S/N and report
   acceptance, residual and interval coverage by layer and depth.
7. Complete: quantify target input coverage and pure-photometry cold-branch selection
   against independent Gaia/DESI labels. Plot, document and sync all results.
8. Complete: correct the paired PS1 statistics in the report, and rerun
   target coverage with a training threshold defined by the tenth *other*
   object (k=11 self-query; k=10 external query). Publish clearly named
   color-box recall, color-plus-support recall, and known-nonstar leakage.
9. Complete: before any source inspection, freeze `configs/next_fields.json`. Its
   RA 205–215/Dec −10–0 field is development calibration; RA 230–240/Dec
   5–15 and RA 50–60/Dec −30–−20 are untouched final tests. The previous
   north/south fields now count as development evidence for any new method.
   After the first final field yielded only 25 cold anchors due the fixed
   E(B−V) cut, register two additional high-latitude fields in
   `configs/supplemental_fields.json` before reading their source rows.
   Evaluate and report all four final fields regardless of their counts.
10. Complete: build a revised noise experiment on the identical anchor stars, identical
    target noise rows and one shared five-band realization for A/B/C. Draw
    noise from a grz mother pool including weak WISE detections; quantify
    source-template noise, all-tier acceptance, strict quality, Mg and final
    distance-modulus/distance residuals and interval coverage. Cluster repeat
    realizations by source for uncertainty.
11. Complete: fit bias correction and conditional 68/95% interval expansion using only
    application-visible Legacy/WISE colors, magnitude and photometric errors.
    Keep fit, calibration and final test identities disjoint. Report bias,
    NMAD, both coverage rates, low/high misses, interval width, count and
    spatial-block uncertainty by color and depth. Compare same-star A/B/C
    under a common training identity set before attributing band benefit. The
    original cold_baseline.py already supplies that common-fit comparison;
    retain its result while assessing new external-field performance.
12. Complete: audit dwarf, giant, nonstar and other/unknown labels across color,
    brightness and sky; state selection of each labeled sample. Evaluate
    class-conditional recall and leakage. Use cold-dwarf-hypothesis distances
    with type-risk and support flags until calibrated type probabilities are
    possible. Keep APOGEE auxiliary and recovery candidates separate.
13. Complete: commit and synchronize programs, manifests, reports and selected figures
    among local, GitHub and NERSC; keep catalogs and models on NERSC. Review
    final-field metrics and unresolved limits with the user.
14. Complete: export a bounded photometry-only candidate demonstration in
    an independent field, with frozen-model median distances and explicit
    support, optical quality, depth and unverified-type flags. No class
    probability or calibrated interval is assigned.
15. Complete: audit the fixed highest-available A/B/C routing rule together
    with color-box and support selection on all four registered final fields.
    Report the selected fraction and direct frozen-model distance performance,
    including Gaia-color diagnostics, before any claim about a selected catalog.
16. Complete: register cold_v4 expanded strict-anchor training, an independent
    route-calibration field and three new untouched 10° final fields before
    reading their source rows. Fit the tier-specific forests, add feature-
    matched support and calibrate the complete routing flow without a center
    correction. Evaluate every registered final field and keep the original
    c3637cb forests/results frozen. All three new final fields completed on
    NERSC; the fixed route selects 4,991/4,784/2,681 strict cold anchors
    and achieves final-modulus 95% coverage 96.0%/94.7%/95.0%.
17. Complete: pass the shared five-band flux-noise experiment through the
    complete cold_v4 color box, feature-matched support, tier fallback,
    forest and route-calibrated intervals. Report acceptance, strict quality,
    final distance-modulus coverage and source-cluster uncertainty by depth.
    The four r=18–22 bins have 95% final-modulus coverage
    92.5%/91.8%/92.4%/91.0%; the deepest bin's strict-quality fraction is
    32.3%, and W2 is selected in only 51/1,784 draws. The observed failure
    is a depth-calibration limit, not a reason to change frozen final tests.
18. Complete: without changing forests, report A/B/C × clean/strict/other
    image-quality and error/support strata for calibration, new final Gaia
    fields, shared noise and real photometric candidates. Include per-tier
    bias/NMAD/68%/95% coverage/tail misses/width, sky- or source-cluster
    uncertainty, and complete-route intersection with grz_clean. The real
    candidate/Gaia audit and new deep source-cluster tier diagnostics are
    published; sparse deep C is explicitly recorded.
19. Complete: applied the complete v4 route to deduplicated, quality-qualified
    Gaia/DESI labels in the new fields. Measure dwarf recall and giant/nonstar
    leakage by tier, clean state, fallback, color, magnitude, errors and sky;
    separately audited calibrated-domain metal-poor labels. Two final fields
    have DESI matches; RA0–10 has none. Defer type probabilities until label
    selection and class priors are supportable.
20. Complete: corrected simulated flux IVAR when template noise exceeds
    target noise, and corrected the report's description of borrowed masks.
    Compare only a few predeclared routing policies on identical five-band
    development observations, considering precision, coverage, width and
    acceptance. Development chose and hash-locked `grz_clean` plus WISE
    S/N>=20; tested it on separately registered fields.
21. Complete: assembled high-S/N templates and WISE-unconditioned target-noise
    environments from several fields; isolate fit/development, calibration
    and final-test identities and sky blocks. Calibrate final distance-modulus
    intervals by chosen tier and measurable noise/support using independent
    sources. RA35–45 calibration completed with one observation per source
    per target-noise depth. Superseded v1 grouping by injected target-noise r;
    v2 groups by measured r and deduplicates source identities in each bin.
    The one-time RA170–180 final test has 95% coverage 92.6%/92.9%/95.1%/92.3%
    and 68% coverage 62.6%–65.2% across measured r18–22. Deep v2 intervals
    remain diagnostic pending a new development/calibration/final sequence.
22. Complete for current scope: used isolated wide-binary/cluster reference systems for color- and
    brightness-dependent distance checks, then assess weak/negative Gaia
    parallaxes with a parallax-space forward model and documented selection.
    Examine metal-poor, binary, extinction and field dependence of bias.
    Initial final-field check has 456 clean-routed pairs and 201 Blanco 1
    members; the pair fainter-minus-brighter median is +0.046 mag.
23. Complete: publish the bounded application-range report, selected figures,
    small manifests and validation; update README/WORKLOG/PLAN, commit and
    sync local, public GitHub and NERSC. Keep catalogs/models on NERSC.

## Scope

Original field RA [0,10), Dec [-10,0) remains a locked baseline. Prior field
roles and source/model hashes are in configs/followup_fields.json; new roles
are in configs/next_fields.json. The strict optical and Gaia criteria remain
the comparison standard. Cold-v4 roles and fixed method are in
configs/cold_v4_fields.json. The full-footprint scan remains stopped.

## Next Checkpoint

Review the infrared conflict flag on a separately registered validation set
before using it to withhold distances or calibrate type probabilities. The
photometric center discrepancy persists in A/B/C on identical stars and in
same-tier, complete-input matched C pairs; further cause work needs an
independent optical reference or a new population/label calibration design.
Retain the frozen distance route and its prior final tests. For any fitted
center or support change, register new calibration and final fields before
scoring their residuals. The metal-poor blue extension remains its own branch.
