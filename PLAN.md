# PLAN

## Current Objective

Complete the latest Pro review of the cold-v4 component by measuring the gap
between strict Gaia validation stars and actual photometric candidates. Keep
the v4 forests, route, three final-field results and prior baselines frozen.
Audit A/B/C, image cleanliness, errors and support; update DESI type and
metal-poor diagnostics; then compare simple model-routing policies, calibrate
deep-noise intervals with independent source/sky splits, and test real stellar
populations with reference systems and weak-parallax forward checks.

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
18. In progress: without changing forests, report A/B/C × clean/strict/other
    image-quality and error/support strata for calibration, new final Gaia
    fields, shared noise and real photometric candidates. Include per-tier
    bias/NMAD/68%/95% coverage/tail misses/width, sky- or source-cluster
    uncertainty, and complete-route intersection with grz_clean. The real
    candidate/Gaia field audit is complete; the noise tier cross-tab remains.
19. Pending: apply the complete v4 route to deduplicated, quality-qualified
    Gaia/DESI labels in the new fields. Measure dwarf recall and giant/nonstar
    leakage by tier, clean state, fallback, color, magnitude, errors and sky;
    separately audit calibrated-domain metal-poor labels. Defer type
    probabilities until label selection and class priors are supportable.
20. In progress: corrected simulated flux IVAR when template noise exceeds
    target noise, and corrected the report's description of borrowed masks.
    Compare only a few predeclared routing policies on identical five-band
    development observations, considering precision, coverage, width and
    acceptance. Freeze the selected policy before new calibration/testing.
21. Pending: assemble high-S/N templates and WISE-unconditioned target-noise
    environments from several fields; isolate fit/development, calibration
    and final-test identities and sky blocks. Calibrate final distance-modulus
    intervals by chosen tier and measurable noise/support using independent
    sources, and assess deep coverage and clean acceptance on unseen sources.
22. Pending: use isolated wide-binary/cluster reference systems for color- and
    brightness-dependent distance checks, then assess weak/negative Gaia
    parallaxes with a parallax-space forward model and documented selection.
    Examine metal-poor, binary, extinction and field dependence of bias.
23. Pending: publish the bounded application-range report, selected figures,
    small manifests and validation; update README/WORKLOG/PLAN, commit and
    sync local, public GitHub and NERSC. Keep catalogs/models on NERSC.

## Scope

Original field RA [0,10), Dec [-10,0) remains a locked baseline. Prior field
roles and source/model hashes are in configs/followup_fields.json; new roles
are in configs/next_fields.json. The strict optical and Gaia criteria remain
the comparison standard. Cold-v4 roles and fixed method are in
configs/cold_v4_fields.json. The full-footprint scan remains stopped.

## Next Checkpoint

First quantify the calibration/application mismatch: calibration and strict
final stars are 92%–94% C, while actual candidates are 57% A, 19% B, 24% C.
Calculate complete-route ∩ grz_clean and conditional performance in existing
fields before choosing new cuts. Depth simulations currently have 91%–93%
95%-interval coverage; the deepest bin is clearly below nominal. Preserve
that finding while planning an independent source/sky calibration experiment.
The v2 center correction remains rejected.
