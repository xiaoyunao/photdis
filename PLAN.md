# PLAN

## Current Objective

Diagnose why the cold-v4 deep calibration and independent field differ in
distance-modulus center and scatter, and correct the weak-parallax diagnostic.
Keep the v4 forests, color box, support domain, WISE20 route, strict-anchor
final fields and deep calibration/final v2 outputs frozen. Use the already
inspected RA35–45 and RA170–180 fields only for cause analysis. If the cause
supports a method change, preregister new calibration/final fields before
examining their residuals. The full detailed objective is in the active goal.

## Current Diagnostic Round

Extinction flow, selected weak-parallax PIT, same-pair wide binaries,
DESI type boundaries and the 2×2 template/noise experiment are complete.
The crossover shows a native template median difference of 0.059 mag,
while matched noise-field swaps change the median by at most 0.011 mag and
mainly affect deep scatter and acceptance. An outcome-blind color/A_g match
retains a +0.065 mag field offset. Four additional disjoint 10° DR11 south
fields are now registered in `configs/cold_v4_native_transfer_fields.json`:
two development diagnostics and two reserved final tests. Next: inspect the
exact-template Gaia-label census, construct the registered fields on NERSC
and apply a fixed native-transfer score. The existing deep final test remains
frozen.

1. Verify DR11 MW transmission, AB fluxes, all five dereddened magnitudes,
   Gaia `M_g` labels and cross-field dust handling in noise injection.
2. Fix weak-parallax PIT selection bias. Build an explicit forest-weighted
   magnitude distribution, convolve with Gaia error, condition on the actual
   corrected-parallax/SNR<10 selection, compare frozen WISE20 with old routing
   on the same objects, and retain negative parallaxes.
3. On previously inspected calibration/final fields, record native and
   degraded predictions per original source. Decompose native residual and
   degradation increment, compare actual fallback with supported fixed tier,
   and report source-cluster uncertainty.
4. Run the 2×2 template-field × noise-environment exchange with common
   target depths, paired accepted-source differences and full acceptance.
   Diagnose error, color, support, E(B-V), source population and tier changes
   before changing the center or interval model.
5. Recheck wide binaries under WISE20; distinguish Blanco 1 internal scatter
   from an absolute zero-point test. Inspect the 12 accepted DESI cold giants
   and 14 optical-box-rejected metal-poor cold dwarfs individually, with
   quality, boundary and CMD/feature-space plots.
6. Decide from the evidence whether to expand development/calibration samples,
   fit a continuous error scale or add a separate stellar branch. Register
   new sky roles before any new independent final evaluation. Preserve all
   failed/negative results, update reports and three-machine synchronization.

Validation: local targeted tests and scientific invariants; NERSC compute
allocation for all catalog scans and simulations; check hashes/role isolation,
independent-source bootstrap and final result manifests. No new model or
nominal deep interval is released solely from old final-field diagnostics.

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

Review the [application decision](docs/cold_v4_application_decision.md) with
the user. The independent final test shows deep interval undercoverage and
positive modulus bias; do not tune these frozen final results. For the next
research round, preregister new development/calibration/final sky roles,
investigate center and low-tail errors with binaries/metallicity and a
forward weak-parallax check, then recalibrate conditionally on measured
inputs. Build type probabilities only after quantifying DESI/Gaia label
selection and application class priors.
