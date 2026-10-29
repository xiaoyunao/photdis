# PLAN

## Current Objective: next RGB luminosity-information test

Status: planning only. The registered DECam conditional RGB transfer is
complete; see `docs/cluster_rgb_conditional_result_v1.md`. Keep its frozen
fit, NGC 5466 identity and blind score as the comparison baseline. The next
experiment should first register its development/whole-cluster test split and
choose additional observables only if they exist for the same RGB members.
Test luminosity information by phase and magnitude bin, propagate coherent
cluster dust and photometric scales, and compare against the current gr/grz
result on identical stars. Add explicit RGB/HB/AGB and abundance mixtures only
with external stage evidence; require a new untouched whole-cluster test and
independent field validation before claiming a field giant-distance product.

## Completed Objective: registered DECam conditional RGB experiment

Status: completed as a bounded model and once-only whole-cluster transfer
test. Followed the Pro review of `7256423` and the frozen protocol in
`docs/cluster_rgb_conditional_protocol_v1.md`; parameters and whole-cluster
identity are recorded in `configs/cluster_rgb_conditional_v1.json`.

1. Register NGC 5466 as a new whole-cluster DECam test before inspecting its
   Legacy photometry. Use its independent RR Lyrae distance and a common
   RRab-to-UVES metallicity scale. Keep its source IDs out of all development.
2. Audit NGC 7006 as a new development system: gr and grz upper RGB locus,
   local color residual, magnitude/radius/image-quality dependence, shared
   distance/dust sensitivity and actual overlap with M53. M3 lower and M53
   upper/lower remain other development stages; M3 upper edge diagnostic only.
3. Fit low-parameter forward gr and grz conditional RGB relations only on
   observed shared magnitude ranges. Propagate shared-band covariance and
   whole-cluster scale uncertainties. Keep existing per-band optical quality.
4. Freeze new model and fixed priors, then extract/score NGC 5466 with an
   allocated NERSC job. Compare bands on identical grz-clean stars and report
   gr-only added coverage. Evaluate known-abundance and three fixed unknown
   abundance mixture priors, by magnitude bin and with the registered metrics.
5. Write the physical result and limitations, run focused validation, update
   WORKLOG, commit milestone and synchronize tracked code/docs/small figures
   across local, GitHub and NERSC. Large catalogs remain on NERSC.

Measured outcome: NGC 5466 supplies 93 supported lower-RGB stars on the exact
gr/grz common set. Adding z improves distance-modulus NMAD from 0.314 to
0.231 mag (paired-bootstrap improvement 95% interval +0.009…+0.140 mag).
The grz median residual swings by 0.686 mag between the registered bright
and faint lower-RGB bins; its 68% interval spans a median 1.50 mag and covers
95.7% of point references. There is no registered grz upper-RGB support.
The model remains a conditional diagnostic and is not a field giant-distance
product. Unknown-abundance priors shift its median residual. A coherent
0.01 change in cluster E(B−V) changes the median residual by 0.152 mag in
the sensitivity experiment. See the result report, manifests and figures.

## Completed Objective: native-system, band-matched RGB audit

Status: completed as a bounded observational audit. Followed the Pro review of `beb15b7` with a bounded
four-cluster experiment. Keep the frozen pooled A/B fit, NGC 7006 test,
cold-v4/XD and blue forward-parallax results unchanged.

1. Complete: confirmed each member's DR11 `RELEASE` and natural camera system against
   sweep files and official Legacy documentation. Separate southern DECam
   from northern BASS/MzLS in all new comparisons. Review the existing
   `MW_TRANSMISSION` dereddening and quantify common cluster distance/dust
   uncertainty. Apply a fixed RGB passband conversion only with a reliable
   source; otherwise leave native systems separate.
2. Complete: re-read the full matched membership parent on allocated NERSC compute.
   Build a new `gr_clean` tier with exactly the existing per-band optical
   requirements on g and r, including their mask bits and BAILOUT, without
   any z condition. Preserve the grz-qualified frozen rows for same-star
   comparisons. Report by cluster and Gaia-CMD stage: gr and grz counts,
   same-row z-only failures, changed crossmatch row, luminosity/color/radius
   and image-quality distributions; inspect M3's bright upper RGB explicitly.
3. Complete: recomputed and plotted empirical loci from exact fitted stage cores: M3 lower,
   M53 upper plus lower, NGC 2419 upper. Show the NGC 2419 lower/red-HB
   overlap as a separate diagnostic. Distinguish within-bin 16–84% width
   from local-curve residuals and propagated photometric errors, including
   shared-band covariance.
4. Complete: compared M3 and M53 at the same M_g within their common DECam range using
   fixed interpolation. Audit residuals against magnitude, cluster radius
   and measured image-quality quantities. Treat modulus and dust errors as
   whole-cluster shifts. Inspect NGC 2419 within its own native system.
5. Complete: documented measured results, limits of four-cluster coverage and the minimum
   additional system/stage/metallicity cells needed for a conditional
   p(color | M_g, metallicity, RGB, system). Register a new whole-cluster
   test before any subsequent fit; NGC 7006 has already been used. Keep
   WISE as an acceptance diagnostic. Targeted tests passed; code, report,
   small manifests and the selected figure are in the milestone sync.

Measured outcome: M3 upper RGB has 97 clean gr vs 2 clean grz candidates,
but remains confined near `M_g≈+0.4…+0.9`; M3–M53 same-DECam lower loci
have `+0.0275 mag` median `(g-r)₀` difference at common `M_g`. NGC 2419
upper has a `0.090 mag` local gr residual NMAD with a strong radius trend;
only 25 quality PS1 counterparts pass the northern-system cross-check.
See `docs/cluster_rgb_native_system_v2.md` and two new manifests.

Next: the new experiment above registers NGC 5466 for its own whole-cluster
test and records NGC 7006's changed role. The old test remains frozen.

## Completed Objective: first pooled cluster RGB experiment

Status: completed as a bounded cluster audit and registered transfer
experiment. The empirical development sequences and once-only whole-cluster
test are documented in `docs/cluster_rgb_calibration_v1.md`; the tested
pooled A/B models are not adopted as a field-star distance product.

Follow the Pro review of `73ad728`. Existing cold-v4/XD, forward-parallax,
warm-giant models and all registered historical test identities stay frozen.
The new label is each cluster member's measured Legacy `g0` minus an
independent whole-system distance modulus, with shared zero-point error.

1. Correct the interpretation of the metal-poor forward model's residual
   scatter and predictive tails, and of the warm giant kernel width. Preserve
   all existing fitted numbers, decisions and products.
2. Recover the complete Vasiliev–Baumgardt M3 and NGC 2419 membership tables
   with their probabilities and Gaia CMD columns. Join every member to actual
   DR11 Legacy grz on allocated NERSC compute. Report identity uniqueness,
   `Pmember>0.9` core, lower-probability diagnostic layer, grz base/clean,
   dust, brightness, position/crowding and usable WISE bands. Spectroscopy is
   diagnostic, not a per-member eligibility condition.
3. Register broad Gaia-CMD phase regions before fitting Legacy sequences.
   Count RGB, HB/RC, AGB, main sequence and ambiguous members separately,
   include independent spectroscopic stage checks where available, and audit
   clean-photometry loss by magnitude, cluster radius, mask and fraction.
   No member selection by model distance or residual.
4. For credible RGB members, construct per-system `(g-r)0`–`M_g` and
   `(r-z)0`–`M_g` Legacy empirical sequences and width summaries. Propagate
   the external modulus and extinction as shared cluster uncertainties;
   contrast M3 low dust with NGC 2419's E(B−V)≈0.06. Compare existing frozen
   model outputs as diagnostics only. Make small inspectable figures/manifests.
5. Use measured RGB number, color/luminosity overlap and label quality to
   decide whether a minimal low-degree cross-system relation is identifiable.
   M3 has 422 clean lower-RGB and only 2 clean upper-RGB candidates; NGC 2419
   has 391 clean upper-RGB candidates and a lower region overlapping red HB.
   Their secure luminosity ranges barely overlap. M53 is preregistered as a
   development bridge and NGC 7006 as a *whole-system reserved test* before
   either system's Legacy RGB photometry is inspected. If a relation is
   identifiable, assess per-system center, scatter and tails. Test
   WISE on the same reliable unblended stars and report its acceptance rate.
   Do not claim photo-only metallicity or phase certainty from two clusters.
   M53 now supplies 156 clean upper and 253 clean lower candidates, bridging
   `M_g≈−0.19` to `+2.73`; NGC 7006 has 19 clean upper candidates reserved.
   The fixed quadratic A/B diagnostic, whole-cluster test and shared-scale
   sensitivity are registered in `docs/cluster_rgb_experiment_protocol_v1.md`.
   The development fit is frozen in `manifests/cluster_rgb_transfer_model_v1.json`:
   model A already has M3/NGC 2419 median offsets +0.688/−0.580 mag.
   It is a transfer diagnostic, not an adopted RGB distance calibration.
   Commit this model before scoring NGC 7006 once; use the holdout to
   quantify the failure or limited domain, not to retune the curve.
   The once-only NGC 7006 test is complete (19 clean upper-RGB stars,
   A median Δμ +0.166 mag, NMAD 0.323 mag, 5/19 |Δμ|>0.5). The small
   holdout and large development-system offsets do not support adopting
   either pooled quadratic. Preserve the cluster loci and report the
   shared distance/dust scale sensitivities; do not retune on NGC 7006.
6. Validate, document in reverse-chronological WORKLOG, commit milestone
   changes and sync code, docs, small manifests and selected plots across
   local, GitHub and NERSC; large catalogs stay on NERSC.

## Next recommended steps

- Add independent low-dust RGB calibrator systems spanning metallicity at
  *overlapping* luminosities, with separate whole-system tests registered
  before fitting. Give their external distance and dust-scale errors a
  shared-cluster treatment.
- Improve phase evidence for red HB/AGB overlap and quantify per-star image
  quality at the bright end. Preserve the existing clean-grz rule until a
  measured recovery layer is validated.
- Only after those systems exist, test a metallicity-marginalized RGB
  luminosity distribution and an explicit RGB/HB/AGB mixture for ordinary
  targets without spectra. Retain the cold-dwarf baseline as a separate
  conditional product.

## Completed Objective: Pro review of caeb218

Keep cold-v4/XD models, routes and earlier final-test identities frozen.
Complete a bounded correction and calibration round with these milestones:

1. Complete: separate general RVS dwarf/giant evidence, cold-dwarf hypothesis conflict,
   4200–5000 K blue-branch training eligibility and calibrated-abundance
   validity. Preserve raw Redrock QSO/galaxy type with a separate quality
   state; the MWS VAC carries `RR_SPECTYPE` and `RR_Z` but no Redrock `ZWARN`
   or `DELTACHI2`. Record CMD white-dwarf evidence independently from the
   spectral label, count conflicts and omit unresolved conflicts from the
   trusted training core. Version the candidate and joint catalogs and
   verify all frozen distance/route columns against earlier products.
2. Complete: review Gaia source 3601173023628559616 by identity, neighbors,
   photometry, image-quality/exposure metadata and available alternate
   measurements; decide its core eligibility without residual-based cuts.
   Extend the joint luminosity plot to include `M_g≈10.19` anchors.
3. Complete: build a targeted giant sample from qualified broad-temperature RVS class
   evidence and actual SpecDis v2.1 / SPDist v1.0 tables. Preserve each
   catalog's identity, quality, distance uncertainty, training correlation
   and applicability; derive Legacy `M_g` from DR11 `g0` and a distance
   distribution. Stratify cool/warm, `logg<1.8`, abundance-valid and
   RGB/RC-ambiguous cases. Audit overlap and label disagreements. The v3
   identity fix yields 2,217/2,044 valid clean SpecDis/SPDist labels;
   both-valid Legacy Mg labels differ by −0.207 mag median. Details:
   `docs/giant_vac_v3.md`.
4. Complete as a bounded, non-adopted model experiment: select independent
   cluster systems with actual
   qualified members and DR11 light; keep whole systems out of fitting.
   Fit and validate a first bounded photo-only giant luminosity relation
   only where label quality and external scale coverage support one.
   The registered grz-only warm evolved-star kernel-mixture test is complete
   (`docs/giant_photo_baseline_v1.md`): 588 reserved stars yield 0.464 mag
   NMAD against SPDist point labels, but five M3 stars show +0.367 mag
   median modulus shift against an independent RR Lyrae reference. Do not
   adopt this model as a giant distance product. A separate
   NGC 2419 RR Lyrae whole-system audit is complete: 13 valid SPDist giants
   have pooled median offset +0.007 mag, while three <5000 K giants have
   median −0.543 mag, so the cool branch needs further scale validation.
   The 4200–5000 K and cooler RGB samples are reported separately. A DESI-first ranking of
   untouched complete DR11 north/south 10° cells identified two southern
   development cells and one separated northern reserved cell. Their DR11
   joint samples have 225/59 conservative 4200–5000 K giants and 173/11
   clean low-dust giants in development/reserved roles. Only 42/2 retain
   valid SPDist labels, so a K-giant model is not fitted from this set;
   `docs/giant_ranked_cool_yield_v1.md` records the selection and label
   bottlenecks. The existing eleven fields supply another 42 SPDist labels
   at 4200–5000 K but do not repair the two-object reserved check. The
   northern reserved optical loss is dominated by bright fraction/mask
   failures, with rejected/retained median `g0` 15.53/18.28.
5. Complete as a bounded spectroscopy-conditioned experiment: combine the seven-field and four targeted-field main-sequence samples
   with source identity deduplication. On development-only data, compare a
   low-dimensional smooth color baseline with and without one continuous
   [Fe/H] term using a forward parallax likelihood for high, medium, weak
   and negative observations. Propagate optical, iron and intrinsic scatter;
   assess parameter/prior sensitivity and reserved-field parallax-space
   predictions. Keep this a feasibility experiment, not a new all-sky release.
   The one-iron model improves reserved parallax log density in five fields;
   β ranges from −0.679 to −0.654 mag/dex in leave-one-development-field-out
   fits. Its use for photo-only targets awaits a metallicity population prior.
6. Complete: validate focused code/data results, update reports and WORKLOG,
   commit coherent milestones and sync programs, manifests and selected
   figures across local, GitHub and NERSC. Large catalogs remain on NERSC
   compute.

## Next recommended science steps

- Keep cold-v4 conditional distances and XD non-adoption decision frozen.
  A global cold-dwarf catalog still needs class probabilities and a
  metallicity-population prior; the spectroscopic iron coefficient alone
  cannot be applied to stars without measured abundance.
- Before another giant fit, identify a low-extinction independent system
  with enough clean K giants and an external distance scale. Audit the
  bright-end fraction/mask losses and SPDist K quality there, using PS1
  only as a photometry cross-check where useful. Preserve RGB, RC and
  subgiant ambiguity explicitly; do not tune to the five M3 members.
- If a photo-only giant experiment is resumed, compare actual grz and
  usable WISE inputs on the same objects, preserve multimodal luminosity
  distributions, and register a new whole-system test before model choice.

## Completed Pro Follow-up Objective

The latest Pro review of `1acf946` is in a bounded joint-calibration round.
Keep cold-v4 models, routes, registered XD adoption gate and historical test
identities frozen. Correct the XD physical interpretation and Gaia point-
reference coverage wording; version the candidate export with existing class
evidence; build a same-star DESI–Gaia–Legacy feasibility sample in seven
registered fields and four DESI-selected DR11 fields. Decide from measured
overlap whether a blue metal-poor
main-sequence relation can be calibrated and identify K giant/cool RGB labels.
No new distance model or deep interval is adopted in this round.

## Current Round Milestones

1. Complete: revise XD result interpretation, label raw interval coverage relative to
   noisy Gaia point references, and correct blue inventory base versus clean,
   low-dust denominators. Keep the original v4 and XD decision records intact.
2. Complete: add per-candidate DESI RVS/Redrock evidence availability, source, quality,
   conservative class and conflict flag. Export versioned v2 FITS products on
   allocated NERSC compute, retaining v1 catalogs and photometric distances.
   Unknown spectra remain unknown. Verify schema and counts field by field.
3. Complete: on allocated compute, assemble a bounded same-star DESI–Gaia–Legacy table
   from `configs/joint_calibration_fields.json`. Start with unique,
   unambiguous DESI matches and report measured/clean grz, low dust,
   4200–5000 K calibrated continuous Fe/H and formal error, conservative
   `logg−sigma_logg>4.2` main-sequence core, transitional/evolved and CMD
   white-dwarf candidates. Retain raw/corrected parallax and weak/negative
   values; derive M_g only for quality high-S/N anchors. Reserve separate
   fields for subsequent validation before any model fitting.
4. Complete: report color, temperature, metallicity, brightness and sky coverage by
   class and parallax layer. Quantify whether the same-star metal-poor core
   supports zero point, color slope and metallicity dependence; otherwise
   state the specific missing overlap. Inventory reliable K giant/cool RGB
   labels and independent luminosity resources for the next science branch.
5. Complete: run focused validation, write concise scientific report and WORKLOG,
   commit milestone(s) and synchronize programs, docs, manifests and selected
   figures through local, GitHub and NERSC. Catalogs remain on NERSC.

Current compute checkpoint: v2 candidate jobs 60484208/209/210 completed
and allocated validation 60484704 confirms all 14 frozen identity/route/
distance columns match v1 for 395,629 objects. The seven-field joint study
60484207 contains 37,214 unique DESI matches, 224 clean low-dust high-S/N
conservative cool-main-sequence stars and only one `[Fe/H]<−1.5` anchor,
none blue of the frozen edge. DESI-first ranking 60484858 identified four
unprocessed DR11 fields with 8/8/6/4 low-dust high-lat metal-poor targets
at *raw* Gaia S/N≥20. Array 60485082 built those fields; dependent job
60485094 found 132,530 unique matches, including 18 high-S/N metal-poor
main-sequence core stars. Five lie blue of the v4 edge (4 development,
1 reserved). The clean metal-poor core has another 42 at S/N 10–20, 228
at S/N 0–10 and 3 negative corrected parallaxes. One blue high-S/N object
has discordant optical and Gaia/spectral colors. Targeted fields include
230 conservative cool giants, 103 clean low-dust, but zero direct high-S/N
geometric giant anchors. No new model is adopted.
The cold-v4 and registered XD decisions remain frozen.

## New Round Milestones

1. Complete: reconcile the prior review's interpretation and publish an explicit
   development/selection protocol and candidate-output schema.
2. Complete: export v4 per-candidate A/B/C choice, higher-tier fallback reasons, support
   ratios, infrared-conflict marker, cold-dwarf-hypothesis status and validated
   depth state without changing distances or deleting conflicts.
3. Complete: build a full-covariance heteroscedastic deconvolved Gaussian mixture
   `p(M_g, colors)` on exactly the v4 A/B/C training identities. Use 16/32/64
   components and propagate shared-band and Gaia label uncertainty.
4. Complete: compare configurations by held-out development sky blocks. On identical
   accepted objects, compare chosen candidate against frozen v4 in bias,
   NMAD, tails, probability quality, g/r perturbation response and error/color
   strata. The registered gate rejected XD K=32; no new final test follows.
   A separate color-stratified g/r response diagnostic confirms the same
   direction in both occupied color bins without altering that decision.
5. Complete: define the metal-poor blue branch from reliable Gaia/DESI labels: coverage,
   color-boundary failures, CMD position and giant/nonstar confusion. Keep it
   distinct from the frozen cold-v4 box. The current four-field inventory
   and label/anchor contract are in `docs/cold_v5_blue_branch.md`.
6. Complete: validate, document, commit and synchronize programs, small manifests and
   figures across local, GitHub and NERSC. Keep catalogs and model files on
   NERSC and use allocated CPUs for scans/training. Three candidate exports
   and single-table checks are complete; 81 tests pass. The final color
   response manifest and figure close the comparison record.

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

## Next Recommended Objective

Keep frozen v4 as the conditional cold-dwarf distance baseline. Build a
targeted K-giant/cool-RGB class and luminosity sample from qualified DESI
spectra, nearby high-S/N Gaia giants, clusters and independently audited
spectroscopic-distance resources. Register development and final identities
before fitting or evaluating a new giant relation. Keep class and metallicity
uncertainty explicit for no-spectrum application targets.

For the metal-poor main-sequence extension, the targeted DR11 sample supplies
18 high-S/N anchors below `[Fe/H]=−1.5`, five blue of the old edge and only
one blue in reserved validation. Audit the discordant-color case and seek
more clean geometric anchors before fitting a blue color–luminosity relation.
Develop a forward-parallax likelihood for the 42 intermediate, 228 weak and
three negative-parallax clean metal-poor core objects; preserve the spectral
selection model and source identities. The XD comparison remains a research
candidate under its registered gate, with no change to v4 output.
