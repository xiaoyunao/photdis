# PLAN

## Current Objective

Review the completed frozen-model follow-up with the user, then refine
conditional interval calibration and star-type probabilities based on the
measured red-end, depth and contamination behavior.

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
8. Next after user review: fit and validate color/depth-dependent interval
   adjustments using an untouched final test, and evaluate probabilistic
   dwarf/giant/nonstar selection. Decide on recovery-sample training only
   after that comparison.

## Scope

Original field RA [0,10), Dec [-10,0) remains a locked baseline. New fields,
roles and source/model hashes are in configs/followup_fields.json. The strict
optical and Gaia criteria remain the comparison standard. The previous full-
footprint scan remains stopped. APOGEE remains auxiliary.

## Next Checkpoint

Present docs/followup_validation.md and the tracked manifests/figures. Ask
the user to review the conditional-calibration and star-type next step.
