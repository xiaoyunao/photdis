# PLAN

## Current Objective

Follow the latest Pro review: repair recovery and spectroscopy comparison
criteria, preserve the existing model and test identities, diagnose the
bright-star offset, and measure interval calibration, noisy-input behavior
and transfer to three preassigned independent fields.

## Milestones

1. Freeze baseline commit, model checksums, original test predictions and
   nside=64 test cells. Preassign one external calibration and two external
   final-test fields before examining model residuals.
2. Reconcile strict and recovery nonoptical conditions; require DESI quality
   and calibration domain in spectral residual bins. Record impact on counts.
3. Decompose the recovered bright-star offset with matched strict controls,
   out-of-fit strict predictions and paired Legacy/PS1 photometry replacement.
4. Run the three external fields with unchanged selection and first measure
   direct transfer of the frozen baseline models.
5. Calibrate 68%/95% intervals on the preassigned external calibration field;
   score them once in the two final-test fields, including red-color bins.
6. Run raw-flux noise injection at observed target-depth S/N and report
   acceptance, residual and interval coverage by layer and depth.
7. Quantify target input coverage and pure-photometry cold-branch selection
   against independent Gaia/DESI labels. Plot, document and sync all results.

## Scope

Original field RA [0,10), Dec [-10,0) remains a locked baseline. New fields,
roles and source/model hashes are in configs/followup_fields.json. The strict
optical and Gaia criteria remain the comparison standard. The previous full-
footprint scan remains stopped. APOGEE remains auxiliary.

## Next Checkpoint

Present corrected recovery counts, paired-photometry diagnosis, calibrated
intervals and cross-field/noise results before deciding whether recovered
stars enter training or the W2 gain generalizes.
