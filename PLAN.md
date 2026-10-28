# PLAN

## Current Objective

Test one sky field and present its sample selection, counts and figures for user
review. The previous full-footprint run exceeded the intended first checkpoint.
Full-footprint finalization is stopped.

## Milestones

1. Use one DR11 south sweep field; the first corrected smoke field is
   sweep-000m005-005p000.fits (RA 0–5 deg, Dec -5–0 deg).
2. Examine its Gaia anchor/extended and spectroscopy selections and plot the
   actual distributions. Explain the cuts with the field results.
3. Present the field result and wait for user confirmation.

## Current State

- GitHub is public; code and prior outputs remain in the repository/NERSC.
- The 1,074-tile scan has completed; its fragments are on NERSC.
- Serial full-table materialization was interrupted. Its partial tables are at
  data/stage1_v1/tables_partial_serial_20261028T060107Z.
- Both compute allocations (60424566, 60426781) have been released.
- No model has been trained. No full-table result has been published.

## Validation Criteria

- Review one field's data and selection at the next checkpoint.
- Check Gaia IDs, parallax signs, photometric cuts, spectral matches and test
  reservations using that field.
- Synchronize the pilot code and figures across local, NERSC and GitHub.

## Next Step

One-field inspection and user review. Keep full-footprint processing stopped.
