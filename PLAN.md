# PLAN

## Current Objective

Completed: archival/reset, private GitHub repository and verified program/figure
synchronization. Stop here for user review before scientific sample construction
or training.

## Milestones

1. Archive old work outside the active directories and recreate empty working
   directories. Complete: local checksums and remote directory identities verified.
2. Initialize this fresh repository and synchronize it with GitHub and NERSC.
   Complete.
3. Verify code deployment and a tiny non-scientific figure round trip; verify
   matching committed versions and that catalog/data files are excluded. Complete.
4. Record the SpecDis/SPDist source assessment and stop for confirmation. Complete.

## Outstanding Issues

- Await the user's review of this completed infrastructure checkpoint.
- LAMOST and independent cluster/wide-binary validation resources remain unverified.
- SpecDis/SPDist require quality and identity joins before any scientific use;
  they are auxiliary comparisons with Gaia-related calibration, not independent
  absolute-distance truth.

## Validation Criteria

- Old local files pass archive checksums; remote directory identities survive
  same-filesystem archival. Archive paths and old Git HEAD are recorded.
- Local, GitHub and NERSC main branches end at the same commit with clean tracked
  worktrees. A tracked SVG has identical bytes on all three.
- Git ignores catalog/large generated-data formats; sync checks reject tracked
  data and large files. No sample or model code is introduced.
- NERSC photodis is empty; photdis_pipeline is the sole active checkout.

## Next Recommended Step

Wait for the user's confirmation of this completed infrastructure checkpoint.
The next scientific milestone, only after approval, is sample definitions,
coverage and protected external-test design.
