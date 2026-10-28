# WORKLOG

## 2026-10-28

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
