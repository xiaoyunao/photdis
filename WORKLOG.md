# WORKLOG

## 2026-10-28

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
