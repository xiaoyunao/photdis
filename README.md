# photdis

Photometric stellar distances from Legacy Survey DR11 `grzW1W2`, following the
GPT Pro restart plan in conversation
`6ae164f5-1a74-83e9-a2c0-768b28e09d1b`.

Current state: Pro step 1 sample construction and coverage audit are authorized.
Model training requires the next user checkpoint.

## Locations

- Local repository: `/Users/yunaoxiao/Desktop/photdis`
- GitHub: https://github.com/xiaoyunao/photdis (public)
- NERSC working checkout: `/pscratch/sd/y/yunao/photdis_pipeline`
- NERSC `/pscratch/sd/y/yunao/photodis` is intentionally empty.
- Source catalogs remain in their shared NERSC locations; see
  [data sources](docs/data_sources.md).

## Environment and operation

Synchronization helpers use Python 3 standard library, Git and SSH.
NERSC login: `ssh yunao@perlmutter.nersc.gov`; renew expired authentication with
`sshproxy -u yunao`. Small metadata/few-row reads may run on login nodes.
Full scans, crossmatches and training must use allocated compute resources.

Stage 1 uses the NERSC DESI environment (`source
/global/common/software/desi/desi_environment.sh main`) for NumPy, fitsio,
Astropy, healpy, SciPy and Matplotlib. `gaiadr3-zeropoint==0.1.0` is installed
without dependencies into ignored `.deps/` for the published Gaia parallax
zero-point correction. Export `PYTHONPATH=$PWD/src:$PWD/.deps` when running.

See [synchronization](docs/sync.md) for exact commands and recovery procedures.
Programs, configuration, documentation and selected figures are versioned.
Catalogs, generated sample tables, models, caches and logs stay out of Git.

## Continuity

Read Git history, [WORKLOG.md](WORKLOG.md), then [PLAN.md](PLAN.md) when resuming.
The previous project is preserved outside the active directories; see
[archive locations](docs/archives.json). Its last commit was
`f846aee9ce0ce39a5b28f48801b77b29506a0ff1`; the new repository has a fresh history.
