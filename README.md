# photdis

Photometric stellar distances from Legacy Survey DR11 `grzW1W2`, following the
GPT Pro restart plan in conversation
`6ae164f5-1a74-83e9-a2c0-768b28e09d1b`.

Current state: fresh repository and synchronization infrastructure only.
Sample construction and model training require the next user checkpoint.

## Locations

- Local repository: `/Users/yunaoxiao/Desktop/photdis`
- GitHub: https://github.com/xiaoyunao/photdis (private)
- NERSC working checkout: `/pscratch/sd/y/yunao/photdis_pipeline`
- NERSC `/pscratch/sd/y/yunao/photodis` is intentionally empty.
- Source catalogs remain in their shared NERSC locations; see
  [data sources](docs/data_sources.md).

## Environment and operation

Synchronization helpers use Python 3 standard library, Git and SSH.
NERSC login: `ssh yunao@perlmutter.nersc.gov`; renew expired authentication with
`sshproxy -u yunao`. Small metadata/few-row reads may run on login nodes.
Full scans, crossmatches and training must use allocated compute resources.

See [synchronization](docs/sync.md) for exact commands and recovery procedures.
Programs, configuration, documentation and selected figures are versioned.
Catalogs, generated sample tables, models, caches and logs stay out of Git.

## Continuity

Read Git history, [WORKLOG.md](WORKLOG.md), then [PLAN.md](PLAN.md) when resuming.
The previous project is preserved outside the active directories; see
[archive locations](docs/archives.json). Its last commit was
`f846aee9ce0ce39a5b28f48801b77b29506a0ff1`; the new repository has a fresh history.
