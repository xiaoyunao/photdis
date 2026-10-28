# Program and figure synchronization

The local checkout coordinates two Git remotes:

- `origin`: `https://github.com/xiaoyunao/photdis.git` (private)
- `nersc`: `yunao@perlmutter.nersc.gov:/pscratch/sd/y/yunao/photdis_pipeline`

NERSC receives commits over the user's existing SSH authentication. It does not
need a GitHub token. Its working checkout uses
`receive.denyCurrentBranch=updateInstead`: a push updates clean tracked files and
Git rejects changes that would overwrite local work. Ignored data stays in place.

## Local commands

Run from `/Users/yunaoxiao/Desktop/photdis`:

```bash
python3 scripts/sync.py status
python3 scripts/sync.py check
git add scripts/ docs/ README.md PLAN.md WORKLOG.md
git diff --cached
git commit -m "Describe the completed change"
python3 scripts/sync.py push
```

The helper requires clean `main` for check/push/pull. It checks the committed tree
and reachable Git history before transfer, rejects catalog/model/log paths and
unsupported binaries, and limits source files to 5 MiB and selected figures to
20 MiB each. Images/PDFs belong under `figures/`. `.gitignore` excludes raw data,
sample tables, models, caches and logs from ordinary staging.

Push updates GitHub and then NERSC. A failure reports which remote was updated;
resolve the cause and rerun. Both pushes use normal fast-forward rules. `status`
compares committed main revisions; it does not assert that remote working files
are clean. Check those with `git status` on NERSC when necessary.

## Return selected figures or program changes from NERSC

On NERSC, review and commit exactly the intended files. For the infrastructure
demonstration, these commands are safe on a login node:

```bash
cd /pscratch/sd/y/yunao/photdis_pipeline
python3 scripts/make_sync_check.py
git status --short
git add figures/sync_check.svg
git diff --cached --stat
git commit -m "Verify NERSC figure synchronization"
```

Then locally:

```bash
cd /Users/yunaoxiao/Desktop/photdis
python3 scripts/sync.py pull
```

This fetches committed NERSC changes, fast-forwards local main and pushes the
result to GitHub. If local main is already ahead, run `push` to converge. If
histories diverge, stop and inspect the commits; the helper does not resolve
conflicts or force-push. Commit intended remote changes before pushing new local
work to NERSC. Use the same procedure for reviewed program changes made there.

## Authentication and recovery

Local repository SSH is configured for batch authentication and a 15-second
connection timeout. If NERSC authentication expires, run `sshproxy -u yunao`
interactively, enter the password/OTP yourself, then retry. No credentials are
stored in this repository or sent to NERSC for GitHub access.

To resume on another machine, clone the private GitHub repository, configure the
`nersc` remote above, inspect status/history/WORKLOG/PLAN, and fetch before editing.
Check clean fast-forward relationships before syncing. Do not replace a checkout
that has uncommitted work or use force-push to resolve a disagreement.

Archives are outside the active repository; see `docs/archives.json`. The old
NERSC `photodis` directory is intentionally empty. The only active runtime
checkout is `photdis_pipeline`.
