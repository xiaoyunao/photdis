#!/usr/bin/env python3
"""Sync committed programs and selected figures through local Git remotes."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRS = {
    "data", "datasets", "catalogs", "models", "results", "results_small",
    "logs", "cache", "archives", "checkpoints", "__pycache__", ".deps", ".venv",
}
SOURCE_SUFFIXES = {
    ".py", ".sh", ".md", ".toml", ".yaml", ".yml", ".json", ".txt",
    ".rst", ".ini", ".cfg", ".ipynb",
}
FIGURE_SUFFIXES = {".svg", ".png", ".pdf", ".jpg", ".jpeg", ".webp"}
SPECIAL_NAMES = {".gitignore", ".gitattributes", ".gitkeep", "LICENSE", "Makefile"}
SOURCE_LIMIT = 5 * 1024 * 1024
FIGURE_LIMIT = 20 * 1024 * 1024
NERSC_HOST = "yunao@perlmutter.nersc.gov"
NERSC_PATH = "/pscratch/sd/y/yunao/photdis_pipeline"


class SyncError(Exception):
    pass


def git(*args: str, input_text: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotePath=false", *args], cwd=ROOT,
        input=input_text, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SyncError(f"git {' '.join(args)} failed:\n{detail}")
    return result.stdout


def require_main(clean: bool = True) -> None:
    if git("branch", "--show-current").strip() != "main":
        raise SyncError("Run synchronization from branch main.")
    if clean:
        status = git("status", "--porcelain", "--untracked-files=normal")
        if status:
            raise SyncError("Commit or resolve working changes before syncing:\n" + status)


def path_limit(name: str) -> int:
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in {".", ".."} for p in path.parts):
        raise SyncError(f"Unsafe tracked path: {name}")
    if any(c in name for c in "\n\r\t") or name.startswith('"'):
        raise SyncError(f"Unsupported tracked filename: {name!r}")
    if any(p.lower() in BLOCKED_DIRS for p in path.parts[:-1]):
        raise SyncError(f"Data/generated directory cannot be synced: {name}")
    if path.parts[0] == "figures":
        if path.name == ".gitkeep":
            return SOURCE_LIMIT
        if path.suffix.lower() not in FIGURE_SUFFIXES:
            raise SyncError(f"Only selected image/PDF files belong in figures/: {name}")
        return FIGURE_LIMIT
    if path.name not in SPECIAL_NAMES and path.suffix.lower() not in SOURCE_SUFFIXES:
        raise SyncError(f"Unsupported source file; catalogs/models/binaries stay on NERSC: {name}")
    return SOURCE_LIMIT


def check_tree(ref: str = "HEAD") -> int:
    """Check tree and reachable history, since Git transfers historical blobs too."""
    records = git("ls-tree", "-r", "-l", "-z", ref).split("\0")
    count = 0
    for record in records:
        if not record:
            continue
        metadata, name = record.split("\t", 1)
        mode, kind, _oid, size = metadata.split()
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise SyncError(f"Symlinks/submodules are excluded from sync: {name}")
        if int(size) > path_limit(name):
            raise SyncError(f"Tracked file exceeds sync size limit: {name} ({size} bytes)")
        count += 1

    objects = []
    for line in git("rev-list", "--objects", ref).splitlines():
        oid, _, name = line.partition(" ")
        objects.append((oid, name))
    details = git("cat-file", "--batch-check", input_text="".join(o + "\n" for o, _ in objects))
    for (oid, name), detail in zip(objects, details.splitlines()):
        actual_oid, kind, size = detail.split()
        if actual_oid != oid:
            raise SyncError("Unexpected Git object inspection result.")
        if kind == "blob" and int(size) > path_limit(name):
            raise SyncError(f"Historical file exceeds sync size limit: {name} ({size} bytes)")
    return count


def remote_head(remote: str) -> str:
    rows = git("ls-remote", "--heads", remote, "refs/heads/main").splitlines()
    return rows[0].split()[0] if rows else "(empty main)"


def preflight_nersc() -> None:
    url = git("remote", "get-url", "nersc").strip()
    host, separator, path = url.partition(":")
    if not separator or host != NERSC_HOST or path.rstrip("/") != NERSC_PATH:
        raise SyncError(f"Unexpected NERSC SSH remote: {url!r}")
    command = f"cd {shlex.quote(path)} && python3 scripts/sync.py check"
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
            cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        raise SyncError("NERSC preflight timed out before any fetch.") from exc
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SyncError("NERSC preflight failed before any fetch:\n" + detail)
    print("NERSC committed files/history passed remote preflight.", flush=True)


def status() -> None:
    require_main(clean=False)
    heads = {"local": git("rev-parse", "HEAD").strip()}
    errors = []
    for remote in ("origin", "nersc"):
        try:
            heads[remote] = remote_head(remote)
        except SyncError as exc:
            heads[remote] = "unreachable"
            errors.append(str(exc))
    for label, head in heads.items():
        print(f"{label:6s} {head}")
    print("All three commits match." if len(set(heads.values())) == 1 else "Commits differ.")
    changes = git("status", "--short", "--untracked-files=normal")
    if changes:
        print("Working changes:\n" + changes, end="")
    if errors:
        raise SyncError("\n".join(errors))


def push() -> None:
    require_main()
    print(f"Checked {check_tree()} committed files and reachable history.", flush=True)
    completed = []
    try:
        for remote in ("origin", "nersc"):
            git("push", remote, "HEAD:refs/heads/main")
            completed.append(remote)
            print(f"Updated {remote}.", flush=True)
    except SyncError as exc:
        raise SyncError(f"Push stopped. Updated remotes: {', '.join(completed) or 'none'}.\n{exc}") from exc
    status()


def pull() -> None:
    require_main()
    check_tree("HEAD")
    preflight_nersc()
    git("fetch", "nersc", "main")
    print(f"Checked {check_tree('FETCH_HEAD')} files in the NERSC revision.", flush=True)
    git("merge", "--ff-only", "FETCH_HEAD")
    print("Local main now includes the committed NERSC changes.", flush=True)
    check_tree("HEAD")
    try:
        git("push", "origin", "HEAD:refs/heads/main")
    except SyncError as exc:
        raise SyncError("Local merge completed; GitHub push failed. Resolve it and run push.\n" + str(exc)) from exc
    status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "check", "push", "pull"), help=(
        "status: compare commits; check: audit committed files/history; "
        "push: local to GitHub and NERSC; pull: NERSC to local then GitHub"))
    args = parser.parse_args()
    try:
        if args.action == "check":
            require_main()
            print(f"Checked {check_tree()} committed files and reachable history.")
        else:
            {"status": status, "push": push, "pull": pull}[args.action]()
    except (SyncError, OSError) as exc:
        print(f"Sync error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
