#!/usr/bin/env python3
"""Check that v3.2 preserves the tracked files of the reviewed release."""
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "7699299b83a152327a83dc503dd3e1da0b437ca5"
PROJECT_FILES = {".gitignore", "PLAN.md", "WORKLOG.md"}


def examine(repo=ROOT):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True)

    def path_set(output):
        return set(output.split("\0")) - {""}

    original = path_set(git("ls-tree", "-r", "--name-only", "-z", BASELINE))
    protected = original - PROJECT_FILES
    head = git("rev-parse", "HEAD").strip()
    views = {}
    for view, options in (
        ("commit", [head]),
        ("index", ["--cached"]),
        ("worktree", []),
    ):
        paths = path_set(git("diff", "--no-ext-diff", "--no-textconv", "--no-renames",
                             "--name-only", "-z", *options, BASELINE, "--"))
        views[view] = {
            "changed_project_files": sorted(paths & PROJECT_FILES),
            "protected_changes": sorted(paths & protected),
        }
    return {
        "baseline_commit": BASELINE,
        "examined_commit": head,
        "protected_baseline_files": len(protected),
        "allowed_project_files": sorted(PROJECT_FILES),
        "views": views,
        "passed": all(not view["protected_changes"] for view in views.values()),
        "scope": "Tracked files. Science payloads and live deployment are outside this Git audit.",
    }


if __name__ == "__main__":
    result = examine()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
