#!/usr/bin/env python3
"""Audit tracked baseline preservation, including staged and working changes."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASELINE = '31f12772fe29c801db7269adf3a8b160e264bb5f'
MUTABLE_PROJECT_FILES = {'.gitignore', 'README.md', 'AGENTS.md', 'WORKLOG.md', 'PLAN.md'}
NEW_PREFIXES = ('src/photdis/v3_1/', 'manifests/v3_1/', 'figures/v3_1/',
                'scripts/v3_1_', 'scripts/run_v3_1_', 'tests/test_v3_1_',
                'configs/v3_1_', 'docs/v3_1_')


def examine(repo=ROOT, baseline=BASELINE):
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo, text=True)

    baseline = git('rev-parse', baseline + '^{commit}').strip()
    original = set(git('ls-tree', '-r', '--name-only', '-z', baseline).split('\0')) - {''}
    differences = {}
    for view, option in [('worktree', []), ('index', ['--cached'])]:
        names = set(git('diff', '--no-ext-diff', '--no-textconv', '--no-renames',
                        '--name-only', '-z', *option, baseline, '--').split('\0')) - {''}
        forbidden = []
        for name in names:
            if name in MUTABLE_PROJECT_FILES:
                continue
            # Existing v3.1-named files, if any, are still baseline artifacts.
            if name in original or not (name.startswith(NEW_PREFIXES) or
                                       name == 'docs/pro_review_31f1277.md'):
                forbidden.append(name)
        differences[view] = dict(changed_paths=len(names), protected_changes=sorted(forbidden))
    result = dict(baseline_commit=baseline, examined_commit=git('rev-parse', 'HEAD').strip(),
                  protected_baseline_files=len(original - MUTABLE_PROJECT_FILES),
                  allowed_project_files=sorted(MUTABLE_PROJECT_FILES), views=differences,
                  passed=not any(v['protected_changes'] for v in differences.values()),
                  scope='Tracked baseline files and additions outside v3.1 namespaces, in both index and worktree. No remote large-payload or deployment-runtime scan is claimed.')
    return result


def main():
    result = examine()
    if not result['passed']:
        raise ValueError('Protected baseline changes: ' + json.dumps(result['views']))
    result['inputs'] = {'scripts/v3_1_preservation_check.py':
                       hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output = ROOT / 'manifests/v3_1/baseline_preservation_current.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['examined_commit', 'protected_baseline_files', 'passed']}))


if __name__ == '__main__':
    main()
