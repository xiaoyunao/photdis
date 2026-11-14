#!/usr/bin/env python3
"""Reconcile the archived v3.2 external forecasts, openings and score tables."""
import csv
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path('manifests/v3_2')
VIEWS = {'candidate', 'qrf', 'nominal_reference', 'old_target'}


def audit(root=ROOT):
    sources = {}
    checks = []

    def require(label, condition):
        if not condition:
            raise ValueError(label)
        checks.append(label)

    def hash_file(path, expected=None):
        digest = sha256((root / path).read_bytes()).hexdigest()
        if expected is not None:
            require(f'SHA256 {path}', digest == expected)
        sources[str(path)] = digest
        return digest

    def record(path):
        hash_file(path)
        return json.loads((root / path).read_text())

    def rows(path):
        with (root / path).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def compact_path(name):
        path = Path(name)
        if path.parts[:2] == ('data', 'v3_2'):
            path = BASE.joinpath(*path.parts[2:])
        return path

    decision_path = BASE / 'branch_decision.json'
    decision = record(decision_path)
    require('Eligible opened-only decision', decision['development_evaluation_complete'] is True
            and decision['new_outcomes_read'] is False
            and decision['approved_for_prospective_evaluation'] is True)
    for name, digest in decision['inputs'].items():
        hash_file(Path(name), digest)

    scopes = {}
    for role, prefix, n, admitted in [('dr20', 'minesweeper', 1117, 1107), ('common', 'common_scale', 103, 80)]:
        folder = BASE / f'final_validation_{role}'
        freeze_path = folder / 'freeze.json'
        freeze = record(freeze_path)
        started = record(folder / 'started.json')
        start_hash = started['sources']['freeze_sha256'] if role == 'dr20' else started['freeze_sha256']
        require(f'{role} freeze before opening', freeze['predictions_frozen'] is True
                and freeze['outcomes_opened'] is False and freeze['n'] == n
                and freeze['job'] == started['job'] and start_hash == sources[str(freeze_path)])
        if role == 'dr20':
            require('DR20 freeze timestamp and views',
                    datetime.fromisoformat(freeze['utc']) <= datetime.fromisoformat(started['utc'])
                    and set(freeze['views']) == VIEWS and freeze['all_PSF'] is True)

        forecast_name = f'data/v3_2/{prefix}_forecast/complete.json'
        forecast = record(compact_path(forecast_name))
        freeze_hashes = freeze['hashes']
        require(f'{role} forecast declares frozen decision and gate',
                forecast['cohort'] == freeze['cohort'] and forecast['n'] == n
                and forecast['all_PSF'] is True and forecast['references_read'] is False
                and forecast['pilot'] is False and set(forecast['views']) == VIEWS
                and forecast['decision_sha256'] == sources[str(decision_path)]
                and forecast['candidate_admission']['admitted'] == admitted)
        for view, entry in {**forecast['views'], 'candidate_admission': forecast['candidate_admission']}.items():
            require(f'{role} payload bound in freeze for {view}',
                    freeze_hashes[entry['payload']] == entry['sha256'])
        require(f'{role} forecast ancestry in freeze',
                all(freeze_hashes.get(name) == digest for name, digest in forecast['inputs'].items()))
        remote_only = []
        for name, digest in freeze_hashes.items():
            path = compact_path(name)
            if (root / path).is_file():
                hash_file(path, digest)
            else:
                require(f'{role} remote payload declared: {name}', Path(name).parts[0] == 'data')
                remote_only.append(name)
        require(f'{role} forecast receipt archived', str(compact_path(forecast_name)) in sources)

        intake = BASE / ('minesweeper_native' if role == 'dr20' else 'common_scale_inventory')
        native = record(intake / 'complete.json')
        require(f'{role} native DR11 PSF admission receipt',
                native['all_admitted_PSF'] is True and native['outputs']['photometry.npz'] == forecast['photometry_sha256'])
        for name in ('history.csv', 'counts.json' if role == 'dr20' else 'systems.csv'):
            hash_file(intake / name, native['outputs'][name])
        diagnostics = rows(intake / 'history.csv')
        require(f'{role} historical overlap audit',
                diagnostics and all(int(r['identity_overlap']) == int(r['sky_overlap']) == 0
                                    if role == 'dr20' else int(r['member_overlap']) == 0 for r in diagnostics))

        if role == 'dr20':
            evidence_path = folder / 'evidence_complete.json'
            evidence = record(evidence_path)
            score_dir = folder / 'scores'
            score = record(score_dir / 'complete.json')
            require('DR20 evidence and score refer to the same freeze and release',
                    evidence['freeze_sha256'] == score['freeze_sha256'] == start_hash
                    and evidence['job'] == score['job'] == freeze['job']
                    and evidence['n'] == score['n'] == native['admitted'] == n
                    and evidence['all_PSF'] is True and score['all_PSF'] is True
                    and evidence['outcomes_opened'] is True
                    and set(score['views']) == VIEWS and score['candidate_admitted'] == admitted
                    and score['evidence_sha256'] == sources[str(evidence_path)])
            hash_file(folder / 'coverage.csv', evidence['outputs']['coverage.csv'])
            for name, digest in score['outputs'].items():
                hash_file(score_dir / name, digest)
            objects = rows(score_dir / 'objects.csv')
            identities = {}
            view_pass = dict.fromkeys(VIEWS, 0)
            for row in objects:
                identity, ls_id = int(row['identity']), int(row['ls_id'])
                require(f'DR20 strict PSF row {identity} {row["view"]}', identity > 0 and ls_id > 0
                        and row['TYPE'] == 'PSF' and row['view'] in VIEWS
                        and row['numerical_pass'] in {'True', 'False'}
                        and row['candidate_admission'] in {'True', 'False'})
                last, gate, seen = identities.setdefault(identity, (ls_id, row['candidate_admission'], set()))
                require(f'DR20 unique aligned view {identity} {row["view"]}',
                        (ls_id, row['candidate_admission']) == (last, gate) and row['view'] not in seen)
                seen.add(row['view'])
                view_pass[row['view']] += row['numerical_pass'] == 'True'
            require('DR20 exact photometric gate and four views per identity',
                    len(identities) == len({row[0] for row in identities.values()}) == n
                    and sum(row[1] == 'True' for row in identities.values()) == admitted
                    and all(row[2] == VIEWS for row in identities.values()))
        else:
            result = record(folder / 'complete.json')
            require('HST scores refer to the same freeze and opening',
                    result['freeze_sha256'] == start_hash and result['job'] == freeze['job']
                    and result['outcomes_opened'] is True and result['all_PSF'] is True
                    and result['n'] == native['admitted_members'] == n
                    and result['n_systems'] == native['admitted_systems'] == 2)
            for name, digest in result['outputs'].items():
                hash_file(folder / name, digest)
            by_system = {r['system']: int(r['admitted']) for r in rows(intake / 'systems.csv') if int(r['admitted'])}
            require('HST PSF membership roster', by_system == {'NGC_104_47Tuc': 77, 'NGC_362': 26})
            scored = [r for r in rows(folder / 'scale_agreement.csv')
                      if r['scope'] == 'native_photometric' and r['candidate_minus_view'] == 'False'
                      and r['metric'] == 'median_relative_residual']
            expected = {(name, view, accepted) for name in by_system for view in VIEWS
                        for accepted in ('True', 'False')}
            require('HST scales use every native member in each unreduced view',
                    len(scored) == len(expected) and {(r['system'], r['view'], r['numerical_only']) for r in scored} == expected
                    and all(int(r['registered']) == by_system[r['system']] for r in scored)
                    and all(int(r['n']) == by_system[r['system']] for r in scored if r['numerical_only'] == 'False'))
            view_pass = {view: sum(int(r['n']) for r in scored if r['view'] == view
                                   and r['numerical_only'] == 'True') for view in VIEWS}
        require(f'{role} retained forecast flags reconcile', view_pass == {
            view: entry['passed'] for view, entry in forecast['views'].items()})
        scopes[role] = dict(n=n, admitted=admitted, numerical_flags={k: n-v for k, v in sorted(view_pass.items())},
                            freeze_job=freeze['job'], remote_payloads_declared=len(remote_only))

    return dict(passed=True, checks=len(checks), scopes=scopes, sources=sources,
                scope='Archived receipts and compact tables. Remote payload digests are bound by the '
                      'compute-step freeze. This local audit does not reread remote arrays or reference outcomes.')


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
