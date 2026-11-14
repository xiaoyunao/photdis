#!/usr/bin/env python3
"""Archive verified small radial recovery tables after Slurm completion."""
import argparse
import csv
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import subprocess
import tarfile

from v3_2_radial_cache_audit import audit as cache_audit

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'manifests/v3_2'
REMOTE = '/pscratch/sd/y/yunao/photdis_pipeline'
HOST = 'yunao@perlmutter.nersc.gov'
FILES = ('complete.json', 'numerical.csv', 'recovery.csv')
RECOVERY_ARRAY = '61266369'


def digest(raw):
    return sha256(raw).hexdigest()


def completed_catalogs(report):
    result = {}
    for line in report.splitlines():
        task, raw, state, code = line.strip().split('|')
        array, sep, catalog = task.partition('_')
        if array != RECOVERY_ARRAY or not sep or not catalog.isdecimal():
            continue
        if state == 'COMPLETED' and code == '0:0':
            result[int(catalog)] = raw
    return result


def discover():
    command = f'sacct -n -X -P -j {RECOVERY_ARRAY} --format=JobID,JobIDRaw,State,ExitCode'
    status = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, command],
                            check=True, capture_output=True, timeout=25, text=True)
    return completed_catalogs(status.stdout)


def collect(names):
    # Names are generated from the frozen catalog IDs and fixed filenames.
    command = f'cd {REMOTE} && tar -cf - ' + ' '.join(names)
    raw = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', HOST, command],
                         check=True, capture_output=True, timeout=35).stdout
    with tarfile.open(fileobj=BytesIO(raw)) as archive:
        members = archive.getmembers()
        if (len(members) != len(names) or {m.name for m in members} != set(names)
                or any(not m.isfile() or m.size > 100_000 for m in members)):
            raise ValueError('Unexpected compact recovery archive')
        return {name: archive.extractfile(name).read() for name in names}


def validate(catalog, payload, design, outputs, design_digest, caches):
    receipt = json.loads(payload['complete.json'])
    if (receipt['catalog'] != catalog or receipt['n'] != int(design['n'])
            or receipt['all_PSF'] is not True
            or receipt['inputs']['data/v3_2/radial_design/complete.json'] != design_digest):
        raise ValueError(f'Mismatched recovery receipt for catalog {catalog}')
    for kind in ('photometry', 'truth'):
        name = f'{kind}_{catalog:03}.npz'
        if receipt['inputs'][f'data/v3_2/radial_design/{name}'] != outputs[name]:
            raise ValueError(f'Wrong {kind} payload for catalog {catalog}')
    if catalog != 0:
        repeat = int(design['repeat'])
        if receipt['inputs'][f'data/v3_2/radial_entry/repeat_{repeat:02}/complete.json'] != caches[repeat]:
            raise ValueError(f'Wrong entry cache for catalog {catalog}')
    for name in FILES[1:]:
        if receipt['outputs'][name] != digest(payload[name]):
            raise ValueError(f'Mismatched {name} for catalog {catalog}')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalogs', nargs='*', type=int,
                        metavar='CATALOG', help='Catalog IDs with completed Slurm jobs')
    parser.add_argument('--completed', action='store_true',
                        help='Find unarchived successful tasks in the production Slurm array')
    args = parser.parse_args()
    if bool(args.catalogs) == args.completed or any(not 0 <= i < 288 for i in args.catalogs):
        parser.error('Choose --completed or explicit catalog IDs in 0–287')
    jobs = discover() if args.completed else {}
    catalogs = sorted(set(args.catalogs)) if not args.completed else [
        i for i in sorted(jobs) if not all((BASE/f'radial_recovery/catalog_{i:03}'/name).is_file()
                                          for name in FILES)
    ][:32]
    if not catalogs:
        print('No new successful production catalogs')
        return
    if len(catalogs) > 32:
        parser.error('Collect at most 32 compact results in each transfer')
    paths = {i: f'radial_recovery/catalog_{i:03}' for i in catalogs}
    design_path = BASE/'radial_design'
    with (design_path/'design.csv').open(newline='') as stream:
        design = {int(r['catalog']): r for r in csv.DictReader(stream)}
    raw_design = (design_path/'complete.json').read_bytes()
    outputs = json.loads(raw_design)['outputs']
    caches = {r['repeat']: r['receipt_sha256'] for r in cache_audit()['records']}
    archive = collect([f'data/v3_2/{folder}/{name}' for folder in paths.values() for name in FILES])
    for i, folder in paths.items():
        payload = {name: archive[f'data/v3_2/{folder}/{name}'] for name in FILES}
        receipt = validate(i, payload, design[i], outputs, digest(raw_design), caches)
        if jobs and receipt['job'] != jobs[i]:
            raise ValueError(f'Recovery receipt differs from the successful Slurm job for catalog {i}')
        dest = BASE/folder
        dest.mkdir(exist_ok=True)
        for name, contents in payload.items():
            try:
                with (dest/name).open('xb') as stream:
                    stream.write(contents)
            except FileExistsError:
                if (dest/name).read_bytes() != contents:
                    raise ValueError(f'Refusing to overwrite different archived evidence: {dest/name}') from None
        print(f'{i:03} {receipt["job"]} {digest(payload["complete.json"])}')


if __name__ == '__main__':
    main()
