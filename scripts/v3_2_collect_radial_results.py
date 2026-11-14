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
RECOVERY_RETRIES = {'61278909': {37, 46}}
PILOT_CATALOGS = {0, 128, 256}
SSH = ('ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
       '-o', 'ServerAliveInterval=8', '-o', 'ServerAliveCountMax=2', HOST)


def digest(raw):
    return sha256(raw).hexdigest()


def completed_tasks(report, array_id):
    result = {}
    for line in report.splitlines():
        task, raw, state, code = line.strip().split('|')
        array, sep, catalog = task.partition('_')
        if array != array_id or not sep or not catalog.isdecimal():
            continue
        if state == 'COMPLETED' and code == '0:0':
            result[int(catalog)] = raw
    return result


def discover(array_id, tasks=()):
    job_spec = ','.join(f'{array_id}_{task}' for task in tasks) if tasks else array_id
    command = f'timeout 15 sacct -n -X -P -j {job_spec} --format=JobID,JobIDRaw,State,ExitCode'
    status = subprocess.run([*SSH, command], check=True, capture_output=True, timeout=45, text=True)
    return completed_tasks(status.stdout, array_id)


def completed_jobs(job_ids):
    command = f'timeout 15 sacct -n -X -P -j {",".join(job_ids)} --format=JobID,State,ExitCode'
    status = subprocess.run([*SSH, command], check=True, capture_output=True, timeout=45, text=True)
    return {job for job, state, code in (line.strip().split('|') for line in status.stdout.splitlines())
            if job in job_ids and state == 'COMPLETED' and code == '0:0'}


def collect(names, *, max_file_bytes=100_000):
    # Names are generated from the frozen catalog IDs and fixed filenames.
    command = f'cd {REMOTE} && timeout -k 2 15 tar -czf - ' + ' '.join(names)
    raw = subprocess.run([*SSH, command], check=True, capture_output=True, timeout=45).stdout
    with tarfile.open(fileobj=BytesIO(raw), mode='r:gz') as archive:
        members = archive.getmembers()
        if (len(members) != len(names) or {m.name for m in members} != set(names)
                or any(not m.isfile() or m.size > max_file_bytes for m in members)):
            raise ValueError('Unexpected compact results archive')
        return {name: archive.extractfile(name).read() for name in names}


def archive_verified(dest, payload):
    dest.mkdir(parents=True, exist_ok=True)
    for name, contents in payload.items():
        try:
            with (dest/name).open('xb') as stream:
                stream.write(contents)
        except FileExistsError:
            if (dest/name).read_bytes() != contents:
                raise ValueError(f'Refusing to overwrite different archived evidence: {dest/name}') from None


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


def is_archived(catalog):
    folder = BASE/f'radial_recovery/catalog_{catalog:03}'
    return all((folder/name).is_file() for name in FILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('catalogs', nargs='*', type=int,
                        metavar='CATALOG', help='Catalog IDs with completed Slurm jobs')
    parser.add_argument('--completed', action='store_true',
                        help='Find unarchived successful tasks in the production Slurm array')
    args = parser.parse_args()
    if bool(args.catalogs) == args.completed or any(not 0 <= i < 288 for i in args.catalogs):
        parser.error('Choose --completed or explicit catalog IDs in 0–287')
    tasks = sorted(set(args.catalogs) - PILOT_CATALOGS)
    if len(set(args.catalogs)) > 32:
        parser.error('Collect at most 32 compact results in each transfer')
    jobs = discover(RECOVERY_ARRAY, tasks) if args.completed or tasks else {}
    for array_id, retry_catalogs in RECOVERY_RETRIES.items():
        requested = ({i for i in retry_catalogs if not is_archived(i)}
                     if args.completed else retry_catalogs.intersection(tasks))
        if requested:
            for catalog, job in discover(array_id, sorted(requested)).items():
                if catalog in jobs and jobs[catalog] != job:
                    raise ValueError(f'Multiple successful recovery jobs for catalog {catalog}')
                jobs[catalog] = job
    if missing := set(tasks) - jobs.keys():
        parser.error('Slurm has no successful completion for catalogs ' + ', '.join(map(str, sorted(missing))))
    catalogs = sorted(set(args.catalogs)) if not args.completed else [
        i for i in sorted(jobs) if not is_archived(i)
    ][:32]
    if not catalogs:
        print('No new successful production catalogs')
        return
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
        if i in jobs and receipt['job'] != jobs[i]:
            raise ValueError(f'Recovery receipt differs from the successful Slurm job for catalog {i}')
        archive_verified(BASE/folder, payload)
        print(f'{i:03} {receipt["job"]} {digest(payload["complete.json"])}')


if __name__ == '__main__':
    main()
