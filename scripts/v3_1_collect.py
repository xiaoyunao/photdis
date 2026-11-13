#!/usr/bin/env python3
"""Retrieve explicit small v3.1 records over SSH, verifying declared hashes."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import shlex
import subprocess
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('files',nargs='*',default=[])
    p.add_argument('--marker',default='complete.json',help='Exact completion marker basename, including pilot-only markers')
    p.add_argument('--figures',action='store_true',help='Allow explicit hash-verified PNG/PDF artifacts, stored under figures/v3_1')
    a=p.parse_args();folder=PurePosixPath(a.folder);names=[a.marker]+a.files
    if Path(a.marker).suffix!='.json':raise ValueError('JSON completion marker required')
    if folder.is_absolute() or '..' in folder.parts:raise ValueError('Relative v3.1 folder required')
    suffixes=['.json','.csv']+(['.png','.pdf'] if a.figures else [])
    if len(names)!=len(set(names)) or any(PurePosixPath(n).name!=n or Path(n).suffix not in suffixes for n in names):
        raise ValueError('Unique explicit supported artifact basenames required')
    remote='/pscratch/sd/y/yunao/photdis_pipeline/data/v3_1/'+str(folder)
    command='tar -C '+shlex.quote(remote)+' -cf - '+' '.join(shlex.quote(n) for n in names)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','yunao@perlmutter.nersc.gov',command],capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode()[-2000:])
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        members=archive.getmembers()
        if len(members)!=len(names) or set(archive.getnames())!=set(names) or any(not m.isfile() or m.size>(20000000 if Path(m.name).suffix in ['.png','.pdf'] else 5000000) for m in members):
            raise ValueError('Unexpected member, link or oversized artifact')
        data={m.name:archive.extractfile(m).read() for m in members}
    record=json.loads(data[a.marker])
    for name,content in data.items():
        if name!=a.marker:
            if hashlib.sha256(content).hexdigest()!=record['outputs'][name]:raise ValueError('Artifact hash mismatch: '+name)
    output=Path(__file__).resolve().parents[1]/'manifests/v3_1'/str(folder);output.mkdir(parents=True,exist_ok=True)
    for name,content in data.items():
        destination=output
        if Path(name).suffix in ['.png','.pdf']:
            destination=Path(__file__).resolve().parents[1]/'figures/v3_1'/str(folder)
            destination.mkdir(parents=True,exist_ok=True)
        (destination/name).write_bytes(content)
    print(json.dumps({k:v for k,v in record.items() if k not in ['inputs','outputs','objects','chunks']},ensure_ascii=False))


if __name__=='__main__':main()
