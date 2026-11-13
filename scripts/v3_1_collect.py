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
    a=p.parse_args();folder=PurePosixPath(a.folder);names=['complete.json']+a.files
    if folder.is_absolute() or '..' in folder.parts:raise ValueError('Relative v3.1 folder required')
    if len(names)!=len(set(names)) or any(PurePosixPath(n).name!=n or Path(n).suffix not in ['.json','.csv'] for n in names):
        raise ValueError('Unique explicit small JSON/CSV basenames required')
    remote='/pscratch/sd/y/yunao/photdis_pipeline/data/v3_1/'+str(folder)
    command='tar -C '+shlex.quote(remote)+' -cf - '+' '.join(shlex.quote(n) for n in names)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','yunao@perlmutter.nersc.gov',command],capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode()[-2000:])
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        members=archive.getmembers()
        if len(members)!=len(names) or set(archive.getnames())!=set(names) or any(not m.isfile() or m.size>5000000 for m in members):
            raise ValueError('Unexpected member, link or oversized artifact')
        data={m.name:archive.extractfile(m).read() for m in members}
    record=json.loads(data['complete.json'])
    for name,content in data.items():
        if name!='complete.json':
            if hashlib.sha256(content).hexdigest()!=record['outputs'][name]:raise ValueError('Artifact hash mismatch: '+name)
    output=Path(__file__).resolve().parents[1]/'manifests/v3_1'/str(folder);output.mkdir(parents=True,exist_ok=True)
    for name,content in data.items():(output/name).write_bytes(content)
    print(json.dumps({k:v for k,v in record.items() if k not in ['inputs','outputs','objects','chunks']},ensure_ascii=False))


if __name__=='__main__':main()
