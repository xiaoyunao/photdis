#!/usr/bin/env python3
"""Extract reproducible age/metallicity brackets from the official DECam grid."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEY = 'cluster_depth_20261120'
CONFIG = ROOT/'configs'/f'{KEY}.json'
URL = 'https://mist.science/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_DECam.txz'


def column_data(raw, iron, ages):
    lines = raw.splitlines()
    cols = next(x.decode().lstrip('# ').split() for x in lines if x.startswith(b'# EEP '))
    age_column = next(c for c in cols if 'isochrone_age_yr' in c)
    fields = ['EEP',age_column,'[Fe/H]_init','phase','DECam_g','DECam_r','DECam_z']
    t = np.loadtxt(BytesIO(raw),usecols=[cols.index(c) for c in fields])
    np.testing.assert_allclose(t[:,2],iron,atol=1e-9,rtol=0)
    available = np.unique(t[:,1])
    needed = {}
    for age in ages:
        k = np.searchsorted(available,age)
        assert 0<k<len(available)
        needed[age] = available[k-1:k+1]
    return {age:{x:t[(t[:,1]==x)&(t[:,3]==0)][:,[0,4,5,6]] for x in keys}
            for age,keys in needed.items()}


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('archive',type=Path)
    archive=args.parse_args().archive
    cfg=json.loads(CONFIG.read_text())
    ages=[np.log10(x['age_gyr']*1e9) for x in cfg['systems'].values()]
    destination=ROOT/'configs'/KEY
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive,'r:xz') as grid:
        names={}
        for name in grid.getnames():
            match=re.search(r'feh_([mp])(\d+\.\d+)_afe_p0\.0.*\.iso\.cmd$',name)
            if match:
                val=float(match[2])*(-1 if match[1]=='m' else 1)
                names[val]=name
        irons=np.array(sorted(names))
        by_system={}
        for name,param in cfg['systems'].items():
            j=np.searchsorted(irons,param['feh'])
            by_system[name]=irons[j-1:j+1]
        data={}
        for iron in sorted(set(np.concatenate(list(by_system.values())))):
            with grid.extractfile(names[iron]) as raw_file:
                data[iron]=column_data(raw_file.read(),iron,ages)
    with archive.open('rb') as f:
        digest=sha256()
        for block in iter(lambda:f.read(1<<20),b''):
            digest.update(block)
    receipt=dict(url=URL,archive_sha256=digest.hexdigest(),rotation=.4,alpha=0,version='1.2',systems={})
    for (name,param),age in zip(cfg['systems'].items(),ages):
        lo,hi=by_system[name]
        weight_iron=(param['feh']-lo)/(hi-lo)
        corners=[]
        for iron,factor in [(lo,1-weight_iron),(hi,weight_iron)]:
            a,b=sorted(data[iron][age])
            frac=(age-a)/(b-a)
            corners.extend([(iron,a,factor*(1-frac),data[iron][age][a]),
                            (iron,b,factor*frac,data[iron][age][b])])
        ids=corners[0][3][:,0]
        for _,_,_,block in corners[1:]:
            ids=np.intersect1d(ids,block[:,0])
        assert np.all(np.diff(ids)==1) and len(ids)>100
        values=np.zeros((len(ids),3))
        for _,_,factor,block in corners:
            ix=np.searchsorted(block[:,0],ids)
            np.testing.assert_array_equal(block[ix,0],ids)
            values+=factor*block[ix,1:]
        path=destination/(name+'_mist_ms.csv')
        np.savetxt(path,np.column_stack((ids,values)),fmt=['%d','%.9f','%.9f','%.9f'],
                   delimiter=',',header='EEP,g,r,z',comments='')
        receipt['systems'][name]=dict(FeH=param['feh'],age_gyr=param['age_gyr'],n=len(ids),
             brackets=[dict(filename=names[fe],feh=fe,log_age=la,weight=w)
                       for fe,la,w,_ in corners],sha256=sha256(path.read_bytes()).hexdigest())
        print(name,len(ids),min(values[:,1]),max(values[:,1]),flush=True)
    (destination/'mist_sources.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
