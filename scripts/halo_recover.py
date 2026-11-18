#!/usr/bin/env python3
"""Recover entire bright sweeps absent from the QRF-supported part manifest."""
from photdis.stage1 import require_compute
from halo_common import *
from halo_cache import old_bright, join
from final_science_catalog import indices
from v3_5_dr11_production import PRODUCTION, FREEZE
from v3_5_dr11_inventory import OUT as V35


if __name__=='__main__':
    require_compute()
    inv_path=OUT/'inventory/tasks.json'
    primary=json.loads(inv_path.read_text())
    admitted={t['sweep'] for g in primary['groups'] for t in g if t['bright']}
    frozen=json.loads(FREEZE.read_text())
    names={Path(n).stem for g in frozen['groups'] for n in g}
    inventory={Path(s['filename']).stem:s for s in json.loads((V35/'inventory/complete.json').read_text())['sweeps']}
    _,overrides,inputs=indices()
    inputs.update(freeze=sha(FREEZE),primary_inventory=sha(inv_path))
    rows=[];extra=[]
    for name in sorted(names-admitted):
        path=PRODUCTION/name/'complete.json'
        rec=json.loads(path.read_text())
        n=rec['exclusive_counts']
        strict=int(n['outside_qrf_support'])+int(n['model_supported'])
        assert int(n['model_supported'])==0,name
        rows.append(dict(sweep=name,strict=strict,unsupported=int(n['outside_qrf_support']),**n))
        inputs[str(path.relative_to(ROOT))]=sha(path)
        if strict:
            task=dict(side='south',sweep=name,bright=True,parts=[],raw=inventory[name])
            t,_=old_bright(task,overrides,inputs)
            if t is not None:extra.append(t)
    d=folder('cache_extra')/'00';d.mkdir(exist_ok=True)
    count=sum(len(t['ls_id']) for t in extra)
    if extra:
        t=join(extra)
        assert (t['TYPE']=='PSF').all() and not t['pdf'].any()
        atomic_npz(d/'photometry.npz',**t)
    csv_rows(d/'absent_sweeps.csv',rows)
    complete(d,inputs,missing_sweeps=len(rows),strict_owned=count,
             strict_before_owner=sum(r['strict'] for r in rows),TYPE='PSF')
    print('absent sweeps',len(rows),'recovered strict',count,flush=True)
