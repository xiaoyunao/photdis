#!/usr/bin/env python3
"""Recover the complete strict imaging parent and reuse its archived PDF masses."""
import argparse
from collections import defaultdict
import fitsio
from photdis.stage1 import galactic, require_compute
from photdis.v3.controls import optical_magnitudes, probe_features
from halo_common import *
from final_science_catalog import indices
from paper_v2_scan import old_grz
from sv18_common import index_join
from v3_0_native import RAW, native_payload
from v3_5_dr11_production import PRODUCTION, FREEZE
from v3_5_dr11_inventory import OUT as V35, CONFIG as V35_CONFIG
from v3_9_catalog import OUT as V39
from v3_9_references import predictor_features
from v3_9_unified import owner_rows


CUTS = 10 + 5*np.log10([10**.8,10,15,26,50,100])


def prepare():
    parts, _, inputs = indices()
    inventory = json.loads((V35/'inventory/complete.json').read_text())
    tasks = {}
    for part in parts:
        if part['region']=='south_v35':
            key = ('south', part['sweep'])
            task = tasks.setdefault(key, dict(side='south', sweep=part['sweep'], bright=True, parts=[]))
            task['bright'] = True
        else:
            key = (part['side'],part['sweep'])
            task = tasks.setdefault(key, dict(side=part['side'],sweep=part['sweep'],bright=False,parts=[]))
        task['parts'].append(part)
    by_name = {Path(s['filename']).stem:s for s in inventory['sweeps']}
    for task in tasks.values():
        if task['bright']:
            task['raw'] = by_name[task['sweep']]
    groups = [[] for _ in range(C['cache_shards'])]
    loads = np.zeros(len(groups))
    for task in sorted(tasks.values(), key=lambda t:-sum(int(p['native_rows']) for p in t['parts'])):
        lane = int(loads.argmin())
        groups[lane].append(task)
        loads[lane] += sum(int(p['native_rows']) for p in task['parts'])
    d = folder('inventory')
    save_json(d/'tasks.json', dict(groups=groups, inputs=inputs, old_inventory=sha(V35/'inventory/complete.json')))
    complete(d,inputs,tasks=len(tasks),groups=len(groups))
    print('tasks',len(tasks),'groups',len(groups),flush=True)


def normalized(t, mag, err, depth, side):
    observed = mag-2.5*np.log10(t['transmission'][:,[0,1,3]])
    x = predictor_features(mag) if side=='north' else probe_features(t,'grz')
    features = np.full((len(mag),16),np.nan,'f4')
    features[:,:x.shape[1]] = x
    return dict(TYPE=t['TYPE'],ls_id=t['ls_id'],source_id=t['source_id'],RA=t['RA'],DEC=t['DEC'],
        EBV=t['EBV'],m=mag.astype('f4'),err=err.astype('f4'),observed=observed.astype('f4'),
        dep=depth.astype('f4'),features=features,native=np.full(len(mag),side=='north','u1'),
        q=np.full((len(mag),5),np.nan,'f4'),mass=np.full((len(mag),7),np.nan),
        pdf=np.zeros(len(mag),bool),mask=t['MASKBITS'].astype('i8'))


def add_pdf(t, part, inputs):
    if not part['pdf_file']:
        return
    path = verified(ROOT/part['pdf_file'],part['pdf_sha256'])
    z = read_columns(path,('TYPE','ls_id','qrf_quantiles_kpc','qrf_mu_edges','qrf_mu_mass'))
    j = index_join(z['ls_id'],t['ls_id'])
    ix = np.flatnonzero(j>=0)
    out = j[ix]
    assert (z['TYPE'][ix]=='PSF').all() and not t['pdf'][out].any()
    for start in range(0,len(ix),2048):
        k = ix[start:start+2048]
        at = j[k]
        edges,mass = z['qrf_mu_edges'][k],z['qrf_mu_mass'][k]
        assert (np.diff(edges,axis=1)>0).all() and (mass>=0).all()
        np.testing.assert_allclose(mass.sum(axis=1),1,atol=2e-7,rtol=0)
        cdf = np.column_stack([np.sum(mass*np.clip((cut-edges[:,:-1])/np.diff(edges,axis=1),0,1),axis=1) for cut in CUTS])
        t['mass'][at] = np.diff(cdf,axis=1,prepend=0,append=1)
    t['q'][out] = z['qrf_quantiles_kpc'][ix]
    t['pdf'][out] = True
    inputs[part['pdf_file']] = part['pdf_sha256']


def old_bright(task, overrides, inputs):
    home = PRODUCTION/task['sweep']
    rec = json.loads((home/'complete.json').read_text())
    inputs[str((home/'complete.json').relative_to(ROOT))] = sha(home/'complete.json')
    status = read_columns(verified(home/'parent_status.npz',rec['outputs']['parent_status.npz']),('status',))['status']
    rows = np.flatnonzero(np.isin(status,(7,8)))
    if not len(rows):
        return None,dict(raw=len(status),missing_TYPE=int((status==2).sum()),non_PSF=int((status==3).sum()),
            strict_before_owner=0,strict_owned=0,recovered_qrf_unsupported=0)
    cfg = json.loads(V35_CONFIG.read_text())
    path = Path(cfg['sweep_root'])/task['raw']['filename']
    st = path.stat()
    assert (st.st_size,st.st_mtime_ns)==(task['raw']['size'],task['raw']['mtime_ns'])
    pieces=[]
    with fitsio.FITS(str(path)) as f:
        names=f[1].get_colnames()
        cols=[k for k in RAW if k in names]+['PSFDEPTH_'+b for b in 'GRZ']
        for start in range(0,len(rows),150000):
            at = rows[start:start+150000]
            raw = f[1].read(columns=cols,rows=at)
            native = native_payload(raw,'DECam_DR11')
            assert (native['TYPE']=='PSF').all()
            mag,err = optical_magnitudes(native)
            assert native['photometric_entry'].all() and (err<.100001).all()
            own,_ = owner_rows('south',native,overrides,frozen=True)
            take = np.flatnonzero(own)
            dep = np.column_stack([raw['PSFDEPTH_'+b] for b in 'GRZ'])
            v = {k:a[take] for k,a in native.items()}
            z = normalized(v,mag[take],err[take],dep[take],'south')
            z['was_unsupported'] = status[at[take]]==7
            pieces.append(z)
    t = join(pieces)
    if len(t['ls_id']):
        for part in task['parts']:
            if part['region']=='south_v35':add_pdf(t,part,inputs)
        np.testing.assert_array_equal(t['pdf'],~t['was_unsupported'])
    return t,dict(raw=len(status),missing_TYPE=int((status==2).sum()),non_PSF=int((status==3).sum()),
       strict_before_owner=len(rows),strict_owned=len(t['ls_id']),recovered_qrf_unsupported=int(t['was_unsupported'].sum()))


def new_part(part, overrides, inputs):
    path=verified(ROOT/part['photo_file'],part['photo_sha256'])
    p=read_columns(path,('TYPE','ls_id','source_id','RA','DEC','EBV','MASKBITS','row_state','north_primary_region',
        'mag_grz','error_grz','transmission','flux','flux_error','band_mask',
        'PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z'))
    assert (p['TYPE']=='PSF').all()
    own,_=owner_rows(part['side'],p,overrides)
    high=abs(galactic(p['RA'],p['DEC'])[1])>30
    keep=own&high&(p['row_state']==7)
    v={k:a[keep] for k,a in p.items()}
    mag,err=v['mag_grz'],v['error_grz']
    assert np.isfinite(mag).all() and np.isfinite(err).all() and (err<.100001).all()
    t=normalized(v,mag,err,np.column_stack([v['PSFDEPTH_'+b] for b in 'GRZ']),part['side'])
    t['was_unsupported']=np.zeros(len(mag),bool)
    add_pdf(t,part,inputs)
    t['was_unsupported']=~t['pdf']
    inputs[part['photo_file']]=part['photo_sha256']
    counts=dict(raw=len(own),missing_TYPE=0,non_PSF=0,strict_before_owner=int((high&(p['row_state']==7)).sum()),
        strict_owned=int(keep.sum()),recovered_qrf_unsupported=int((~t['pdf']).sum()),
        lower_precision_owned=int((own&high&(p['row_state']==6)).sum()),low_latitude_owned=int((own&~high).sum()))
    return t,counts


def join(pieces):
    return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}


def scan(shard, smoke=False):
    marker=OUT/'inventory/tasks.json'
    tasks=json.loads(marker.read_text())['groups'][shard]
    if smoke:
        all_tasks=[t for group in json.loads(marker.read_text())['groups'] for t in group]
        tasks=[next(t for t in all_tasks if t['bright']),next(t for t in all_tasks if t['side']=='north')]
    _,overrides,inputs=indices()
    inputs['inventory']=sha(marker)
    output=[];ledger=[]
    for task in tasks:
        if task['bright']:
            t,c=old_bright(task,overrides,inputs)
            if t is not None:output.append(t)
            ledger.append(dict(side='south',sweep=task['sweep'],region='south_bright',**c))
        for p in task['parts']:
            if p['region']=='south_v35':continue
            t,c=new_part(p,overrides,inputs)
            output.append(t);ledger.append(dict(side=p['side'],sweep=p['sweep'],region=p['region'],**c))
        print(shard,task['side'],task['sweep'],sum(len(p['ls_id']) for p in output),flush=True)
    t=join(output)
    assert (t['TYPE']=='PSF').all() and len(np.unique(t['ls_id']))==len(t['ls_id'])
    np.testing.assert_allclose(t['mass'][t['pdf']].sum(axis=1),1,rtol=0,atol=2e-7)
    d=folder('cache_smoke' if smoke else 'cache')/f'{shard:02}'
    d.mkdir(exist_ok=True)
    atomic_npz(d/'photometry.npz',**t)
    csv_rows(d/'admission.csv',ledger)
    complete(d,inputs,strict_owned=len(t['ls_id']),pdf=int(t['pdf'].sum()),
             recovered_qrf_unsupported=int(t['was_unsupported'].sum()),TYPE='PSF')


if __name__=='__main__':
    require_compute()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('prepare','scan'))
    parser.add_argument('--shard',type=int,default=int(os.getenv('SLURM_ARRAY_TASK_ID','0')))
    parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args()
    prepare() if a.task=='prepare' else scan(a.shard,a.smoke)
