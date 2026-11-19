#!/usr/bin/env python3
"""Unique native identities and exact 61-node all-sky CMD counts."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor,as_completed
from multiprocessing import get_context
import healpy as hp
from photdis.stage1 import require_compute
from sky6_common import *
from stream_local import accept,curves
from halo_filtering import channel_weights

NPIX=hp.nside2npix(C['nside']);MU=np.round(np.arange(14,20.001,.1),1)


def assert_unique_partitioned(ids,label):
    # Equal identities necessarily have equal brick-bit buckets. Exact local
    # sorts therefore prove global uniqueness without a billion-row serial sort.
    bucket=((ids>>20)&63).astype('u1')
    def check(k):
        part=ids[bucket==k];assert len(np.unique(part))==len(part),(label,k)
        return len(part)
    with ThreadPoolExecutor(8) as pool:total=sum(pool.map(check,range(64)))
    assert total==len(ids);print('UNIQUE_NATIVE_PASS',label,total,flush=True)


def gaia_aliases(ids,gaia):
    bucket=(gaia>>56).astype('u1');keys=np.unique(bucket[gaia>0])
    def aliases(k):
        j=np.flatnonzero((gaia>0)&(bucket==k));jj=j[np.lexsort((ids[j],gaia[j]))]
        return ids[jj[np.r_[False,np.diff(gaia[jj])==0]]]
    with ThreadPoolExecutor(8) as pool:parts=list(pool.map(aliases,keys))
    return np.sort(np.concatenate(parts)) if parts else np.array([],dtype=ids.dtype)


def identities():
    require_compute();d=folder('identities');paths=sorted((OUT/'cache').glob('*/complete.json'))
    # Exact audit of all owned native PSFs, not only accepted CMD members.
    ids=np.concatenate([read_columns(p.parent/'owned_ids.npz',('ls_id',))['ls_id'] for p in paths]);nraw=len(ids)
    assert_unique_partitioned(ids,'all raw owned PSFs')
    del ids
    parts=[read_columns(p.parent/'photometry.npz',('TYPE','ls_id','source_id')) for p in paths]
    assert all((z['TYPE']=='PSF').all() for z in parts)
    ids=np.concatenate([z['ls_id'] for z in parts]);gaia=np.concatenate([z['source_id'] for z in parts]);del parts
    assert_unique_partitioned(ids,'valid gr');positive=int((gaia>0).sum());exclude=gaia_aliases(ids,gaia)
    npz(d/'alias_exclusions.npz',TYPE=np.full(len(exclude),'PSF'),ls_id=exclude)
    save_json(d/'identity_counts.json',dict(native_PSF_unique=nraw,valid_gr_native_unique=len(ids),positive_gaia=positive,positive_gaia_aliases=len(exclude),valid_gr_after_aliases=len(ids)-len(exclude),rule='actual PSF/native first; minimum native ID per repeated positive G3 identity; unlinked sources retain native IDs',execution='64 deterministic native identity buckets and native Gaia HEALPix-prefix buckets; exact sorts in8 shared-memory workers'))
    complete(d,{},raw_native_unique_verified=True)


def count_one(p):
    p=Path(p);d=folder('nodes')/p.name;d.mkdir(exist_ok=True);r=json.loads((p/'complete.json').read_text());marker=d/'complete.json'
    if marker.exists():
        z=json.loads(marker.read_text());assert z['source']==sha(p/'complete.json') and z['config']==CH;return str(d)
    t=read_columns(verified(p/'photometry.npz',r['outputs']['photometry.npz']),('TYPE','ls_id','m','extinction','RA','DEC','lat','EBV','old_domain','strict','legacy','shallow'))
    with np.load(OUT/'identities/alias_exclusions.npz') as a:keep=~np.isin(t['ls_id'],a['ls_id'])
    t={k:v[keep] for k,v in t.items()};assert (t['TYPE']=='PSF').all();native=int(p.name.startswith('north_'));inputs={}
    temp=read(HALO/'native',('template.npz','south','north','phase','north_support'),inputs)
    phase=temp['phase'].copy()
    if native:phase[~temp['north_support']]=-99
    cc=curves(temp[('south','north')[native]],phase,'MS_TO')
    with np.load(ROOT/C['inputs']['error_curve']) as a:ec=(a['g_centers'],a['sigma_g'])
    pix=hp.ang2pix(C['nside'],t['RA'],t['DEC'],lonlat=True,nest=True);up,inv=np.unique(pix,return_inverse=True);counts=np.zeros((2,len(MU),len(up)),'i4');parent=np.zeros((2,len(up)),'i4');union=np.zeros((2,len(pix)),bool);uncorrected=np.zeros((3,len(up)));weights=channel_weights(MU,C['rgb']['overview_BGR_kpc'])
    for start in range(0,len(pix),20000):
        stop=min(start+20000,len(pix));accepted=accept(t['m'][start:stop],MU,cc,ec,'MS_TO')
        observed=t['m'][start:stop].copy();observed[:,:2]+=t['extinction'][start:stop]
        raw=accept(observed,MU,cc,ec,'MS_TO')@weights.T
        for j in range(3):uncorrected[j]+=np.bincount(inv[start:stop],weights=raw[:,j],minlength=len(up))
        for view,eligible in enumerate((np.ones(stop-start,bool),t['old_domain'][start:stop])):
            a=accepted&eligible[:,None];union[view,start:stop]=a.any(axis=1)
            for j in range(len(MU)):counts[view,j]+=np.bincount(inv[start:stop][a[:,j]],minlength=len(up)).astype('i4')
    for view,flag in enumerate((np.ones(len(pix),bool),t['old_domain'])):parent[view]=np.bincount(inv[flag],minlength=len(up)).astype('i4')
    rows=[]
    domain=dict(all=np.ones(len(pix),bool),low_latitude=abs(t['lat'])<=30,high_EBV=t['EBV']>=.1,shallower_23=t['shallow'],old_domain=t['old_domain'])
    for name,flag in domain.items():rows.append(dict(domain=name,valid_gr=int(flag.sum()),strict_companion=int((flag&t['strict']).sum()),legacy_entrance=int((flag&t['legacy']).sum()),CMD_union=int((flag&union[0]).sum())))
    strict=np.bincount(inv[t['strict']],minlength=len(up))
    npz(d/'counts.npz',pixels=up,counts=counts,parents=parent,strict=strict,uncorrected=uncorrected,native=np.array(native),union=union.sum(axis=1),totals=counts.sum(axis=2));csv_rows(d/'domains.csv',rows)
    save_json(marker,dict(config=CH,source=sha(p/'complete.json'),error_curve=sha(ROOT/C['inputs']['error_curve']),code=sha(Path(__file__)),outputs={q.name:sha(q) for q in d.iterdir() if q.is_file() and q.name!='complete.json'}))
    return str(d)


def count():
    require_compute();d=folder('counts');paths=sorted((OUT/'cache').glob('*/complete.json'));counts=np.zeros((2,2,len(MU),NPIX),'i8');parents=np.zeros((2,2,NPIX),'i8');geometry=np.zeros((2,2,NPIX),'i8');uncorrected=np.zeros((2,3,NPIX));strict=np.zeros((2,NPIX),'i8');rows=[];domains=[]
    with ProcessPoolExecutor(C['workers'],mp_context=get_context('spawn')) as pool:
        fs=[pool.submit(count_one,str(p.parent)) for p in paths]
        for i,f in enumerate(as_completed(fs)):
            p=Path(f.result());side=p.name.startswith('north_')
            with np.load(p/'counts.npz') as a:
                pix=a['pixels'];counts[int(side)][:,:,pix]+=a['counts'];parents[int(side)][:,pix]+=a['parents'];uncorrected[int(side)][:,pix]+=a['uncorrected'];strict[int(side),pix]+=a['strict'];rows.extend(dict(part=p.name,view=v,union=int(a['union'][v]),acceptances=int(a['totals'][v].sum())) for v in (0,1))
            with np.load(OUT/'cache'/p.name/'geometry.npz') as a:geometry[int(side)][:,a['pixels']]+=a['counts']
            domains.extend(dict(part=p.name,side='north' if side else 'south',**r) for r in csv.DictReader((p/'domains.csv').open()))
            if i%50==0:print('COUNTED',i+1,len(paths),flush=True)
    assert counts.max()<np.iinfo('i4').max
    npz(d/'node_counts.npz',counts=counts.astype('i4'),parents=parents,geometry=geometry,strict=strict,uncorrected=uncorrected,mu=MU)
    csv_rows(d/'part_counts.csv',rows);csv_rows(d/'domain_counts.csv',domains)
    ident=json.loads((OUT/'identities/identity_counts.json').read_text());assert geometry[:,0].sum()==ident['native_PSF_unique'];assert parents[:,0].sum()==ident['valid_gr_after_aliases']
    complete(d,{},parent_count_conservation=True,node_acceptances=int(counts.sum()),scope='views all display and same display within old geographic/dust/depth domain; not a new distance posterior')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['identities','count']);v=a.parse_args();globals()[v.stage]()
