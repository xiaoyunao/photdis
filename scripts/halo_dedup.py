#!/usr/bin/env python3
"""Verify the full imaging inventory and resolve newly recovered seam duplicates."""
import hashlib
import healpy as hp
from photdis.stage1 import require_compute
from halo_common import *
from v3_9_seam import match, BOUNDARY


def main():
    require_compute()
    expected=[OUT/'cache'/f'{i:02}'/'complete.json' for i in range(C['cache_shards'])]
    assert all(p.exists() for p in expected) and (OUT/'cache_extra/00/complete.json').exists()
    inputs={};keys=[];edge=[];stats=[]
    hist=np.zeros((2,92,101),'i8')
    g_edges=np.linspace(4,27,93);e_edges=np.linspace(0,.101,102)
    parent=np.zeros((2,hp.nside2npix(C['nside'])),'i8')
    for d in cache_folders():
        t=read(d,('photometry.npz','TYPE','ls_id','source_id','RA','DEC','native','m','err','observed','dep','pdf','was_unsupported'),inputs)
        assert (t['TYPE']=='PSF').all()
        keys.append(t['ls_id'])
        take=abs(t['DEC']-BOUNDARY)<.01
        edge.append({k:v[take] for k,v in t.items() if k in ('TYPE','ls_id','source_id','RA','DEC','native')})
        pix=hp.ang2pix(C['nside'],t['RA'],t['DEC'],lonlat=True,nest=True)
        for code in (0,1):
            use=t['native']==code
            parent[code]+=np.bincount(pix[use],minlength=parent.shape[1])
            hist[code]+=np.histogram2d(t['observed'][use,0],t['err'][use,0],(g_edges,e_edges))[0].astype('i8')
        stats.append(dict(cache=str(d.relative_to(OUT)),strict=len(t['ls_id']),pdf=int(t['pdf'].sum()),
            qrf_unsupported=int(t['was_unsupported'].sum())))
    ids=np.sort(np.concatenate(keys));assert not (np.diff(ids)==0).any()
    digest=hashlib.sha256(ids.tobytes()).hexdigest()
    e={k:np.concatenate([x[k] for x in edge]) for k in edge[0]}
    n={k:v[e['native']==1] for k,v in e.items() if k!='native'}
    s={k:v[e['native']==0] for k,v in e.items() if k!='native'}
    i,j,_,_,_,_=match(n,s)
    keep_north=(n['DEC'][i]+s['DEC'][j])/2>=BOUNDARY
    excluded=np.where(keep_north,s['ls_id'][j],n['ls_id'][i])
    at=np.flatnonzero(np.isin(e['ls_id'],excluded))
    if len(at):
        pix=hp.ang2pix(C['nside'],e['RA'][at],e['DEC'][at],lonlat=True,nest=True)
        np.add.at(parent,(e['native'][at],pix),-1)
    d=folder('parent')
    atomic_npz(d/'exclusions.npz',TYPE=np.full(len(excluded),'PSF'),ls_id=np.sort(excluded))
    atomic_npz(d/'parent_density.npz',count=parent)
    centers=(g_edges[:-1]+g_edges[1:])/2
    errors=(e_edges[:-1]+e_edges[1:])/2
    med=np.zeros((2,len(centers)))
    for code,h in enumerate(hist):
        c=np.cumsum(h,axis=1);count=c[:,-1]
        good=count>=30
        at=np.argmax(c>=count[:,None]/2,axis=1)
        med[code]=np.interp(centers,centers[good],errors[at[good]])
    atomic_npz(d/'error_curve.npz',g_centers=centers,sigma_g=med,histogram=hist,g_edges=g_edges,error_edges=e_edges)
    csv_rows(d/'imaging_parent.csv',stats)
    csv_rows(d/'error_curve.csv',[dict(native=code,g=g,sigma_g=s,n=int(h.sum())) for code in (0,1)
       for g,s,h in zip(centers,med[code],hist[code])])
    complete(d,inputs,strict_before_physical_dedup=len(ids),duplicate_physical_pairs=len(excluded),
        strict_after_physical_dedup=len(ids)-len(excluded),sorted_native_id_sha256=digest,
        error_curve_scope='median error binned at 0.001mag resolution in full strict imaging parent, >=30 per g bin; interpolate missing error-curve bins',
        dedup_scope='all native IDs unique; newly recovered opposite-native owners matched within 0.01deg of boundary using original 1.5arcsec spherical rule')


if __name__=='__main__':main()
