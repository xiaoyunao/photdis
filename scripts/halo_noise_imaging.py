#!/usr/bin/env python3
"""Independent full-imaging donor control with strict photometry reapplied."""
from scipy.spatial import cKDTree
import healpy as hp
from photdis.stage1 import require_compute
from photdis.v3.data import stable_rank
from photdis.v3_3 import frozen_qrf
from halo_common import *
from halo_noise import effective_errors,models
from sv18_common import metric


def depths(dep):
    return 22.5-2.5*np.log10((2.5/np.log(10)/.1)/np.sqrt(dep))


def main():
    require_compute();d=folder('noise_imaging');inputs={};library={}
    density=read(OUT/'parent',('parent_density.npz','count'),inputs)['count']
    ex=read(OUT/'parent',('exclusions.npz','TYPE','ls_id'),inputs)['ls_id']
    for p in cache_folders():
        t=read(p,('photometry.npz','TYPE','ls_id','native','observed','m','err','dep','RA','DEC'),inputs)
        pix=hp.ang2pix(C['nside'],t['RA'],t['DEC'],nest=True,lonlat=True)
        gr=t['m'][:,0]-t['m'][:,1]
        valid=~np.isin(t['ls_id'],ex)&np.isfinite(t['dep']).all(axis=1)&(t['dep']>0).all(axis=1)
        for code in (0,1):
            for lo in (20,21,22):
                for g in range(14):
                    use=valid&(t['native']==code)&(t['observed'][:,1]>=lo)&(t['observed'][:,1]<lo+1)&(gr>=-.2+g*.15)&(gr<-.2+(g+1)*.15)
                    j=np.flatnonzero(use)
                    if not len(j):continue
                    rank=stable_rank(t['ls_id'][j],C['seed']+44);j=j[np.argsort(rank)[:300]]
                    rec=dict(TYPE=t['TYPE'][j],ls_id=t['ls_id'][j],m=t['m'][j],err=t['err'][j],dep=t['dep'][j],
                             density=density[code,pix[j]],rank=stable_rank(t['ls_id'][j],C['seed']+44))
                    key=code,lo,g
                    if key in library:rec={k:np.concatenate((library[key][k],v)) for k,v in rec.items()}
                    keep=np.argsort(rec['rank'])[:300];library[key]={k:v[keep] for k,v in rec.items()}
    pools={}
    for code in (0,1):
        for lo in (20,21,22):
            a=[v for (c,l,g),v in library.items() if (c,l)==(code,lo)]
            if a:pools[code,lo]={k:np.concatenate([v[k] for v in a]) for k in a[0]}
    t=read(OUT/'noise_fixed',('receivers.npz','TYPE','ls_id','source_id','native','m','err','dep','q','r','reference','RA','DEC'),inputs)
    assert (t['TYPE']=='PSF').all()
    rows=[];parts=[];library_summary=[]
    for code,side in enumerate(('south','north')):
        idx=np.flatnonzero(t['native']==code);a={k:v[idx] for k,v in t.items()}
        pix=hp.ang2pix(C['nside'],a['RA'],a['DEC'],nest=True,lonlat=True)
        crowd=density[code,pix]
        def features(m,dep,crowd):
            return np.column_stack(((m[:,0]-m[:,1])/.15,(m[:,1]-m[:,2])/.15,
                depths(dep)/.5,np.log10(np.maximum(crowd,1))/.5))
        x=features(a['m'],a['dep'],crowd)
        forest,offset=models(bool(code),inputs)
        for lo in (20,21,22):
            pool=pools[code,lo];assert (pool['TYPE']=='PSF').all()
            tree=cKDTree(features(pool['m'],pool['dep'],pool['density']))
            distance,nearest=tree.query(x,k=np.arange(1,33))
            library_summary.append(dict(native=side,r_lo=lo,donors=len(pool['ls_id']),receivers=len(idx),
                median_nearest_distance=np.median(distance[:,0]),missing_receiver_crowding=int((crowd==0).sum())))
            for draw in range(16):
                j=nearest[np.arange(len(idx)),RNG.integers(32,size=len(idx))]
                extra,eff=effective_errors(a['err'],pool['err'][j]);normal=RNG.normal(size=a['m'].shape)
                m=a['m']+normal*extra
                col=(abs(pool['m'][j,0]-pool['m'][j,1]-(a['m'][:,0]-a['m'][:,1]))<.15)
                col&=abs(pool['m'][j,1]-pool['m'][j,2]-(a['m'][:,1]-a['m'][:,2]))<.15
                depth=(abs(depths(pool['dep'][j])-depths(a['dep']))<1).all(axis=1)
                crowdok=(crowd>0)&(abs(np.log10(np.maximum(pool['density'][j],1)/np.maximum(crowd,1)))<1)
                donor_support=col&depth&crowdok
                observed_r=a['r']+normal[:,1]*extra[:,1]
                quality=np.isfinite(m).all(axis=1)&(eff>0).all(axis=1)&(eff<.1).all(axis=1)&(observed_r>=16)&(observed_r<18)
                pred=frozen_qrf(forest,m,eff,a['source_id'])
                q=pred['distance']*10**(offset/5)
                final=quality&donor_support&pred['supported']
                stat=metric(q[final],a['reference'][final]);baseline=metric(a['q'][final],a['reference'][final])
                rows.append(dict(native=side,r_lo=lo,draw=draw,parent=len(idx),photometric_quality=int(quality.sum()),
                    color_match=int(col.sum()),depth_match=int(depth.sum()),crowding_match=int(crowdok.sum()),
                    donor_common=int(donor_support.sum()),supported=int(pred['supported'].sum()),scored=int(final.sum()),
                    retained_fraction=final.mean(),**stat,baseline_bias=baseline['bias'],baseline_nmad=baseline['nmad']))
                parts.append(dict(TYPE=a['TYPE'],ls_id=a['ls_id'],donor_ls_id=pool['ls_id'][j],native=a['native'],
                   r_lo=np.full(len(idx),lo),draw=np.full(len(idx),draw),final=final,quality=quality,
                   donor_support=donor_support,normal=normal,effective_error=eff,
                   distance=q[:,2],reference=a['reference'],unperturbed=a['q'][:,2],
                   donor_distance=np.linalg.norm(features(pool['m'][j],pool['dep'][j],pool['density'][j])-x,axis=1)))
    atomic_npz(d/'paired_realizations.npz',**{k:np.concatenate([p[k] for p in parts]) for k in parts[0]})
    atomic_npz(d/'donor_library.npz',**{k:np.concatenate([v[k] for v in pools.values()]) for k in next(iter(pools.values()))})
    csv_rows(d/'imaging_noise_metrics.csv',rows);csv_rows(d/'donor_matching.csv',library_summary)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.1),layout='constrained')
        for code,side,color in ((0,'south',BLUE),(1,'north',ORANGE)):
            values=np.concatenate([p['distance'][p['final']]/p['reference'][p['final']]-1 for p in parts if p['native'][0]==code])
            axes[0].hist(values,bins=np.linspace(-.6,.6,65),histtype='step',color=color,density=True,label=side.capitalize())
            for base,ls in ((False,'-'),(True,':')):
                y=[]
                for lo in (20,21,22):
                    z=[r for r in rows if r['native']==side and r['r_lo']==lo]
                    y.append(np.nanmedian([r['baseline_nmad' if base else 'nmad'] for r in z]))
                axes[1].plot([20.5,21.5,22.5],y,'o',ls=ls,color=color,ms=3)
        axes[0].set(xlabel='Relative-distance residual',ylabel='Density');axes[0].legend(frameon=False)
        axes[1].set(xlabel='Imaging error-donor r (mag)',ylabel='Conditional NMAD')
        figure(d,'whole_imaging_noise_control',f,rows)
    complete(d,inputs,donor_population='full strict PSF imaging parent; stable-hash reservoirs up to300 per native/r/0.15gr cell; no spectroscopic-reference gate',
        matching='gr/rz/three depth sensitivities/local PSF-count density; 32 nearest, conditional color<.15mag/depth<1mag/crowding<1dex common support',
        crowding='local Nside512 strict PSF count is a density proxy, not a deblend completeness measurement',
        noise='fixed intrinsic bright population/distance; actual incremental variance and passed error=max(original,donor); strict r16..18,error<.1 and forest support reapplied',
        baseline='unperturbed frozen predictions on the identical final surviving IDs each draw')


if __name__=='__main__':main()
