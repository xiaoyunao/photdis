#!/usr/bin/env python3
"""Recover same-star native colors and bind the northern template transfer."""
from collections import defaultdict
from scipy.spatial import ConvexHull, Delaunay
from photdis.stage1 import require_compute, tile_bounds
from photdis.v3.data import stable_rank
from halo_common import *
from sv18_common import index_join
from final_science_catalog import indices
from paper_v2_scan import old_grz
from v3_5_dr11_inventory import OUT as V35, CONFIG as V35_CONFIG
from v3_9_delivery import DEST as RELEASE
from v3_9_seam import within_strip
from v3_13_clusters import SYSTEMS


def design(gr,rz):
    return np.column_stack((np.ones(len(gr)),gr,rz,gr*gr,gr*rz,rz*rz))


def extract(parts,ids,side,inputs):
    n=len(ids)
    result=dict(TYPE=np.full(n,'PSF'),ls_id=ids,m=np.full((n,3),np.nan),err=np.full((n,3),np.nan),
                dep=np.full((n,3),np.nan),RA=np.full(n,np.nan),DEC=np.full(n,np.nan),r=np.full(n,np.nan))
    seen=np.zeros(n,bool)
    inv={Path(s['filename']).stem:s for s in json.loads((V35/'inventory/complete.json').read_text())['sweeps']}
    cfg=json.loads(V35_CONFIG.read_text())
    groups=defaultdict(list)
    for p in parts:
        if p['side']==side and within_strip(tile_bounds(p['sweep']+'.fits')):
            groups[p['region'],p['sweep']].append(p)
    for (region,sweep),items in groups.items():
        old=region=='south_v35'
        payload=[]
        for p in items:
            cols=('parent_row','observed_r') if old else ('mag_grz','error_grz','r','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z')
            path=verified(ROOT/p['photo_file'],p['photo_sha256'])
            t=read_columns(path,('TYPE','ls_id','RA','DEC',*cols))
            at=index_join(t['ls_id'],ids)
            if not (at>=0).any():continue
            payload.append((p,t,at))
        if old:
            recovered=old_grz([t for _,t,_ in payload],sweep,cfg,inv) if payload else []
        else:
            recovered=((t['mag_grz'],t['error_grz'],np.column_stack([t['PSFDEPTH_'+b] for b in 'GRZ']),None)
                       for _,t,_ in payload)
        for (p,t,at),(mag,err,dep,_) in zip(payload,recovered):
            use=at>=0;rows=at[use]
            assert not seen[rows].any() and (t['TYPE'][use]=='PSF').all()
            seen[rows]=True
            for key in ('RA','DEC'):result[key][rows]=t[key][use]
            result['r'][rows]=t['observed_r' if old else 'r'][use]
            result['m'][rows]=mag[use];result['err'][rows]=err[use];result['dep'][rows]=dep[use]
            inputs[p['photo_file']]=p['photo_sha256']
        print(side,sweep,int(seen.sum()),flush=True)
    assert seen.all(),(side,int((~seen).sum()))
    return result


def template():
    p=ROOT/'configs/halo_templates/mist10_a12_z0001_decam.txt'
    names=next(l for l in p.read_text().splitlines() if l.startswith('# EEP ')).lstrip('#').split()
    a=np.loadtxt(p)
    t={name:a[:,i] for i,name in enumerate(names)}
    assert np.allclose(t['isochrone_age_yr'],12e9) and '1.00000E-04' in p.read_text()
    return t,p


def main():
    require_compute()
    d=folder('native');inputs={}
    parts,_,base=indices();inputs.update(base)
    h=RELEASE/'seam/complete.json';r=json.loads(h.read_text())
    pairs=read_columns(verified(h.with_name('paired_sources.npz'),r['outputs']['paired_sources.npz']),
                      ('TYPE','north_ls_id','south_ls_id','north_source_id','south_source_id'))
    assert (pairs['TYPE']=='PSF').all()
    inputs['pairs']=sha(h)
    data={side:extract(parts,pairs[side+'_ls_id'],side,inputs) for side in ('north','south')}
    for side,t in data.items():
        p=ROOT/f'data/v3_10/seam/{side}.npz'
        z=read_columns(p,('TYPE','ls_id','state','qrf_quantiles_kpc','distance_shell_mass'))
        at=index_join(t['ls_id'],z['ls_id']);assert (at>=0).all() and (z['TYPE'][at]=='PSF').all()
        t.update(state=z['state'][at],q=z['qrf_quantiles_kpc'][at],shell=z['distance_shell_mass'][at])
        atomic_npz(d/(side+'_pairs.npz'),**t)
        inputs['final_'+side]=sha(p)
    n,s=data['north'],data['south']
    gr=s['m'][:,0]-s['m'][:,1];rz=s['m'][:,1]-s['m'][:,2]
    selected=np.isfinite(n['m']).all(axis=1)&np.isfinite(s['m']).all(axis=1)
    selected&=(n['err']<.03).all(axis=1)&(s['err']<.03).all(axis=1)
    selected&=(s['r']>=16)&(s['r']<19)&(gr>=-.2)&(gr<1.2)
    for center,_ in SYSTEMS.values():
        dra=(s['RA']-center[0]+180)%360-180
        selected&=np.hypot(dra*np.cos(np.deg2rad(center[1])),s['DEC']-center[1])>=.8
    rank=stable_rank(s['ls_id'],C['seed'])
    train=selected&(rank%5!=4);test=selected&(rank%5==4)
    j=np.flatnonzero(train);j=j[np.argsort(rank[j],kind='stable')[:C['native_transfer']['max_rows']]]
    x=design(gr,rz);y=n['m']-s['m']
    keep=np.ones(len(j),bool)
    for _ in range(3):
        beta=np.linalg.lstsq(x[j[keep]],y[j[keep]],rcond=None)[0]
        res=y[j]-x[j]@beta
        sig=1.4826*np.median(abs(res[keep]-np.median(res[keep],axis=0)),axis=0)
        keep=(abs(res)<np.maximum(5*sig,.03)).all(axis=1)
    beta=np.linalg.lstsq(x[j[keep]],y[j[keep]],rcond=None)[0]
    res=y[test]-x[test]@beta
    sig=1.4826*np.median(abs(res-np.median(res,axis=0)),axis=0)
    physical,path=template()
    mag=np.column_stack([physical['DECam_'+b] for b in 'grz'])
    col=np.column_stack((mag[:,0]-mag[:,1],mag[:,1]-mag[:,2]))
    cloud=np.column_stack((gr[j[keep]],rz[j[keep]]))
    hull=cloud[ConvexHull(cloud).vertices]
    expanded=np.concatenate([hull+offset for offset in ((-.05,-.05),(-.05,.05),(.05,-.05),(.05,.05))])
    support=Delaunay(expanded).find_simplex(col)>=0
    transformed=mag+design(*col.T)@beta
    atomic_npz(d/'template.npz',south=mag,north=transformed,north_support=support,phase=physical['phase'],
               initial_mass=physical['initial_mass'],EEP=physical['EEP'],beta=beta,scatter=sig,hull=expanded)
    csv_rows(d/'native_transfer.csv',[dict(band=b,train=int(keep.sum()),heldout=int(test.sum()),
        median_residual_mag=np.median(res[:,i]),nmad_residual_mag=sig[i],
        basis='1 gr rz gr2 gr*rz rz2',**{f'b{k}':beta[k,i] for k in range(6)}) for i,b in enumerate('grz')])
    atomic_npz(d/'transfer_roles.npz',TYPE=s['TYPE'],ls_id=s['ls_id'],training=np.isin(np.arange(len(gr)),j[keep]),validation=test)
    seam=[]
    for lo,hi in ((14,18),(18,20),(20,21),(21,22),(22,23)):
        for a,b in ((-.2,.3),(.3,.6),(.6,1),(1,1.6)):
            for e0,e1 in ((0,.03),(.03,.06),(.06,.1),(.1,np.inf)):
                domain=(s['r']>=lo)&(s['r']<hi)&(gr>=a)&(gr<b)
                error=np.maximum(n['err'].max(axis=1),s['err'].max(axis=1))
                domain&=(error>=e0)&(error<e1)
                both=domain&(n['state']==3)&(s['state']==3)
                delta=5*np.log10(n['q'][both,2]/s['q'][both,2])
                seam.append(dict(r_lo=lo,r_hi=hi,gr_lo=a,gr_hi=b,error_lo=e0,error_hi=e1,parent=int(domain.sum()),
                    both=int(both.sum()),north_supported=int((domain&(n['state']==3)).sum()),
                    south_supported=int((domain&(s['state']==3)).sum()),
                    median_delta_mu=np.median(delta) if len(delta) else None))
    csv_rows(d/'seam_error_support.csv',seam)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,3,figsize=(7.1,2.9),layout='constrained')
        rows=[]
        for i,(ax,b) in enumerate(zip(axes,'grz')):
            h,xe,ye=np.histogram2d(gr[test],res[:,i],bins=(np.linspace(-.2,1.2,45),np.linspace(-.1,.1,61)))
            from matplotlib.colors import LogNorm
            image=ax.pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap='viridis',norm=LogNorm(1,max(2,h.max())),rasterized=True)
            ax.axhline(0,color='.5',ls=':',lw=.7)
            ax.set(xlabel=r'$(g-r)_{0,S}$',ylabel=b+' transfer residual (mag)')
            f.colorbar(image,ax=ax,fraction=.035,label='Stars / bin')
            rows.append(dict(band=b,parent=int(test.sum()),inside_axes=int(h.sum()),outside_axes=int(test.sum()-h.sum())))
        figure(d,'native_template_transfer',f,rows)
    complete(d,inputs,theory_source_sha256=sha(path),transfer_scope='DECam MIST colors transformed to observed BASS/MzLS from same-star overlap; finite color support and independent hash validation',
             native_template_support=int(support.sum()),template_rows=len(mag))


if __name__=='__main__':main()
