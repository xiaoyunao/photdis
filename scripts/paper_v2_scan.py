#!/usr/bin/env python3
"""Read all owned DR11 parents and full frozen PDFs on allocated CPUs."""
from collections import defaultdict
import json
import os
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np
from scipy.special import ndtr

from photdis.gaia import NPIX, SOURCE_ID_HEALPIX_FACTOR, derive_gaia_labels, join_gaia
from photdis.stage1 import galactic, legacy_ids, require_compute
from v2_1_near import ROOT, atomic_npz, save_json, sha
from v3_3_sed import verified
from v3_4_near_reference_support import separation
from v3_4_near_validate import GAIA, tile_inventory
from v3_5_dr11_inventory import CONFIG as SOUTH_CONFIG, OUT as SOUTH
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_6_vac_distances import read_columns
from v3_9_references import OLDER, join, lookup, matched, raw_catalogs, sorted_key
from v3_9_unified import owner_rows
from v3_10_redrock import DEST as REDROCK
from final_science_catalog import REGIONS, indices
from paper_v1_scan import locate

OUT=ROOT/'data/paper_v2'
SHARDS=48
R=(-np.inf,18.,20.,21.,22.,23.,np.inf)
GR=(-np.inf,.0,.35,.5,.65,.8,1.05,1.4,np.inf)
RZ=(-np.inf,.15,.35,.65,np.inf)
CLASSES=('unknown','ordinary_dwarf','metal_poor_dwarf','dwarf_unknown_feh','intermediate_gravity',
         'giant','turnoff_proxy','other_stellar','nonstellar','class_conflict','white_dwarf_candidate')
REFS=('gaia','specdis','spdist')
SHELLS=(0.,1.,2.,5.,10.,20.,np.inf)
NSIDE=128
NP=hp.nside2npix(NSIDE)
CROSS=(('g','gaia'),('s','specdis'),('t','spdist'))
AXES=4,2,len(R),4,3,5
GK=np.array([-.7745966692414834,0,.7745966692414834])
GW=np.array([5,8,5])/18


def bin_axis(values,edges):
    return np.where(np.isfinite(values),np.searchsorted(edges,values,'right')-1,len(edges)-1)


def roster(inputs):
    cats, provenance=raw_catalogs()
    inputs.update(provenance)
    base=OLDER.parent
    p=base/'sources.json'
    u=base/'source_evidence.json'
    a=json.loads(p.read_text())['source_sha256']
    b=json.loads(u.read_text())
    assert a==b['input_sha256']['sources.npz']
    r=read_columns(verified(base/'sources.npz',a),('source_id','ra','dec','rvs_feh','rvs_feh_err',
                'rvs_logg','rvs_logg_err','rvs_teff','rvs_teff_err'))
    label=read_columns(verified(base/'source_evidence.npz',b['output_sha256']),
        ('source_id','class_label','class_identity_conflict','target_identity_conflict'))
    np.testing.assert_array_equal(r['source_id'],label['source_id'])
    r.update(reference_ra=r['ra'],reference_dec=r['dec'],label=label['class_label'])
    j=np.flatnonzero((r['source_id']>0)&(r['label']>=0)&
                     ~label['class_identity_conflict']&~label['target_identity_conflict'])
    r['_gaia']=sorted_key(r['source_id'],j)
    inputs['raw_class']=sha(u)
    rr=REDROCK/'complete.json'
    red=read_columns(verified(rr.with_name('qualified.npz'),json.loads(rr.read_text())['outputs']['qualified.npz']),
                    ('ls_id','TYPE','qualified'))
    assert (red['TYPE']=='PSF').all()
    inputs['redrock']=sha(rr)
    rr=np.unique(red['ls_id'][red['qualified']!=''])
    # Spectroscopic white-dwarf and turnoff categories are never inferred from a distance prediction.
    return cats,r,rr


def training(inputs):
    from v3_9_catalog import OUT as V39
    s=ROOT/'data/v3_0/controls/complete.json'
    x=json.loads(s.read_text())
    t=read_columns(verified(s.with_name('training.npz'),x['outputs']['training.npz']),('identity','TYPE','rows'))
    a=read_columns(verified(s.parent.parent/'references.npz',x['inputs']['data/v3_0/references.npz']),
                   ('ls_id','source_id','identity','TYPE'))
    i=t['rows']
    assert np.all(t['TYPE']=='PSF') and np.all(a['TYPE'][i]=='PSF')
    np.testing.assert_array_equal(a['identity'][i],t['identity'])
    n=V39/'north/native_model/complete.json'
    y=json.loads(n.read_text())
    b=read_columns(verified(n.with_name('roles.npz'),y['outputs']['roles.npz']),('ls_id','source_id','TYPE','role'))
    assert np.all(b['TYPE']=='PSF')
    inputs.update(south_training=sha(s),north_training=sha(n))
    return [(name,np.unique(data['ls_id'][j]),np.unique(data['source_id'][j]))
            for name,data,j in (('south_fit',a,i),*((f'north_{v}',b,np.flatnonzero(b['role']==v))
                for v in ('gaia_fit','specdis_fit','calibration','test','class_fit')))]


def old_grz(t,sweep,cfg,inventory):
    entry=inventory[sweep]
    path=Path(cfg['sweep_root'])/entry['filename']
    stat=path.stat()
    assert (stat.st_size,stat.st_mtime_ns)==(entry['size'],entry['mtime_ns'])
    rows=np.concatenate([x['parent_row'] for x in t])
    order=np.argsort(rows)
    cols=['RELEASE','BRICKID','OBJID','TYPE','MASKBITS',
          *(f'{a}_{b}' for a in ('FLUX','FLUX_IVAR','MW_TRANSMISSION','PSFDEPTH') for b in 'GRZ')]
    with fitsio.FITS(path) as f:
        raw=f[1].read(columns=cols,rows=rows[order])[np.argsort(order)]
    np.testing.assert_array_equal(legacy_ids(raw),np.concatenate([x['ls_id'] for x in t]))
    assert np.all(np.char.strip(raw['TYPE'].astype(str))=='PSF')
    flux,iv,tr,depth=[np.column_stack([raw[f'{a}_{b}'] for b in 'GRZ'])
                      for a in ('FLUX','FLUX_IVAR','MW_TRANSMISSION','PSFDEPTH')]
    with np.errstate(invalid='ignore',divide='ignore'):
        mag=22.5-2.5*np.log10(flux.astype(float)/tr)
        err=2.5/np.log(10)/flux/np.sqrt(iv)
    end=np.cumsum([len(x['ls_id']) for x in t])[:-1]
    return zip(np.split(mag,end),np.split(err,end),np.split(depth,end),np.split(raw['MASKBITS'],end))


def photometry(parts,overrides,inputs):
    s=SOUTH/'inventory/complete.json'
    inv={Path(x['filename']).stem:x for x in json.loads(s.read_text())['sweeps']}
    cfg=json.loads(SOUTH_CONFIG.read_text())
    inputs['old_inventory']=sha(s)
    blocks=defaultdict(list)
    for i,p in parts:
        blocks[p['region'],p['sweep']].append((i,p))
    out,handles=[],[]
    for (domain,sweep),items in blocks.items():
        frozen=domain=='south_v35'
        region=REGIONS.index(domain)
        base=('ls_id','source_id','TYPE','RA','DEC','EBV','observed_r','dereddened_g_r')
        extra=('parent_row','primary') if frozen else ('mag_grz','error_grz','MASKBITS','row_state',
            'north_primary_region','PSFDEPTH_G','PSFDEPTH_R','PSFDEPTH_Z')
        native=[read_columns(verified(ROOT/p['photo_file'],p['photo_sha256']),base+extra) for i,p in items]
        recovered=old_grz(native,sweep,cfg,inv) if frozen else (
            (t['mag_grz'],t['error_grz'],np.column_stack([t[f'PSFDEPTH_{b}'] for b in 'GRZ']),t['MASKBITS'])
            for t in native)
        for (i,p),t,(m,e,d,b) in zip(items,native,recovered):
            keep,_=owner_rows(p['side'],t,overrides,frozen=frozen)
            count=np.flatnonzero(keep)
            assert len(count)==int(p['owned_rows'])
            if frozen:
                state=np.full(len(keep),3,'u1')
                strict=np.full(len(keep),7,'u1')
            else:
                z=read_columns(verified(ROOT/p['state_file'],p['state_sha256']),('ls_id','TYPE','state'))
                np.testing.assert_array_equal(z['ls_id'],t['ls_id'])
                np.testing.assert_array_equal(z['TYPE'],t['TYPE'])
                state,strict=z['state'],t['row_state']
            assert (state[count]==3).sum()==int(p['owned_pdf_rows'])
            a={k:t[k][count] for k in ('ls_id','TYPE','source_id','RA','DEC','EBV','observed_r')}
            a.update(region=np.full(len(count),region,'u1'),m=m[count].astype('f4'),err=e[count].astype('f4'),
                     dep=d[count].astype('f4'),mask=b[count],quality=strict[count],state=state[count],
                     is_primary=(t['primary'][count] if frozen else np.zeros(len(count),bool)))
            handles.append((p,count,np.flatnonzero(state==3),len(a['ls_id'])))
            out.append(a)
    return join(out),handles


def spectra(t,cats,rvs,rr):
    p=np.zeros(len(t['ls_id']),'u1')
    values=np.full((len(p),6),np.nan,'f4')
    j=matched(t,rvs)
    good=j>=0
    label=rvs['label'][j[good]]
    for i,k in enumerate(('rvs_feh','rvs_feh_err','rvs_logg','rvs_logg_err','rvs_teff','rvs_teff_err')):
        y=rvs[k][j[good]]
        values[good,i]=np.where(y>-9000,y,np.nan)
    p[good]=np.choose(label,[3,5,7])
    f,df,g,dg,temp,dt=values.T
    qualified_f=np.isfinite(f)&(df>0)&(df<=.3)
    qualified_g=np.isfinite(g)&(dg>0)&(dg<=.3)
    p[(p==3)&qualified_f]=np.where(f[(p==3)&qualified_f]<-1.5,2,1)
    p[(p==7)&qualified_g&(g>=3.2)&(g<4.2)]=4
    gr=t['m'][:,0]-t['m'][:,1]
    p[(p==4)&(gr<.5)&(g>=3.7)&(temp>=5500)&(dt>0)&(dt<=200)]=6
    non=np.isin(t['ls_id'],rr)
    p[non]=np.where(p[non]==0,8,9)
    for key in ('specdis','spdist'):
        c=cats[key]
        ids=c['_gaia'][0]
        recorded=(np.searchsorted(ids,t['source_id'],'right')>np.searchsorted(ids,t['source_id'],'left'))&(t['source_id']>0)
        j=matched(t,c)
        pos=j>=0
        d=np.full(len(p),np.nan,'f4'); sigma=d.copy()
        d[pos]=c['distance_kpc'][j[pos]]; sigma[pos]=c['sigma_mu'][j[pos]]
        valid=pos&np.isfinite(d)&(d>0)
        qual=valid.copy()
        qual[valid]=c['historical_quality'][j[valid]]&c['usable_uncertainty'][j[valid]]
        t.update({f'{key}_d':d,f'{key}_sig':sigma,
                  f'{key}_stage':np.where(qual,3,np.where(valid,2,recorded)).astype('u1')})
    t.update(pop=p,physical=values)


def attach_gaia(t,inputs):
    src=t['source_id']
    before=tile_inventory(src[(src>0)&(src<NPIX*SOURCE_ID_HEALPIX_FACTOR)])
    valid=(src>0)&(src<NPIX*SOURCE_ID_HEALPIX_FACTOR)
    g=join_gaia(np.where(valid,src,0),GAIA)
    f=derive_gaia_labels(g)
    state,ref,lo,hi=distances(t,valid,f,g,json.loads(CUTS.read_text()))
    close=(state>=3)
    t.update(gaia_state=state,gaia_d=ref.astype('f4'),gaia_lo=lo.astype('f4'),gaia_hi=hi.astype('f4'),
             gaia_stage=np.where(state==6,3,np.where(close&f['finite_parallax'],2,g['gaia_match'])).astype('u1'),
             gmissing=g['gaia_file_missing'],p_corr=f['p_corr'],p_err=g['parallax_error'])
    if before!=tile_inventory(src[valid]):
        raise ValueError('Gaia archive tiles changed during census')
    inputs['gaia_tile_count']=len(before)


def radial_prob(edges,mass,cut):
    return np.sum(mass*np.clip((cut-edges[:,:-1])/np.diff(edges,axis=1),0,1),axis=1)


def signed(edges,mass,y,e,states):
    lo=np.where(states==4,-np.inf,5)[:,None]
    hi=np.where(states==4,5.,10.)[:,None]
    mid=(edges[:,:-1]+edges[:,1:])*.5
    dw=.5*np.diff(edges,axis=1)
    u=np.zeros(len(y))
    z=np.zeros(len(y))
    for node,wt in zip(GK,GW):
        p=10**((10-mid-dw*node)/5)/e[:,None]
        a=ndtr(lo-p)
        u+=wt*np.sum(mass*(ndtr(y[:,None]/e[:,None]-p)-a),axis=1)
        z+=wt*np.sum(mass*(ndtr(hi-p)-a),axis=1)
    return np.divide(u,z,out=np.full(len(y),np.nan),where=z>1e-13)


def posterior(t,handles,sky):
    p=0
    for part,owner,indices,n in handles:
        p+=n
        if not len(indices):
            continue
        path=verified(ROOT/part['pdf_file'],part['pdf_sha256'])
        j,k=locate(indices,owner)
        if not len(j):
            continue
        ii=p-n+j
        a=read_columns(path,('ls_id','TYPE','qrf_quantiles_kpc','qrf_mu_edges','qrf_mu_mass',
                              *(() if part['region']=='south_v35' else ('photo_row',))))
        if 'photo_row' in a: np.testing.assert_array_equal(a['photo_row'],indices)
        np.testing.assert_array_equal(a['ls_id'][k],t['ls_id'][ii])
        assert (a['TYPE'][k]=='PSF').all()
        e,w=a['qrf_mu_edges'][k],a['qrf_mu_mass'][k]
        np.testing.assert_allclose(w.sum(axis=1),1,rtol=0,atol=1e-10)
        assert np.all(np.diff(e,axis=1)>0)
        q=a['qrf_quantiles_kpc'][k]
        t['q'][ii]=q
        # Binned local maxima, with directly reported resolution and threshold.
        den=w/np.diff(e,axis=1)
        smooth=(den[:,:-2]+2*den[:,1:-1]+den[:,2:])*.25
        peak=(smooth[:,1:-1]>smooth[:,:-2])&(smooth[:,1:-1]>=smooth[:,2:])
        peak &= smooth[:,1:-1]>=.2*smooth.max(axis=1)[:,None]
        t['peaks'][ii]=peak.sum(axis=1)
        weak=np.flatnonzero(np.isin(t['gaia_state'][ii],(4,5)))
        for v in np.array_split(weak,max(1,(len(weak)+511)//512)):
            krow=ii[v]
            if len(v):
                t['pit'][krow]=signed(e[v],w[v],t['p_corr'][krow],t['p_err'][krow],t['gaia_state'][krow])
        p1=radial_prob(e,w,10)
        p10=radial_prob(e,w,15)
        pix=hp.ang2pix(NSIDE,t['RA'][ii],t['DEC'][ii],lonlat=True,nest=True)
        layer=np.searchsorted([1,10],q[:,2],side='right')
        # Native region matrices preserve exact positions used for the random-area denominator.
        r=int(t['region'][ii[0]])
        for k,f in enumerate((p1,p10-p1,1-p10)):
            np.add.at(sky[r,0,k],pix,layer==k)
            np.add.at(sky[r,1,k],pix,f)
    assert p==len(t['ls_id']) and np.array_equal(np.isfinite(t['q'][:,2]),t['state']==3)


def scan(shard):
    parts,overrides,inputs=indices()
    seg=np.array_split(np.arange(len(parts)),SHARDS)[shard]
    t,handles=photometry([(i,parts[i]) for i in seg],overrides,inputs)
    n=len(t['ls_id'])
    cats,rvs,red=roster(inputs)
    spectra(t,cats,rvs,red)
    del cats,rvs
    attach_gaia(t,inputs)
    train=training(inputs)
    flags=np.zeros(n,'u1')
    for i,(name,ls,ids) in enumerate(train):
        flags|=((np.isin(t['ls_id'],ls[ls>0])|np.isin(t['source_id'],ids[ids>0])).astype('u1')<<i)
    # Any distance fit or zero point removes the row from both native holdouts.
    role=(np.floor(t['RA']/10).astype(int)+3*np.floor((t['DEC']+90)/10).astype(int))%5
    hold=(role==4)&((flags&15)==0)&(t['quality']==7)
    t.update(training=flags,heldout=hold,q=np.full((n,5),np.nan,'f4'),pit=np.full(n,np.nan,'f4'),
             peaks=np.zeros(n,'u1'))
    l,lat=galactic(t['RA'],t['DEC'])
    t['low_lat']=np.abs(lat)<=30
    sky=np.zeros((4,2,3,NP))
    posterior(t,handles,sky)
    # Retain one compact row for every source with a usable external association or
    # qualified spectral class. All other rows remain represented by exact counts.
    common=(*[t[k+'_stage'] for k in REFS],)
    key=np.ravel_multi_index((t['region'],t['low_lat'].astype(int),bin_axis(t['observed_r'],R),
                               t['state'],np.zeros(n,int),np.zeros(n,int)),AXES)
    counts=np.zeros((3,AXES[0]*AXES[1]*AXES[2]*AXES[3],5),'i8')
    index=key//(AXES[-1]*AXES[-2])
    for i,state in enumerate(common):
        z=(index*5+state).astype('i8')
        counts[i]=np.bincount(z,minlength=counts.shape[1]*5).reshape(-1,5)
    retain=np.any(np.stack(common)>0,axis=0)|(t['pop']>0)
    keys=('ls_id','source_id','TYPE','region','RA','DEC','EBV','observed_r','m','err','dep','mask','quality',
          'state','is_primary','gaia_state','gaia_lo','gaia_hi','gmissing','p_corr','p_err','pit','q','peaks',
          'pop','physical','training','heldout','low_lat',*(k+suffix for k in REFS
               for suffix in (('_stage','_d') if k=='gaia' else ('_stage','_d','_sig'))))
    dest=OUT/'scan'
    dest.mkdir(exist_ok=True,parents=True)
    a=dest/f'{shard:02}_summary.npz'
    b=dest/f'{shard:02}_matched.npz'
    atomic_npz(a,counts=counts[:3],sky=sky)
    atomic_npz(b,**{k:t[k][retain] for k in keys})
    save_json(dest/f'{shard:02}.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,parts=len(seg),
        parent=n,pdf=int((t['state']==3).sum()),matched=int(retain.sum()),outputs={p.name:sha(p) for p in (a,b)}))
    print(shard,n,int((t['state']==3).sum()),int(retain.sum()),flush=True)


if __name__=='__main__':
    require_compute()
    scan(int(os.environ['SLURM_ARRAY_TASK_ID']))
