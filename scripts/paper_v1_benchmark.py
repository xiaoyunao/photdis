#!/usr/bin/env python3
"""Replay paired comparisons, score complete PDFs, and measure observable overlap."""
import csv
import json
import os

import numpy as np
from scipy.special import ndtr

from photdis.gaia import derive_gaia_labels, join_gaia
from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, save_json, sha
from v3_3_sed import verified
from v3_4_near_validate import GAIA, tile_inventory
from v3_5_dr11_production import CUTS
from v3_6_gaia_census import distances
from v3_9_references import matched, raw_catalogs
from final_catalogue_benchmark import load_parent, masks, published
from final_catalogue_population import phase
from final_science_catalog import REGIONS
from release_readiness_compare import available, stats
from paper_v1_scan import AXES, DETAIL, DIMS, EDGES, OUT, SHARDS, concatenate_reference, locate

REFS = ('SpecDis','SPDist')
CLASSES = ('dwarf_feh_ge_minus1p5','dwarf_feh_lt_minus1p5','intermediate_gravity','giant')
NMIN = 20
MAX_RATIO = 10
PIT_EDGES = np.linspace(0,1,11)
GAUSS_X, GAUSS_W = np.polynomial.legendre.leggauss(8)


def read_scan(inputs):
    count = np.zeros((3,int(np.prod(DIMS))),'i8')
    arrays,links = [],[]
    parts = 0
    for i in range(SHARDS):
        p = OUT/'scan'/f'{i:02}.json'
        r = json.loads(p.read_text())
        inputs[f'scan_{i:02}'] = sha(p)
        parts += r['partitions']
        for key,target,fields in (('counts',None,('counts',)),('links',links,('ls_id','cell','tier')),
                                   ('reference',arrays,DETAIL)):
            name = f'{i:02}_{key}.npz'
            with np.load(verified(p.with_name(name),r['outputs'][name])) as data:
                if key=='counts':
                    count += data['counts']
                elif data.files:
                    target.append({k:data[k] for k in fields})
    np.testing.assert_array_equal(count.sum(axis=1),(43347612,1115077,1165946))
    assert parts==28917
    joined = [concatenate_reference(blocks,blocks[0]) for blocks in (arrays,links)]
    for t in joined:
        assert len(np.unique(t['ls_id']))==len(t['ls_id'])
    assert len(joined[0]['ls_id'])==48714 and len(joined[1]['ls_id'])==2281023
    return count,*joined


def full_curves(t,q,detail,inputs):
    order = np.argsort(detail['ls_id'])
    j,k = locate(detail['ls_id'][order],t['ls_id'])
    t, detail, q = ({x:v[i] for x,v in a.items()} for a,i in ((t,j),(detail,order[k]),(q,j)))
    assert (detail['TYPE']=='PSF').all()
    np.testing.assert_array_equal(detail['source_id'],t['source_id'])
    np.testing.assert_array_equal(np.array(REGIONS)[detail['region']],t['region'])
    np.testing.assert_allclose(t['g0']-t['r0'],detail['g0']-detail['r0'],atol=2e-5,rtol=0)
    np.testing.assert_array_equal(t['r'],detail['r'])
    np.testing.assert_allclose(detail['qrf_mu_mass'].sum(axis=1),1,atol=1e-10,rtol=0)
    assert (np.diff(detail['qrf_mu_edges'],axis=1)>0).all()
    cats, archive_inputs = raw_catalogs()
    inputs.update({f'raw_{k}':v for k,v in archive_inputs.items()})
    for ref in REFS:
        name = ref.lower()
        raw = cats[name]
        ix = matched(t,raw)
        ok = np.isfinite(t[name+'_kpc'])
        assert np.all(ix[ok]>=0)
        uncertainty = np.full(len(ix),np.nan)
        uncertainty[ok] = raw['sigma_mu'][ix[ok]]
        np.testing.assert_allclose(raw['distance_kpc'][ix[ok]],t[name+'_kpc'][ok],rtol=1e-6)
        t[name+'_sigma_mu'] = uncertainty
    return t,q,detail


def weak(t,q,dest):
    rows = []
    m = masks(t)
    for ref in REFS:
        name = ref.lower()
        d = t[name+'_kpc']
        mu = np.full_like(d,np.nan)
        sig = t[name+'_sigma_mu']
        pos = np.isfinite(d)
        mu[pos] = 10+5*np.log10(d[pos])
        pair = available(q['Photdis'])&available(q['BJ_photogeometric'])&pos
        for sample,flag in (('weak_union',m['weak_gaia']),('state3',t['gaia_state']==3),
                            ('state4_5',np.isin(t['gaia_state'],(4,5)))):
            for group in ('dwarfs','ordinary_dwarfs','poor_dwarfs','intermediate_gravity'):
                for label,cut in (('2_to_5',(d>=2)&(d<5)),('buffer_2p2_4p7',(d>=2.2)&(d<4.7)),
                                  ('one_reported_sigma_inside',(mu-sig>=10+5*np.log10(2))&
                                                                 (mu+sig<10+5*np.log10(5))),
                                  ('all_reference',pos)):
                    j = pair&m[group]&flag&cut
                    color = (t['g0']-t['r0'])[j]
                    for method in ('Photdis','BJ_photogeometric'):
                        rows.append(dict(reference=ref,gaia=sample,population=group,reference_slice=label,
                            method=method,r_med=np.median(t['r'][j]) if j.any() else None,
                            gr_med=np.median(color) if j.any() else None,
                            n_north=int((j&(t['region']=='north')).sum()),**stats(q[method][j],d[j])))
    csv_rows(dest/'paired_sensitivity.csv',rows)
    sets = [r['n'] for r in rows if r['gaia']=='weak_union' and r['population']=='dwarfs' and
              r['method']=='Photdis' and r['reference_slice']=='2_to_5']
    assert sets==[9355,8927]


def forward_nodes(edges,mass):
    half = .5*np.diff(edges,axis=1)
    mid = (edges[:,:-1]+edges[:,1:])*.5
    values = mid[:,:,None]+half[:,:,None]*GAUSS_X
    weights = mass[:,:,None]*(GAUSS_W*.5)
    return values.reshape(len(mid),-1),weights.reshape(len(mid),-1)


def distribution(t,q,detail,group,dest,inputs):
    rows,pits = [],{}
    edges,mass = (detail[k] for k in ('qrf_mu_edges','qrf_mu_mass'))
    for ref in REFS:
        r = ref.lower()
        d,s = t[r+'_kpc'],t[r+'_sigma_mu']
        good = np.isfinite(d)&np.isfinite(s)&(s>0)
        data = np.full((len(d),7),np.nan)
        for j in np.array_split(np.flatnonzero(good),max(1,(good.sum()+511)//512)):
            if not len(j):
                continue
            e,wt = edges[j],mass[j]
            mu = 10+5*np.log10(d[j])
            qm = 10+5*np.log10(q['Photdis'][j])
            q95 = 10+5*np.log10(t['qrf_quantiles_kpc'][j][:,[0,4]])
            center = np.sum(wt*np.clip((mu[:,None]-e[:,:-1])/np.diff(e,axis=1),0,1),axis=1)
            x,w = forward_nodes(e,wt)
            pred = np.sum(w*ndtr((mu[:,None]-x)/s[j,None]),axis=1)
            def contains(limits):
                lo,hi = limits[:,0],limits[:,-1]
                return ((lo<=mu)&(mu<=hi),ndtr((hi-mu)/s[j])-ndtr((lo-mu)/s[j]))
            h68,b68 = contains(qm)
            h95,b95 = contains(q95)
            data[j] = np.column_stack((center,pred,h68,b68,h95,b95,qm[:,2]-qm[:,0]))
        for region in ('all','north','south_low'):
            obs=(t['region']==region) if region!='all' else np.ones(len(d),bool)
            for jpop,pop in enumerate(('all',*CLASSES)):
                label=(group==pop) if jpop else np.ones(len(d),bool)
                for lo,hi in ((18,19),(19,20),(18,20)):
                    for slab,(lower,upper) in (('all',(0,np.inf)),('2_5',(2,5))):
                        j=good&obs&label&(t['r']>=lo)&(t['r']<hi)&(d>=lower)&(d<upper)
                        u=data[j]
                        key=f'{ref}_{region}_{jpop}_{lo}_{hi}_{slab}'
                        pits[key+'_center']=np.histogram(u[:,0],PIT_EDGES)[0]
                        pits[key+'_forward']=np.histogram(u[:,1],PIT_EDGES)[0]
                        means = u[:,2:].mean(axis=0) if len(u) else [None]*5
                        rows.append(dict(key=key,reference=ref,region=region,population=pop,
                            r_lo=lo,r_hi=hi,reference_slab=slab,n=len(u),c68_center=means[0],
                            c68_error_overlap=means[1],c95_center=means[2],c95_error_overlap=means[3],
                            mean_width68_mu=means[4],sigma_mu_median=np.median(s[j]) if len(u) else None))
    csv_rows(dest/'cdf_diagnostics.csv',rows)
    atomic_npz(dest/'spectral_pit.npz',edges=PIT_EDGES,**pits)
    signed(t,detail,group,dest,inputs)


def signed(t,detail,group,dest,inputs):
    use=np.isin(t['gaia_state'],(4,5))&t['unique_gaia']
    j=np.flatnonzero(use)
    tile=tile_inventory(t['source_id'][j])
    gaia=join_gaia(t['source_id'][j],GAIA)
    flag=derive_gaia_labels(gaia)
    state,_,_,_=distances({k:t[k][j] for k in ('RA','DEC')},np.ones(len(j),bool),
                         flag,gaia,json.loads(CUTS.read_text()))
    if tile!=tile_inventory(t['source_id'][j]):
        raise ValueError('Gaia archive changed during replay')
    inputs['gaia_tiles']=tile
    # Require the exact type of eligible measurement in this file. Preserve any mismatches.
    matched_state=state==t['gaia_state'][j]
    csv_rows(dest/'signed_gaia_admission.csv',[dict(original_state=k,replay_state=z,
        n=int(((t['gaia_state'][j]==k)&(state==z)).sum())) for k in (4,5) for z in np.unique(state)])
    index=j[matched_state]
    y=flag['p_corr'][matched_state]
    err=gaia['parallax_error'][matched_state]
    pit=np.empty(len(index))
    point=np.empty(len(index))
    for block in np.array_split(np.arange(len(index)),max(1,(len(index)+511)//512)):
        if not len(block):
            continue
        j=index[block]
        u,w=forward_nodes(detail['qrf_mu_edges'][j],detail['qrf_mu_mass'][j])
        p=10**((10-u)/5)
        v,e=y[block,None],err[block,None]
        low=np.where(t['gaia_state'][j]==4,-np.inf,5)[:,None]
        high=np.where(t['gaia_state'][j]==4,5.,10.)[:,None]
        def conditional(p,w):
            a=ndtr(low-p/e)
            b=ndtr(high-p/e)
            numerator=np.sum(w*(ndtr((v-p)/e)-a),axis=1)
            denominator=np.sum(w*(b-a),axis=1)
            return np.divide(numerator,denominator,out=np.full_like(numerator,np.nan),where=denominator>1e-13)
        pit[block]=conditional(p,w)
        point[block]=conditional(1/t['qrf_quantiles_kpc'][j,2,None],np.ones((len(j),1)))
    rows=[]
    hist={}
    for state in (4,5):
        for region in ('all','north','south_low'):
            for pop in ('all',*CLASSES):
                keep=(t['gaia_state'][index]==state)
                if region!='all': keep &= t['region'][index]==region
                if pop!='all': keep &= group[index]==pop
                for method,u in (('qrf_pdf',pit),('qrf_median',point)):
                    j=keep&np.isfinite(u)&(u>=-1e-7)&(u<=1+1e-7)
                    key=f'{state}_{region}_{pop}_{method}'
                    hist[key]=np.histogram(np.clip(u[j],0,1),PIT_EDGES)[0]
                    rows.append(dict(key=key,state=state,region=region,population=pop,method=method,
                        n=int(keep.sum()),valid=int(j.sum()),negative=int((j&(y<0)).sum()),
                        c68=((u[j]>=.16)&(u[j]<=.84)).mean() if j.any() else None,
                        median_pit=np.median(u[j]) if j.any() else None))
    csv_rows(dest/'signed_gaia.csv',rows)
    atomic_npz(dest/'signed_pit.npz',edges=PIT_EDGES,**hist)


def light_summary(ratio,w):
    def wmed(x):
        i=np.argsort(x)
        c=np.cumsum(w[i])
        return x[i[np.searchsorted(c,.5*c[-1])]]
    resid=ratio-1
    bias=wmed(resid)
    tail=np.average(np.abs(5*np.log10(ratio))>1,weights=w)
    return dict(bias=bias,nmad=1.4826*wmed(np.abs(resid-bias)),tail=tail)


def write_support(dest,links,reference_counts):
    from astropy.io import fits
    o=np.argsort(links['ls_id'])
    ids=links['ls_id'][o]
    cell=links['cell'][o]
    assert np.all(np.diff(ids)>0)
    columns=[fits.Column(name='LS_ID',format='K',array=ids),
             fits.Column(name='OBS_CELL',format='J',array=cell.astype('i4')),
             fits.Column(name='SCIENCE_TIER',format='B',array=links['tier'][o])]
    for name,count in zip(REFS,reference_counts):
        n=count[cell]
        columns.extend((fits.Column(name=f'N_{name.upper()}',format='J',array=n.astype('i4')),
                        fits.Column(name=f'{name.upper()}_SUPPORT',format='L',array=n>=NMIN)))
    p=fits.BinTableHDU.from_columns(columns,name='OBS_SUPPORT')
    p.header['MINDW']=(NMIN,'Paired F15 dwarfs in the observable cell')
    p.header['BASELINE']='87069408'
    out=dest/'core_observable_support.fits'
    tmp=out.with_suffix('.tmp')
    fits.HDUList([fits.PrimaryHDU(),p]).writeto(tmp,overwrite=True,checksum=True)
    with fits.open(tmp,checksum=True) as f:
        np.testing.assert_array_equal(f[1].data['LS_ID'],ids)
        assert f[1].verify_checksum()==f[1].verify_datasum()==1
    tmp.replace(out)


def coverage(t,q,detail,count,links,pop,dest):
    cell=detail['cell']
    m=masks(t)
    pair=available(q['Photdis'])&available(q['BJ_photogeometric'])
    index=np.flatnonzero((count[0]>0))
    regions,rbin,gr,rz,err,dust,sky=np.unravel_index(index,DIMS)
    head=dict()
    reference_counts=[]
    all_rows=[]
    weights=[]
    compact={}
    measured=[]
    for ref in REFS:
        d=t[ref.lower()+'_kpc']
        good=pair&m['weak_gaia_dwarfs']&(d>=2)&(d<5)
        head[ref]=good
        n=np.bincount(cell[good],minlength=count.shape[1])
        reference_counts.append(n)
        h=(count[0]>0)&(n>=NMIN)
        qual=np.bincount(cell[np.isfinite(d)],minlength=count.shape[1])
        label={p:np.bincount(cell[np.isfinite(d)&(pop==p)],minlength=count.shape[1]) for p in CLASSES}
        label['unknown']=np.bincount(cell[np.isfinite(d)&(pop=='unclassified')],minlength=count.shape[1])
        # Reference distances are diagnostic columns; observational cells never include distance.
        metrics={}
        for s in np.unique(cell[good]):
            cohort=good&(cell==s)
            metrics[s]=stats(q['Photdis'][cohort],d[cohort])
        for i,c in enumerate(index):
            r=metrics.get(c,{k:None for k in ('bias','nmad','tail','coverage68')})
            all_rows.append(dict(reference=ref,cell_id=c,region=REGIONS[regions[i]],r_bin=rbin[i],
                 gr_bin=gr[i],rz_bin=rz[i],error_bin=err[i],ebv_bin=dust[i],sky_nested=sky[i],
                 full_pdf=count[0,c],near_core=count[1,c],mid_core=count[2,c],paired_dwarf=n[c],
                 distance_reference=qual[c],**{f'n_{p}':x[c] for p,x in label.items()},
                 status=('common' if h[c] else 'sparse' if n[c] else 'none'),
                 **{k:r[k] for k in ('bias','nmad','tail','coverage68')}))
        for i,region in enumerate(REGIONS):
            select=regions==i
            for lower,upper in ((-np.inf,np.inf),(-np.inf,18),(18,20),(20,21),(21,22),(22,23)):
                r_edges=np.array(EDGES[0])
                rb=np.array([k for k,(a,b) in enumerate(zip(r_edges[:-1],r_edges[1:]))
                             if a>=lower and b<=upper])
                j=index[select&np.isin(rbin,rb)]
                for k,s in enumerate(('full_pdfs','near_core','mid_core')):
                    measured.append(dict(reference=ref,region=region,r_min=lower,r_max=upper,view=s,
                          n=int(count[k,j].sum()),common_n=int(count[k,j[h[j]]].sum()),
                          sparse_n=int(count[k,j[(n[j]>0)&~h[j]]].sum()),
                          unsupported_n=int(count[k,j[n[j]==0]].sum()),reference_n=int(n[j].sum())))
        for i,region in enumerate(REGIONS):
            for k,view in enumerate(('full_pdfs','near_core','mid_core')):
                use=good&h[cell]&(detail['region']==i)&(count[k,cell]>0)
                if not use.any():
                    continue
                c=cell[use]
                w=count[k,c]/n[c]
                w=w/w.mean()
                clipped=np.clip(w,1/MAX_RATIO,MAX_RATIO)
                for method in ('Photdis','BJ_photogeometric'):
                    v=q[method][use,1]/d[use]
                    for weight_kind,x in (('uniform',np.ones(len(v))),('clipped_ratio',clipped)):
                        weights.append(dict(reference=ref,region=region,view=view,method=method,
                           weighting=weight_kind,n=len(v),n_cells=len(np.unique(c)),
                           ess=x.sum()**2/np.sum(x*x),min_weight=x.min(),max_weight=x.max(),
                           clipped_n=int((w!=clipped).sum()) if weight_kind=='clipped_ratio' else 0,
                           **light_summary(v,x)))
        compact[ref+'_good']=good
    assert [int(head[s].sum()) for s in REFS]==[9355,8927]
    csv_rows(dest/'observable_cells.csv',all_rows)
    csv_rows(dest/'coverage.csv',measured)
    csv_rows(dest/'reweight.csv',weights)
    atomic_npz(dest/'observables.npz',counts=count,reference_counts=np.array(reference_counts))
    write_support(dest,links,reference_counts)
    stringify=lambda x:[float(j) if np.isfinite(j) else ('-inf' if j<0 else 'inf') for j in x]
    save_json(dest/'support_schema.json',dict(axis_order=['region',*AXES],dimensions=DIMS,
        bounds={a:stringify(b) for a,b in zip(AXES,EDGES)},sky='nested equatorial HEALPix nside 2',
        regions=REGIONS,unknown_bin='final bin in each continuous axis',min_pair=NMIN,
        score_support='counts of weak-geometry, reference 2–5 kpc F15 paired dwarfs',
        caution='observable overlap; the target tier does not enter the F15 admission',
        weight='source-to-reference ratio within common selected cells, rescaled mean 1 then clipped 0.1–10'))
    return compact


def cohort_composition(t,q,pop,dest):
    rows=[]
    d50=q['Photdis'][:,1]
    core=(t['state']==4)&(d50>=2)&(d50<5)
    nonstellar=np.isin(pop,('qualified_nonstellar','class_conflict'))
    assessed=t['redrock_quality_assessed']
    for region in ('north','south_low','all'):
        obs=(t['region']==region) if region!='all' else np.ones(len(core),bool)
        for ref in REFS:
            d=t[ref.lower()+'_kpc']
            j=core&obs&np.isfinite(d)&~nonstellar
            severe=np.zeros(len(d),bool)
            severe[j]=np.abs(5*np.log10(d50[j]/d[j]))>1
            count=lambda mask: int(mask.sum())
            rows.append(dict(reference=ref,region=region,n_spectral_distance=count(j),
                n_stellar_label=count(j&np.isin(pop,(*CLASSES,'other_stellar','dwarf_feh_unknown'))),
                n_unknown=count(j&(pop=='unclassified')),giant=count(j&(pop=='giant')),
                intermediate=count(j&(pop=='intermediate_gravity')),severe=count(severe),
                giant_severe=count(severe&(pop=='giant')),intermediate_severe=count(severe&(pop=='intermediate_gravity')),
                redrock_assessed_in_observable_frame=count(core&obs&assessed),
                redrock_nonstellar_in_assessed=count(core&obs&assessed&np.isin(t['redrock_SPECTYPE'],('QSO','GALAXY'))),
                phase_nonstellar_in_observable_frame=count(core&obs&nonstellar)))
    csv_rows(dest/'conditional_composition.csv',rows)


def branch_conditions(t,q,detail,group,dest):
    rows=[]
    c=detail['g0']-detail['r0']
    for ref in REFS:
        d=t[ref.lower()+'_kpc']
        for system,region in (('DECam','south_low'),('BASS_MzLS','north')):
            for r0,r1 in ((18,19),(19,20)):
                for c0,c1 in ((-.5,.5),(.5,.8),(.8,1.4)):
                    for a,b in ((2,5),(5,10)):
                        base=(t['region']==region)&(t['r']>=r0)&(t['r']<r1)&(c>=c0)&(c<c1)&(d>=a)&(d<b)
                        for pop in CLASSES:
                            j=base&(group==pop)
                            rows.append(dict(reference=ref,system=system,r_lo=r0,r_hi=r1,
                              gr_lo=c0,gr_hi=c1,d_lo=a,d_hi=b,group=pop,
                              n_quality_feh=int((j&(t['rvs_feh_err']>0)&(t['rvs_feh_err']<=.3)).sum()),
                              n_quality_logg=int((j&(t['rvs_logg_err']>0)&(t['rvs_logg_err']<=.3)).sum()),
                              median_rz=np.median((detail['r0']-detail['z0'])[j]) if j.any() else None,
                              **stats(q['Photdis'][j],d[j])))
    csv_rows(dest/'conditional_branch.csv',rows)


def main():
    require_compute()
    dest=OUT/'benchmark'
    dest.mkdir(exist_ok=True,parents=True)
    t,b=load_parent()
    q,_,inputs=published(t,b['outputs']['ids.npy'])
    count,detail,links=read_scan(inputs)
    t,q,detail=full_curves(t,q,detail,inputs)
    group=phase(t,inputs)
    weak(t,q,dest)
    distribution(t,q,detail,group,dest,inputs)
    masks=coverage(t,q,detail,count,links,group,dest)
    cohort_composition(t,q,group,dest)
    branch_conditions(t,q,detail,group,dest)
    atomic_npz(dest/'figure_stars.npz',**{k:detail[k] for k in ('ls_id','TYPE','region','r','g0','r0','z0','grz_error','cell')},
      **{k:t[k] for k in ('specdis_kpc','spdist_kpc','specdis_sigma_mu','spdist_sigma_mu','rvs_feh','rvs_feh_err',
                         'rvs_logg','rvs_logg_err','gaia_state')},**masks,
      group=group,photdis=q['Photdis'],bj=q['BJ_photogeometric'])
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),inputs=inputs,
        outputs={p.name:sha(p) for p in dest.iterdir() if p.name!='complete.json'}))


if __name__=='__main__':
    main()
