#!/usr/bin/env python3
"""Spatially resampled environment contrasts and paired dust comparisons."""
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp

from environment_pilot import C, MAG, OUT, finish
from environment_dust import modulus_quantiles
from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, sha

SOURCES = ('gaia','specdis','spdist','astro')
METRICS = ('bias','nmad','severe','log_score','crps','coverage68','width68','retained')


def histogram_scores(edges, mass, target, sigma):
    """Gaussian-reference log density and exact point-reference CRPS in modulus."""
    width = np.diff(edges,axis=1)
    positive = (width>0)&(mass>0)
    safe_width = np.where(positive,width,1.)
    low = (edges[:,:-1]-target[:,None])/sigma[:,None]
    high = (edges[:,1:]-target[:,None])/sigma[:,None]
    reflected = low+high>0
    a = log_ndtr(np.where(reflected,-low,high))
    b = log_ndtr(np.where(reflected,-high,low))
    with np.errstate(invalid='ignore',divide='ignore'):
        lp = np.where(positive,np.log(mass)+a+np.log(-np.expm1(b-a))-np.log(safe_width),-np.inf)
    log_score = logsumexp(lp,axis=1)
    midpoint=(edges[:,:-1]+edges[:,1:])/2
    d=abs(midpoint-target[:,None])
    mean_abs=np.where(d>=width/2,d, (d**2+width**2/4)/safe_width)
    cum_mass=np.cumsum(mass,axis=1)-mass
    first=midpoint*mass
    cum_first=np.cumsum(first,axis=1)-first
    half_pair=(mass*(midpoint*cum_mass-cum_first)+mass**2*width/6).sum(axis=1)
    crps=(mass*mean_abs).sum(axis=1)-half_pair
    assert np.isfinite(log_score).all() and (crps>=-1e-10).all()
    return log_score, crps


def measure(edges, mass, mu, sigma, retained):
    np.testing.assert_allclose(mass.sum(axis=1),1.,atol=1e-10)
    quants=modulus_quantiles(mass,edges)
    log,crps=histogram_scores(edges,mass,mu,sigma)
    return dict(bias=quants[:,2]-mu, severe=abs(quants[:,2]-mu)>C['severe_error_mu_mag'],
                log_score=log,crps=crps,coverage68=(quants[:,1]<=mu)&(mu<=quants[:,3]),
                width68=quants[:,3]-quants[:,1],retained=retained)


def median(x,w):
    positive = w>0
    x, w = x[positive], w[positive]
    order=np.argsort(x)
    s=np.cumsum(w[order])
    at=np.searchsorted(s,s[-1]/2,side='left')
    if at+1<len(x) and np.isclose(s[at],s[-1]/2,atol=1e-12,rtol=0):
        return (x[order[at]]+x[order[at+1]])/2
    return x[order[at]]


def summary(values,w):
    mid=median(values['bias'],w)
    nmad=1.4826*median(abs(values['bias']-mid),w)
    return np.array([mid,nmad]+[np.average(values[key],weights=w) for key in METRICS[2:]])


def covariance_blocks(t):
    x=np.floor(t['longitude']/C['spatial_block_deg']).astype(int)
    y=np.floor((t['latitude']+90)/C['spatial_block_deg']).astype(int)
    return x*10000+y


def weighted_cells(cells,a,b):
    groups=np.unique(np.r_[cells[a],cells[b]])
    na=np.array([(cells[a]==x).sum() for x in groups])
    nb=np.array([(cells[b]==x).sum() for x in groups])
    count=np.minimum(na,nb)
    use=count>=C['matching']['minimum_per_cell_per_arm']
    g,w=groups[use],count[use]
    aa=a[np.isin(cells[a],g)]
    bb=b[np.isin(cells[b],g)]
    wa=(w/na[use])[np.searchsorted(g,cells[aa])]
    wb=(w/nb[use])[np.searchsorted(g,cells[bb])]
    return aa,bb,wa,wb,len(g)


def infer(a,b,wa,wb,data,blocks,*,bdata=None):
    bdata = data if bdata is None else bdata
    if not len(a) or not len(b):
        return {'a_n':len(a),'b_n':len(b),'precise_010':False}
    av={k:v[a] for k,v in data.items()}
    bv={k:v[b] for k,v in bdata.items()}
    sa,sb=summary(av,wa),summary(bv,wb)
    delta=sb-sa
    au,ai=np.unique(blocks[a],return_inverse=True)
    bu,bi=np.unique(blocks[b],return_inverse=True)
    an=wa.sum()**2/(wa@wa)
    bn=wb.sum()**2/(wb@wb)
    record=dict(a_n=len(a),b_n=len(b),a_effective_n=an,b_effective_n=bn,a_blocks=len(au),b_blocks=len(bu))
    boot=[]
    rng=np.random.default_rng(C['sample_seed'])
    # Draw from the union so nearby stars in both arms carry the same block multiplicity.
    union=np.union1d(au,bu)
    ai=np.searchsorted(union,au[ai])
    bi=np.searchsorted(union,bu[bi])
    if min(len(au),len(bu))>=2:
        for count in rng.multinomial(len(union),np.repeat(1/len(union),len(union)),size=C['bootstrap_draws']):
            ra,rb=count[ai]*wa,count[bi]*wb
            if min(ra.sum(),rb.sum())>0:
                boot.append(summary(bv,rb)-summary(av,ra))
    intervals=np.quantile(boot,[.025,.975],axis=0) if len(boot) else np.full((2,len(delta)),np.nan)
    record.update(bootstrap_draws=len(boot))
    for k,x,y,d,lo,hi in zip(METRICS,sa,sb,delta,*intervals):
        record.update({f'a_{k}':x,f'b_{k}':y,f'delta_{k}':d,f'{k}_lo':lo,f'{k}_hi':hi})
    half=max(delta[0]-intervals[0,0],intervals[1,0]-delta[0])
    met=(min(an,bn)>=C['minimum_common_effective_n_per_arm']
        and min(len(au),len(bu))>=C['minimum_spatial_blocks_per_arm'])
    record['bias_interval_max_error']=half
    record['precise_010']=bool(met and half<=C['precision_target_mu_mag'])
    record['sign_resolved_at_prespecified_precision']=bool(record['precise_010'] and
        intervals[0,0]*intervals[1,0]>0)
    return record


def tag(t):
    labels=np.array(t['gaia_proxy'],dtype='U15')
    classes = np.where(t['classification']>=0,t['classification'],t['astro_classification'])
    conflict = (t['classification']>=0)&(t['astro_classification']>=0)
    conflict &= t['classification']!=t['astro_classification']
    labels[classes==0]='spectra_dwarf'
    labels[classes==1]='spectra_giant'
    labels[classes==2]='spectra_other'
    labels[conflict]='class_conflict'
    return labels


def code(columns):
    return np.unique(np.column_stack(columns),axis=0,return_inverse=True)[1]


def area():
    from astropy.coordinates import SkyCoord
    import healpy as hp
    path=ROOT/'data/sky_sequence_20261119/area/south/geometry.npz'
    z=load(path)
    nside=hp.npix2nside(len(z['area'][0]))
    rows=[]
    for name,l,b in C['fields']:
        center=SkyCoord(l,b,frame='galactic',unit='deg').icrs
        theta,phi=np.deg2rad([90-center.dec.deg,center.ra.deg])
        vec=hp.ang2vec(theta,phi)
        fine=nside*16
        ips=hp.query_disc(fine,vec,np.deg2rad(C['field_radius_deg']),nest=True)
        counts=np.bincount(ips//256,minlength=len(z['area'][0]))/256
        rows.append(dict(field=name,longitude=l,latitude=b,area_gr_random_deg2=float(counts@z['area'][0]),
                         aperture_deg2=float(len(ips)*hp.nside2pixarea(fine,degrees=True))))
    return rows,sha(path)


def census_rows(t, pdf, areas):
    r=t['r']
    support=t['support']
    apps=t['application_sample']
    working=np.zeros(len(apps),bool)
    working[pdf['rows']]=(pdf['dwarf_score']>=.8)&(pdf['mass_2_10']>=.5)
    out=[]
    for field in areas:
        same=t['field']==field['field']
        for a,b in [(14.,23.),*zip(MAG[:-1],MAG[1:])]:
            mask=same&(r>=a)&(r<b)
            n=mask.sum()
            denominator=(mask&apps).sum()
            row=dict(**field,r_min=a,r_max=b,precise_grz=int(n),unique_G3=int((mask&t['unique_G3']).sum()),
                     apogee_matched=int((mask&t['astro_matched']).sum()),
                     gaia_signed=int((mask&(t['gaia_state']>=4)).sum()),
                     qrf_nominal_support=int((mask&support).sum()),
                     qrf_train_overlap=int((mask&t['qrf_fit_overlap']).sum()),
                     history_pool_overlap=int((mask&t['historical_overlap']).sum()),
                     application_draw_N=int(denominator),
                     application_draw_selected=int((mask&apps&working).sum()),
                     working_rule_fraction=float((mask&apps&working).sum()/denominator) if denominator else np.nan,
                     median_sfd=float(np.median(t['EBV'][mask])) if n else np.nan,
                     median_fracflux_r=float(np.median(t['FRACFLUX_R'][mask])) if n else np.nan)
            for source in SOURCES:
                use=mask&np.isfinite(t[source+'_distance_kpc'])
                row[source+'_geometry_or_quality']=int(use.sum())
                row[source+'_support_nonfit']=int((use&support&~t['qrf_fit_overlap']).sum())
                row[source+'_evaluation']=int((mask&t[source+'_evaluation']).sum())
            out.append(row)
    return out


def provider_stats(t,pdf):
    rows=pdf['rows']
    small={k:v[rows] for k,v in t.items()}
    blocks=covariance_blocks(small)
    labels=tag(small)
    lon=small['longitude']
    lat=abs(small['latitude'])
    ebv=small['EBV']
    crowd=small['FRACFLUX_R']
    gr=np.diff(small['mag_grz'][:,:2],axis=1)[:,0]*-1
    retention=(pdf['dwarf_score']>=.8)&(pdf['mass_2_10']>=.5)
    margins, contrasts, cells_out, all_scores, strata = [], [], [], [], []
    for source in SOURCES:
        available=small[source+'_evaluation']
        take=np.flatnonzero(available)
        if not len(take):
            continue
        distance=small[source+'_distance_kpc'][take]
        ref=10+5*np.log10(distance)
        sigma=small[source+'_sigma_mu'][take]
        data=measure(pdf['mu_edges'][take],pdf['mass'][take],ref,sigma,retention[take])
        s={k:x[take] for k,x in small.items()}
        for k in range(len(take)):
            all_scores.append(dict(TYPE=s['TYPE'][k],ls_id=int(s['ls_id'][k]),field=s['field'][k],source=source,
                label=labels[take[k]],r=s['r'][k],gr0=gr[take[k]],reference_kpc=distance[k],
                gal_l=lon[take[k]],gal_b=s['latitude'][k],ebv=s['EBV'][k],fracflux_r=s['FRACFLUX_R'][k],
                historical_overlap=bool(s['historical_overlap'][k]),**{m:v[k] for m,v in data.items()}))
        for name,_,_ in C['fields']:
            field=s['field']==name
            splits=[('all',np.ones(len(take),bool)),('history_clean',~s['historical_overlap']),
                    ('dwarf_proxy',labels[take]=='dwarf_proxy'),('spectra_dwarf',labels[take]=='spectra_dwarf')]
            for group,mask in splits:
                k=np.flatnonzero(field&mask)
                row=dict(field=name,source=source,subsample=group,n=len(k))
                if len(k):
                    row.update(zip(METRICS,summary({m:v[k] for m,v in data.items()},np.ones(len(k)))))
                margins.append(row)
        match=C['matching']
        encoded_labels=np.unique(labels[take],return_inverse=True)[1]
        variables=[np.digitize(gr[take],match['gr0'][1:-1]),np.digitize(s['r'],match['r'][1:-1]),
                   np.digitize(distance,match['distance_kpc'][1:-1]),encoded_labels]
        low=np.isin(s['field'],['L219','L233','L240'])
        mid=np.isin(s['field'],['M240','M037'])
        high=s['field']=='H240'
        dust_bin=np.digitize(ebv[take],[.08,.2,.5])
        crowd_bin=np.digitize(crowd[take],[.02,.10])
        latitude_bin=np.digitize(lat[take],[10,15,30])
        for variable, codes, edges in [
            ('observed_r', np.digitize(s['r'], MAG[1:-1]), MAG),
            ('reference_kpc', np.digitize(distance, match['distance_kpc'][1:-1]),
             match['distance_kpc']),
            ('abs_b_deg', latitude_bin, C['latitude_edges_deg']),
            ('sfd_ebv', dust_bin, C['sfd_edges']),
            ('fracflux_r', crowd_bin, C['fracflux_r_edges']),
        ]:
            for k,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                for cohort, chosen in [('all_nonfit',np.ones(len(take),bool)),
                                       ('dwarf_proxy',labels[take]=='dwarf_proxy')]:
                    rows = np.flatnonzero((codes==k)&chosen)
                    item = dict(source=source, cohort=cohort, variable=variable,
                                lower=lo, upper=hi, n=len(rows))
                    if len(rows):
                        item.update(zip(METRICS, summary({m:v[rows] for m,v in data.items()},
                                                        np.ones(len(rows)))))
                    strata.append(item)
        designs=[
            ('high_vs_low_fields',high,low,[dust_bin,crowd_bin]),
            ('high_vs_mid_fields',high,mid,[dust_bin,crowd_bin]),
            ('higher_sfd_column',ebv[take]<.08,ebv[take]>=.2,[latitude_bin,crowd_bin]),
            ('higher_fracflux_r',crowd[take]<.02,crowd[take]>=.10,[latitude_bin,dust_bin]),
        ]
        # Matched estimates over the observed joint support. Field-pair comparisons
        # also report the practical total-environment difference at common stellar properties.
        for target in ('L219','L233','L240','M240','M037'):
            designs.append((target+'_vs_H240',high,s['field']==target,[]))
        for name,a,b,hold in designs:
            for label in ('all_nonfit','history_clean','dwarf_proxy','spectra_dwarf'):
                cohort=np.ones(len(take),bool)
                if label=='history_clean':
                    cohort=~s['historical_overlap']
                elif label=='dwarf_proxy':
                    cohort=labels[take]=='dwarf_proxy'
                elif label=='spectra_dwarf':
                    cohort=labels[take]=='spectra_dwarf'
                aa=np.flatnonzero(a&cohort)
                bb=np.flatnonzero(b&cohort)
                cells=code(variables+hold)
                x,y,wx,wy,n=weighted_cells(cells,aa,bb)
                result=infer(x,y,wx,wy,data,blocks[take])
                contrasts.append(dict(source=source,contrast=name,cohort=label,
                    available_a=len(aa),available_b=len(bb),common_cells=n,**result))
                for c in np.intersect1d(cells[x],cells[y]):
                    ax=x[cells[x]==c];by=y[cells[y]==c]
                    cells_out.append(dict(source=source,contrast=name,cohort=label,cell=int(c),n_a=len(ax),n_b=len(by)))
    return margins,contrasts,cells_out,all_scores,strata


def dust_stats(t,pdf,d):
    rows=d['rows']
    order=np.searchsorted(pdf['rows'],rows)
    np.testing.assert_array_equal(pdf['rows'][order],rows)
    mu=d['reference_mu']
    sig=t['gaia_sigma_mu'][rows]
    overlap=(mu>=d['mu_min'])&(mu<=d['mu_max'])
    blocks=covariance_blocks({k:v[rows] for k,v in t.items()})
    sufficient=d['base_probability']>=C['dust']['minimum_histogram_mass']
    defined=d['iteration_state']==5
    oracle=d['oracle_state']==3
    audits, stats, changes=[],[],[]
    for name in ('ALL',*[x[0] for x in C['fields']]):
        field=np.ones(len(mu),bool) if name=='ALL' else t['field'][rows]==name
        audits.append(dict(field=name,inputs=int(field.sum()),
            reliable_map=int((field&(d['mu_min']<d['mu_max'])).sum()),
            reference_in_map=int((field&overlap).sum()),
            sfd_has_half_map_mass=int((field&sufficient).sum()),
            converged=int((field&defined).sum()),converged_ref_in_map=int((field&defined&overlap).sum()),
            oracle_common=int((field&sufficient&oracle).sum()),
            **{f'iteration_state_{k}':int((field&(d['iteration_state']==k)).sum()) for k in range(6)}))
        for typ, select in [('all',field),('history_clean',field&~t['historical_overlap'][rows]),
                           ('dwarf_proxy',field&(tag(t)[rows]=='dwarf_proxy'))]:
            for mode, pick in [('iteration',defined&overlap),('known_distance',sufficient&oracle)]:
                keep=np.flatnonzero(select&pick)
                result=dict(field=name,cohort=typ,method=mode,n=len(keep))
                if not len(keep):
                    stats.append(result)
                    continue
                k=keep
                base=measure(d['base_edges'][k],d['base_mass'][k],mu[k],sig[k],np.full(len(k),np.nan))
                base_full=measure(pdf['mu_edges'][order[k]],pdf['mass'][order[k]],mu[k],sig[k],np.full(len(k),np.nan))
                result['sfd_full_bias']=float(np.median(base_full['bias']))
                if mode=='iteration':
                    new=measure(d['iteration_edges'][k],d['iteration_mass'][k],mu[k],sig[k],np.full(len(k),np.nan))
                else:
                    new={m:np.full(len(k),np.nan) for m in base}
                    new['bias']=d['oracle_center'][k]-mu[k]
                    new['severe']=abs(new['bias'])>C['severe_error_mu_mag']
                    base={m:(v if m in ('bias','severe') else np.full(len(k),np.nan)) for m,v in base.items()}
                with np.errstate(invalid='ignore'):
                    result.update(infer(np.arange(len(k)),np.arange(len(k)),np.ones(len(k)),np.ones(len(k)),
                                        base,blocks[k],bdata=new))
                stats.append(result)
                if typ=='all' and name!='ALL':
                    for i,a,b in zip(k,base['bias'],new['bias']):
                        changes.append(dict(field=name,TYPE=t['TYPE'][rows[i]],ls_id=int(d['ls_id'][i]),
                            mode=mode,reference_mu=mu[i],sfd_full_ebv=t['EBV'][rows[i]],
                            map_reliable_lo=d['mu_min'][i],map_reliable_hi=d['mu_max'][i],
                            sfd_full_bias=base_full['bias'][np.searchsorted(k,i)],sfd_same_range_bias=a,
                            bayestar_bias=b,base_probability=d['base_probability'][i]))
    return audits,stats,changes


def main():
    require_compute()
    p=OUT/'results'
    p.mkdir(parents=True,exist_ok=True)
    t=load(OUT/'forecast/roster.npz')
    pdf=load(OUT/'forecast/pdf.npz')
    d=load(OUT/'dust/forecasts.npz')
    apogee=load(OUT/'apogee/references.npz')
    np.testing.assert_array_equal(t['ls_id'],apogee['ls_id'])
    assert not (apogee['astro_evaluation']&t['qrf_fit_overlap']).any()
    t.update({key:value for key,value in apogee.items() if key.startswith('astro_')})
    apogee_pdf=OUT/'apogee/pdf.npz'
    if apogee_pdf.exists():
        supplemental=load(apogee_pdf)
        pdf={key:np.concatenate([value,supplemental[key]]) for key,value in pdf.items()}
        permutation=np.argsort(pdf['rows'])
        assert len(np.unique(pdf['rows']))==len(permutation)
        pdf={key:value[permutation] for key,value in pdf.items()}
    assert (t['TYPE']=='PSF').all() and (pdf['TYPE']=='PSF').all() and (d['TYPE']=='PSF').all()
    areas,area_sha=area()
    csv_rows(p/'census.csv',census_rows(t,pdf,areas))
    metrics,cross,cells,scores,strata=provider_stats(t,pdf)
    dust_counts,dust,changes=dust_stats(t,pdf,d)
    for name,data in [('field_metrics',metrics),('environment_contrasts',cross),
                      ('matched_cells',cells),('source_scores',scores),('band_metrics',strata),('dust_flow',dust_counts),
                      ('dust_pairs',dust),('dust_source_pairs',changes)]:
        csv_rows(p/(name+'.csv'),data)
    finish(p,{'geometry_south':area_sha,'forecast':sha(OUT/'forecast/complete.json'),
              'dust':sha(OUT/'dust/complete.json'),'apogee':sha(OUT/'apogee/complete.json'),
              'script':sha(Path(__file__))})
    print('Census, paired dust and matched environment metrics written',flush=True)


if __name__=='__main__':
    main()
