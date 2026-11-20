#!/usr/bin/env python3
"""Common-CMD, quality and shared-reference comparisons on the fixed roster."""
import csv
import json

import numpy as np

from cluster_depth_inputs import C, CONFIG, OLD_CONFIG, OUT, ROOT
from cluster_depth_physics import MODELS, OPTICAL_VETO, imaging_arrays
from environment_results import median
from photdis.stage1 import require_compute
from v2_1_near import csv_rows, load, save_json, sha

PRODUCT = OUT/'results'
LABELS = ('all','lower','faint_lower','faint_lower_clean','faint_lower_common_clean',
          'faint_lower_outer_quality','faint_lower_outer_common','faint_lower_low_crowd_common')
METRICS = ('crps','log','width68','width95','c68','c95','severe')


def blocks_ci(x,block,w,sigma=0.):
    cells=[np.flatnonzero(block==k) for k in np.unique(block)]
    if len(cells)<2:
        return np.nan,np.nan
    rng=np.random.default_rng(C['seed'])
    draws=rng.integers(0,len(cells),size=(C['bootstrap'],len(cells)))
    no_weights=np.all(w==w[0])
    result=[]
    for draw in draws:
        j=np.concatenate([cells[i] for i in draw])
        result.append(np.median(x[j]) if no_weights else median(x[j],w[j]))
    ref_error=rng.normal(0,sigma,len(result))
    return tuple(np.quantile(np.asarray(result)+ref_error,[.025,.975]))


def summary(a,mask,weight,arm='qrf',sigma=0.,ci=False):
    out=dict(N=int(mask.sum()),blocks=len(np.unique(a['block'][mask])))
    if not mask.any():
        return out
    w=weight[mask]
    d=a[arm+'_bias'][mask]
    b=median(d,w)
    out.update(weight=float(w.sum()),bias=b,nmad=1.4826*median(abs(d-b),w),
               **{k:float(np.average(a[f'{arm}_{k}'][mask],weights=w)) for k in METRICS})
    if ci:
        out['bias_lo'],out['bias_hi']=blocks_ci(d,a['block'][mask],w,sigma)
    return out


def indices(a):
    spec=C['common_cells']
    names=('Mr','gr','rz')
    j=[np.searchsorted(spec[k+'_edges'],a[k],side='right')-1 for k in names]
    shape=tuple(len(spec[k+'_edges'])-1 for k in names)
    k=np.stack(j,axis=0)
    ok=((k>=0)&(k<np.array(shape)[:,None])).all(0)
    return np.where(ok,np.ravel_multi_index(np.clip(k,0,np.array(shape)[:,None]-1),shape),-1)


def weighted_overlap(data,quality="clean"):
    spec=C['common_cells']
    keys=C['primary']
    cell={key:indices(a) for key,a in data.items()}
    count={}
    for key in keys:
        c=cell[key]
        a=data[key]
        idx=c[a[quality]&a['faint']&a['lower']&(c>=0)]
        count[key]={int(j):int(sum(idx==j)) for j in np.unique(idx)}
    shared={j:min(count[key].get(j,0) for key in keys) for j in set().union(*(count[k] for k in keys))}
    kept={j:n for j,n in shared.items() if n>=spec['minimum_per_cluster']}
    weights={}
    for key,c in cell.items():
        if key in keys:
            weights[key]=np.array([kept.get(j,0)/max(1,count[key].get(j,0)) for j in c])
        else:
            weights[key]=(np.isin(c,list(kept))).astype(float)
    rows=[dict(cell=j,eligible=bool(n>=spec['minimum_per_cluster']),
               **{k:count[k].get(j,0) for k in keys})
          for j,n in sorted(shared.items())]
    return weights,rows


def partitions(a,w):
    all_rows=np.ones(len(w),bool)
    depth=a['faint']&a['lower']
    return dict(all=all_rows,bright=a['r']<20,lower=a['lower'],bright_lower=(a['r']<20)&a['lower'],
                faint_lower=depth,faint_lower_clean=depth&a['clean'],
                faint_lower_common_clean=depth&a['clean']&(w>0))


def continuous(a,key):
    edges={'r':np.arange(15,23.51,.5),'Mr':np.arange(1.5,9.51,.5),
           'gr':np.arange(-.1,1.51,.1),'rz':np.arange(-.3,.851,.10),
           'radius':[0,4,8,12,18,24,31], 'fracflux':[-1,0.01,.03,.05,.1,.5,3],
           'fracmasked':[-1,.01,.025,.05,.1,.3,1.1],
           'sigma':[0,.01,.02,.035,.05,.075,.101],
           'depth_margin':[-2,0,.25,.5,.75,1,1.5,2,3,5,10],
           'EBV':[0,.04,.08,.12,.16,.2,.3,1],
           'nobs':[0,2,3,4,6,9,100]}
    rows=[]
    full=np.ones(len(a['r']))
    for axis,values in edges.items():
        for lo,hi in zip(values[:-1],values[1:]):
            k=(a[axis]>=lo)&(a[axis]<hi)
            for group,eligible in [('all',k),('lower',k&a['lower']),
              ('clean_lower',k&a['lower']&a['clean']),('outer_quality_lower',k&a['lower']&a['outer_quality']),
              ('faint_lower',k&a['lower']&a['faint'])]:
                if eligible.any():
                    rows.append(dict(system=key,axis=axis,lo=lo,hi=hi,group=group,
                            **summary(a,eligible,full,ci=int(eligible.sum())>=20)))
    return rows


def paired(a,k,key,group):
    rows=[]
    for name in MODELS:
        s=k&a[name+'_support']
        d=dict(system=key,subset=group,model=name,eligible=int(k.sum()),support=int(s.sum()))
        if s.any():
            w=np.ones(len(s))
            for ar in ('qrf',name):
                for metric,x in summary(a,s,w,arm=ar).items():
                    d[('qrf_' if ar=='qrf' else 'cmd_')+metric]=x
            delta=a[name+'_crps'][s]-a['qrf_crps'][s]
            d['mean_delta_crps']=float(delta.mean())
            d['delta_abs_median_bias']=abs(d['cmd_bias'])-abs(d['qrf_bias'])
            # Use paired resampling of full bricks for the proper-score gain.
            block=a['block'][s]
            cells=[delta[block==v] for v in np.unique(block)]
            if len(cells)>=2:
                rng=np.random.default_rng(C['seed'])
                j=rng.integers(0,len(cells),size=(C['bootstrap'],len(cells)))
                sums=np.array([c.sum() for c in cells]);n=np.array([len(c) for c in cells])
                d['crps_lo'],d['crps_hi']=np.quantile(sums[j].sum(1)/n[j].sum(1),[.025,.975])
        rows.append(d)
    return rows


def quality_on_common_cmd(a,key):
    """Balance the clean and remaining stars inside the same fixed CMD cells."""
    cell=indices(a)
    base=a['faint']&a['lower']&(cell>=0)
    left=base&a['clean']
    right=base&~a['clean']
    counts=[{int(c):int(np.count_nonzero(use&(cell==c))) for c in np.unique(cell[use])}
            for use in (left,right)]
    relevant={k:min(n,counts[1].get(k,0)) for k,n in counts[0].items()}
    relevant={k:n for k,n in relevant.items() if n>=5}
    rows=[]
    for group,mask,denom in [('clean',left,counts[0]),('remaining',right,counts[1])]:
        w=np.array([relevant.get(j,0)/max(1,denom.get(j,0)) for j in cell])
        rows.append(dict(system=key,subset=group,cells=len(relevant),
                         **summary(a,mask&(w>0),w,ci=True)))
    return rows


def cmd_bins(a,key):
    rows=[]
    for col,dx,xlo,xhi in [('gr',.045,-.1,1.6),('rz',.035,-.3,.9)]:
        y=a['Mr'];x=a[col]
        assert np.isfinite(x+y).all()
        xlo=min(xlo,float(x.min()));xhi=max(xhi,float(x.max())+dx)
        ix=np.floor((x-xlo)/dx).astype(int);iy=np.floor((y-y.min())/.20).astype(int)
        px=int(np.ceil((xhi-xlo)/dx))
        ids=iy*px+ix
        for i in np.unique(ids):
            take=ids==i
            rows.append(dict(system=key,color=col,coord=float(np.median(x[take])),
                Mr=float(np.median(y[take])),N=int(take.sum()),bias=float(np.median(a['qrf_bias'][take]))))
    return rows


def main():
    require_compute()
    PRODUCT.mkdir(exist_ok=True)
    data={k:load(OUT/(k+'_diagnostics.npz')) for k in C['systems']}
    for key,a in data.items():
        native=load(OUT/(key+'_native.npz'))
        np.testing.assert_array_equal(a['ls_id'],native['ls_id'])
        metrics,flags=imaging_arrays(native)
        a.update(**metrics,**flags)
        a['outer_quality']=a['outer']&a['low_crowd']&a['phot_precise']&a['deep']
        a['low_crowd_quality']=a['low_crowd']&a['phot_precise']&a['deep']
    weights,overlap=weighted_overlap(data)
    outer_w,outer_cells=weighted_overlap(data,'outer_quality')
    crowd_w,crowd_cells=weighted_overlap(data,'low_crowd_quality')
    prior=json.loads(OLD_CONFIG.read_text())['cluster_candidates']
    results={k:[] for k in ('populations','quality','mask_audit','quality_common','continuous','paired','cmd','scales','color_residuals','growth')}
    choices={}
    for key,a in data.items():
        assert (a['TYPE']=='PSF').all()
        w=np.ones(len(a['r']))
        res=partitions(a,weights[key])
        base=res['faint_lower']
        res.update(faint_lower_outer_quality=base&a['outer_quality'],
                   faint_lower_outer_common=base&a['outer_quality']&(outer_w[key]>0),
                   faint_lower_low_crowd_common=base&a['low_crowd_quality']&(crowd_w[key]>0))
        rrl=prior[key]['rrl_mu'];p=C['systems'][key]
        sigma=np.hypot(prior[key]['rrl_sigma_mag'],C['model']['score_ref_common_mag'])
        bv=10+5*np.log10(p['bv_distance_kpc'])
        refs={'rrl_hst':(rrl,sigma),'rrl_statistical':(rrl-.32,sigma),
                    'rrl_positive_stress':(rrl+.32,sigma),
                    'baumgardt_vasiliev':(bv,np.hypot(.10,5/np.log(10)*p['bv_sigma_kpc']/p['bv_distance_kpc'])),
                    'secondary_published':(p['other_mu'],np.hypot(.10,p['other_sigma'] or 0.)) }
        for label,mask in res.items():
            kw={'faint_lower_common_clean':weights[key],'faint_lower_outer_common':outer_w[key],
                'faint_lower_low_crowd_common':crowd_w[key]}.get(label,w)
            row=dict(system=key,subset=label,**summary(a,mask,kw,sigma=sigma,ci=True))
            results['populations'].append(row)
            if label in LABELS:
                results['paired'].extend(paired(a,mask,key,label))
            if label!='all' and mask.any():
                for anchor,(mu,err) in refs.items():
                    x=a['qrf_median'][mask]-mu
                    lo,hi=blocks_ci(x,a['block'][mask],kw[mask],err)
                    results['scales'].append(dict(system=key,subset=label,anchor=anchor,mu=mu,assumed_sigma=err,
                                         N=row['N'],blocks=row['blocks'],bias=median(x,kw[mask]),lo=lo,hi=hi))
                for method in MODELS:
                    for color in ('gr','rz'):
                        v=a[method+'_residual_'+color]
                        valid=np.isfinite(v)&mask
                        if valid.any():
                            results['color_residuals'].append(dict(system=key,subset=label,method=method,
                               color=color,N=int(valid.sum()),median=float(np.median(v[valid]))))
        first=res['faint_lower_common_clean']
        b=next(r for r in results['populations'] if r['system']==key and r['subset']=='faint_lower_common_clean')
        chosen=[r for r in results['scales'] if r['system']==key and r['subset']=='faint_lower_common_clean']
        lit=[r for r in chosen if r['anchor']!='rrl_positive_stress']
        enough=bool(first.sum()>=C['minimum_clean'] and b['blocks']>=C['minimum_spatial_blocks'])
        choices[key]=dict(n=int(first.sum()),blocks=b['blocks'],enough=enough,
                           robust=bool(enough and all(r['bias']>=C['threshold_mag'] and r['lo']>0 for r in lit)),
                           minimum_published_bias=min((r['bias'] for r in lit),default=None))
        for q in ('outer','low_crowd','imaging_unmasked','phot_precise','deep','high_membership','clean','clean_all_bits'):
            for side,sel in [('passes',a[q]),('remaining',~a[q])]:
                mask=sel&res['faint_lower']
                results['quality'].append(dict(system=key,cut=q,side=side,
                                **summary(a,mask,w,sigma=sigma,ci=True)))
        bits=a['MASKBITS'].astype('i8')
        for bit in range(20):
            use=res['faint_lower']&((bits&(1<<bit))!=0)
            results['mask_audit'].append(dict(system=key,bit=bit,optical_veto=bool(OPTICAL_VETO&(1<<bit)),
                                         N=int(use.sum()),faint_lower=int(res['faint_lower'].sum())))
        results['quality_common'].extend(quality_on_common_cmd(a,key))
        cont=continuous(a,key)
        results['continuous'].extend(cont)
        for axis in ('r','Mr'):
            for g in ('lower','clean_lower','outer_quality_lower'):
                good=[r for r in cont if r['axis']==axis and r['group']==g and r['N']>=30 and
                    r['blocks']>=3 and r.get('bias_lo',0)>=C['threshold_mag']]
                results['growth'].append(dict(system=key,axis=axis,group=g,
                         first_lo=good[0]['lo'] if good else '',first_hi=good[0]['hi'] if good else '',
                         N=good[0]['N'] if good else 0,bias=good[0]['bias'] if good else ''))
        results['cmd'].extend(cmd_bins(a,key))
    results['matched_cells']=[dict(control=group,**r) for group,tab in
                [('strict_clean',overlap),('outer_quality',outer_cells),('low_crowd',crowd_cells)] for r in tab]
    results['model_support']=[]
    for key in data:
        with (OUT/(key+'_model_support.csv')).open() as f:
            results['model_support'].extend(csv.DictReader(f))
    for name,rr in results.items():
        csv_rows(PRODUCT/(name+'.csv'),rr)
    save_json(PRODUCT/'decision.json',dict(config=sha(CONFIG),shared_cells=int(sum(r['eligible'] for r in overlap)),
                    primary_pass=all(choices[k]['robust'] for k in C['primary']),systems=choices,
                    outputs={f.name:sha(f) for f in sorted(PRODUCT.glob('*.csv'))}))
    print('Fixed common-CMD decision',choices,flush=True)


if __name__=='__main__':
    main()
