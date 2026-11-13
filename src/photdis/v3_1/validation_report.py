"""Fixed external strata and shared-system reference uncertainty summaries."""
import numpy as np
from photdis.v3.data import WEAK, SIGNIFICANT
from photdis.v3_1.comparison import paired_mean

MAIN_VIEW = 'candidate/bootstrap_evidence'


def external_strata(photo, evidence):
    np.testing.assert_array_equal(photo['identity'],evidence['identity'])
    if not (photo['TYPE']=='PSF').all() or not (evidence['TYPE']=='PSF').all():
        raise ValueError('Verified PSF required in both inputs')
    n=len(photo['identity']); masks={'all':np.ones(n,bool)}
    classes={name:evidence['class_label']==c for c,name in [(-1,'unknown'),(0,'dwarf'),(1,'giant')]}
    valid=evidence['feh_valid']; feh=evidence['feh']
    metals={'poor':valid & (feh < -1),'not_poor':valid & (feh >= -1),'unavailable':~valid}
    for name,mask in classes.items():masks['class/'+name]=mask
    for name,mask in metals.items():masks['feh/'+name]=mask
    for lo,hi in [(-np.inf,-1.5),(-1.5,-.5),(-.5,np.inf)]:
        masks[f'feh_original/{lo}_{hi}']=valid & (feh>=lo) & (feh<hi)
    masks['feh_original/unavailable']=~valid
    for c,cm in classes.items():
        for f,fm in metals.items():masks[f'class_feh/{c}/{f}']=cm & fm
    valid=evidence['distance_available']; d=evidence['distance_kpc']; distances={}
    for lo,hi in [(0,2),(2,5),(5,10),(10,20),(20,50),(50,np.inf)]:
        distances[f'{lo}_{hi}']=valid & (d>=lo) & (d<hi)
    distances['unavailable']=~valid
    for name,mask in distances.items():masks['distance/'+name]=mask
    for c,cm in classes.items():
        for d,dm in distances.items():masks[f'class_distance/{c}/{d}']=cm & dm
    for channel in ['specdis','spdist']:
        if channel+'_valid' not in evidence:continue
        valid=evidence[channel+'_valid']; distance=evidence[channel+'_distance']
        for lo,hi in [(0,2),(2,5),(5,10),(10,20),(20,50),(50,np.inf)]:
            masks[f'{channel}_distance/{lo}_{hi}']=valid & (distance>=lo) & (distance<hi)
        masks[channel+'_distance/unavailable']=~valid
    if 'mws_class_label' in evidence:
        mws_classes={name:evidence['mws_class_label']==c for c,name in [(-1,'unknown'),(0,'dwarf'),(1,'giant'),(2,'other')]}
        quality=evidence['mws_parameter_quality'] & np.isfinite(evidence['rvs_feh']) & np.isfinite(evidence['rvs_feh_err']) & (evidence['rvs_feh_err']>=0)
        mws_metals={'poor':quality & (evidence['rvs_feh'] < -1),'not_poor':quality & (evidence['rvs_feh']>=-1),'unavailable':~quality}
        for name,mask in mws_classes.items():masks['mws_class/'+name]=mask
        for name,mask in mws_metals.items():masks['mws_feh/'+name]=mask
        for lo,hi in [(-np.inf,-1.5),(-1.5,-.5),(-.5,np.inf)]:
            masks[f'mws_feh_original/{lo}_{hi}']=quality & (evidence['rvs_feh']>=lo) & (evidence['rvs_feh']<hi)
        masks['mws_feh_original/unavailable']=~quality
        for c,cm in mws_classes.items():
            for f,fm in mws_metals.items():masks[f'mws_class_feh/{c}/{f}']=cm & fm
    for lo,hi in [(-np.inf,18),(18,20),(20,22),(22,np.inf)]:
        masks[f'r0/{lo}_{hi}']=(photo['r0']>=lo) & (photo['r0']<hi)
    available=np.isin(photo['band_state'][:,4],[WEAK,SIGNIFICANT])
    f=photo['flux'][:,4]; e=photo['flux_error'][:,4]
    available &= np.isfinite(f) & np.isfinite(e) & (e>0)
    snr=np.divide(f,e,out=np.full(n,np.nan),where=available)
    masks.update({'W1/weak':available & (snr<5),'W1/significant':available & (snr>=5),'W1/unavailable':~available})
    return masks


def metric_arrays(records, view):
    """No unavailable label becomes a numerical zero or class2 observation."""
    n=len(records); result={}; accepted=np.zeros(n,bool)
    def put(name,i,value):
        if name not in result:result[name]=np.full(n,np.nan)
        result[name][i]=value
    for i,record in enumerate(records):
        value=record['views'][view]
        accepted[i]=next(c['accepted'] for c in record['checks'] if c['view']==view)
        for name in ('width68','width95','qnear'):put(name,i,value[name])
        for channel,r in value['references'].items():
            for name in ('log_score','mu_residual','relative_residual','severe','reference_agreement68','reference_agreement95'):
                put(channel+'/'+name,i,r[name])
            put(channel+'/absolute_mu_error',i,abs(r['mu_residual']))
        if 'gaia' in value:
            for name in ('log_score','pit','tail'):put('gaia/'+name,i,value['gaia'][name])
        for channel in ['class','mws_class']:
            if channel not in value:continue
            r=value[channel]
            for name in ('probability','brier','zero_probability'):put(channel+'/'+name,i,r[name])
            put(channel+'/log_score',i,-np.inf if r['zero_probability'] else r['log_score'])
            put(channel+'/correct',i,r['label']==r['predicted'])
    return result,accepted


def field_statistic(values,cells):
    """Mean with whole-sky-cell interval; nonfinite scores cannot disappear."""
    values=np.asarray(values,float); cells=np.asarray(cells)
    if values.shape!=cells.shape:raise ValueError('Aligned sky cells required')
    defined=~np.isnan(values); x=values[defined]; sky=cells[defined]
    nonfinite=int((~np.isfinite(x)).sum())
    if nonfinite:
        return dict(n=len(x),cells=len(np.unique(sky)),mean=None,lo95=None,hi95=None,
            nonfinite=nonfinite,median_descriptive=None,nmad_descriptive=None,
            status='nonfinite_score_retained_not_trimmed')
    stats=paired_mean(x,np.ones(len(x)),sky,repeats=1024,seed=320511)
    median=float(np.median(x)) if len(x) else None
    stats.update(nonfinite=0,median_descriptive=median,
        nmad_descriptive=float(1.4826*np.median(abs(x-median))) if len(x) else None,
        status='empty' if not len(x) else 'single_sky_cell_no_interval' if stats['cells']==1 else 'sky_block_interval')
    return stats


def paired_field_statistic(candidate,baseline,cells):
    a,b,cells=np.asarray(candidate,float),np.asarray(baseline,float),np.asarray(cells)
    if a.shape!=b.shape or a.shape!=cells.shape:raise ValueError('Aligned pairs required')
    qualified=~np.isnan(a) & ~np.isnan(b)
    with np.errstate(invalid='ignore'):
        delta=a[qualified]-b[qualified]
    # inf-inf is an undefined score contrast, not a missing reference to discard.
    undefined=np.isnan(delta); delta[undefined]=np.inf
    result=field_statistic(delta,cells[qualified])
    result['undefined_nonfinite_pairs']=int(undefined.sum())
    return result


def common_distance_draws(distance,error,*,draws=4096,seed=320501):
    if not np.isfinite([distance,error]).all() or min(distance,error)<=0 or draws<1:
        raise ValueError('Positive finite common-scale parameters required')
    rng=np.random.default_rng(seed); result=[]; count=0
    while count<draws:
        x=rng.normal(distance,error,draws-count); x=x[x>0]
        result.append(x); count+=len(x)
    return np.concatenate(result)


def common_scale_statistics(quantiles,distances):
    """One shared distance per draw for every star; no independent-star errors.

    These are reference-agreement statistics of fixed marginal predictions, not
    posterior realized member counts or a new structure likelihood.
    """
    q=np.asarray(quantiles,float); d=np.asarray(distances,float)
    if q.ndim!=2 or q.shape[1]!=5 or not np.isfinite(q).all() or (np.diff(q,axis=1)<0).any():
        raise ValueError('Ordered N by5 quantiles required')
    if d.ndim!=1 or not np.isfinite(d).all() or (d<=0).any():
        raise ValueError('Positive shared distances required')
    names=['median_mu_residual','median_relative_residual','mean_absolute_mu_error',
           'severe_fraction','reference_agreement68','reference_agreement95']
    result={name:np.full(len(d),np.nan) for name in names}
    if not len(q):return result
    mu=10+5*np.log10(d)
    for start in range(0,len(d),64):
        m=mu[start:start+64,None]; delta=q[None,:,2]-m; ix=slice(start,start+len(m))
        result['median_mu_residual'][ix]=np.median(delta,axis=1)
        result['median_relative_residual'][ix]=np.median(np.expm1(np.log(10)*delta/5),axis=1)
        result['mean_absolute_mu_error'][ix]=abs(delta).mean(axis=1)
        result['severe_fraction'][ix]=(abs(delta)>1).mean(axis=1)
        result['reference_agreement68'][ix]=((m>=q[None,:,1]) & (m<=q[None,:,3])).mean(axis=1)
        result['reference_agreement95'][ix]=((m>=q[None,:,0]) & (m<=q[None,:,4])).mean(axis=1)
    return result
