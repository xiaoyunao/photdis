#!/usr/bin/env python3
"""A fixed coarse color–luminosity baseline on the production training identities."""
import joblib
from scipy.spatial import cKDTree
from photdis.stage1 import require_compute
from photdis.baseline_v1 import qmc_noise, shift_pdf, quantiles_interpolated
from photdis.v3.controls import optical_magnitudes
from sv18_common import *
from final_catalogue_benchmark import load_parent, masks, published, parent
from release_readiness_compare import available
from v3_9_catalog import OUT as V39, from_north
from v3_9_references import join
from photdis.gaia import join_gaia, derive_gaia_labels
from v3_4_near_validate import GAIA


def training(side,inputs):
    if side=='south':
        a=read(ROOT/'data/v3_0/controls',('training.npz','TYPE','identity','rows','Mg','class_label'),inputs)
        b=read_columns(ROOT/'data/v3_0/references.npz',('TYPE','ls_id','source_id','flux','flux_error','band_mask','transmission','role'))
        j=a['rows']
        assert (a['TYPE']=='PSF').all() and (b['TYPE'][j]=='PSF').all()
        mag,_=optical_magnitudes(b)
        return dict(TYPE=a['TYPE'],m=mag[j],y=a['Mg'],ls_id=b['ls_id'][j],source_id=b['source_id'][j],class_label=a['class_label']),0.
    role=read(V39/'north/native_model',('roles.npz','TYPE','ls_id','role'),inputs)
    use=np.isin(role['role'],('gaia_fit','specdis_fit'))
    ids=role['ls_id'][use];typ=role['role'][use]
    chunks=[]
    for p in sorted((V39/'north/reference').glob('[0-9][0-9].json')):
        r=json.loads(p.read_text())
        inputs[str(p.relative_to(ROOT))]=sha(p)
        for name,h in r['outputs'].items():
            if name.endswith('.npz'):
                t=read_columns(verified(p.with_name(name),h),
                    ('TYPE','ls_id','source_id','mag_grz','reference_kpc','specdis_kpc','class_label'))
                j=index_join(t['ls_id'],ids);k=j>=0
                if k.any():
                    z=j[k];d=np.where(typ[z]=='gaia_fit',t['reference_kpc'][k],t['specdis_kpc'][k])
                    assert (t['TYPE'][k]=='PSF').all()
                    chunks.append(dict(TYPE=t['TYPE'][k],ls_id=t['ls_id'][k],source_id=t['source_id'][k],
                                       m=t['mag_grz'][k],y=t['mag_grz'][k,0]-10-5*np.log10(d),class_label=t['class_label'][k]))
    t=join(chunks)
    assert len(t['ls_id'])==len(ids) and len(np.unique(t['ls_id']))==len(ids)
    model,hd=from_north();inputs['north_qrf']=hd
    return t,float(model['mu_offset'])


def fit(t):
    """Empirical Mg histograms in fixed .10-mag color cells, four-cell smoothing."""
    x=t['m'][:,:2]-t['m'][:,1:3]
    k=np.floor(x/P['simple_baseline']['color_bin_width']).astype('i4')
    cells,inv=np.unique(k,axis=0,return_inverse=True)
    counts=np.bincount(inv)
    support=np.flatnonzero(counts>=P['simple_baseline']['minimum_cell_training'])
    centers=(cells[support]+.5)*P['simple_baseline']['color_bin_width']
    edges=np.arange(np.floor(t['y'].min())-1,np.ceil(t['y'].max())+1.01,.1)
    h=np.array([np.histogram(t['y'][inv==j],edges)[0] for j in support],float)
    h/=h.sum(axis=1)[:,None]
    return cKDTree(centers),h,edges,counts[support],centers


def predict(model,m,e,ids,offset):
    tree,h,edges,counts,centers=model
    pdf=np.zeros((len(m),len(edges)-1))
    for draw in range(P['simple_baseline']['noise_draws']):
        delta=qmc_noise(ids,draw)[:,:3]*e
        pert=m+delta
        x=pert[:,:2]-pert[:,1:3]
        dist,ind=tree.query(x,k=np.arange(1,min(4,len(h))+1))
        w=1/np.maximum(dist,.025)**2
        w/=w.sum(axis=1)[:,None]
        p=np.sum(h[ind]*w[:,:,None],axis=1)
        pdf+=shift_pdf(p,delta[:,0],edges)
    pdf/=P['simple_baseline']['noise_draws']
    qm=quantiles_interpolated(pdf,edges)
    q=10**((m[:,:1]-qm[:,::-1]-10+offset)/5)
    dist,ind=tree.query(m[:,:2]-m[:,1:3],k=1)
    return q,dist,counts[ind]


def comparison(t,qs,groups,dist,block,d,frame):
    rows=[];delta=[]
    for name,base in groups.items():
        use=base&np.isfinite(dist)&(dist>0)
        # Same eligible objects for every method.
        use &= np.logical_and.reduce([np.isfinite(q).all(axis=1)&(q>0).all(axis=1) for q in qs.values()])
        if not use.any():
            continue
        idx=np.flatnonzero(use); bb=block[idx];keys=np.unique(bb)
        values={k:metric(q[idx],dist[idx]) for k,q in qs.items()}
        y=10+5*np.log10(dist[idx])
        for method,q in qs.items():
            mu=10+5*np.log10(q[idx])
            width=mu[:,3]-mu[:,1]
            score=width+(2/.32)*(np.maximum(mu[:,1]-y,0)+np.maximum(y-mu[:,3],0))
            values[method]['median_width68_mu']=np.median(width)
            values[method]['mean_interval_score68_mu']=np.mean(score)
        if 'BJ' in values:
            values['BJ']['c95']=None
        for k,v in values.items():
            rows.append(dict(frame=frame,group=name,method=k,sky_blocks=len(keys),**v))
        estimates=[]
        if len(keys)>1:
            inv=np.searchsorted(keys,bb)
            lists=[np.flatnonzero(inv==k) for k in range(len(keys))]
            for draw in range(P['bootstrap_draws']):
                pick=RNG.integers(len(keys),size=len(keys))
                jj=np.concatenate([lists[k] for k in pick]);ii=idx[jj]
                met={k:metric(q[ii],dist[ii]) for k,q in qs.items()}
                for other in ('Simple','BJ'):
                    if other not in qs:
                        continue
                    estimates.append(dict(draw=draw,other=other,
                       **{key:met['Photdis'][key]-met[other][key] for key in ('bias','nmad','severe','c68')}))
        for other in ('Simple','BJ'):
            if other not in qs:
                continue
            for key in ('bias','nmad','severe','c68'):
                draws=[r[key] for r in estimates if r['other']==other]
                delta.append(dict(frame=frame,group=name,other=other,statistic=key,n=len(idx),
                    sky_blocks=len(keys),difference=values['Photdis'][key]-values[other][key],**interval(draws)))
    csv_rows(d/(frame+'_metrics.csv'),rows)
    csv_rows(d/(frame+'_paired_ci.csv'),delta)
    return rows


def main():
    require_compute()
    inputs={};d=out('benchmark')
    keys=('TYPE','ls_id','source_id','RA','DEC','region','m','err','observed_r','heldout','state','pop','physical',
          'quality','gaia_state','p_corr','p_err','q','specdis_stage','specdis_d','spdist_stage','spdist_d')
    t=cached(inputs,keys)
    keep=t['heldout']&(t['state']==3)&(t['observed_r']>=18)&(t['observed_r']<20)
    keep &= ((t['specdis_stage']==3)|(t['spdist_stage']==3))&np.isin(t['pop'],(1,2,3))
    t={k:v[keep] for k,v in t.items()}
    labels=derive_gaia_labels(join_gaia(t['source_id'],GAIA))
    models={};simple=np.full_like(t['q'],np.nan);near=np.full(len(simple),np.nan);nn=near.copy()
    histories={}
    for side,north in (('south',False),('north',True)):
        train,offset=training(side,inputs)
        atomic_npz(d/(side+'_training_sample.npz'),**train)
        model=fit(train); models[side]=(model,offset)
        histories[side]=dict(ls_id=train['ls_id'],source_id=train['source_id'])
        k=np.flatnonzero((t['region']==3)==north)
        for start in range(0,len(k),512):
            ii=k[start:start+512]
            simple[ii],near[ii],nn[ii]=predict(model,t['m'][ii],t['err'][ii],t['source_id'][ii],offset)
        atomic_npz(d/(side+'_simple_relation.npz'),centers=model[-1],hist=model[1],
                   Mg_edges=model[2],training_count=model[3],training_ls_id=train['ls_id'])
    v,vm=load_parent();pub,_,pmeta=published(v,vm['outputs']['ids.npy']);inputs.update(pmeta)
    cross=index_join(t['ls_id'],v['ls_id'])
    bq=np.full((len(cross),5),np.nan)
    j=cross>=0
    bq[j,1:4]=pub['BJ_photogeometric'][cross[j]]
    bq[:,0]=bq[:,1];bq[:,4]=bq[:,3]
    was=np.zeros((len(cross),2),bool)
    for i,ref in enumerate(('specdis','spdist')):
        oldmask=masks(v)['weak_gaia_dwarfs']&available(pub['Photdis'])&available(pub['BJ_photogeometric'])
        oldmask&=(v[ref+'_kpc']>=2)&(v[ref+'_kpc']<5)
        was[:,i]=np.isin(t['ls_id'],v['ls_id'][oldmask])
        assert was[:,i].sum()==(9355,8927)[i]
    block=sky(t)
    metrics=[]
    for i,ref in enumerate(('specdis','spdist')):
        dist=t[ref+'_d']
        good=(t[ref+'_stage']==3)&(dist>=2)&(dist<5)&np.isin(t['gaia_state'],(3,4,5))
        masks0={'headline':was[:,i],'expanded_common':good,
            'ordinary':was[:,i]&(t['pop']==1),'metal_poor':was[:,i]&(t['pop']==2),
            'north':was[:,i]&(t['region']==3),'south':was[:,i]&(t['region']!=3),
            'blue':was[:,i]&((t['m'][:,0]-t['m'][:,1])<.5),
            'red':was[:,i]&((t['m'][:,0]-t['m'][:,1])>=.5),
            'low_snr':was[:,i]&(t['p_corr']/t['p_err']<10),
            'weak_astrometry':was[:,i]&~labels['astro_quality'],
            'missing_zero_point':was[:,i]&~labels['zpt_valid']}
        metrics+=comparison(t,dict(Photdis=t['q'],Simple=simple,BJ=bq),masks0,dist,block,d,ref)
    replay=[]
    for i,ref in enumerate(('specdis','spdist')):
        dist=t[ref+'_d']
        base=(t[ref+'_stage']==3)&(dist>=2)&(dist<5)&np.isin(t['gaia_state'],(3,4,5))
        anchor=was[:,i]; active=base.copy()
        predicates=[('native_photometry',t['quality']==7),('qualified_reference',t[ref+'_stage']==3),
           ('weak_geometry',np.isin(t['gaia_state'],(3,4,5))),('spectroscopic_dwarf',np.isin(t['pop'],(1,2,3))),
           ('reference_2_5',(dist>=2)&(dist<5)),('historical_parent',cross>=0),
           ('BJ_available',np.isfinite(bq[:,2])),('original_F15',anchor)]
        for label,mask in predicates:
            before=active.copy();active&=mask
            removed=before&~active;added=anchor&~active
            replay.append(dict(reference=ref,step=label,n=int(active.sum()),removed=int(removed.sum()),
                original_missing=int(added.sum()),extra=int((active&~anchor).sum()),**{
                k:v for k,v in metric(t['q'][active],dist[active]).items() if k!='n'}))
            # Full per-source deltas remain on NERSC.
            atomic_npz(d/(ref+'_'+label+'_delta.npz'),removed=t['ls_id'][removed],original_missing=t['ls_id'][added],
                       extra=t['ls_id'][active&~anchor],removed_TYPE=t['TYPE'][removed],
                       original_missing_TYPE=t['TYPE'][added],extra_TYPE=t['TYPE'][active&~anchor])
    csv_rows(d/'mask_replay.csv',replay)
    atomic_npz(d/'evaluated.npz',**t,simple=simple,bj=bq,was_f15=was,baseline_color_distance=near,
               baseline_neighbor_training_count=nn)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained',sharey=True)
        density=[]
        edges=np.linspace(-2,2,101)
        for i,(ax,ref) in enumerate(zip(axes,('specdis','spdist'))):
            for method,color in zip(('Photdis','Simple','BJ'),(BLUE,ORANGE,GREEN)):
                values={'Photdis':t['q'],'Simple':simple,'BJ':bq}[method]
                use=was[:,i]
                residual=5*np.log10(values[use,2]/t[ref+'_d'][use])
                counts=np.histogram(residual,edges)[0]
                hist=counts/(len(residual)*np.diff(edges))
                ax.stairs(hist,edges,color=color,lw=1,label=method)
                density.extend(dict(reference=ref,method=method,lo=a,hi=b,n=int(n),density=y,
                    total=len(residual),outside_plot=int(((residual<edges[0])|(residual>edges[-1])).sum()))
                    for a,b,n,y in zip(edges[:-1],edges[1:],counts,hist))
            ax.axvline(0,color='.6',lw=.7,ls=':')
            ax.set(xlabel=r'$\Delta\mu$ (mag)',title=REFERENCE_NAMES[ref],xlim=(-2,2))
        axes[0].set_ylabel('Probability density')
        axes[0].legend(fontsize=7,frameon=False)
        figure(d,'same_training_baseline',f,density)
    complete(d,inputs,model_scope='fixed coarse empirical relation on same source IDs, same labels and offset',
             methods='Photdis and BJ frozen, simple intervals are conditional training distributions')


if __name__=='__main__':
    main()
