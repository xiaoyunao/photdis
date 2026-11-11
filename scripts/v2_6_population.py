#!/usr/bin/env python3
"""Cache-preserving observed-catalog inference and restricted response controls."""
import json
import os
import itertools
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from v2_1_near import ROOT,load,subset,sha,save_json,csv_rows,atomic_npz
from v2_5_population import draw,response,populations
from baseline_v1_4_experiments import arrays
from photdis.audit import rvs_metallicity
from photdis.stage1 import require_compute
from photdis.population_v2_5 import (NCELL,COLOR_EDGES,DISTANCES,DISTANCE_PRIOR,
    observation_cell,component_weights,posterior_grid,weighted_quantile,selected_probabilities)
from photdis.population_v2_6 import latent_counts,response_logposterior

D=ROOT/'data/v2_6';T=D/'tables';OLD=ROOT/'data/v2_5'
DESIGNS=('reference_like','ordinary_dominated','giant_rich')


def interval(values,w):
    return np.array([weighted_quantile(v,w)[[0,2]] for v in values.T])


def count_records(cat,i,joint,counts,method,mc):
    u=cat['catalog']==i;rows=[]
    for target,vals,truth in [('distance',counts.reshape(-1,4,5).sum(axis=1),np.bincount(cat['distance_bin'][u],minlength=5)),
                              ('group',counts.reshape(-1,4,5).sum(axis=2),np.bincount(cat['group'][u],minlength=4))]:
        mean=joint.sum(axis=0).reshape(4,5).sum(axis=0 if target=='distance' else 1)
        lo,hi=np.quantile(vals,[.025,.975],axis=0)
        for k in range(len(mean)):
            rows.append(dict(catalog=int(i),design=str(cat['design'][u][0]),regime=int(cat['regime'][u][0]),method=method,target=target,bin=k,
                n=int(u.sum()),realized_count=int(truth[k]),mean_count=mean[k],low=lo[k],high=hi[k],covered=bool(lo[k]<=truth[k]<=hi[k]),
                marginal_half_delta=mc,draw_mean_error=float(abs(vals.mean(axis=0)-mean).max())))
    return rows


def cached(cat,base):
    records=[];ends=[];reach=[]
    for regime in (0,1):
        p=load(OLD/f'posterior_regime{regime}.npz');eta=component_weights(p['theta'],base)
        values=np.column_stack((p['selected'].reshape(-1,4,5).sum(axis=2),p['selected'].reshape(-1,4,5).sum(axis=1)))
        for k,i in enumerate(p['catalog']):
            u=cat['catalog']==i;cells=observation_cell(cat['mags'][u]);rng=np.random.default_rng(261100+int(i))
            ix=rng.choice(len(eta),8192,p=p['weights'][k]);joint,counts,mc=latent_counts(eta[ix],p['response'],cells,seed=261200+int(i))
            atomic_npz(D/'observed'/f'{i:03}.npz',identity=cat['identity'][u],TYPE=cat['TYPE'][u],rows=np.flatnonzero(u),joint=joint,
                count_draws=counts,count_covariance=np.cov(counts.T),theta_index=ix)
            records+=count_records(cat,i,joint,counts,'fixed_response',mc)
            full=interval(values,p['weights'][k]);n=len(eta)//2;half=interval(values[:n],p['weights'][k,:n])
            for j in range(9):ends.append(dict(catalog=int(i),design=str(cat['design'][u][0]),regime=regime,target='group' if j<4 else 'distance',bin=j if j<4 else j-4,
                low=full[j,0],high=full[j,1],half_low=half[j,0],half_high=half[j,1],max_delta=float(abs(full[j]-half[j]).max())))
        # At fixed tilt, extrema of the ratio in each independent group odds
        # coordinate lie at its bounds. Reduce to eight scalar functions.
        alpha=p['response'].sum(axis=1).reshape(4,5)
        for b in range(5):
            minima=[];maxima=[]
            for corner in itertools.product((-3.,3.),repeat=3):
                def fraction(tilt):
                    m=component_weights(np.array([*corner,tilt]),base).reshape(4,5)*alpha
                    return float(m[:,b].sum()/m.sum())
                grid=np.linspace(-3,3,121);v=np.array([fraction(x) for x in grid])
                for sign,out in [(1,minima),(-1,maxima)]:
                    candidates=[(sign*v[0],grid[0]),(sign*v[-1],grid[-1])]
                    for j in range(1,len(grid)-1):
                        if sign*v[j]<=min(sign*v[j-1],sign*v[j+1]):
                            r=minimize_scalar(lambda z:sign*fraction(z),bounds=(grid[j-1],grid[j+1]),method='bounded')
                            candidates.append((r.fun,r.x))
                    out.append(sign*min(candidates)[0])
            reach.append(dict(regime=regime,bin=b,minimum=min(minima),maximum=max(maxima),
                grid_min=float(values[:,4+b].min()),grid_max=float(values[:,4+b].max()),status='corner reduction plus scalar numerical extrema; not interval-certified'))
        print('CACHED JOINT',regime,flush=True)
    csv_rows(T/'observed_counts.csv',records);csv_rows(T/'interval_endpoints.csv',ends);csv_rows(T/'reachable_range.csv',reach)


def controls(c,test,base,bins,curve,cat):
    r=load(OLD/'posterior_regime0.npz')['response'];pops=populations(base);rng=np.random.default_rng(261113)
    counts=[];meta=[];latents=[];tr=response(test,bins,curve,seed=251114)[0]
    # Independent same-library response supplies truth, with 8 seeds to reduce MC.
    same_truth=np.mean([response(c,bins,curve,seed=261120+j)[0] for j in range(8)],axis=0)
    for mode in ('known_response','same_template','independent_transfer'):
        for design in DESIGNS:
            group,tilt=pops[design];dp=DISTANCE_PRIOR*(DISTANCES/5)**tilt;dp/=dp.sum();eta=(np.array(group)[:,None]*dp).ravel()
            rt=r if mode=='known_response' else (same_truth if mode=='same_template' else tr)
            truth=eta*rt.sum(axis=1);truth/=truth.sum()
            for repeat in range(12):
                if mode=='known_response':
                    flat=eta[:,None]*r;flat/=flat.sum();z=rng.choice(flat.size,128,p=flat.ravel());cells=z%NCELL;latent=z//NCELL
                elif mode=='same_template':
                    cells=[];latent=[]
                    while len(cells)<128:
                        g=rng.choice(4,512,p=group);d=rng.choice(5,512,p=dp);m,e,s,ix=draw(c,g,d,rng,bins,curve)
                        take=np.flatnonzero(s)[:128-len(cells)];cells.extend(observation_cell(m[take]));latent.extend(5*g[take]+d[take])
                    cells=np.array(cells);latent=np.array(latent)
                else:
                    u=(cat['design']==design)&(cat['regime']==0)&(cat['repeat']==repeat)
                    cells=observation_cell(cat['mags'][u]);latent=5*cat['group'][u]+cat['distance_bin'][u]
                counts.append(np.bincount(cells,minlength=NCELL));latents.append(np.bincount(latent,minlength=20))
                meta.append((mode,design,repeat,truth))
    theta,selected,w=posterior_grid(counts,r,base,power=18);dv=selected.reshape(-1,4,5).sum(axis=1);rows=[]
    for k,(mode,design,rep,truth) in enumerate(meta):
        target=truth.reshape(4,5).sum(axis=0);mean=w[k]@dv;ci=interval(dv,w[k]);small=interval(dv[:len(dv)//2],w[k,:len(dv)//2])
        for j in range(5):rows.append(dict(mode=mode,design=design,repeat=rep,bin=j,truth=target[j],mean=mean[j],low=ci[j,0],high=ci[j,1],
            covered=bool(ci[j,0]<=target[j]<=ci[j,1]),endpoint_delta=float(abs(ci[j]-small[j]).max()),ess=float(1/(w[k]@w[k]))))
    atomic_npz(D/'control_counts.npz',counts=np.array(counts),latent=np.array(latents));csv_rows(T/'response_controls.csv',rows)
    print('CONTROLS COMPLETE',flush=True)


def template_pool(testids):
    t=load(ROOT/'data/v2_1/development.npz');fit=load(ROOT/'data/v2_3/training.npz')['identity']
    feh,valid=rvs_metallicity(t['rvs_feh'],t['rvs_teff'],t['rvs_logg']);quality=valid&t['rvs_parameter_quality']&np.isfinite(t['rvs_feh_err'])&(t['rvs_feh_err']>0)
    base=(t['split']==1)&(t['TYPE']=='PSF')&np.isfinite(t['ref_chosen'])&(t['optical_error']<.02)&~np.isin(t['identity'],np.r_[fit,testids])
    masks=[base&(t['class_label']==0)&quality&(feh>=-.5),base&(t['class_label']==0)&quality&(feh< -1.5),base&(t['class_label']==1),base&(t['class_label']==2),base&(t['class_label']<0)]
    pools=[]
    for m in masks:
        rows=np.flatnonzero(m)
        with np.errstate(over='ignore'):rank=t['identity'][rows].astype('u8')*np.uint64(11400714819323198485)
        pools.append(rows[np.argsort(rank,kind='stable')])
    output=[]
    for quotas in ([64,64,48,48,32],[192,192,144,144,96],[512,512,384,384,256]):
        rr=np.concatenate([r[:n] for r,n in zip(pools,quotas)]);labels=np.concatenate([np.full(min(len(r),n),k) for k,(r,n) in enumerate(zip(pools,quotas))]);q=subset(t,rr)
        m,e,ids=arrays(q,np.arange(len(rr)));c=dict(identity=ids,TYPE=q['TYPE'],absolute=m[:,:3]-q['ref_chosen'][:,None],error=e[:,:3],sigma=q['sigma_mu'],label_available=labels<4)
        feature=np.column_stack((c['absolute'][:,0]-c['absolute'][:,1],c['absolute'][:,1]-c['absolute'][:,2],c['absolute'][:,0]));known=labels<4
        member=np.eye(4)[np.minimum(labels,3)];kernel=np.exp(-.5*cdist(feature[~known]/[.2,.2,1],feature[known]/[.2,.2,1])**2)+1e-20
        soft=kernel@member[known];member[~known]=soft/soft.sum(axis=1)[:,None];c['member']=member;output.append(c)
    save_json(D/'template_admission.json',dict(eligible=[int(m.sum()) for m in masks],sizes=[len(c['identity']) for c in output],nonPSF=int(((t['split']==1)&(t['TYPE']!='PSF')).sum()),missing_TYPE=int(((t['split']==1)&(t['TYPE']=='')).sum()),fit_overlap=0,test_overlap=0))
    return output


def source_responses(c,bins,curve):
    n=len(c['identity']);full=np.zeros((n,5,NCELL));half=full.copy();rng=np.random.default_rng(261130)
    for start in range(0,n,64):
        stop=min(start+64,n);ix=np.repeat(np.arange(start,stop),5*256);d=np.tile(np.repeat(np.arange(5),256),stop-start)
        clean=c['absolute'][ix]+(10+5*np.log10(DISTANCES[d]))[:,None];slot=np.clip(np.searchsorted(bins,clean[:,1])-1,0,len(curve)-1)
        e=np.maximum(curve[slot],c['error'][ix]);m=clean+rng.normal(size=clean.shape)*e;sel=(m[:,1]>=14)&(m[:,1]<20)&(e.max(axis=1)<.1);cell=observation_cell(m)
        flat=(ix*5+d)*NCELL+cell;v=np.bincount(flat[sel],minlength=n*5*NCELL).reshape(n,5,NCELL)/256
        full+=v;take=sel&(np.tile(np.arange(256),len(ix)//256)<128)
        half+=np.bincount(flat[take],minlength=n*5*NCELL).reshape(n,5,NCELL)/128
    return full,half


def aggregate(c,source,w=None):
    w=np.ones(len(c['identity'])) if w is None else w
    member=c['member']*w[:,None];member/=member.sum(axis=0)[None,:]
    r=np.einsum('ng,ndb->gdb',member,source).reshape(20,NCELL)
    ri=np.unravel_index(np.arange(NCELL),tuple(len(e)-1 for e in COLOR_EDGES))[-1];allowed=(COLOR_EDGES[-1][ri]>=14)&(COLOR_EDGES[-1][ri+1]<=20)
    return (1-1e-4)*r+1e-4*r.sum(axis=1)[:,None]*allowed[None,:]/allowed.sum()


def uncertain(test,base,bins,curve,cat):
    pools=template_pool(test['identity']);shape=[];residual=[];responses=[];convergence=[];tr=response(test,bins,curve,seed=251114)[0]
    ids=np.array([i for i in np.unique(cat['catalog']) if cat['regime'][cat['catalog']==i][0]==0 and cat['design'][cat['catalog']==i][0] in DESIGNS])
    cc=np.array([np.bincount(observation_cell(cat['mags'][cat['catalog']==i]),minlength=NCELL) for i in ids]);theta=6*qmc.Sobol(4,scramble=True,seed=20261111).random_base2(16)-3;eta=component_weights(theta,base)
    for scale,c in enumerate(pools):
        assert (c['TYPE']=='PSF').all() and not np.intersect1d(c['identity'],test['identity']).size
        atomic_npz(D/f'templates_{scale}.npz',**c);source,half=source_responses(c,bins,curve);r=aggregate(c,source);rh=aggregate(c,half);responses.append(r)
        for label,q in [('cal',c),('test',test)]:
            member=q.get('member',np.eye(4)[q.get('group',np.zeros(len(q['identity']),int))])
            for g in (1,3):
                for feature,v in [('gr',q['absolute'][:,0]-q['absolute'][:,1]),('rz',q['absolute'][:,1]-q['absolute'][:,2]),('Mg',q['absolute'][:,0])]:
                    lo,med,hi=weighted_quantile(v,member[:,g]);shape.append(dict(scale=scale,library=label,group=g,feature=feature,n=len(v),effective_sources=float(member[:,g].sum()**2/np.sum(member[:,g]**2)),low=lo,median=med,high=hi))
        for g in (1,3):
            for d in range(5):
                j=5*g+d
                for b in range(NCELL):residual.append(dict(scale=scale,group=g,distance_bin=d,cell=b,cal_response=r[j,b],test_response=tr[j,b],difference=r[j,b]-tr[j,b]))
        ww,_,_=response_logposterior(cc,eta,[r,rh]);sv=eta*r.sum(axis=1);sv/=sv.sum(axis=1)[:,None];hv=eta*rh.sum(axis=1);hv/=hv.sum(axis=1)[:,None]
        for k,i in enumerate(ids):
            mean=(ww[0,k]@sv).reshape(4,5).sum(axis=0);hmean=(ww[1,k]@hv).reshape(4,5).sum(axis=0)
            for b in range(5):convergence.append(dict(scale=scale,sources=len(c['identity']),catalog=int(i),design=str(cat['design'][cat['catalog']==i][0]),bin=b,mean=mean[b],mc_half_mean=hmean[b]))
        if scale==2:
            rng=np.random.default_rng(261140);rr=np.array([aggregate(c,source,rng.exponential(size=len(c['identity']))) for _ in range(16)])
            atomic_npz(D/'response_prior.npz',responses=rr,source_response=source,identity=c['identity'],TYPE=c['TYPE'])
        print('SOURCE SCALE',scale,len(c['identity']),flush=True)
    w,rw,z=response_logposterior(cc,eta,rr);atomic_npz(D/'uncertain_grid.npz',theta=theta,catalog=ids,response_weight=rw,theta_weight=w,log_evidence=z)
    records=[];pop=[];diag=[];rng=np.random.default_rng(261150)
    for k,i in enumerate(ids):
        u=cat['catalog']==i;cells=observation_cell(cat['mags'][u]);ri=rng.choice(16,8192,p=rw[k]);joints=[];draws=[];selected_draws=[];deltas=[]
        for l in range(16):
            n=int((ri==l).sum())
            if not n:continue
            ix=rng.choice(len(theta),n,p=w[l,k]);joint,counts,mc=latent_counts(eta[ix],rr[l],cells,seed=261160+int(i)*16+l)
            joints.append(joint*n);draws.append(counts);deltas.append(mc)
            ss=eta[ix]*rr[l].sum(axis=1);selected_draws.append(ss/ss.sum(axis=1)[:,None])
        joint=sum(joints)/8192;counts=np.concatenate(draws);ss=np.concatenate(selected_draws).reshape(-1,4,5)
        atomic_npz(D/'uncertain_observed'/f'{i:03}.npz',identity=cat['identity'][u],TYPE=cat['TYPE'][u],joint=joint,count_draws=counts,count_covariance=np.cov(counts.T))
        records+=count_records(cat,i,joint,counts,'response_marginalized',max(deltas))
        for b in range(5):
            v=ss.sum(axis=1)[:,b];lo,hi=np.quantile(v,[.025,.975]);pop.append(dict(catalog=int(i),design=str(cat['design'][u][0]),bin=b,mean=v.mean(),low=lo,high=hi))
        # Evidence convergence in theta quadrature and finite response-prior draws.
        zhalf=[];means=[];halves=[]
        from scipy.special import logsumexp
        for l in range(16):
            nh=len(theta)//2;mass=w[l,k,:nh].sum();zhalf.append(z[k,l]+np.log(mass*2))
            sv=eta*rr[l].sum(axis=1);sv/=sv.sum(axis=1)[:,None];means.append(w[l,k]@sv);halves.append(w[l,k,:nh]@sv[:nh]/mass)
        rhalf=np.exp(zhalf-logsumexp(zhalf));mean=rw[k]@means;half=rhalf@halves;eight=rw[k,:8]@np.array(means[:8])/rw[k,:8].sum()
        diag.append(dict(catalog=int(i),design=str(cat['design'][u][0]),response_ess=float(1/(rw[k]@rw[k])),max_response_weight=float(rw[k].max()),
            joint_theta_ess=float(1/np.sum((rw[k,:,None]*w[:,k])**2)),theta_half_max_delta=float(abs(mean-half).max()),response_half_max_delta=float(abs(mean-eight).max())))
    csv_rows(T/'template_shapes.csv',shape);csv_rows(T/'response_cell_residuals.csv',residual);csv_rows(T/'nested_response.csv',convergence)
    csv_rows(T/'uncertain_counts.csv',records);csv_rows(T/'uncertain_population.csv',pop);csv_rows(T/'response_integration.csv',diag)
    print('UNCERTAINTY COMPLETE',flush=True)


def main():
    require_compute();T.mkdir(parents=True,exist_ok=True)
    files=[ROOT/'docs/v2_6_protocol.md',ROOT/'scripts/v2_6_population.py',ROOT/'src/photdis/population_v2_6.py',OLD/'catalogs.npz',OLD/'posterior_regime0.npz',OLD/'posterior_regime1.npz',ROOT/'data/v2_4/prospective_identity_reservation.npz']
    freeze={str(p.relative_to(ROOT)):sha(p) for p in files};marker=D/'population_freeze.json'
    if marker.exists():assert json.loads(marker.read_text())==freeze
    else:save_json(marker,freeze)
    meta=json.loads((OLD/'templates.json').read_text());base=np.array(meta['base']);bins=np.array(meta['noise_bins']);curve=np.array(meta['noise_curve'])
    cat=load(OLD/'catalogs.npz');c=load(OLD/'cal_templates.npz');test=load(OLD/'test_templates.npz');assert (cat['TYPE']=='PSF').all()
    for name,fn in [('cached',lambda:cached(cat,base)),('controls',lambda:controls(c,test,base,bins,curve,cat)),('uncertain',lambda:uncertain(test,base,bins,curve,cat))]:
        done=D/(name+'_complete.json')
        if not done.exists():fn();save_json(done,dict(job=os.getenv('SLURM_JOB_ID'),completed=True))
    assert all(sha(ROOT/k)==v for k,v in freeze.items())
    print('V2.6 POPULATION COMPLETE',flush=True)


if __name__=='__main__':main()
