#!/usr/bin/env python3
"""Same-source scales and proper scores with shared observational bootstrap."""
from itertools import combinations
from seq7_common import *
from seq7_predict import load
from seq7_math import row_interp
from stream_sequence import count_median
from photdis.stage1 import require_compute


def CI(v):
    v=np.asarray(v);return np.quantile(v[np.isfinite(v)],[.025,.975]) if np.isfinite(v).any() else np.array([np.nan,np.nan])


def weighted_mean(v,w):
    total=w.sum(axis=0);return np.divide((v*w).sum(axis=0),total,out=np.full(total.shape,np.nan),where=total>0)


def measure(z,ii,t,x,mult,shift):
    # ii indexes the global frozen roster; z has an independent explicit row map.
    j=np.searchsorted(z['ii'],ii);assert (z['ii'][j]==ii).all();qq=z['q'][j];ref=10+5*np.log10(t['ref'][ii]);res=qq[:,2]-ref;center=float(np.median(res));w=mult[:,ii].T
    cb=np.array([count_median(res-ss,w[:,k]) for k,ss in enumerate(shift)])
    # A full-parent bootstrap can contain no member of a sparse diagnostic bin.
    # Evaluate a harmless finite placeholder, but omit that empty draw in every CI.
    draw_center=np.where(np.isfinite(cb),cb,center-shift)
    values={};boots={};per={};norm=np.sum(w,axis=0)
    values.update(N=len(ii),median_mu_residual=center,NMAD=float(1.4826*np.median(abs(res-center))),scale_target_met=abs(center)<=.1,median_width68=float(np.median(qq[:,3]-qq[:,1])),median_width95=float(np.median(qq[:,4]-qq[:,0])),median_left68=float(np.median(qq[:,2]-qq[:,1])),median_right68=float(np.median(qq[:,3]-qq[:,2])),median_left95=float(np.median(qq[:,2]-qq[:,0])),median_right95=float(np.median(qq[:,4]-qq[:,2])))
    values['center_lo'],values['center_hi']=CI(cb);boots['center']=cb
    nb=np.array([1.4826*count_median(abs(res-shift[k]-cb[k]),w[:,k]) for k in range(len(shift))]);values['NMAD_lo'],values['NMAD_hi']=CI(nb)
    for state,point,draw,med in [('raw',ref,ref[:,None]+shift[None],0),('centered',ref+center,ref[:,None]+shift[None]+draw_center[None],center)]:
        for label,key in [('logscore','logpdf'),('CRPS','crps'),('CDF','cdf')]:
            pv=row_interp(x,point,z[key][j],cubic=label=='logscore');bv=row_interp(x,draw,z[key][j],cubic=label=='logscore');per[state+'_'+label]=pv
            if label=='CDF':
                for level,lo,hi in [('68',.16,.84),('95',.025,.975)]:
                    score=((pv>=lo)&(pv<=hi)).astype(float);boot=weighted_mean(((bv>=lo)&(bv<=hi)).astype(float),w);name=state+'_convolved_C'+level;values[name]=float(score.mean());values[name+'_lo'],values[name+'_hi']=CI(boot);boots[name]=boot
                continue
            name=state+'_'+label;boot=weighted_mean(bv,w);values[name]=float(pv.mean());values[name+'_lo'],values[name+'_hi']=CI(boot);boots[name]=boot
        for level,lo,hi in [('68',1,3),('95',0,4)]:
            pv=(qq[:,lo]-med<=ref)&(ref<=qq[:,hi]-med);dv=(qq[:,lo,None]<=draw)&(draw<=qq[:,hi,None]);name=state+'_center_C'+level;values[name]=float(pv.mean());bv=weighted_mean(dv.astype(float),w);values[name+'_lo'],values[name+'_hi']=CI(bv);boots[name]=bv
    values.update(post_mean=float(np.nanmean(z['post'][j])) if np.isfinite(z['post'][j]).any() else np.nan,r_min=float(t['r'][ii].min()),r_max=float(t['r'][ii].max()),Mr_min=float((t['m'][ii,1]-ref).min()),Mr_max=float((t['m'][ii,1]-ref).max()),gr_min=float((t['m'][ii,0]-t['m'][ii,1]).min()),gr_max=float((t['m'][ii,0]-t['m'][ii,1]).max()))
    if np.isfinite(z['post'][j]).all():
        pb=weighted_mean(z['post'][j,None],w);values['post_mean_lo'],values['post_mean_hi']=CI(pb);boots['post_mean']=pb
    for metric in ('logscore','CRPS'):
        key='centered_minus_raw_'+metric;values[key]=values['centered_'+metric]-values['raw_'+metric];delta=boots['centered_'+metric]-boots['raw_'+metric];values[key+'_lo'],values[key+'_hi']=CI(delta);boots[key]=delta
    values['bootstrap_valid_draws']=int((w.sum(axis=0)>0).sum())
    return values,boots,per


def main():
    require_compute();d=folder('evaluate');inputs={'curves':sha(OUT/'curves/complete.json')};t=load(OUT/'prepare/roster.npz');x=load(OUT/'curves/grid.npz')['mu'];summary=[];paired=[];perstar=[];gaps=[];bootrows=[]
    for s in C['systems']:
        rows=np.flatnonzero(t['system']==s);shift=5*np.log10(t['scale_'+s]);z={m:load(OUT/'curves'/f'{s}_{m}.npz') for m in METHODS};supported={m:z[m]['ii'][z[m]['support']] for m in METHODS};bs={}
        common=rows[t['v5'][rows]]
        for m in METHODS[:-1]:common=np.intersect1d(common,supported[m])
        for cohort,base,mult in [('primary',rows,t['mult']),('v5',rows[t['v5'][rows]],t['v5_mult']),('v5_commonABCDE',common,t['v5_mult'])]:
            groups=[(phase,base if phase=='all' else base[t['phase'][base]==phase]) for phase in ('all','lower_MS','turnoff','other')]
            if cohort=='primary':
                gr=t['m'][base,0]-t['m'][base,1];mg=t['m'][base,0]-(10+5*np.log10(t['ref'][base]))
                for label,v,edges in [('gr',gr,C['diagnostic_gr_edges']),('Mg',mg,C['diagnostic_Mg_edges'])]:
                    groups.extend((f'{label}_{lo:g}_{hi:g}',base[(v>=lo)&(v<hi)]) for lo,hi in zip(edges[:-1],edges[1:]))
            for phase,parent in groups:
                if not len(parent):continue
                for m in METHODS:
                    ii=np.intersect1d(parent,supported[m]);baseinfo=dict(system=s,cohort=cohort,phase=phase,method=m,parent_N=len(parent),supported_N=len(ii),support_fraction=len(ii)/len(parent))
                    if not len(ii):summary.append(baseinfo);continue
                    val,b,p=measure(z[m],ii,t,x,mult,shift);summary.append(dict(**baseinfo,**val));bs[cohort,phase,m]=b
                    if phase in ('all','lower_MS','turnoff','other'):
                        for k,i in enumerate(ii):perstar.append(dict(system=s,cohort=cohort,phase=phase,method=m,ls_id=int(t['ls_id'][i]),median_residual=float(z[m]['q'][np.searchsorted(z[m]['ii'],i),2]-10-5*np.log10(t['ref'][i])),**{key:float(v[k]) for key,v in p.items()}))
                if phase not in ('all','lower_MS','turnoff','other'):continue
                for a,b in combinations(METHODS,2):
                    ii=np.intersect1d(parent,np.intersect1d(supported[a],supported[b]));info=dict(system=s,cohort=cohort,phase=phase,first=a,second=b,parent_N=len(parent),N=len(ii))
                    if not len(ii):paired.append(info);continue
                    va,ba,_=measure(z[a],ii,t,x,mult,shift);vb,bb,_=measure(z[b],ii,t,x,mult,shift)
                    for state in ('raw','centered'):
                        for metric in ('logscore','CRPS'):
                            key=state+'_'+metric;dd=bb[key]-ba[key];lo,hi=CI(dd);info.update({key+'_delta':vb[key]-va[key],key+'_lo':lo,key+'_hi':hi})
                    info['center_delta']=vb['median_mu_residual']-va['median_mu_residual'];info['center_delta_lo'],info['center_delta_hi']=CI(bb['center']-ba['center']);paired.append(info)
            for m in METHODS:
                a=bs.get((cohort,'lower_MS',m));b=bs.get((cohort,'turnoff',m))
                if a and b:
                    lookup=[r for r in summary if r.get('system')==s and r.get('cohort')==cohort and r.get('method')==m];aa=next(r for r in lookup if r['phase']=='lower_MS');bb=next(r for r in lookup if r['phase']=='turnoff');lo,hi=CI(b['center']-a['center']);gaps.append(dict(system=s,cohort=cohort,method=m,turnoff_minus_lowerMS=bb['median_mu_residual']-aa['median_mu_residual'],lo=lo,hi=hi,N_lower=aa['N'],N_TO=bb['N']))
        for (cohort,phase,m),data in bs.items():
            if phase in ('lower_MS','turnoff'):
                for k in range(256):bootrows.append(dict(system=s,cohort=cohort,phase=phase,method=m,replicate=k,**{key:float(v[k]) for key,v in data.items()}))
        print('Evaluated',s,flush=True)
    csv_rows(d/'summary.csv',summary);csv_rows(d/'paired.csv',paired);csv_rows(d/'phase_gaps.csv',gaps);csv_rows(d/'per_star_scores.csv',perstar);csv_rows(d/'shared_bootstrap_summary.csv',bootrows)
    complete(d,inputs,score='mean per-star marginal predictive log density, Gaussian reference convolution, modulus coordinate; not joint cluster likelihood',bootstrap='shared whole-parent or original-v5 multinomial, one system-common observed reference draw; same convolved kernel, not extra posterior broadening',centering='diagnostic evaluated-group median recomputed for every shared replicate',CRPS='integral of convolved CDF squared against observation step; lower is better');print('Evaluation complete',flush=True)


if __name__=='__main__':main()
