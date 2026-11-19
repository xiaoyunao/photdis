#!/usr/bin/env python3
"""Shared-refit conditional scores and separately marginalized external PDFs."""
from nat8_common import *
from photdis.stage1 import require_compute
from nat8_statsmath import hist_curves,curve_values,direct_queries,sparse_CDF_components
from seq7_math import curve_crps,log_density_grid
from stream_sequence import count_median
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from scipy.special import ndtr
from itertools import combinations


def ci(v):
    v=np.asarray(v);v=v[~np.isnan(v)]
    return np.quantile(v,[.025,.975],method='inverted_cdf') if len(v) else [np.nan,np.nan]


def weighted(v,w):
    value=np.multiply(v,w,out=np.zeros_like(v,dtype=float),where=w>0);n=w.sum(axis=0);return np.divide(value.sum(axis=0),n,out=np.full(n.shape,np.nan),where=n>0)


@lru_cache(maxsize=12)
def cached(p):return load(p)


def source_scores(task):
    s,method,i,family_ids=task;t=cached(OUT/'prepare/roster.npz');z=cached(OUT/'prepare'/f'{s}_E.npz');k=int(np.flatnonzero(z['ii']==i)[0]);pred=cached(OUT/'predict'/f'{s}_calibrated.npz');draws=cached(OUT/'fit/parameter_draws.npz');centers=cached(OUT/'evaluate/group_centers.npz');params=json.loads((OUT/'fit/parameters.json').read_text());ref=float(10+5*np.log10(t['ref'][i]));shift=t['shift_'+s];ids=np.asarray(family_ids);c0=centers['point'][ids];cb=centers['boot'][ids].T;cb=np.nan_to_num(cb);y0=ref+np.r_[0,c0];yb=ref+shift[:,None]+np.column_stack((np.zeros(256),cb));crps0=None;ref_sigma=0. if s=='M5' else float(t['reference_sigma_'+s]);mm=z['means'][k];lw=z['logw'][k];sg=z['sigma'][k];xx=z['x']
    if method=='CalE':
        label=str(t['fold'][i]) if s=='M5' else 'full';a=np.array(params[label]['point']['a']);ab=draws[label+'_a'];point_mm=mm-a[0]-a[1]*xx;boot_mm=mm[None]-ab[:,0,None]-ab[:,1,None]*xx
        p0,l0=direct_queries(mm,lw,np.hypot(sg,ref_sigma),xx,a,y0[None]);pb,lb=direct_queries(mm,lw,np.hypot(sg,ref_sigma),xx,ab,yb);active,w=sparse_CDF_components(boot_mm,lw);gx,cf,cr,_=hist_curves(active,w,np.hypot(sg,ref_sigma),query=yb);crb=curve_values(gx,cr,yb)
        active,w=sparse_CDF_components(point_mm,lw);gx,cf,cr,_=hist_curves(active,w,np.hypot(sg,ref_sigma),query=y0);cr0=curve_values(gx,cr,y0[None])[0];p0=p0[0];l0=l0[0]
    elif method=='CalE_marg':
        from scipy.interpolate import CubicSpline
        ab=draws['full_a'];means=(mm[None]-ab[:,0,None]-ab[:,1,None]*xx).ravel();logw=np.tile(lw-np.log(len(ab)),len(ab));allquery=np.r_[y0,yb.ravel()];h=.00125;grid=np.arange(np.floor((allquery.min()-.01)/h)*h,np.ceil((allquery.max()+.01)/h)*h+h/2,h);lp=log_density_grid(grid,means,logw,np.hypot(sg,ref_sigma));interp=CubicSpline(grid,lp);l0=interp(y0);lb=interp(yb)
        atom,w=sparse_CDF_components(mm[None]-ab[:,0,None]-ab[:,1,None]*xx,lw);flat=atom.ravel();weights=np.tile(w/len(ab),len(ab));gx,cf,cr,density=hist_curves(flat,weights,np.hypot(sg,ref_sigma),.00125,query=allquery);p0=curve_values(gx,cf,y0[None])[0];pb=curve_values(gx,cf,yb);cr0=curve_values(gx,cr,y0[None])[0];crb=curve_values(gx,cr,yb)
    else:
        oldcurve=cached(OLD/'curves'/f'{s}_{"QRF" if method=="QRF" else "E"}.npz');at=int(np.flatnonzero(oldcurve['ii']==i)[0]);gx=cached(OLD/'curves/grid.npz')['mu'];cf=oldcurve['raw' if s=='M5' else 'cdf'][at];cr=curve_crps(gx,cf) if s=='M5' else oldcurve['crps'][at];p0=np.interp(y0,gx,cf);pb=np.interp(yb.ravel(),gx,cf).reshape(yb.shape);cr0=np.interp(y0,gx,cr);crb=np.interp(yb.ravel(),gx,cr).reshape(yb.shape)
        if method=='E':
            p0,l0=direct_queries(mm,lw,np.hypot(sg,ref_sigma),xx,np.zeros(2),y0[None]);pb,lb=direct_queries(mm,lw,np.hypot(sg,ref_sigma),xx,np.zeros((256,2)),yb);p0=p0[0];l0=l0[0]
        elif s=='M5':
            q=cached(OLD/'predict/QRF_south.npz');at=int(np.flatnonzero(q['ii']==i)[0]);edges=q['edges'][at];mass=q['mass'][at]
            def logdensity(y):
                j=np.searchsorted(edges,y,side='right')-1;good=(j>=0)&(j<len(mass));value=np.zeros(y.shape);value[good]=(mass/np.diff(edges))[j[good]]
                with np.errstate(divide='ignore'):return np.log(value)
            l0=logdensity(y0);lb=logdensity(yb)
        else:
            from seq7_math import row_interp
            values=oldcurve['logpdf'][at:at+1];l0=row_interp(gx,y0[None],values,cubic=True)[0];lb=row_interp(gx,yb.reshape(1,-1),values,cubic=True).reshape(yb.shape)
    return dict(i=i,family_ids=ids,point_CDF=p0,boot_CDF=pb,point_logscore=l0,boot_logscore=lb,point_CRPS=cr0,boot_CRPS=crb)


def main():
    require_compute();d=folder('evaluate');t=load(OUT/'prepare/roster.npz');inputs={'predict':sha(OUT/'predict/complete.json'),'fit':sha(OUT/'fit/complete.json')};families=[];lookup={};aliases=[];pairs=[];method_data={}
    def family(s,m,ii):
        key=(s,m,tuple(ii))
        if key not in lookup:
            lookup[key]=len(families);families.append(dict(system=s,method=m,ii=ii))
        return lookup[key]
    for s in C['systems']:
        rows=np.flatnonzero(t['system']==s);base=load(OUT/'prepare'/f'{s}_E.npz');cal=load(OUT/'predict'/f'{s}_calibrated.npz');oldq=load(OLD/'curves'/f'{s}_QRF.npz');methods=['E','CalE','QRF']+(['CalE_marg'] if s!='M5' else []);reference=10+5*np.log10(t['ref'][rows]);shift=t['shift_'+s]
        for m in methods:
            if m=='QRF':
                support=np.isin(rows,oldq['ii'][oldq['support']]);qs=np.full((len(rows),5),np.nan);qs[support]=oldq['q'][np.searchsorted(oldq['ii'],rows[support])];qb=np.broadcast_to(qs[:,None],(len(rows),256,5))
            else:
                support=base['support'];qs=base['q'] if m=='E' else (cal['marginal_q'] if m=='CalE_marg' else cal['q']);qb=cal['qboot'] if m=='CalE' else np.broadcast_to(qs[:,None],(len(rows),256,5))
            method_data[s,m]=dict(ii=rows,support=support,q=qs,qboot=qb,post=base['post'] if m!='QRF' else np.full(len(rows),np.nan),pout=cal['pout'])
        common=rows[np.logical_and.reduce([method_data[s,m]['support'] for m in methods])]
        cohorts=[('primary',rows),('v5',rows[t['v5'][rows]]),('common',common)]
        if s=='M5':cohorts.append(('calibration_region',rows[t['calibration_eligible'][rows]]))
        for cohort,parent in cohorts:
            groups=[(phase,parent if phase=='all' else parent[t['phase'][parent]==phase],'all') for phase in ('all','lower_MS','turnoff','other')]
            if cohort=='primary':
                gr=t['m'][parent,0]-t['m'][parent,1]
                groups.extend((f'gr_{lo:g}_{hi:g}',parent[(gr>=lo)&(gr<hi)],'all') for lo,hi in zip(C['gr_edges'][:-1],C['gr_edges'][1:]))
                for phase in ('all','lower_MS','turnoff'):
                    for k,label in enumerate(('overlap','transition','extrapolation')):groups.append((phase,parent[(cal['domain'][np.searchsorted(rows,parent)]==k)&((t['phase'][parent]==phase) if phase!='all' else True)],label))
            for phase,group,domain in groups:
                if not len(group):continue
                info=dict(system=s,cohort=cohort,phase=phase,domain=domain,parent_N=len(group))
                for m in methods:
                    ii=np.intersect1d(group,rows[method_data[s,m]['support']]);aliases.append(dict(**info,method=m,family=family(s,m,ii) if len(ii) else -1,supported_N=len(ii)))
                if phase.startswith('gr_'):continue
                for a,b in combinations(methods,2):
                    ii=np.intersect1d(group,np.intersect1d(rows[method_data[s,a]['support']],rows[method_data[s,b]['support']]));pairs.append(dict(**info,first=a,second=b,N=len(ii),first_family=family(s,a,ii) if len(ii) else -1,second_family=family(s,b,ii) if len(ii) else -1))
    centers=[];boots=[]
    for f in families:
        s,m,ii=f['system'],f['method'],f['ii'];z=method_data[s,m];j=np.searchsorted(z['ii'],ii);ref=10+5*np.log10(t['ref'][ii]);res=z['q'][j,2]-ref;rb=z['qboot'][j,:,2]-ref[:,None]-t['shift_'+s][None];mult=t['mult'][:,ii];centers.append(float(np.median(res)));boots.append(np.array([count_median(rb[:,k],mult[k]) for k in range(256)]))
    npz(d/'group_centers.npz',point=np.array(centers),boot=np.array(boots));save_json(d/'group_design.json',[dict(**f,family=i) for i,f in enumerate(families)])
    values={};pointrows=[];summary=[]
    for s in C['systems']:
        for method in ['E','CalE','QRF']+(['CalE_marg'] if s!='M5' else []):
            relevant=[(k,f) for k,f in enumerate(families) if f['system']==s and f['method']==method];all_ids=np.unique(np.concatenate([f['ii'] for _,f in relevant]));tasks=[(s,method,int(i),[k for k,f in relevant if i in f['ii']]) for i in all_ids]
            with ProcessPoolExecutor(max_workers=C['workers']) as pool:results=list(pool.map(source_scores,tasks,chunksize=1))
            by_id={r['i']:r for r in results};md=method_data[s,method]
            score_dir=folder('score_cache')
            for result in results:npz(score_dir/f'{s}_{method}_{result["i"]:05d}.npz',**result)
            for k,f in relevant:
                ii=f['ii'];j=np.searchsorted(md['ii'],ii);q=md['q'][j];qb=md['qboot'][j];ref=10+5*np.log10(t['ref'][ii]);refb=ref[:,None]+t['shift_'+s][None];res=q[:,2]-ref;rb=qb[:,:,2]-refb;w=t['mult'][:,ii].T;center=centers[k];cb=boots[k];record=dict(N=len(ii),median_mu_residual=center,NMAD=float(1.4826*np.median(abs(res-center))),scale_target_met=abs(center)<=.1,score_convention='shared_reference_conditional' if s=='M5' else 'independent_target_reference_kernel',parameter_forecast='empirical_bootstrap_parameter_mixture' if method=='CalE_marg' else 'point_parameter_conditional',bootstrap_valid_draws=int((w.sum(axis=0)>0).sum()),post_mean=float(np.nanmean(md['post'][j])) if np.isfinite(md['post'][j]).any() else np.nan,outside_color_mean=float(np.mean(md['pout'][j])))
                record['center_lo'],record['center_hi']=ci(cb);bd=dict(center=cb)
                nmad=np.array([1.4826*count_median(abs(rb[:,r]-cb[r]),w[:,r]) for r in range(256)]);record['NMAD_lo'],record['NMAD_hi']=ci(nmad);bd['NMAD']=nmad
                for name,low,high in [('W68',1,3),('W95',0,4),('left68',1,2),('right68',2,3),('left95',0,2),('right95',2,4)]:
                    val=q[:,high]-q[:,low];boot=qb[:,:,high]-qb[:,:,low];v=np.array([count_median(boot[:,r],w[:,r]) for r in range(256)]);record[name]=float(np.median(val));record[name+'_lo'],record[name+'_hi']=ci(v);bd[name]=v
                for state,idx in [('raw',0),('centered',1)]:
                    scorepoints={};scoreboots={}
                    for metric in ('CDF','logscore','CRPS'):
                        pv=[];bv=[]
                        for i in ii:
                            r=by_id[int(i)];at=int(np.flatnonzero(r['family_ids']==k)[0])+1 if idx else 0;pv.append(r['point_'+metric][at]);bv.append(r['boot_'+metric][:,at])
                        pv=np.array(pv);bv=np.array(bv);scorepoints[metric]=pv;scoreboots[metric]=bv
                        if metric=='CDF':
                            for level,lo,hi in [('68',.16,.84),('95',.025,.975)]:
                                name=state+'_CDF_C'+level;v=weighted(((bv>=lo)&(bv<=hi)).astype(float),w);record[name]=float(np.mean((pv>=lo)&(pv<=hi)));record[name+'_lo'],record[name+'_hi']=ci(v);bd[name]=v
                                coupled=np.mean(bv,axis=1);record[state+'_coupled_reference_CDF_C'+level]=float(np.mean((coupled>=lo)&(coupled<=hi)))
                        else:
                            name=state+'_'+metric;v=weighted(bv,w);record[name]=float(np.mean(pv));record[name+'_lo'],record[name+'_hi']=ci(v);bd[name]=v
                            if metric=='logscore':record[state+'_zero_density_N']=int(np.sum(np.isneginf(pv)))
                    for level,lo,hi in [('68',1,3),('95',0,4)]:
                        query=ref+(center if idx else 0);queryb=refb+(cb[None] if idx else 0);pv=(q[:,lo]<=query)&(query<=q[:,hi]);bv=(qb[:,:,lo]<=queryb)&(queryb<=qb[:,:,hi]);name=state+'_center_C'+level;v=weighted(bv.astype(float),w);record[name]=float(pv.mean());record[name+'_lo'],record[name+'_hi']=ci(v);bd[name]=v
                record.update(r_min=float(t['r'][ii].min()),r_max=float(t['r'][ii].max()),gr_min=float((t['m'][ii,0]-t['m'][ii,1]).min()),gr_max=float((t['m'][ii,0]-t['m'][ii,1]).max()))
                values[k]=dict(point=record,bootstrap=bd)
            print('Evaluated',s,method,len(all_ids),flush=True)
    for r in aliases:
        f=r['family'];summary.append(dict(**r,**values[f]['point']) if f>=0 else r)
    paired=[]
    for r in pairs:
        if r['N']==0:paired.append(r);continue
        a=values[r['first_family']];b=values[r['second_family']];record=dict(r)
        for metric in ('raw_logscore','raw_CRPS','centered_logscore','centered_CRPS','median_mu_residual','W68','W95'):
            bk='center' if metric=='median_mu_residual' else metric
            with np.errstate(invalid='ignore'):delta=b['bootstrap'][bk]-a['bootstrap'][bk];point=b['point'][metric]-a['point'][metric]
            record[metric+'_delta']=point;record[metric+'_lo'],record[metric+'_hi']=ci(delta)
        paired.append(record)
    gaps=[]
    for s in C['systems']:
        for cohort in ('primary','v5','common'):
            for m in ['E','CalE','QRF']+(['CalE_marg'] if s!='M5' else []):
                a=next((r for r in aliases if r['system']==s and r['cohort']==cohort and r['phase']=='lower_MS' and r['domain']=='all' and r['method']==m),None);b=next((r for r in aliases if r['system']==s and r['cohort']==cohort and r['phase']=='turnoff' and r['domain']=='all' and r['method']==m),None)
                if a and b and a['family']>=0 and b['family']>=0:
                    aa=values[a['family']];bb=values[b['family']];lo,hi=ci(bb['bootstrap']['center']-aa['bootstrap']['center']);gaps.append(dict(system=s,cohort=cohort,method=m,turnoff_minus_lower_MS=bb['point']['median_mu_residual']-aa['point']['median_mu_residual'],lo=lo,hi=hi))
    csv_rows(d/'summary.csv',summary);csv_rows(d/'paired.csv',paired);csv_rows(d/'phase_gaps.csv',gaps);save_json(d/'family_metrics.json',values)
    complete(d,inputs,M5='conditional point forecasts at common reference, reference uncertainty coupled to calibration refits, no independent M5 measurement kernel',external='once-convolved target reference, conditional calibration refits or separately parameter-marginalized forecast',marginal_CIs='conditional on frozen calibration-parameter ensemble; target member/reference resampling; full parameter effects already inside forecast',v7_scores_preserved=True);print('All calibrated evaluations complete',flush=True)


if __name__=='__main__':main()
