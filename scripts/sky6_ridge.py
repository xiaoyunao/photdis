#!/usr/bin/env python3
"""Native empirical sequences: separate gray displacement from two-color shape."""
from scipy.spatial import Delaunay
from photdis.stage1 import require_compute
from halo_native import design
from halo_common import BLUE,ORANGE,GREEN
from sv18_common import index_join
from sky6_common import *


def model_tables(template,inputs):
    result={'FOS10':dict(south=template['south'],north=template['north'],phase=template['phase'],north_support=template['north_support'])}
    hull=Delaunay(template['hull'])
    for spec in C['ridge']['models']:
        key=spec['key'];path=ROOT/'configs/v6_models'/(key+'.txt');record=json.loads(path.with_suffix('.json').read_text());verified(path,record['sha256']);inputs[str(path.relative_to(ROOT))]=sha(path)
        cols=next(l.lstrip('# ').split() for l in path.read_text().splitlines() if l.startswith('# EEP '));x=np.loadtxt(path)
        np.testing.assert_allclose(x[:,cols.index('isochrone_age_yr')],np.log10(spec['age_Gyr']*1e9),atol=1e-10)
        np.testing.assert_allclose(x[:,cols.index('[Fe/H]_init')],spec['FeH'],atol=1e-10)
        mag=x[:,[cols.index('DECam_'+b) for b in 'grz']];color=np.column_stack((mag[:,0]-mag[:,1],mag[:,1]-mag[:,2]));north=mag+design(*color.T)@template['beta']
        result[key]=dict(south=mag,north=north,phase=x[:,cols.index('phase')],north_support=hull.find_simplex(color)>=0)
    return result


def physical(model,native,MS=False):
    m=model[('south','north')[native]];p=model['phase'];ok=(p==0) if MS else np.isin(p,(0,1,2))
    if native:ok&=model['north_support']
    points=np.column_stack((m[:,0],m[:,0]-m[:,1],m[:,1]-m[:,2]));i=np.flatnonzero(ok);left=i[:-1][np.diff(i)==1]
    return points,left


def intersections(points,left,axis,value):
    a=points[left];b=points[left+1];dx=b[:,axis]-a[:,axis]
    with np.errstate(divide='ignore',invalid='ignore'):u=(value-a[:,axis])/dx
    take=np.isfinite(u)&(u>=0)&(u<=1)&(abs(dx)>1e-10)
    return a[take]+u[take,None]*(b[take]-a[take])


def residuals(mag,model,native,MS):
    obs=np.column_stack((mag[:,0],mag[:,0]-mag[:,1],mag[:,1]-mag[:,2]));out=np.full((len(mag),4),np.nan);number=np.zeros((len(mag),3),'i2')
    for i,row in enumerate(obs):
        points,left=physical(model,native,bool(MS[i]))
        candidates=intersections(points,left,0,row[0]);number[i,0]=len(candidates)
        if len(candidates):
            best=candidates[np.argmin(((candidates[:,1:]-row[1:])**2).sum(axis=1))];out[i,:2]=row[1:]-best[1:]
        for col in (1,2):
            candidates=intersections(points,left,col,row[col]);number[i,col]=len(candidates)
            if len(candidates):out[i,col+1]=row[0]-candidates[np.argmin(abs(candidates[:,0]-row[0])),0]
    return out,number


def ridge(m,edges):
    mg=m[:,0];color=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]));result=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        original=(mg>=lo)&(mg<hi);take=original.copy()
        if take.sum():
            for _ in range(2):
                med=np.median(color[take],axis=0);take=original&(abs(color-med)<.10).all(axis=1)
                if not take.any():break
        enough=bool(take.sum()>=8 and take.sum()>=.6*original.sum())
        result.append(dict(Mg_lo=lo,Mg_hi=hi,N=int(original.sum()),core_N=int(take.sum()),usable=enough,Mg=float(np.median(mg[take])) if take.any() else np.nan,
            gr=float(np.median(color[take,0])) if take.any() else np.nan,rz=float(np.median(color[take,1])) if take.any() else np.nan,
            gr_scatter=float(1.4826*np.median(abs(color[original,0]-np.median(color[original,0])))) if original.any() else np.nan,
            rz_scatter=float(1.4826*np.median(abs(color[original,1]-np.median(color[original,1])))) if original.any() else np.nan))
    return result


def color_support(m,e,template,native):
    from stream_sequence import polyline
    nodes,_=polyline(template,('south','north')[native]);c=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]));mc=np.column_stack((nodes[:,0]-nodes[:,1],nodes[:,1]-nodes[:,2]));diff=c[:,None,:]-mc[None]
    s=np.maximum(e,.001);v1=s[:,0]**2+s[:,1]**2;v2=s[:,1]**2+s[:,2]**2;cov=-s[:,1]**2;det=v1*v2-cov**2
    chi=(diff[:,:,0]**2*v2[:,None]+diff[:,:,1]**2*v1[:,None]-2*cov[:,None]*diff[:,:,0]*diff[:,:,1])/det[:,None]
    return chi.min(axis=1)<=25


def render(a,d):
    model_keys=a['model_keys'];models={key:{k:a['model_'+key+'_'+k] for k in ('south','north','phase','north_support')} for key in model_keys}
    for systems,name in [(['M13','M5'],'native_ridge_M13_M5'),(['M30'],'native_ridge_M30')]:
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(len(systems),2,figsize=(7.15,5.3 if len(systems)>1 else 2.9),layout='constrained',squeeze=False)
            subset=np.isin(a['system'],systems)&~a['mask'];absolute=a['m'][:,0]-(10+5*np.log10(a['ref']));ylo=np.floor(absolute[subset].min()-.1);yhi=np.ceil(absolute[subset].max()+.1)
            for row,system in enumerate(systems):
                ii=(a['system']==system)&~a['mask'];native=int(system=='M13');mg=absolute[ii];col=np.column_stack((a['m'][ii,0]-a['m'][ii,1],a['m'][ii,1]-a['m'][ii,2]));support=a['support'][ii];rr=a['ridge_system']==system
                for j,ax in enumerate(axes[row]):
                    ax.scatter(col[support,j],mg[support],s=4,lw=0,alpha=.45,color=BLUE,rasterized=True,label='Members')
                    ax.scatter(col[~support,j],mg[~support],s=5,lw=.4,facecolor='none',edgecolor='.45',rasterized=True,label='Outside MS model support')
                    for key,label,color in [('FOS10','Fixed MIST1.0','.5'),('version_control','Same age/FeH, MIST1.2',GREEN),('matched_'+system,'Literature match',ORANGE)]:
                        model=models[key];points,left=physical(model,native);valid=np.zeros(len(points),bool);valid[left]=True;valid[left+1]=True
                        ids=np.flatnonzero(valid)
                        for seg in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                            if not len(seg):continue
                            ax.plot(points[seg,j+1],points[seg,0],color=color,lw=.8,label=label if seg[0]==ids[0] else None)
                            if key.startswith('matched_'):ax.plot(points[seg,j+1],points[seg,0]+C['ridge']['binary_shift_mag'],color=color,lw=.7,ls=':',label='Equal-mass binary reference' if seg[0]==ids[0] else None)
                    good=rr&a['ridge_usable'];x=a['ridge_colors'][good,j];y=a['ridge_Mg'][good];ax.errorbar(x,y,xerr=a['ridge_scatter'][good,j],fmt='o',ms=2,color='.1',lw=.5,label='Median ± observed NMAD')
                    mederr=np.median(a['err'][ii],axis=0);xerr=np.hypot(mederr[0 if j==0 else 1],mederr[1 if j==0 else 2]);ax.errorbar(np.quantile(col[:,j],.05),yhi-.5,xerr=xerr,yerr=mederr[0],fmt='none',color='.3',lw=.6)
                    xlo=min(-.05,float(col[:,j].min())-.05);xhi=float(col[:,j].max())+.05
                    ax.set(xlim=(xlo,xhi),ylim=(yhi,ylo),xlabel=r'$(g-r)_0$' if j==0 else r'$(r-z)_0$',ylabel=r'$M_{g,\mathrm{ref}}$ (mag)',title=f'{system} · N={int(ii.sum())}')
            axes[0,0].legend(frameon=False,fontsize=5,loc='lower right');figure(d,name,f,[dict(system=s,primary=int(((a['system']==s)&~a['mask']).sum())) for s in systems])
    if 'position_system' in a:
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(3,3,figsize=(7.15,7),layout='constrained')
            for row,system in enumerate(('M13','M5','M30')):
                for col,(component,ylabel) in enumerate([(0,r'$\Delta(g-r)_0$ (mag)'),(1,r'$\Delta(r-z)_0$ (mag)'),(2,r'$\Delta M_g$ at $(g-r)_0$ (mag)')]):
                    ax=axes[row,col]
                    for key,label,color in [('FOS10','Fixed MIST1.0','.5'),('version_control','MIST1.2 control',GREEN),('matched_'+system,'Literature match',ORANGE)]:
                        use=(a['position_system']==system)&(a['position_model']==key)&(a['position_N'][:,component]>=8)
                        ax.errorbar(a['position_Mg'][use],a['position_median'][use,component],yerr=a['position_scatter'][use,component],fmt='o-',ms=2,lw=.5,color=color,label=label)
                    ax.axhline(0,color='.4',ls=':',lw=.5);ax.set(xlabel=r'$M_{g,\mathrm{ref}}$ (mag)',ylabel=ylabel,title=system)
            axes[0,0].legend(frameon=False,fontsize=5);figure(d,'sequence_shape_residuals',f,[dict(scope='All primary unmasked members in reference-Mg bins; medians and observed NMAD, bins with at least8 finite intersections; reference branch choice is diagnostic only')])


def main():
    require_compute();d=folder('ridge');inputs={};t=read(OUT/'ridge_inputs',('roster.npz','TYPE','ls_id','system','native','mask','m','err','r','ref','sigma_ref','q','original_phase'),inputs)
    raw=[]
    for p in sorted((OUT/'cache').glob('*/member_photometry.npz')):
        marker=json.loads((p.parent/'complete.json').read_text());raw.append(read_columns(verified(p,marker['outputs'][p.name]),('TYPE','ls_id','m','err','EBV','extinction','native')))
    raw={k:np.concatenate([r[k] for r in raw]) for k in raw[0]};assert len(np.unique(raw['ls_id']))==len(raw['ls_id']);j=index_join(t['ls_id'],raw['ls_id']);assert (j>=0).all();assert (raw['TYPE'][j]=='PSF').all();np.testing.assert_array_equal(raw['native'][j],t['native'])
    replay=float(np.nanmax(abs(t['m']-raw['m'][j])));t.update(m_raw=raw['m'][j],EBV=raw['EBV'][j],extinction=raw['extinction'][j]);assert replay<2e-5,replay
    template=read(HALO/'native',('template.npz','south','north','phase','north_support','hull','beta','scatter'),inputs);models=model_tables(template,inputs)
    transfer=HALO/'native/native_transfer.csv';inputs['native_transfer_table']=sha(transfer)
    csv_rows(d/'north_transfer.csv',[dict(**r,scope='frozen bright16..19 native overlap; total measured transfer residual, not an independent atmosphere calibration') for r in csv.DictReader(transfer.open())])
    bootstrap=read(STREAM/'sequence',('shared_bootstrap.npz',*['scale_'+s for s in C['ridge']['systems']]),inputs)
    support=np.zeros(len(j),bool);rows=[];resrows=[];sensitivity=[];per_star=[];position=[];position_plot=[]
    for code in (0,1):
        use=t['native']==code;support[use]=color_support(t['m'][use],t['err'][use],template,code)
    for name in C['ridge']['systems']:
        ii=(t['system']==name)&~t['mask'];mag=t['m'][ii]-(10+5*np.log10(t['ref'][ii]))[:,None];native=int(name=='M13');mu=10+5*np.log10(t['ref'][ii]);gr=mag[:,0]-mag[:,1];mr=mag[:,1]
        phases=np.full(len(mag),'other',dtype='U12');phases[(gr>=.3)&(gr<=.85)&(mr>=4.5)&(mr<=8)]='lower_MS';phases[(gr>=.15)&(gr<.3)&(mr>=3)&(mr<4.5)]='turnoff';MS=np.isin(phases,('lower_MS','turnoff'))
        edges=np.arange(np.floor(mag[:,0].min()*4)/4,np.ceil(mag[:,0].max()*4)/4+.251,.25);rr=ridge(mag,edges);rng=np.random.default_rng(C['seed']+native+len(mag));draws=[]
        # One common distance draw and member resample for both colors/all bins.
        for k,scale in enumerate(bootstrap['scale_'+name]):
            ix=rng.integers(len(mag),size=len(mag));pert=mag[ix]+rng.normal(size=(len(mag),3))*t['err'][ii][ix]-5*np.log10(scale);r=ridge(pert,edges);draws.append([[x['gr'],x['rz']] if x['usable'] else [np.nan,np.nan] for x in r])
        draws=np.array(draws)
        for k,row in enumerate(rr):
            keep=(mag[:,0]>=row['Mg_lo'])&(mag[:,0]<row['Mg_hi']);row.update(system=name,native=('south','north')[native],median_sigma_gr=np.median(np.hypot(t['err'][ii][keep,0],t['err'][ii][keep,1])) if keep.any() else np.nan,median_sigma_rz=np.median(np.hypot(t['err'][ii][keep,1],t['err'][ii][keep,2])) if keep.any() else np.nan)
            for col,label in enumerate(('gr','rz')):
                values=draws[:,k,col];values=values[np.isfinite(values)];row[label+'_lo']=np.quantile(values,.025) if len(values) else np.nan;row[label+'_hi']=np.quantile(values,.975) if len(values) else np.nan
            rows.append(row)
        for key in ('FOS10','version_control','matched_'+name):
            res,ncandidate=residuals(mag,models[key],native,np.zeros(len(MS),bool))
            for lo,hi in zip(edges[:-1],edges[1:]):
                use=(mag[:,0]>=lo)&(mag[:,0]<hi)
                if not use.any():continue
                median=[];scatter=[];number=[]
                for k,label in enumerate(('gr_at_Mg','rz_at_Mg','Mg_at_gr','Mg_at_rz')):
                    value=res[use,k];value=value[np.isfinite(value)];center=float(np.median(value)) if len(value) else np.nan;scale=float(1.4826*np.median(abs(value-center))) if len(value) else np.nan
                    record=dict(system=name,model=key,Mg_lo=lo,Mg_hi=hi,Mg_median=float(np.median(mag[use,0])),component=label,N=int(use.sum()),finite=len(value),median=center,nmad=scale)
                    for q in (.05,.16,.84,.95):record[f'q{int(q*100):02d}']=float(np.quantile(value,q)) if len(value) else np.nan
                    position.append(record);median.append(center);scatter.append(scale);number.append(len(value))
                position_plot.append(dict(system=name,model=key,Mg=float(np.median(mag[use,0])),median=median,scatter=scatter,N=number))
            for phase in ('all','lower_MS','turnoff','other'):
                use=np.ones(len(mag),bool) if phase=='all' else phases==phase
                for k,label in enumerate(('gr_at_Mg','rz_at_Mg','Mg_at_gr','Mg_at_rz')):
                    z=res[use,k];finite=np.isfinite(z);v=z[finite];value=float(np.median(v)) if len(v) else np.nan
                    resrows.append(dict(system=name,model=key,projection='MS_SGB_RGB',phase=phase,component=label,N=int(use.sum()),supported=int(finite.sum()),median=value,nmad=float(1.4826*np.median(abs(v-value))) if len(v) else np.nan,
                        r_min=float(t['r'][ii][use].min()) if use.any() else np.nan,r_max=float(t['r'][ii][use].max()) if use.any() else np.nan,Mr_min=float(mr[use].min()) if use.any() else np.nan,Mr_max=float(mr[use].max()) if use.any() else np.nan))
            restricted,_=residuals(mag,models[key],native,MS)
            for phase in ('lower_MS','turnoff'):
                use=phases==phase
                for k,label in enumerate(('gr_at_Mg','rz_at_Mg','Mg_at_gr','Mg_at_rz')):
                    v=restricted[use,k];v=v[np.isfinite(v)];center=float(np.median(v)) if len(v) else np.nan
                    resrows.append(dict(system=name,model=key,projection='phase0_for_reference_MS_boxes',phase=phase,component=label,N=int(use.sum()),supported=len(v),median=center,nmad=float(1.4826*np.median(abs(v-center))) if len(v) else np.nan))
            for scenario in ('column075','column125','EBV_minus001','EBV_plus001','distance_minus_sigma','distance_plus_sigma'):
                changed=mag.copy();E=t['EBV'][ii];ext=t['extinction'][ii]
                if scenario.startswith('column'):changed+=ext*(.25 if scenario=='column075' else -.25)
                elif scenario.startswith('EBV'):
                    shift=-.01 if 'minus' in scenario else .01;coefficient=np.nanmedian(np.divide(ext,E[:,None],out=np.full_like(ext,np.nan),where=E[:,None]>1e-5),axis=0);changed-=(np.maximum(E+shift,0)-E)[:,None]*coefficient
                else:
                    sigma=t['sigma_ref'][ii][0];D=t['ref'][ii][0];changed-=5*np.log10((D+(-sigma if 'minus' in scenario else sigma))/D)
                r,_=residuals(changed,models[key],native,np.zeros(len(MS),bool))
                for k,label in enumerate(('gr_at_Mg','rz_at_Mg','Mg_at_gr','Mg_at_rz')):sensitivity.append(dict(system=name,model=key,scenario=scenario,component=label,N=int(np.isfinite(r[:,k]).sum()),median=float(np.nanmedian(r[:,k]))))
            per_star.append(dict(ls_id=t['ls_id'][ii],system=np.full(len(mag),name),model=np.full(len(mag),key),residual=res,candidate_count=ncandidate,reference_phase=phases))
    csv_rows(d/'ridge.csv',rows);csv_rows(d/'model_residuals.csv',resrows);csv_rows(d/'sequence_position_residuals.csv',position);csv_rows(d/'distance_dust_sensitivity.csv',sensitivity)
    npz(d/'per_member_residuals.npz',**{k:np.concatenate([p[k] for p in per_star]) for k in per_star[0]})
    npz(d/'members.npz',**t,color_support=support)
    rdata=dict(system=t['system'],mask=t['mask'],m=t['m'],err=t['err'],ref=t['ref'],support=support,model_keys=np.array(list(models)),ridge_system=np.array([r['system'] for r in rows]),ridge_usable=np.array([r['usable'] for r in rows]),ridge_Mg=np.array([r['Mg'] for r in rows]),ridge_colors=np.array([[r['gr'],r['rz']] for r in rows]),ridge_scatter=np.array([[r['gr_scatter'],r['rz_scatter']] for r in rows]))
    rdata.update({'model_'+key+'_'+field:value for key,model in models.items() for field,value in model.items()})
    for key in ('system','model','Mg','median','scatter','N'):rdata['position_'+key]=np.array([r[key] for r in position_plot])
    npz(d/'ridge_redraw.npz',**rdata);render(rdata,d)
    csv_rows(d/'support.csv',[dict(system=name,parent=int((t['system']==name).sum()),unmasked=int(((t['system']==name)&~t['mask']).sum()),primary_MS_color_supported=int(((t['system']==name)&~t['mask']&support).sum()),masked=int(((t['system']==name)&t['mask']).sum())) for name in C['ridge']['systems']])
    complete(d,inputs,native_replay_max_mag_difference=replay,ridge='observed binned two-color median/core, not deconvolved single-star fiducial; ambiguous/sparse bins explicit',projection='original adjacent evolutionary segments; fixed-Mg colors and fixed-color gray luminosity residuals, reference luminosity resolves branches diagnostically; not a new photometry-only estimator',north='frozen DECam-to-BASS/MzLS transfer, scatter and hull inherited',no_new_distance_correction=True)


if __name__=='__main__':main()
