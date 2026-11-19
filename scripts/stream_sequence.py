#!/usr/bin/env python3
"""Integrate the original native evolutionary polyline without sorting its colors."""
from scipy.special import logsumexp
from photdis.baseline_v1 import qmc_noise
from photdis.stage1 import require_compute
from stream_common import *
from halo_common import BLUE,ORANGE,GREEN
from sv18_common import metric,interval,cached,index_join

SIGMA={'M5':.06,'M13':.08,'M30':.09,'M92':.07}


def polyline(template,side):
    m=template[side];valid=(template['phase']==0)&(m[:,1]>=3)&(m[:,1]<=8)
    if side=='north':valid&=template['north_support']
    ii=np.flatnonzero(valid);left=ii[:-1][np.diff(ii)==1];right=left+1
    # Midpoint quadrature subdivides each actual evolutionary segment fourfold.
    u=(np.arange(4)+.5)/4
    nodes=(m[left,None,:]*(1-u[None,:,None])+m[right,None,:]*u[None,:,None]).reshape(-1,3)
    length=np.linalg.norm(m[right]-m[left],axis=1)
    weights=np.repeat(length/4,4);take=weights>1e-12
    return nodes[take],weights[take]/weights[take].sum()


def posterior(m,e,ids,model):
    nodes,prior=model;out=np.full((len(m),5),np.nan);fold=np.zeros(len(m))
    intrinsic=C['sequence']['intrinsic_sigma_mag']
    color=np.column_stack((nodes[:,0]-nodes[:,1],nodes[:,1]-nodes[:,2]))
    for start in range(0,len(m),128):
        stop=min(start+128,len(m));a=m[start:stop];s=np.maximum(e[start:stop],.001);identity=ids[start:stop]
        c=np.column_stack((a[:,0]-a[:,1],a[:,1]-a[:,2]));diff=c[:,None,:]-color[None,:,:]
        v1=s[:,0]**2+s[:,1]**2;v2=s[:,1]**2+s[:,2]**2;cross=-s[:,1]**2;det=v1*v2-cross**2
        chi=(diff[:,:,0]**2*v2[:,None]+diff[:,:,1]**2*v1[:,None]-2*cross[:,None]*diff[:,:,0]*diff[:,:,1])/det[:,None]
        logw=-.5*chi+np.log(prior);logw-=logsumexp(logw,axis=1)[:,None];weight=np.exp(logw);cdf=np.cumsum(weight,axis=1);cdf[:,-1]=1
        inv=1/s**2;total=inv.sum(axis=1)
        mean=((a[:,None,:]-nodes[None,:,:])*inv[:,None,:]).sum(axis=2)/total[:,None]
        # Full Gaussian color likelihood plus the conditional common distance
        # amplitude, rather than perturbing colors and counting noise twice.
        draws=[];hits=np.zeros(len(a))
        from scipy.special import ndtr
        for j in range(C['sequence']['draws']):
            z=qmc_noise(identity,j);u=ndtr(z[:,3]);ix=(cdf>=u[:,None]).argmax(axis=1)
            draws.append(mean[np.arange(len(a)),ix]+z[:,4]*np.sqrt(1/total+intrinsic**2))
            noisy=c[:,0]+z[:,0]*s[:,0]-z[:,1]*s[:,1];hits+=(noisy>=.15545)&(noisy<=.178745)
        out[start:stop]=10**((np.quantile(np.array(draws),[.025,.16,.5,.84,.975],axis=0).T-10)/5)
        # Reject extrapolation when every physical polyline point is >5 sigma.
        out[start:stop][chi.min(axis=1)>25]=np.nan
        fold[start:stop]=hits/C['sequence']['draws']
    return out,fold


def main():
    require_compute();d=folder('sequence');inputs={}
    t=read(OLD/'sequence',('same_source_predictions.npz','TYPE','ls_id','source_id','system','native','m','err','r','q','ref','phase','mask','simple','sequence','common'),inputs)
    template=read(OLD/'native',('template.npz','south','north','phase','north_support'),inputs)
    ids=np.where(t['source_id']>0,t['source_id'],-t['ls_id']);new=np.full_like(t['q'],np.nan);fold=np.zeros(len(new))
    for code,side in enumerate(('south','north')):
        ii=np.flatnonzero(t['native']==code);new[ii],fold[ii]=posterior(t['m'][ii],t['err'][ii],ids[ii],polyline(template,side))
    gr=t['m'][:,0]-t['m'][:,1];mr=t['m'][:,1]-10-5*np.log10(t['ref']);phase=np.full(len(gr),'other',dtype='U12')
    phase[(gr>=.3)&(gr<=.85)&(mr>=4.5)&(mr<=8)]='lower_MS'
    phase[(gr>=.15)&(gr<.3)&(mr>=3)&(mr<4.5)]='turnoff'
    methods={'QRF':t['q'],'Label grid':t['simple'],'Old sorted':t['sequence'],'Fixed branches':new}
    common=np.logical_and.reduce([np.isfinite(x).all(axis=1)&(x>0).all(axis=1) for x in methods.values()])
    cal=(t['system']=='M13')&(phase=='lower_MS')&~t['mask']&common
    assert cal.sum()>=30,('registered calibration too small',cal.sum())
    calres=5*np.log10(new[cal,2]/t['ref'][cal]);delta=float(np.median(calres));adjusted=new*10**(-delta/5)
    methods['Calibrated']=adjusted
    caldraw=[]
    for draw in range(256):
        dc=max(.01,RNG.normal(7.42,SIGMA['M13']));caldraw.append(np.median(calres[RNG.integers(len(calres),size=len(calres))])+5*np.log10(7.42/dc))
    rows=[];cross=[];audit=[]
    for i in range(len(gr)):
        audit.append(dict(TYPE=t['TYPE'][i],ls_id=int(t['ls_id'][i]),source_id=int(t['source_id'][i]),system=t['system'][i],native=int(t['native'][i]),
          r=float(t['r'][i]),gr=float(gr[i]),rz=float(t['m'][i,1]-t['m'][i,2]),reference_Mr=float(mr[i]),
          independent_member_phase=t['phase'][i],fixed_reference_phase=phase[i],mask=bool(t['mask'][i]),old_common=bool(t['common'][i]),
          new_common=bool(common[i]),fold_noise_fraction=float(fold[i]),old_d50=float(t['sequence'][i,2]),new_d50=float(new[i,2]),
          qrf_only_not_new_common=bool(np.isfinite(t['q'][i]).all() and not common[i])))
    for name in np.unique(t['system']):
        for low,high in ((14,18),(18,20),(20,21),(21,22),(22,23),(14,23)):
            for p in ('all','lower_MS','turnoff','other'):
                for masked in (False,True):
                    parent=(t['system']==name)&(t['r']>=low)&(t['r']<high)&((phase==p) if p!='all' else True)&(~t['mask'] if not masked else True)
                    for code in np.unique(t['native'][parent]):
                        for support in (False,True):cross.append(dict(system=name,r_lo=low,r_hi=high,phase=p,include_masked=masked,native=int(code),common=support,n=int((parent&(t['native']==code)&(common==support)).sum())))
                    use=parent&common;n=int(use.sum())
                    if not n:continue
                    dd=t['ref'][use];native=t['native'][use];col=gr[use]
                    # A single common-scale draw for all members and methods of
                    # each system, coupled with the M13 calibration uncertainty.
                    resamples=[RNG.integers(n,size=n) for _ in range(256)]
                    scale=RNG.normal(1,SIGMA.get(name,0)/np.median(dd),256)
                    if name=='M13':scale=10**((np.median(calres)-np.array(caldraw))/5)
                    for method,q in methods.items():
                        z=q[use];vals=[]
                        for draw,(j,s) in enumerate(zip(resamples,scale)):
                            correction=10**((delta-caldraw[draw])/5) if method=='Calibrated' else 1
                            vals.append(np.median(z[j,2]*correction/(dd[j]*s)-1))
                        stat=metric(z,dd)
                        slope=np.polyfit(col,5*np.log10(z[:,2]/dd),1)[0] if n>=3 and np.std(col)>.01 else np.nan
                        rows.append(dict(system=name,r_lo=low,r_hi=high,phase=p,include_masked=masked,method=method,**stat,**interval(vals),
                             color_trend_mag_per_mag=float(slope),n_south=int((native==0).sum()),n_north=int((native==1).sum()),
                             enough=n>=30,bias_pass=bool(n>=30 and abs(stat['bias'])<=.1),
                             role='calibration' if name=='M13' else 'transfer; historically examined development system'))
    # Restore full QRF-only membership, including the old reference-Mr restriction.
    full=[]
    for name in ('M5','M13','M30','M92'):
        z=read(ROOT/f'data/v3_14/audit/{name}',('native_join.npz','TYPE','ls_id','source_id','r','g0','r0','z0','qrf_kpc','member','target','training_or_calibration','low_crowding','cluster_mask'),inputs)
        valid=z['target']&z['member']&~z['training_or_calibration']&z['low_crowding']&np.isfinite(z['qrf_kpc']).all(axis=1)
        j=index_join(z['ls_id'],t['ls_id']);found=j>=0
        for i in np.flatnonzero(valid):full.append(dict(TYPE=z['TYPE'][i],system=name,ls_id=int(z['ls_id'][i]),source_id=int(z['source_id'][i]),r=float(z['r'][i]),mask=bool(z['cluster_mask'][i]),old_sequence_parent=bool(found[i]),old_common=bool(found[i] and t['common'][j[i]]),new_common=bool(found[i] and common[j[i]])))
    csv_rows(d/'qrf_only_identity_audit.csv',full);csv_rows(d/'per_star_branches.csv',audit);csv_rows(d/'cross_counts.csv',cross);csv_rows(d/'distance_scale.csv',rows)
    npz(d/'predictions.npz',**t,new=new,calibrated=adjusted,new_common=common,reference_phase=phase,fold_noise_fraction=fold)
    save_json(d/'calibration.json',dict(system='M13',phase='lower_MS',n=int(cal.sum()),delta_Mg=delta,zero_point_draws=caldraw,color_term=0,
         interpretation='one common absolute-magnitude offset; calibration uncertainty shared by all transfer members; all-band offset preserves colors'))
    ordinary_control(d,template,delta,caldraw,inputs)
    draw(d,rows,t,phase,new,common,delta,template)
    complete(d,inputs,branch_measure='arc length of physical absolute-grz polyline, four midpoint quadrature nodes per original segment; explicitly not an IMF or LF',
        posterior='Gaussian joint gr/rz with negative shared-r covariance; exact conditional distance amplitude; 128 Sobol mixture draws plus 0.1mag gray intrinsic scatter; >5sigma rejection',
        reference_uncertainty='one common cluster draw per bootstrap shared across methods and members; same M13 reference draw used in zero point',
        ancestry='M13/M92 native construction and all systems examined in earlier development; transfer is independent of this zero point only; GD1 reference curve uncertainty unavailable, conditional centers retained')


def ordinary_control(d,template,delta,caldraw,inputs):
    # Reuse an already role-excluded held-out dwarf control; no new fitting.
    z=read(OLD/'noise_fixed',('receivers.npz','TYPE','ls_id','source_id','m','err','native','reference','q'),inputs)
    roles=read(OLD/'roles',('reference_roles.npz','TYPE','ls_id','direct_fit_or_calibration'),inputs)
    j=index_join(z['ls_id'],roles['ls_id']);assert (j>=0).all()
    excluded=roles['direct_fit_or_calibration'][j];z={k:v[~excluded] for k,v in z.items()}
    ids=np.where(z['source_id']>0,z['source_id'],-z['ls_id']);out=np.full((len(ids),5),np.nan)
    for code,side in enumerate(('south','north')):
        j=np.flatnonzero(z['native']==code);out[j],_=posterior(z['m'][j],z['err'][j],ids[j],polyline(template,side))
    rows=[]
    for code,side in enumerate(('south','north')):
        good=(z['native']==code)&np.isfinite(out).all(axis=1)
        if good.any():
            for name,q in (('QRF',z['q']),('Fixed branches',out),('Calibrated',out*10**(-delta/5))):rows.append(dict(native=side,method=name,**metric(q[good],z['reference'][good]),role_exclusions=int(excluded.sum()),original_parent=len(excluded)))
    csv_rows(d/'ordinary_dwarf_control.csv',rows)
    npz(d/'ordinary_predictions.npz',**z,sequence=out,calibrated=out*10**(-delta/5))


def draw(d,rows,t,phase,new,common,delta,template):
    with plt.rc_context(PAPER):
        f,axs=plt.subplots(2,3,figsize=(7.15,5.7),layout='constrained')
        for ax,name in zip(axs[0],('M13','M5','M30')):
            j=common&~t['mask']&(t['system']==name);gr=t['m'][j,0]-t['m'][j,1]
            dr=5*np.log10(new[j,2]/t['ref'][j]);ax.scatter(gr,dr,s=5,lw=0,alpha=.55,color=BLUE,rasterized=True)
            ax.axhline(delta,color=ORANGE,lw=.8,label='M13 lower-MS zero point');ax.axhline(0,color='.4',ls=':',lw=.6)
            ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$M_{g,\mathrm{ref}}-M_{g,\mathrm{seq}}$ (mag)',title=name,xlim=(.1,.9),ylim=(-1.5,1.5))
        axs[0,0].legend(fontsize=6,frameon=False)
        for ax,p in zip(axs[1,:2],('lower_MS','turnoff')):
            names=['M13','M5','M30','M92','GD1_thin','GD1_cocoon']
            for k,(method,col) in enumerate(zip(('QRF','Label grid','Fixed branches','Calibrated'),(BLUE,GREEN,ORANGE,'#8c6bb1'))):
                for i,name in enumerate(names):
                    a=[r for r in rows if r['system']==name and r['phase']==p and r['r_lo']==14 and r['r_hi']==23 and not r['include_masked'] and r['method']==method]
                    if not a:continue
                    r=a[0];x=i+(k-1.5)*.14;ax.errorbar(x,r['bias'],yerr=[[max(0,r['bias']-r['ci_lo'])],[max(0,r['ci_hi']-r['bias'])]],fmt='o' if r['n']>=30 else 'o',mfc=col if r['n']>=30 else 'none',ms=3,color=col,lw=.7,label=method if i==0 else None)
            ax.axhspan(-.1,.1,color='.85',alpha=.4);ax.axhline(0,color='.4',ls=':',lw=.6);ax.set(xticks=range(len(names)),xticklabels=['M13','M5','M30','M92','GD-1\nthin','GD-1\ncocoon'],ylabel='Median relative-distance residual',title=p.replace('_',' '));ax.tick_params(axis='x',labelsize=6)
        axs[1,0].legend(frameon=False,fontsize=6)
        ax=axs[1,2]
        for side,ls in (('south','-'),('north','--')):
            m=template[side];use=(template['phase']==0)&(m[:,1]>=3)&(m[:,1]<=8)
            if side=='north':use&=template['north_support']
            ax.plot(m[use,0]-m[use,1],m[use,0],ls=ls,color=ORANGE,lw=.8,label=side.capitalize())
        ax.axvspan(.15545,.178745,color='.85',alpha=.5);ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$M_g$ (mag)',xlim=(.13,.32),ylim=(6.5,2.5));ax.legend(frameon=False,fontsize=6)
        figure(d,'known_system_branch_calibration',f,rows)


if __name__=='__main__':main()
