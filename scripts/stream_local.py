#!/usr/bin/env python3
"""Raw-imaging on/off stream experiment with exactly the plotted CMD boundaries."""
from scipy.ndimage import gaussian_filter
from matplotlib.colors import SymLogNorm
from photdis.stage1 import require_compute
from stream_common import *
from stream_extract import ROIS,VARS
from halo_filtering import envelope
from halo_common import BLUE,ORANGE,GREEN

MU=np.round(np.arange(14,20.001,.1),1)


def templates(inputs):
    a=read(OLD/'native',('template.npz','south','phase'),inputs);result={'FOS':(a['south'],a['phase'])}
    p=ROOT/'configs/stream_inputs/pal5_mist_11p5_fehm1p4_decam.txt';inputs[str(p.relative_to(ROOT))]=sha(p)
    names=[]
    for line in p.read_text().splitlines():
        if line.startswith('# EEP'):names=line.lstrip('# ').split();break
    x=np.loadtxt(p);mag=x[:,[names.index('DECam_'+b) for b in 'grz']]
    assert abs(x[0,names.index('isochrone_age_yr')]-np.log10(11.5e9))<1e-8
    result['Pal5_control']=(mag,x[:,names.index('phase')])
    return result


def curves(mag,phase,kind):
    out=[]
    if kind=='blue_box':return []
    allowed=(0,2) if kind=='MS_SGB_RGB' else (0,) if kind=='MS_TO' else (2,)
    for p in allowed:
        ii=np.flatnonzero(phase==p)
        for segment in np.split(ii,np.flatnonzero(np.diff(ii)>1)+1):
            if len(segment)>1:out.append(envelope(mag[segment,0],mag[segment,0]-mag[segment,1]))
    return out


def boundaries(g0,mu,cc,error_curve,kind):
    sigma=np.interp(g0,error_curve[0],error_curve[1]);lower=[];upper=[]
    if kind=='blue_box':
        # Mr3.5..5.0 and gr.15..45, with the same 2sigma_g expansion.
        lower.append(np.maximum(.15-2*sigma,g0-mu-5.0-2*sigma))
        upper.append(np.minimum(.45+2*sigma,g0-mu-3.5+2*sigma))
    else:
        for grid,lo,hi in cc:
            lower.append(np.interp(g0-mu,grid,lo,left=np.inf,right=np.inf)-.05-2*sigma)
            upper.append(np.interp(g0-mu,grid,hi,left=-np.inf,right=-np.inf)+.10+2*sigma)
    return np.array(lower),np.array(upper)


def accept(m,mu,cc,errcurve,kind):
    gr=m[:,0]-m[:,1];out=np.zeros((len(m),len(mu)),bool)
    for j,value in enumerate(mu):
        low,high=boundaries(m[:,0],value,cc,errcurve,kind)
        out[:,j]=((gr[None,:]>=low)&(gr[None,:]<=high)).any(axis=0)&(gr>=-.2)&(gr<1.2)
    return out


def uncertainty(non,noff,ron,roff,density,seed):
    if ron<=0 or roff<=0:return np.nan,np.nan,np.nan
    rng=np.random.default_rng(seed);n=4096
    # Jeffreys Poisson-rate draws for counts AND independent random area.
    aon=rng.gamma(ron+.5,1,n)/density;aoff=rng.gamma(roff+.5,1,n)/density
    excess=rng.gamma(non+.5,1,n)/aon-rng.gamma(noff+.5,1,n)/aoff
    return *np.quantile(excess,[.025,.975]),float(np.quantile(excess,.95))


def region_masks(off,cfg):
    a,b=cfg['off_abs_deg'];width=cfg['on_halfwidth_deg'];mid=(a+b)/2
    return [abs(off)<width,(abs(off)>=a)&(abs(off)<b),abs(off-mid)<width,abs(off+mid)<width]


def main():
    require_compute();d=folder('local');inputs={};tm=templates(inputs)
    density=json.loads((OUT/'local_area/complete.json').read_text())['density'];summary=[];responses=[];phasecounts=[];visible=[];recovery={};arrays={}
    for roi in ROIS:
        cfg=C['local'][roi];t=read(OUT/'local_source',(roi+'.npz','TYPE','ls_id','source_id','RA','DEC','m','err','observed','EBV','offset','along','selected'),inputs)
        r=read(OUT/'local_area',(roi+'.npz','RA','DEC','depth0','EBV','offset','along','selected'),inputs)
        assert (t['TYPE']=='PSF').all()
        mu=10+5*np.log10(cfg['distance_kpc']) if 'distance_kpc' in cfg else cfg['reference_mu']
        nodes=np.unique(np.r_[MU,mu,mu-1,mu+1]);at=np.argmin(abs(nodes-mu));sr=region_masks(t['offset'],cfg);rr=region_masks(r['offset'],cfg)
        # Pool on and both backgrounds before any CMD/phase selection. Width
        # depends only on dereddened apparent g; plots call this exact function.
        gcent=np.arange(14.25,24,.5);base=t['selected'][2];sig=[]
        for center in gcent:
            x=t['err'][base&(abs(t['m'][:,0]-center)<.25),0]
            sig.append(np.median(x) if len(x) else np.nan)
        sig=np.array(sig);good=np.isfinite(sig);sig=np.interp(gcent,gcent[good],sig[good]);errcurve=(gcent,sig)
        modelnames=('FOS','Pal5_control') if roi.startswith('Pal') else ('FOS',)
        roi_arrays=dict(g_centers=gcent,sigma_g=sig,mu=nodes)
        for model in modelnames:
            mag,phase=tm[model]
            for kind in ('MS_SGB_RGB','MS_TO','blue_box','SGB_RGB'):
                cc=curves(mag,phase,kind);accepted=accept(t['m'],nodes,cc,errcurve,kind)
                for vi,var in enumerate(VARS):
                    ss=t['selected'][vi];ar=r['selected'][vi]
                    counts=np.array([accepted[ss&region].sum(axis=0) for region in sr]);random=np.array([int((ar&region).sum()) for region in rr]);areas=random/density
                    ex=counts[0]/areas[0]-counts[1]/areas[1] if areas[:2].min()>0 else np.full(len(nodes),np.nan)
                    for j,value in enumerate(nodes):responses.append(dict(roi=roi,model=model,phase=kind,variant=var,mu=value,on=int(counts[0,j]),off=int(counts[1,j]),on_area=areas[0],off_area=areas[1],signed_excess=ex[j]))
                    for segment in range(cfg['segments']):
                        ts=(t['along']>=segment/cfg['segments'])&(t['along']<(segment+1)/cfg['segments']);rs=(r['along']>=segment/cfg['segments'])&(r['along']<(segment+1)/cfg['segments'])
                        nn=np.array([int((ss&ts&region&accepted[:,at]).sum()) for region in sr]);rn=np.array([int((ar&rs&region).sum()) for region in rr]);aa=rn/density
                        cil,cih,upper=uncertainty(nn[0],nn[1],rn[0],rn[1],density,C['seed']+segment)
                        values=np.divide(nn,aa,out=np.full(4,np.nan),where=aa>0)
                        passed=bool(np.isfinite(cil) and cil>0 and aa[0]>=.1 and values[0]>values[2:].max())
                        summary.append(dict(roi=roi,model=model,phase=kind,variant=var,segment=segment,n_on=int(nn[0]),n_off=int(nn[1]),
                             area_on=aa[0],area_off=aa[1],random_on=int(rn[0]),random_off=int(rn[1]),signed_excess=values[0]-values[1],ci_lo=cil,ci_hi=cih,upper95=upper,
                             right_density=values[2],left_density=values[3],angular_pass=passed,mu=mu))
                    if model=='FOS' and kind=='MS_TO' and var=='gr_complete':recovery[roi]=sum(x['angular_pass'] for x in summary if x['roi']==roi and x['model']==model and x['phase']==kind and x['variant']==var)>=2
                    if model=='FOS' and var in ('strict_original','gr_E010','gr_complete'):
                        # Counts are intentionally nonexclusive among phases.
                        for j,zone in enumerate(('target','off','shift_right','shift_left')):phasecounts.append(dict(roi=roi,phase=kind,variant=var,zone=zone,n_at_reference=int(counts[j,at]),near_mu14_15=int(counts[j,(nodes>=14)&(nodes<=15)].sum()),far_mu19_20=int(counts[j,(nodes>=19)&(nodes<=20)].sum()),note='node acceptances overlap; not distinct physical stars'))
                if kind=='MS_TO':
                    use=phase==0;lim=t['selected'][4]&sr[0]
                    maxdepth=float(np.quantile(r['depth0'][r['selected'][4]&rr[0],:2].min(axis=1),.1)) if (r['selected'][4]&rr[0]).any() else np.nan
                    visible.append(dict(roi=roi,model=model,phase=kind,reference_mu=mu,formal_depth_p10=maxdepth,brightest_g0=float(np.min(mag[use,0]+mu)),turnoff_g0=float(np.min(mag[use,0]+mu)),template_nodes_g0_lt23=int((mag[use,0]+mu<23).sum()),warning='template-point visibility is not stellar-population completeness'))
            roi_arrays.update(make_maps(roi,t,r,tm[model],errcurve,mu,model,d,density,responses))
        npz(d/(roi+'_redraw.npz'),**roi_arrays)
        arrays[roi]=roi_arrays
        print(roi,'done',len(t['ls_id']),flush=True)
    csv_rows(d/'segment_excess.csv',summary);csv_rows(d/'modulus_response.csv',responses);csv_rows(d/'phase_acceptances.csv',phasecounts);csv_rows(d/'phase_visibility.csv',visible)
    passed=bool((recovery.get('Pal5_tail',False) or recovery.get('Pal5_leading',False)) and (recovery.get('Sgr_north',False) or recovery.get('Sgr_south',False)))
    save_json(d/'recovery_decision.json',dict(per_ROI=recovery,both_systems_pass=passed,
         full_footprint_action='same gr_complete+MS_TO definition authorized for accumulation' if passed else 'no new full-footprint accumulation; retain existing true cube RGB and all local negative evidence',
         fixed_scheme='FOS template, gr_complete, MS_TO, >=2 predeclared angular segments with positive95% on-minus-off interval and on density above both equal-width shifted controls',
         diagnostic_only='Pal5_core never determines stream recovery'))
    complete(d,inputs,acceptance='union of physical phase envelopes; .5mag modulus breadth, asymmetric .05/.10 color widths plus full 2sigma_g from pooled local pre-CMD sources. Blue box has its own explicit Mr/colour geometry. Count/plot use identical boundary function.',
       uncertainty='raw unsmoothed Poisson on/off rates and independent Poisson random area with Jeffreys gamma posterior; intervals are local descriptive, not global trials-corrected discoveries',
       systematics='off background spatial gradients and selection-depth mismatch are tested with both adjacent shifted controls; quantitative counts retain signed excess; smoothing only for display')


def make_maps(roi,t,r,model,ec,mu,label,d,density,responses):
    cfg=C['local'][roi];mag,phase=model;kind='MS_TO';cc=curves(mag,phase,kind)
    a=accept(t['m'],[mu],cc,ec,kind)[:,0];sx=np.linspace(0,1,61);oy=np.linspace(-cfg['off_abs_deg'][1],cfg['off_abs_deg'][1],81)
    xe=np.linspace(-.2,1.2,71);ye=np.linspace(14,24,101);sr=region_masks(t['offset'],cfg);rr=region_masks(r['offset'],cfg)
    images={};hesses={};tables=[]
    for vi in (0,1,2,3,4,5):
        ss=t['selected'][vi];ar=r['selected'][vi];areas=np.array([int((ar&z).sum())/density for z in rr[:2]])
        hist=[]
        for zone,area in zip(sr[:2],areas):hist.append(np.histogram2d(t['m'][ss&zone,0]-t['m'][ss&zone,1],t['m'][ss&zone,0],(xe,ye))[0]/max(area,1e-20))
        hesses[VARS[vi]]=np.array(hist);counts=np.histogram2d(t['along'][ss&a],t['offset'][ss&a],(sx,oy))[0];area=np.histogram2d(r['along'][ar],r['offset'][ar],(sx,oy))[0]/density
        images[VARS[vi]]=np.stack((counts,area))
        tables.append(dict(roi=roi,model=label,variant=VARS[vi],source_count=int(ss.sum()),area_on=areas[0],area_off=areas[1]))
    with plt.rc_context(PAPER):
        f,axs=plt.subplots(2,3,figsize=(7.15,5.3),layout='constrained')
        for ax,var in zip(axs[0],('strict_original','gr_E010','gr_complete')):
            h=hesses[var][0]-hesses[var][1];cap=max(float(np.quantile(abs(h),.99)),.1)
            z=ax.pcolormesh(xe,ye,h.T,cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
            g=np.linspace(14,24,600);lower,upper=boundaries(g,mu,cc,ec,kind)
            for lo,hi in zip(lower,upper):ax.plot(lo,g,color=ORANGE,lw=.6);ax.plot(hi,g,color=ORANGE,lw=.6)
            ax.set(xlim=(-.2,1.2),ylim=(24,14),xlabel=r'$(g-r)_0$',ylabel=r'$g_0$ (mag)',title=var.replace('_',' '));f.colorbar(z,ax=ax,fraction=.04,label='Target − control (deg⁻² bin⁻¹)')
        ax=axs[1,0]
        for var,color in zip(('strict_original','strict_E010','gr_E010','gr_complete'),('.5',GREEN,BLUE,ORANGE)):
            rr0=[z for z in responses if z['roi']==roi and z['model']==label and z['phase']=='MS_TO' and z['variant']==var]
            ax.plot([z['mu'] for z in rr0],[z['signed_excess'] for z in rr0],color=color,lw=.8,label=var.replace('_',' '))
        ax.axvline(mu,color='.3',lw=.7,ls=':');ax.axhline(0,color='.5',lw=.5);ax.set(xlabel='Distance modulus (mag)',ylabel='Target − control (deg⁻²)');ax.legend(frameon=False,fontsize=5.5)
        raw,area=images['gr_complete'];width_ra=(cfg['ra_range'][1]-cfg['ra_range'][0])*np.cos(np.deg2rad(np.median(t['DEC'])))
        for ax,sigma in zip(axs[1,1:],(.1,.3)):
            kernel=(sigma/(width_ra*np.diff(sx)[0]),sigma/np.diff(oy)[0]);count_s=gaussian_filter(raw,kernel);area_s=gaussian_filter(area,kernel)
            values=np.divide(count_s,area_s,out=np.full_like(raw,np.nan),where=(area>0)&(area_s>0))
            off=(abs((oy[:-1]+oy[1:])/2)>=cfg['off_abs_deg'][0]);background=np.nanmedian(values[:,off],axis=1)
            residual=values-background[:,None];cap=max(float(np.nanquantile(abs(residual),.99)),.1)
            z=ax.pcolormesh(sx,oy,np.ma.masked_invalid(residual.T),cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
            ax.axhline(0,color='.3',lw=.5,ls=':');ax.set(xlabel='Fraction along fixed RA segment',ylabel='Cross-track offset (deg)',title=f'Fixed μ · σ={sigma:g}°');f.colorbar(z,ax=ax,fraction=.04,label='Signed excess (deg⁻²)')
            images['smoothed_'+str(sigma)]=np.stack((count_s,area_s,residual))
        figure(d,roi+'_'+label+'_recovery',f,tables)
    out={label+'_hess_'+k:v for k,v in hesses.items()};out.update({label+'_sky_'+k:v for k,v in images.items()});out.update(hess_gr_edges=xe,hess_g_edges=ye,sky_along_edges=sx,sky_offset_edges=oy)
    g=np.linspace(14,24,600);lo,hi=boundaries(g,mu,cc,ec,kind);out.update({label+'_envelope_g':g,label+'_envelope_lower':lo,label+'_envelope_upper':hi})
    return out


if __name__=='__main__':main()
