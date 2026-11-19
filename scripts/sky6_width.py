#!/usr/bin/env python3
"""Same-member location/width diagnostics with shared scale and zero-point draws."""
from scipy.special import ndtr,log_ndtr,logsumexp
from photdis.stage1 import require_compute
from photdis.v3_3 import frozen_qrf
from stream_sequence import polyline,count_median
from halo_common import BLUE,ORANGE
from sv18_common import index_join
from sky6_common import *


def sequence_mixture(m,e,nodes,prior):
    s=np.maximum(e,.001);inv=1/s**2;total=inv.sum(axis=1);mean=((m[:,None,:]-nodes[None])*inv[:,None]).sum(axis=2)/total[:,None]
    obs=np.column_stack((m[:,0]-m[:,1],m[:,1]-m[:,2]));col=np.column_stack((nodes[:,0]-nodes[:,1],nodes[:,1]-nodes[:,2]));diff=obs[:,None]-col[None]
    v1=s[:,0]**2+s[:,1]**2;v2=s[:,1]**2+s[:,2]**2;cross=-s[:,1]**2;det=v1*v2-cross**2
    chi=(diff[:,:,0]**2*v2[:,None]+diff[:,:,1]**2*v1[:,None]-2*cross[:,None]*diff[:,:,0]*diff[:,:,1])/det[:,None]
    logw=-.5*chi+np.log(prior);logw-=logsumexp(logw,axis=1)[:,None]
    return mean,logw,np.sqrt(1/total+.1**2)


def mixture_probability(y,mean,logw,sigma):
    z=(np.asarray(y)[:,None]-mean)/sigma[:,None]
    return (np.exp(logw)*ndtr(z)).sum(axis=1),logsumexp(logw-.5*z*z-np.log(sigma[:,None]*np.sqrt(2*np.pi)),axis=1)


def uniform_convolution(edges,mass,y,sigma):
    """Exact Gaussian-reference convolution of the frozen piecewise-uniform PDF."""
    lo=(edges[:,:-1]-y[:,None])/sigma;hi=(edges[:,1:]-y[:,None])/sigma;width=np.diff(edges,axis=1)
    reflect=lo+hi>0;upper=np.where(reflect,-lo,hi);lower=np.where(reflect,-hi,lo);a=log_ndtr(upper);b=log_ndtr(lower)
    with np.errstate(divide='ignore',invalid='ignore'):logdiff=a+np.log(-np.expm1(b-a));logmass=np.log(mass)
    logdensity=logsumexp(logmass+logdiff-np.log(width),axis=1)
    def G(z):return z*ndtr(z)+np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    probability=(mass*sigma*(G(-lo)-G(-hi))/width).sum(axis=1)
    return np.clip(probability,0,1),logdensity


def render(a,d):
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,3,figsize=(7.15,5.3),layout='constrained')
        xmax=max(.3,float(np.ceil(a['halfwidth'].max()*10)/10));allres=np.r_[a['residual'],a['residual']-a['group_center']];ylo=min(-.7,float(np.floor(allres.min()*10)/10));yhi=max(.7,float(np.ceil(allres.max()*10)/10))
        for col,name in enumerate(('M13','M5','M30')):
            for row in (0,1):
                ax=axes[row,col]
                for phase,color in [('lower_MS',BLUE),('turnoff',ORANGE)]:
                    use=(a['system']==name)&(a['phase']==phase);value=a['residual'][use]-(a['group_center'][use] if row else 0)
                    ax.scatter(a['halfwidth'][use],value,s=5,lw=0,alpha=.5,color=color,rasterized=True,label='Lower MS' if phase=='lower_MS' else 'Turnoff')
                xx=np.linspace(0,xmax,100);ax.plot(xx,xx,':',color='.5',lw=.6);ax.plot(xx,-xx,':',color='.5',lw=.6);ax.axhline(0,color='.4',ls=':',lw=.5)
                ax.set(xlim=(0,xmax),ylim=(ylo,yhi),xlabel='68% half-width (mag)',ylabel='Modulus residual (mag)' if not row else 'Centered residual (mag)',title=name)
        axes[0,0].legend(frameon=False,fontsize=5);figure(d,'sequence_center_width',f,[dict(scope='Calibrated fixed sequence; exact same members; symmetric half-width diagonals are a visual guide, exact asymmetric interval results in table')])


def main():
    require_compute();d=folder('width');inputs={};t=read(STREAM/'sequence',('predictions.npz','TYPE','ls_id','source_id','system','native','mask','m','err','ref','q','new','calibrated','new_common','reference_phase'),inputs)
    selected=np.isin(t['system'],C['ridge']['systems'])&~t['mask']&t['new_common'];t={k:v[selected] for k,v in t.items()};assert (t['TYPE']=='PSF').all()
    b=read(STREAM/'sequence',('shared_bootstrap.npz','ls_id','multiplicity',*['scale_'+s for s in C['ridge']['systems']]),inputs);j=index_join(t['ls_id'],b['ls_id']);assert (j>=0).all();multiplicity=b['multiplicity'][:,j]
    calibration=json.loads((STREAM/'sequence/calibration.json').read_text());inputs['calibration']=sha(STREAM/'sequence/calibration.json');delta=calibration['delta_Mg'];delta_draw=np.array(calibration['zero_point_draws'])
    template=read(HALO/'native',('template.npz','south','north','phase','north_support'),inputs);mean=[None]*len(t['ls_id']);logw=[None]*len(mean);sigma=np.zeros(len(mean));qrf=[None]*len(mean)
    from v3_13_clusters import model_and_history
    for code,side in enumerate(('south','north')):
        ii=np.flatnonzero(t['native']==code)
        if not len(ii):continue
        nodes,prior=polyline(template,side);mm,ww,ss=sequence_mixture(t['m'][ii],t['err'][ii],nodes,prior)
        for at,m,w,s in zip(ii,mm,ww,ss):mean[at]=m;logw[at]=w;sigma[at]=s
        model,offset,_=model_and_history(side,inputs)
        for start in range(0,len(ii),128):
            use=ii[start:start+128];identity=np.where(t['source_id'][use]>0,t['source_id'][use],-t['ls_id'][use]);z=frozen_qrf(model,t['m'][use],t['err'][use],identity);assert z['supported'].all()
            np.testing.assert_allclose(z['distance']*10**(offset/5),t['q'][use],atol=2e-6,rtol=2e-6)
            for k,at in enumerate(use):qrf[at]=(z['mu_edges'][k]+offset,z['mass'][k])
    rows=[];points=[]
    for name in C['ridge']['systems']:
        for phase in ('all','lower_MS','turnoff','other'):
            ii=np.flatnonzero((t['system']==name)&((t['reference_phase']==phase) if phase!='all' else True))
            if not len(ii):continue
            ref_mu=10+5*np.log10(t['ref'][ii]);ref_shift=5*np.log10(b['scale_'+name]);ref_sigma=float(np.std(ref_shift,ddof=1));mult=multiplicity[:,ii]
            for method,key in [('QRF','q'),('Fixed sequence','new'),('Calibrated sequence','calibrated')]:
                qmu=10+5*np.log10(t[key][ii]);res=qmu[:,2]-ref_mu;center=float(np.median(res));width=qmu[:,3]-qmu[:,1];raw68=(qmu[:,1]<=ref_mu)&(ref_mu<=qmu[:,3]);raw95=(qmu[:,0]<=ref_mu)&(ref_mu<=qmu[:,4]);center68=(qmu[:,1]-center<=ref_mu)&(ref_mu<=qmu[:,3]-center)
                mu0=np.array([count_median(res-ss+(delta-delta_draw[k] if key=='calibrated' else 0),mult[k]) for k,ss in enumerate(ref_shift)])
                if method=='QRF':
                    edges=np.array([qrf[i][0] for i in ii]);mass=np.array([qrf[i][1] for i in ii]);np.testing.assert_allclose(mass.sum(axis=1),1,atol=1e-10)
                    u,logdensity=uniform_convolution(edges,mass,ref_mu,ref_sigma);uc,lc=uniform_convolution(edges,mass,ref_mu+center,ref_sigma)
                    raw_u=(mass*np.clip((ref_mu[:,None]-edges[:,:-1])/np.diff(edges,axis=1),0,1)).sum(axis=1)
                else:
                    mm=np.array([mean[i] for i in ii]);ww=np.array([logw[i] for i in ii]);ss=sigma[ii];point_delta=delta if key=='calibrated' else 0
                    raw_u,_=mixture_probability(ref_mu+point_delta,mm,ww,ss);CDF=[];LP=[];CC=[];LC=[]
                    for draw,rshift in enumerate(ref_shift):
                        adjust=delta_draw[draw] if key=='calibrated' else 0;u0,lp=mixture_probability(ref_mu+rshift+adjust,mm,ww,ss);c0,clp=mixture_probability(ref_mu+rshift+adjust+center,mm,ww,ss);CDF.append(u0);LP.append(lp);CC.append(c0);LC.append(clp)
                    u=np.mean(CDF,axis=0);uc=np.mean(CC,axis=0);logdensity=logsumexp(LP,axis=0)-np.log(len(ref_shift));lc=logsumexp(LC,axis=0)-np.log(len(ref_shift))
                rows.append(dict(system=name,phase=phase,method=method,N=len(ii),median_mu_residual=center,centered_NMAD=1.4826*np.median(abs(res-center)),median_width68=np.median(width),median_width95=np.median(qmu[:,4]-qmu[:,0]),raw_quantile_C68=raw68.mean(),raw_quantile_C95=raw95.mean(),centered_quantile_C68=center68.mean(),raw_analytic_C68=((raw_u>=.16)&(raw_u<=.84)).mean(),reference_convolved_C68=((u>=.16)&(u<=.84)).mean(),centered_reference_convolved_C68=((uc>=.16)&(uc<=.84)).mean(),reference_convolved_logscore=np.mean(logdensity),centered_reference_convolved_logscore=np.mean(lc),shared_center_lo=np.quantile(mu0,.025),shared_center_hi=np.quantile(mu0,.975),reference_sigma_mu=ref_sigma,calibration_draws_shared=key=='calibrated'))
                if phase in ('lower_MS','turnoff') and key=='calibrated':points.append(dict(system=np.full(len(ii),name),phase=np.full(len(ii),phase),halfwidth=width/2,residual=res,group_center=np.full(len(ii),center)))
                print(name,phase,method,len(ii),flush=True)
    csv_rows(d/'center_width.csv',rows);npz(d/'width_redraw.npz',**{k:np.concatenate([p[k] for p in points]) for k in points[0]});a={k:np.concatenate([p[k] for p in points]) for k in points[0]};render(a,d)
    complete(d,inputs,scope='unchanged current v5 common unmasked members, external reference centers and original quantiles preserved; no new distance model fit',probability='fixed sequence analytic Gaussian mixture reconstructs the original branch likelihood; all256 shared reference/M13-zero draws coupled. Frozen QRF full PDFs independently replayed and quantiles verified; exact uniform-bin Gaussian-reference convolution using small-error modulus approximation; no Gaussian approximation of QRF PDF',centering='subtract each evaluated system/phase/method sample median for diagnostic only; not a transferable correction',shared_uncertainty='original member multiplicities and system distances/zero points reused; common uncertainties do not independently shrink with star count')


if __name__=='__main__':main()
