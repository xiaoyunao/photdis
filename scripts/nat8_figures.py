#!/usr/bin/env python3
"""Three fixed science groups, rendered with the locked Pal5/v3.8 style."""
from nat8_common import *
from halo_common import BLUE,ORANGE,GREEN
from paper_v1_figures import NAVY,GRAY
from scipy.special import ndtr
import argparse,shutil

COLORS={'E':BLUE,'CalE':ORANGE,'QRF':'.1','CalE_marg':GREEN}
LABELS={'E':'Original E','CalE':'Calibrated E','QRF':'Frozen QRF','CalE_marg':'Parameter mixture'}


def table(p):return list(csv.DictReader(p.open()))


def errorpoint(ax,x,r,value,lo,hi,color,marker='o',label=None):
    y=float(r[value]);a=float(r[lo]);b=float(r[hi])
    if np.isfinite([y,a,b]).all():ax.errorbar(x,y,yerr=[[max(0,y-a)],[max(0,b-y)]],fmt=marker,ms=3,lw=.7,color=color,label=label)


def render(src,d):
    d.mkdir(parents=True,exist_ok=True);summary=table(src/'summary.csv');paired=table(src/'paired.csv');geo=load(src/'geometry.npz');cases=json.loads((src/'cases.json').read_text());curves=load(src/'example_curves.npz');models=json.loads((src/'models.json').read_text());pars=json.loads((src/'parameters.json').read_text());x=curves['mu']
    def row(s,m,phase,cohort='common'):return next(r for r in summary if r['system']==s and r['method']==m and r['phase']==phase and r['domain']=='all' and r['cohort']==cohort)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,3,figsize=(7.15,4.9),layout='constrained')
        for col,s in enumerate(C['systems']):
            obs=geo[s+'_colors'];ok=geo[s+'_support'];mc=geo[s+'_model_colors'];ax=axes[0,col];ax.scatter(obs[ok,0],obs[ok,1],s=4,lw=0,alpha=.45,color=BLUE,rasterized=True,label='E supported');ax.scatter(obs[~ok,0],obs[~ok,1],s=9,lw=.5,facecolor='none',edgecolor=GRAY,label='Outside E support')
            for j,(left,right) in enumerate(zip(geo[s+'_left'],geo[s+'_right'])):ax.plot(mc[[left,right],0],mc[[left,right],1],color=NAVY,lw=.7,label='Original E curve' if j==0 else None)
            ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$(r-z)_0$',title=s)
            ax=axes[1,col]
            for use,color,openmark in [(ok,BLUE,False),(~ok,GRAY,True)]:ax.scatter(obs[use,0],geo[s+'_normal'][use],s=6,lw=.5 if openmark else 0,facecolor='none' if openmark else color,edgecolor=color,alpha=.6,rasterized=True)
            ax.axhspan(-3,3,color='.93',lw=0,zorder=-1);ax.axhline(0,color='.4',ls=':',lw=.6);ax.set_yscale('symlog',linthresh=3);ax.set(xlabel=r'$(g-r)_0$',ylabel='Orthogonal residual / error')
        axes[0,0].legend(frameon=False,fontsize=5,loc='upper left');figure(d,'two_color_geometry',f,[dict(group=1,note='All 933 primary PSFs; original physical segments and full shared-r covariance. Support is frozen E node support. Orthogonal residual shown in sigma; symlog preserves all outliers. Tangent is zero at interior nearest segment by construction, endpoints in table.')])
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(3,3,figsize=(7.15,6.8),layout='constrained')
        for col,s in enumerate(C['systems']):
            methods=['E','CalE','QRF']+(['CalE_marg'] if s!='M5' else []);ax=axes[0,col];ax.axhspan(-.1,.1,color='.93',lw=0);ax.axhline(0,color='.4',ls=':',lw=.6)
            for phase,color,offset,marker in [('lower_MS',BLUE,-.12,'o'),('turnoff',ORANGE,.12,'s')]:
                for j,m in enumerate(methods):errorpoint(ax,j+offset,row(s,m,phase),'median_mu_residual','center_lo','center_hi',color,marker,'Lower MS' if phase=='lower_MS' and j==0 else ('Turnoff' if phase=='turnoff' and j==0 else None))
            ax.set(xticks=range(len(methods)),xticklabels=['E','Cal E','QRF']+(['Mix'] if s!='M5' else []),ylabel=r'Median $\Delta\mu$ (mag)',title=s+(' · cross prediction' if s=='M5' else ' · fixed transfer'))
            comparisons=[('E','CalE','E→Cal E'),('CalE','QRF','Cal E→QRF')]+([('CalE','CalE_marg','Cal E→Mix')] if s!='M5' else [])
            for rr,metric in [(1,'raw_logscore'),(2,'raw_CRPS')]:
                ax=axes[rr,col];ax.axhline(0,color='.4',ls=':',lw=.6)
                for phase,color,offset,marker in [('lower_MS',BLUE,-.12,'o'),('turnoff',ORANGE,.12,'s')]:
                    for j,(a,b,label) in enumerate(comparisons):
                        r=next(r for r in paired if r['system']==s and r['cohort']=='common' and r['domain']=='all' and r['phase']==phase and r['first']==a and r['second']==b);errorpoint(ax,j+offset,r,metric+'_delta',metric+'_lo',metric+'_hi',color,marker)
                ax.set(xticks=range(len(comparisons)),xticklabels=[v[2] for v in comparisons],ylabel=r'$\Delta$ log score (nat)' if rr==1 else r'$\Delta$ CRPS (mag)');ax.tick_params(axis='x',labelsize=5)
        axes[0,0].legend(frameon=False,fontsize=5);figure(d,'cross_prediction_transfer_scores',f,[dict(group=2,note='Common E/QRF-supported identities, 95% shared member/reference refit intervals. Positive log-score and negative CRPS changes improve. M5 conditional shared-reference definition differs from archived v7; QRF zero-density points remain -infinity and nonfinite contrasts are not plotted. All values/counts retained in tables. Mix is external parameter-mixture prediction.')])
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(3,3,figsize=(7.15,6.5),layout='constrained')
        for col,s in enumerate(C['systems']):
            ax=axes[0,col];model=models[s+'_E'];mag=np.array(model['mag']);phase=np.array(model['phase']);valid=np.array(model['supported'])&(mag[:,1]>=3)&(mag[:,1]<=8)&np.isin(phase,(0,2));ids=np.flatnonzero(valid);gr=mag[:,0]-mag[:,1];a=np.array(pars['full']['point']['a'])
            for seg in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                if len(seg)<2:continue
                ax.plot(gr[seg],mag[seg,1],color=BLUE,lw=.8,label='Original E' if seg[0]==ids[0] else None);ax.plot(gr[seg],mag[seg,1]+a[0]+a[1]*(gr[seg]-.3),color=ORANGE,lw=.8,label='Full-M5 mapping' if seg[0]==ids[0] else None)
            ax.set(xlabel='Native node g-r',ylabel=r'$M_r$ (mag)',title=s);ax.invert_yaxis();methods=['E','CalE','QRF']+(['CalE_marg'] if s!='M5' else [])
            for r,metric in [(1,'W68'),(2,'W95')]:
                ax=axes[r,col]
                for phase,color,offset,mark in [('lower_MS',BLUE,-.12,'o'),('turnoff',ORANGE,.12,'s')]:
                    for j,m in enumerate(methods):errorpoint(ax,j+offset,row(s,m,phase),metric,metric+'_lo',metric+'_hi',color,mark)
                ax.set(xticks=range(len(methods)),xticklabels=['E','Cal E','QRF']+(['Mix'] if s!='M5' else []),ylabel=metric+' (mag)',ylim=(0,None))
        axes[0,0].legend(frameon=False,fontsize=5);figure(d,'luminosity_mapping_widths',f,[dict(group=2,note='Same original node set remains after gray correction, including nodes shifted outside original Mr range. Top: frozen full-M5 transfer map, not evaluated in-sample M5 forecast; M5 width/score panels use two-fold cross predictions. Bottom: original conditional predictive widths, without target reference kernel. No extra scatter fit.')])
    for name,cc,tail in [('calibration_complete_PDF',[r for r in cases if r['kind']!='QRF_decline'],False),('fixed_three_QRF_tails',[r for r in cases if r['kind']=='QRF_decline'],True)]:
        cols=3 if tail else 2;nr=(len(cc)+cols-1)//cols
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(2*nr,cols,figsize=(7.15,3.6*nr),height_ratios=[3,1]*nr,layout='constrained',squeeze=False);pa=axes[::2];ca=axes[1::2]
            for ax,cax,c in zip(pa.ravel(),ca.ravel(),cc):
                k=c['case'];ref=c['ref_mu'];rs=c['ref_sigma'];methods=['QRF'] if tail else list(c['methods']);lo=ref-.6;hi=ref+.6;peak=0.;full=[]
                for m in methods:
                    y=curves[f'{k}_{m}_density'];cf=curves[f'{k}_{m}_CDF'];ax.plot(x-ref,y,color=COLORS[m],lw=.8,label=LABELS[m]);cax.plot(x-ref,cf,color=COLORS[m],lw=.6);q=np.array(c['methods'][m]['q']);lo=min(lo,q[0]-.15);hi=max(hi,q[-1]+.15);peak=max(peak,float(y.max()));full.extend([x[np.searchsorted(cf,1e-9)],x[min(len(x)-1,np.searchsorted(cf,1-1e-9))]])
                if tail:
                    y=curves[f'{k}_QRF_centered_density'];cf=curves[f'{k}_QRF_centered_CDF'];shift=float(curves[f'{k}_QRF_center'][0]);ax.plot(x-ref,y,color=ORANGE,ls='--',lw=.8,label='Diagnostic centered');cax.plot(x-ref,cf,color=ORANGE,ls='--',lw=.6);peak=max(peak,float(y.max()));lo=min(lo,q[0]-shift-.15);hi=max(hi,q[-1]-shift+.15);full.extend([min(full)-shift,max(full)-shift])
                ax.fill_between(x-ref,np.exp(-.5*((x-ref)/rs)**2)*.8*peak,color='.85',lw=0,label='Reference (scaled)',zorder=0);ax.set(xlim=(lo-ref,hi-ref),ylim=(0,1.12*peak),ylabel=r'Density (mag$^{-1}$)',title=f"{c['system']} · {c['ls_id']}");cax.plot(x-ref,ndtr((x-ref)/rs),color='.6',lw=.6,ls=':');cax.set(xlim=(min(full)-ref-.1,max(full)-ref+.1),ylim=(0,1),yticks=[0,.5,1],ylabel='Full CDF',xlabel=r'$\mu-\mu_{\rm ref}$ (mag)');cax.tick_params(labelsize=6)
            for ax in list(pa.ravel()[len(cc):])+list(ca.ravel()[len(cc):]):ax.set_visible(False)
            handles,labels=pa[0,0].get_legend_handles_labels();f.legend(handles,labels,loc='outside upper center',frameon=False,fontsize=5,ncol=3);figure(d,name,f,[dict(group=3,case=c['case'],TYPE=c['TYPE'],ls_id=c['ls_id'],system=c['system'],note='Fixed original v7 identities. Complete CDF in separate row; displayed gray reference profile height only scaled. Conditional raw PDFs, no scoring-reference convolution applied to plotted prediction.') for c in cc])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.source:src=args.source;d=args.out
    else:
        from photdis.stage1 import require_compute
        require_compute();src=folder('public_data');d=folder('figures')
        for stage,files in [('evaluate',{'summary.csv':'summary.csv','paired.csv':'paired.csv','phase_gaps.csv':'phase_gaps.csv'}),('geometry',{'redraw.npz':'geometry.npz'}),('examples',{'curves.npz':'example_curves.npz','cases.json':'cases.json','component_parameters.npz':'component_parameters.npz'}),('fit',{'parameters.json':'parameters.json'})]:
            for name,target in files.items():shutil.copy2(OUT/stage/name,src/target)
        shutil.copy2(OLD/'prepare/curves.json',src/'models.json')
    check_templates();render(src,d);print('Rendered five plates in three scientific groups',flush=True)


if __name__=='__main__':main()
