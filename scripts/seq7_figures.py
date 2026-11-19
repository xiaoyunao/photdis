#!/usr/bin/env python3
"""Portable Pal5/v3.8-style plots from compact fixed-control evidence."""
from seq7_common import *
from halo_common import BLUE,ORANGE,GREEN
from paper_v1_figures import NAVY,GRAY
from seq7_predict import load
from scipy.special import ndtr
import argparse

COLORS={'A':GRAY,'B':BLUE,'C':ORANGE,'D':GREEN,'E':NAVY,'QRF':'.1'}
LABELS={'A':'A: fixed MIST1.0','B':'B: MIST1.2 control','C':'C: matched locus','D':'D: initial-mass weight','E':'E: early post-MS','QRF':'Frozen QRF'}


def table(p):return list(csv.DictReader(p.open()))


def render(src,d):
    d.mkdir(parents=True,exist_ok=True);summary=table(src/'summary.csv');paired=table(src/'paired.csv');cases=json.loads((src/'cases.json').read_text());curves=load(src/'example_curves.npz');geo=load(src/'geometry.npz');models=json.loads((src/'curves.json').read_text());mu=curves['mu']
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(2,3,figsize=(7.15,5.3),layout='constrained')
        for col,s in enumerate(C['systems']):
            ax=axes[0,col];ax.axhspan(-.1,.1,color='.92',lw=0);ax.axhline(0,color='.4',ls=':',lw=.6)
            for phase,color,offset,mark in [('lower_MS',BLUE,-.12,'o'),('turnoff',ORANGE,.12,'s')]:
                rr=[next(r for r in summary if r['system']==s and r['cohort']=='v5_commonABCDE' and r['phase']==phase and r['method']==m) for m in METHODS[:-1]]
                yy=np.array([float(r['median_mu_residual']) for r in rr]);lo=np.array([float(r['center_lo']) for r in rr]);hi=np.array([float(r['center_hi']) for r in rr]);xx=np.arange(5)+offset
                ax.errorbar(xx,yy,yerr=np.vstack((np.maximum(0,yy-lo),np.maximum(0,hi-yy))),fmt=mark+'-',ms=3,lw=.75,color=color,label='Lower MS' if phase=='lower_MS' else 'Turnoff')
            ax.set(xticks=range(5),xticklabels=list('ABCDE'),xlabel='Fixed control',ylabel=r'Median $\mu-\mu_{\rm ref}$ (mag)',title=s)
            ax=axes[1,col]
            for phase,color,offset,mark in [('lower_MS',BLUE,-.1,'o'),('turnoff',ORANGE,.1,'s')]:
                rr=[next(r for r in paired if r['system']==s and r['cohort']=='primary' and r['phase']==phase and r['first']==a and r['second']==b) for a,b in zip('ABCD','BCDE')]
                yy=np.array([float(r['raw_logscore_delta']) for r in rr]);lo=np.array([float(r['raw_logscore_lo']) for r in rr]);hi=np.array([float(r['raw_logscore_hi']) for r in rr]);ax.errorbar(np.arange(4)+offset,yy,yerr=np.vstack((np.maximum(0,yy-lo),np.maximum(0,hi-yy))),fmt=mark,ms=3,lw=.7,color=color)
            ax.axhline(0,color='.4',ls=':',lw=.6);ax.set(xticks=range(4),xticklabels=['A→B','B→C','C→D','D→E'],xlabel='Same-source change',ylabel=r'$\Delta$ mean marginal log score (nat)')
        axes[0,0].legend(frameon=False,fontsize=6);figure(d,'controlled_scales_scores',f,[dict(note='Top: unchanged-v5 cohort intersected with common support of all five controls, same stars at every point. Bottom: full primary, each pair common support. 95% shared member/reference bootstrap intervals; tables include exact N. Gray band is registered absolute-scale target, not applied correction.')])
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(3,3,figsize=(7.15,7.1),layout='constrained')
        for row,s in enumerate(C['systems']):
            for col in (0,1):
                ax=axes[row,col];ax.scatter(geo[s+('_gr' if col==0 else '_rz')],geo[s+'_Mg'],s=4,lw=0,alpha=.4,color=BLUE,rasterized=True,label='All primary members')
                for m in ('A','B','C','E'):
                    z=models[s+'_'+m];mag=np.array(z['mag']);phase=np.array(z['phase']);valid=np.array(z['supported'])&(mag[:,1]>=3)&(mag[:,1]<=8)&np.isin(phase,(0,2) if m=='E' else (0,));ids=np.flatnonzero(valid)
                    for segment in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
                        if len(segment)<2:continue
                        color=mag[segment,col]-mag[segment,col+1];ax.plot(color,mag[segment,0],lw=.8,color=COLORS[m],ls='--' if m=='E' else '-',label=LABELS[m] if segment[0]==ids[0] else None)
                ax.set(xlabel=r'$(g-r)_0$' if col==0 else r'$(r-z)_0$',ylabel=r'$M_{g,\rm ref}$ (mag)',title=s,ylim=(8.5,2.5))
            ax=axes[row,2];gr=geo[s+'_gr'];ok=geo[s+'_E_support'];ax.scatter(gr[ok],geo[s+'_E_post'][ok],s=5,lw=0,alpha=.5,color=ORANGE,rasterized=True);ax.scatter(gr[~ok],geo[s+'_E_post'][~ok],s=7,lw=.5,facecolor='none',edgecolor=GRAY);ax.set(xlabel=r'$(g-r)_0$',ylabel='Early post-MS posterior mass',ylim=(-.03,1.03),title=s)
        axes[0,0].legend(frameon=False,fontsize=5,loc='lower right');figure(d,'native_loci_branch_mass',f,[dict(note='Native photometry; reference distance used only for vertical CMD display. E branch restricted to original Mr3..8. Open points outside E color support; full parent retained.')])
    groups=[('QRF_complete_PDF',[c for c in cases if c['kind'].startswith('QRF')],['QRF']),('sequence_complete_PDF',[c for c in cases if c['kind'].startswith('sequence')],list('ABCDE')),('branch_complete_PDF',[c for c in cases if c['kind']=='branch'],['D','E','QRF'])]
    for name,cc,methods in groups:
        ncol=3 if name.startswith('branch') else 2;nrow=(len(cc)+ncol-1)//ncol
        with plt.rc_context(PAPER):
            f,all_axes=plt.subplots(2*nrow,ncol,figsize=(7.15,3.65*nrow),height_ratios=[3,1]*nrow,layout='constrained',squeeze=False);axes=all_axes[::2];caxes=all_axes[1::2]
            for ax,cax,c in zip(axes.ravel(),caxes.ravel(),cc):
                k=c['case'];ref=c['ref_mu'];rs=c['ref_sigma'];reference=np.exp(-.5*((mu-ref)/rs)**2)
                lo=ref-.6;hi=ref+.6;peak=0.
                for m in methods:
                    if m not in c['methods']:continue
                    yy=curves[f'{k}_{m}_density'];ax.plot(mu-ref,yy,color=COLORS[m],lw=.8,ls='-' if c['methods'][m]['supported'] else ':',label=LABELS[m]);qq=np.array(c['methods'][m]['q']);peak=max(peak,float(yy.max()))
                    if not (name.startswith('branch') and m=='QRF'):lo=min(lo,qq[0]-.15);hi=max(hi,qq[-1]+.15)
                if methods==['QRF']:
                    ax.plot(mu-ref,curves[f'{k}_QRF_centered_density'],color=ORANGE,lw=.8,ls='--',label='Diagnostic centered');peak=max(peak,float(curves[f'{k}_QRF_centered_density'].max()));shift=float(curves[f'{k}_QRF_center'][0]);qq=np.array(c['methods']['QRF']['q'])-shift;lo=min(lo,qq[0]-.15);hi=max(hi,qq[-1]+.15)
                # Reference uncertainty is much narrower than these predictions.
                # Scale only its displayed height, preserving the actual kernel
                # in the case metadata and every scientific score.
                ax.fill_between(mu-ref,reference*.8*peak,color='.85',lw=0,label='Reference (scaled)',zorder=0)
                ax.set(xlim=(lo-ref,hi-ref),ylim=(0,1.12*peak),ylabel=r'Predictive density (mag$^{-1}$)',title=f"{c['system']} · {c['ls_id']}")
                nonzero=[]
                for m in methods:
                    if m not in c['methods']:continue
                    vv=curves[f'{k}_{m}_CDF'];cax.plot(mu-ref,vv,color=COLORS[m],lw=.6);nonzero.extend([mu[np.searchsorted(vv,1e-9)],mu[min(len(mu)-1,np.searchsorted(vv,1-1e-9))]])
                if methods==['QRF']:cax.plot(mu-ref,curves[f'{k}_QRF_centered_CDF'],color=ORANGE,ls='--',lw=.6);nonzero.extend([min(nonzero)-shift,max(nonzero)-shift])
                cax.plot(mu-ref,ndtr((mu-ref)/rs),color='.65',lw=.6,ls=':');cax.set(xlim=(min(nonzero)-ref-.1,max(nonzero)-ref+.1),ylim=(0,1),yticks=[0,.5,1],ylabel='Full CDF',xlabel=r'$\mu-\mu_{\rm ref}$ (mag)');cax.tick_params(labelsize=6)
            for ax in list(axes.ravel()[len(cc):])+list(caxes.ravel()[len(cc):]):ax.set_visible(False)
            handles,labels=axes[0,0].get_legend_handles_labels();f.legend(handles,labels,frameon=False,fontsize=5,loc='outside upper center',ncol=min(3,len(labels)));figure(d,name,f,[{k:v for k,v in c.items() if k!='methods'} for c in cc])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--out',type=Path);args=parser.parse_args()
    if args.source:src=args.source;d=args.out
    else:
        from photdis.stage1 import require_compute
        require_compute();src=folder('public_data');d=folder('figures')
        import shutil
        for stage,files in [('evaluate',['summary.csv','paired.csv','phase_gaps.csv']),('examples',['cases.json','example_curves.npz','geometry.npz','selected.csv','prediction_color_residuals.csv','QRF_M13_lowerMS_per_star.csv','QRF_centering_explanation.json']),('prepare',['curves.json'])]:
            for file in files:shutil.copy2(OUT/stage/file,src/file)
    check_templates();render(src,d);print('Rendered five science plates',d,flush=True)


if __name__=='__main__':main()
