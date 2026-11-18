#!/usr/bin/env python3
"""Show reference sequence and conditional luminosity intervals on identical stars."""
from matplotlib.colors import LogNorm
from photdis.stage1 import require_compute
from halo_common import *


def main():
    require_compute();d=folder('sequence_views');inputs={}
    t=read(OUT/'sequence',('same_source_predictions.npz','TYPE','ls_id','system','native','m','r','q','ref','mask','common','phase','simple','sequence'),inputs)
    theory=read(OUT/'native',('template.npz','south','north','phase','north_support'),inputs)
    rows=[]
    with plt.rc_context(PAPER):
        for name in ('M5','M13','M30','GD1_thin'):
            use=(t['system']==name)&t['common']&~t['mask'];ii=np.flatnonzero(use)
            if not len(ii):continue
            code=int(t['native'][ii[0]]);side=('south','north')[code]
            tr=read(ROOT/'data/science_validation_20261118/benchmark',(side+'_training_sample.npz','TYPE','ls_id','m','y','class_label'),inputs)
            colors=t['m'][ii,0]-t['m'][ii,1];mg=t['m'][ii,0]-10-5*np.log10(t['ref'][ii])
            # Display a predeclared thin color strip through the real members.
            c0,c1=np.quantile(colors,[.2,.8]);strip=(tr['m'][:,0]-tr['m'][:,1]>=c0)&(tr['m'][:,0]-tr['m'][:,1]<=c1)
            f,axes=plt.subplots(1,3,figsize=(7.15,3.1),layout='constrained')
            h,xe,ye=np.histogram2d(tr['m'][:,0]-tr['m'][:,1],tr['y'],(np.linspace(-.1,1.1,65),np.linspace(1,10,70)))
            z=axes[0].pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap='viridis',norm=LogNorm(1,max(2,h.max())),rasterized=True)
            f.colorbar(z,ax=axes[0],fraction=.035,label='Training stars / bin')
            axes[0].scatter(colors,mg,s=5,lw=0,color=ORANGE,alpha=.65,rasterized=True)
            axes[0].set(xlabel=r'$(g-r)_0$',ylabel=r'$M_g$ (mag)',ylim=(10,1),xlim=(-.1,1.1))
            for method,color in (('q',BLUE),('simple',GREEN),('sequence',ORANGE)):
                q=t[method][ii];lum=t['m'][ii,0,None]-10-5*np.log10(q[:,::-1])
                order=np.argsort(colors);c=colors[order];m=lum[order]
                axes[1].vlines(c,m[:,1],m[:,3],color=color,alpha=.2,lw=.6,rasterized=True)
                axes[1].scatter(c,m[:,2],s=3,lw=0,alpha=.55,color=color,rasterized=True,label={'q':'QRF','simple':'Label grid','sequence':'Old sequence'}[method])
                delta=5*np.log10(q[:,2]/t['ref'][ii])
                axes[2].scatter(t['r'][ii],delta,s=4,alpha=.45,lw=0,color=color,rasterized=True)
                rows.append(dict(system=name,method=method,n=len(ii),color_strip_lo=c0,color_strip_hi=c1,
                    training_total=len(tr['ls_id']),training_strip=int(strip.sum()),
                    interval_points_outside_axes=int(((c<-.1)|(c>1.1)|(m[:,2]<1)|(m[:,2]>10)).sum()),
                    residual_points_outside_axes=int(((t['r'][ii]<14)|(t['r'][ii]>23)|(delta<-2.5)|(delta>2.5)).sum())))
            axes[1].scatter(colors,mg,s=8,facecolors='none',edgecolors='.2',lw=.4,rasterized=True)
            axes[1].set(xlabel=r'$(g-r)_0$',ylabel=r'Conditional $M_g$ (mag)',ylim=(10,1),xlim=(-.1,1.1));axes[1].legend(frameon=False,fontsize=5)
            axes[2].axhline(0,color='.4',ls=':',lw=.6)
            axes[2].set(xlabel='Observed r (mag)',ylabel=r'$\Delta\mu$ (mag)',xlim=(14,23),ylim=(-2.5,2.5))
            figure(d,'sequence_luminosity_'+name,f,[r for r in rows if r['system']==name])
    csv_rows(d/'conditional_luminosity_census.csv',rows)
    complete(d,inputs,scope='same source identities for reference positions and model luminosity intervals; QRF/grid/sequence 16-84 intervals and medians, not a reconstructed full PDF',
        training_scope='original same-label independent object counts and class labels retained; overlap/development history follows sequence role records')


if __name__=='__main__':main()
