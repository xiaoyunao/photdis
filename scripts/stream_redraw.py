#!/usr/bin/env python3
"""Redraw public scientific panels without source catalogs or the private cube."""
import argparse
import healpy as hp
from matplotlib.colors import SymLogNorm
from stream_common import *
from v3_final_atlas import COLORMAP


def load(p):
    with np.load(p) as a:return {k:a[k] for k in a.files}


def rgb(root,d):
    for p in sorted((root/'rgb').glob('rgb_*.npz')):
        a=load(p);bounds=a['channel_kpc'];rows=list(csv.DictReader(p.with_suffix('.csv').open()))
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(2,1,figsize=(7.15,7),subplot_kw={'projection':'aitoff'},layout='constrained')
            for ax,image,label in zip(axes,a['rgb'],('CMD density','Positive residual')):
                mesh=ax.pcolormesh(a['l'],a['b'],np.zeros(a['mask'].shape),shading='flat',rasterized=True)
                mesh.set_array(None);mesh.set_facecolor(image.reshape(-1,3));ax.set_axisbelow(False)
                ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°']);ax.set_yticks(np.deg2rad([-60,-30,0,30,60]));ax.tick_params(labelsize=7);ax.grid(ls=':',lw=.5,color='.5');ax.set_title(label);ax.set_facecolor('.88')
                for name in ('Pal5.ibata2024','Sagittarius.antoja2020'):
                    t,_=track(name);ll,bb=hp.Rotator(coord=('C','G'))(t['ra'][::200],t['dec'][::200],lonlat=True)
                    ax.plot(np.deg2rad(-((ll+180)%360-180)),np.deg2rad(bb),'.',ms=1.5,color='.3',label='Reference track' if name.startswith('Pal') else None)
            axes[0].legend(frameon=False,fontsize=6,loc='lower right');f.supxlabel('   |   '.join(f'{c}: {lo:.2f}–{hi:.2f} kpc' for c,(lo,hi) in zip('BGR',bounds))+'\nGalactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
            figure(d,p.stem,f,rows)
            f,axs=plt.subplots(2,3,figsize=(7.15,5.1),subplot_kw={'projection':'aitoff'},layout='constrained')
            for row,values in enumerate((a['density'],a['signed_residual'])):
                cap=float(a['scalar_cap'][row]);lin=cap/100
                for j,ax in enumerate(axs[row]):
                    z=ax.pcolormesh(a['l'],a['b'],np.ma.masked_invalid(values[j]),cmap=COLORMAP if row==0 else 'RdBu_r',norm=SymLogNorm(lin,vmin=0 if row==0 else -cap,vmax=cap),rasterized=True)
                    ax.set_axisbelow(False);ax.set_xticks(np.deg2rad([-120,0,120]),labels=['120°','0°','240°']);ax.tick_params(labelsize=6);ax.grid(ls=':',lw=.4);ax.set_title(f'{bounds[j,0]:.2f}–{bounds[j,1]:.2f} kpc')
                ticks=10.**np.arange(np.ceil(np.log10(lin)),np.floor(np.log10(cap))+1)
                f.colorbar(z,ax=axs[row].tolist(),location='bottom',fraction=.03,pad=.06,label='CMD-selected source density (deg⁻²)' if row==0 else 'Signed residual (deg⁻²)',extend='max' if row==0 else 'both',ticks=np.r_[0,ticks] if row==0 else np.r_[-ticks[::-1],0,ticks],format=plt.FormatStrFormatter('%g'))
            figure(d,p.stem.replace('rgb_','physical_'),f,rows)


def local(root,d):
    from halo_common import BLUE,ORANGE,GREEN
    responses=list(csv.DictReader((root/'local/modulus_response.csv').open()))
    for p in sorted((root/'local').glob('*_redraw.npz')):
        a=load(p);roi=p.stem.replace('_redraw','');cfg=C['local'][roi]
        for model in ('FOS','Pal5_control') if roi.startswith('Pal') else ('FOS',):
            with plt.rc_context(PAPER):
                f,axs=plt.subplots(2,3,figsize=(7.15,5.3),layout='constrained')
                for ax,var in zip(axs[0],('strict_original','gr_E010','gr_complete')):
                    pair=a[model+'_hess_'+var];h=pair[0]-pair[1];cap=max(float(np.quantile(abs(h),.99)),.1)
                    z=ax.pcolormesh(a['hess_gr_edges'],a['hess_g_edges'],h.T,cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
                    for lo,hi in zip(a[model+'_envelope_lower'],a[model+'_envelope_upper']):ax.plot(lo,a[model+'_envelope_g'],color=ORANGE,lw=.6);ax.plot(hi,a[model+'_envelope_g'],color=ORANGE,lw=.6)
                    ax.set(xlim=(-.2,1.2),ylim=(24,14),xlabel=r'$(g-r)_0$',ylabel=r'$g_0$ (mag)',title=var.replace('_',' '));f.colorbar(z,ax=ax,fraction=.04,label='Target − control (deg⁻² bin⁻¹)')
                mu=10+5*np.log10(cfg['distance_kpc']) if 'distance_kpc' in cfg else cfg['reference_mu'];ax=axs[1,0]
                for var,color in zip(('strict_original','strict_E010','gr_E010','gr_complete'),('.5',GREEN,BLUE,ORANGE)):
                    rows=[r for r in responses if r['roi']==roi and r['model']==model and r['phase']=='MS_TO' and r['variant']==var]
                    ax.plot([float(r['mu']) for r in rows],[float(r['signed_excess']) for r in rows],color=color,lw=.8,label=var.replace('_',' '))
                ax.axvline(mu,color='.3',lw=.7,ls=':');ax.axhline(0,color='.5',lw=.5);ax.set(xlabel='Distance modulus (mag)',ylabel='Target − control (deg⁻²)');ax.legend(frameon=False,fontsize=5.5)
                for ax,sigma in zip(axs[1,1:],(.1,.3)):
                    values=a[model+'_sky_smoothed_'+str(sigma)][2];finite=values[np.isfinite(values)];cap=max(float(np.quantile(abs(finite),.99)),.1) if len(finite) else .1
                    z=ax.pcolormesh(a['sky_along_edges'],a['sky_offset_edges'],np.ma.masked_invalid(values.T),cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
                    ax.axhline(0,color='.3',lw=.5,ls=':');ax.set(xlabel='Fraction along fixed RA segment',ylabel='Cross-track offset (deg)',title=f'Fixed μ · σ={sigma:g}°');f.colorbar(z,ax=ax,fraction=.04,label='Signed excess (deg⁻²)')
                figure(d,roi+'_'+model+'_recovery',f,[dict(roi=roi,model=model)])


def numeric_rows(p):
    rows=list(csv.DictReader(p.open()))
    for r in rows:
        for k,x in r.items():
            if x in ('True','False'):r[k]=x=='True'
            elif x=='':r[k]=None
            else:
                try:r[k]=float(x)
                except ValueError:pass
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('rgb','local','sequence','conditioning'));p.add_argument('--root',type=Path,default=ROOT/'manifests/stream_response_20261119');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.stage=='rgb':rgb(a.root,a.output)
    elif a.stage=='local':local(a.root,a.output)
    elif a.stage=='sequence':
        from stream_sequence import plot_results
        t=load(a.root/'sequence/predictions.npz');template=load(a.root/'sequence/template_redraw.npz');delta=json.loads((a.root/'sequence/calibration.json').read_text())['delta_Mg']
        plot_results(a.output,numeric_rows(a.root/'sequence/distance_scale.csv'),t,t['reference_phase'],t['new'],t['new_common'],delta,template)
    else:
        from stream_conditioning import plot_results
        plot_results(a.output,numeric_rows(a.root/'conditioning/estimands.csv'))


if __name__=='__main__':main()
