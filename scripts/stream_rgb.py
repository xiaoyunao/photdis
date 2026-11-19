#!/usr/bin/env python3
"""True three-response RGBs derived from all61 frozen modulus slices."""
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
from photdis.stage1 import require_compute
from stream_common import *
from halo_filtering import channel_weights
from halo_atlas import grid,smooth
from v3_final_atlas import COLORMAP
from matplotlib.colors import SymLogNorm


def stretch(values,mask,residual=False):
    out=[];rows=[]
    for j,v in enumerate(values):
        valid=v[mask];valid=valid[np.isfinite(valid)]
        if residual:valid=valid[valid>0]
        lo=0. if residual else np.quantile(valid,.05) if len(valid) else 0.
        hi=np.quantile(valid,.995) if len(valid) else lo+1
        hi=max(hi,lo+1e-8)
        t=np.arcsinh(10*np.clip((v-lo)/(hi-lo),0,1))/np.arcsinh(10)
        out.append(t);rows.append(dict(channel=('B','G','R')[j],lo=float(lo),hi=float(hi),saturated_fraction=float((v[mask]>hi).mean())))
    return np.stack(out[::-1],axis=-1),rows


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--redraw',action='store_true');parser.add_argument('--cube-stage',type=Path,default=OLD/'atlas');parser.add_argument('--output-stage',default='rgb');args=parser.parse_args()
    require_compute();d=folder(args.output_stage);inputs={}
    record=json.loads((args.cube_stage/'complete.json').read_text())
    path=verified(args.cube_stage/'cmd_filter_cube.fits',record['outputs']['cmd_filter_cube.fits']);inputs['raw_cube']=sha(path);inputs['cube_receipt']=sha(args.cube_stage/'complete.json')
    redraw=args.redraw
    with fits.open(path,memmap=True) as h:
        mu=h['MU'].data;area=h['NATIVE_AREAS'].data[:,2].sum(axis=0);mask=h['DISPLAY_MASK'].data.astype(bool)
        groups=[('overview',np.array([[10,15],[15,26],[26,100]],float))]
        for a,b in ((10**.8,10),(10,15),(15,26),(26,50),(50,100)):
            edges=a*(b/a)**(np.arange(4)/3);groups.append((f'bin_{a:.1f}_{b:g}',np.column_stack((edges[:-1],edges[1:]))))
        intervals=np.concatenate([x[1] for x in groups]);w=channel_weights(mu,intervals)
        counts=None if redraw else np.tensordot(w,h['COUNTS'].data[1,0],axes=(1,0))
    if redraw:
        with np.load(d/'physical_channels.npz') as a:maps=np.stack((a['density'],a['signed_residual']),axis=1)
    sa=np.maximum(smooth(area,.3),0) if not redraw else None;ba=np.maximum(smooth(area,5),0) if not redraw else None
    def work(c):
        s=np.maximum(smooth(c,.3),0);b=np.maximum(smooth(c,5),0)
        density=np.divide(s,sa,out=np.full_like(s,np.nan),where=sa>0)
        background=np.divide(b,ba,out=np.full_like(b,np.nan),where=ba>0)
        density[~mask]=np.nan;background[~mask]=np.nan
        return density,density-background
    if not redraw:
        with ThreadPoolExecutor(6) as pool:maps=np.array(list(pool.map(work,counts)))
        npz(d/'physical_channels.npz',mu=mu,intervals=intervals,weights=w,weighted_counts=counts.astype('f4'),area=area,
            density=maps[:,0].astype('f4'),signed_residual=maps[:,1].astype('f4'),mask=mask)
    l,b,pix=grid();allmeta=[]
    for gi,(name,bounds) in enumerate(groups):
        density=maps[3*gi:3*gi+3,0];residual=maps[3*gi:3*gi+3,1]
        rendered=[];stats=[]
        caps=np.array([max(float(np.quantile(abs(v[np.isfinite(v)]),.995)),1e-5) for v in (density,residual)])
        with plt.rc_context(PAPER):
            f,axes=plt.subplots(2,1,figsize=(7.15,7.0),subplot_kw={'projection':'aitoff'},layout='constrained')
            for kind,ax,values in zip(('CMD density','Positive residual'),axes,(density,residual)):
                rgb,st=stretch(values,mask,kind=='Positive residual');image=rgb[pix]
                image[~mask[pix]]=.88
                # imshow on a non-rectilinear projection is replaced by per-cell RGBA.
                mesh=ax.pcolormesh(l,b,np.zeros(pix.shape),shading='flat',rasterized=True)
                mesh.set_array(None);mesh.set_facecolor(np.nan_to_num(image.reshape(-1,3),nan=.88))
                ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°'])
                ax.set_yticks(np.deg2rad([-60,-30,0,30,60]));ax.tick_params(labelsize=7);ax.grid(ls=':',lw=.5,color='.5');ax.set_title(kind)
                ax.set_facecolor('.88');ax.set_axisbelow(False)
                for tname in ('Pal5.ibata2024','Sagittarius.antoja2020'):
                    import healpy as hp
                    t,p=track(tname);inputs[str(p.relative_to(ROOT))]=sha(p)
                    ll,bb=hp.Rotator(coord=('C','G'))(t['ra'][::200],t['dec'][::200],lonlat=True)
                    ax.plot(np.deg2rad(-((ll+180)%360-180)),np.deg2rad(bb),'.',ms=1.5,color='.3',label='Reference track' if tname.startswith('Pal') else None)
                for j,(row,ab) in enumerate(zip(st,bounds)):
                    stats.append(dict(view=kind,d_lo=ab[0],d_hi=ab[1],**row));allmeta.append(dict(group=name,view=kind,d_lo=ab[0],d_hi=ab[1],**row))
                rendered.append(image.astype('f4'))
            axes[0].legend(frameon=False,fontsize=6,loc='lower right')
            labels=[f'{c}: {a:.2f}–{b:.2f} kpc' for c,(a,b) in zip(('B','G','R'),bounds)]
            f.supxlabel('   |   '.join(labels)+'\nGalactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
            figure(d,'rgb_'+name,f,stats,dict(l=l,b=b,rgb=np.array(rendered),density=density[:,pix].astype('f4'),signed_residual=residual[:,pix].astype('f4'),mask=mask[pix],channel_kpc=bounds,scalar_cap=caps))
            f,axs=plt.subplots(2,3,figsize=(7.15,5.1),subplot_kw={'projection':'aitoff'},layout='constrained')
            for row,values in enumerate((density,residual)):
                finite=values[np.isfinite(values)];cap=max(float(np.quantile(abs(finite),.995)),1e-5);lin=cap/100
                for j,ax in enumerate(axs[row]):
                    z=ax.pcolormesh(l,b,np.ma.masked_invalid(values[j,pix]),cmap=COLORMAP if row==0 else 'RdBu_r',norm=SymLogNorm(lin,vmin=0 if row==0 else -cap,vmax=cap),rasterized=True)
                    ax.set_axisbelow(False);ax.set_xticks(np.deg2rad([-120,0,120]),labels=['120°','0°','240°']);ax.tick_params(labelsize=6);ax.grid(ls=':',lw=.4)
                    ax.set_title(f'{bounds[j,0]:.2f}–{bounds[j,1]:.2f} kpc')
                decades=np.array([10.**np.floor(np.log10(cap))])
                f.colorbar(z,ax=axs[row].tolist(),location='bottom',fraction=.03,pad=.06,label='CMD-selected source density (deg⁻²)' if row==0 else 'Signed residual (deg⁻²)',extend='max' if row==0 else 'both',ticks=np.r_[0,decades] if row==0 else np.r_[-decades[::-1],0,decades],format=plt.FormatStrFormatter('%g'))
            f.supxlabel('Galactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
            figure(d,'physical_'+name,f,stats)
    csv_rows(d/'stretch.csv',allmeta)
    csv_rows(d/'node_weights.csv',[dict(group=g,channel=c,d_lo=ab[0],d_hi=ab[1],mu=m,weight=float(w[3*i+j,k])) for i,(g,bs) in enumerate(groups) for j,(c,ab) in enumerate(zip('BGR',bs)) for k,m in enumerate(mu) if w[3*i+j,k]>0])
    np.testing.assert_allclose(w.sum(axis=1),1)
    complete(d,inputs,channel_order='B near,G middle,R far; sub-bins equal in modulus',geometry='Galactic Aitoff l0 center, increasing longitude left',stretch=C['rgb'],signed_data_retained=True)


if __name__=='__main__':main()
