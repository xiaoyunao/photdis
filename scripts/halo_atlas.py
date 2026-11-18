#!/usr/bin/env python3
"""Assemble real native CMD counts and render the fixed Galactic density atlas."""
from concurrent.futures import ThreadPoolExecutor
import healpy as hp
from astropy.io import fits
from matplotlib.colors import SymLogNorm, Normalize
from photdis.stage1 import require_compute
from halo_common import *
from halo_filtering import channel_weights
from v3_final_atlas import COLORMAP

NPIX=hp.nside2npix(C['nside'])
PIXAREA=hp.nside2pixarea(C['nside'],degrees=True)


def smooth(x,sigma):
    v=hp.smoothing(hp.reorder(np.asarray(x,float),n2r=True),sigma=np.deg2rad(sigma),
                   lmax=C['smoothing']['lmax'],iter=0)
    return hp.reorder(v,r2n=True)


def grid():
    l=np.linspace(-np.pi,np.pi,1441);b=np.linspace(-np.pi/2,np.pi/2,721)
    lc,bc=np.meshgrid((l[:-1]+l[1:])/2,(b[:-1]+b[1:])/2)
    ra,dec=hp.Rotator(coord=('G','C'))(-np.rad2deg(lc.ravel()),np.rad2deg(bc.ravel()),lonlat=True)
    return l,b,hp.ang2pix(C['nside'],ra,dec,lonlat=True,nest=True).reshape(lc.shape)


def panels(d,name,maps,labels,unit,signed=False,annotated=False,limits=None):
    l,b,pix=grid();maps=np.asarray(maps)
    values=maps[np.isfinite(maps)]
    cap=float(np.quantile(abs(values),.995)) if len(values) else 1.
    cap=max(cap,1e-5) if limits is None else limits[1]
    lin=max(cap/100,.001)
    norm=SymLogNorm(linthresh=lin,linscale=.32,vmin=-cap if signed else 0,vmax=cap)
    decades=10.**np.arange(np.ceil(np.log10(lin)),np.floor(np.log10(cap))+1)
    ncols=2 if len(maps)>1 else 1;nrows=(len(maps)+ncols-1)//ncols
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(nrows,ncols,figsize=(7.15,2.35*nrows+.35),
            subplot_kw={'projection':'aitoff'},layout='constrained',squeeze=False)
        stats=[]
        for i,(ax,a,label) in enumerate(zip(axes.flat,maps,labels)):
            z=ax.pcolormesh(l,b,np.ma.masked_invalid(a[pix]),cmap='RdBu_r' if signed else COLORMAP,
                norm=norm,rasterized=True,shading='flat')
            ax.set_axisbelow(False);ax.grid(color='.49',alpha=.48,ls=':',lw=.5)
            ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°'])
            ax.set_yticks(np.deg2rad([-60,-30,0,30,60]))
            ax.tick_params(labelsize=7);ax.set_title(label)
            if annotated:
                from v3_13_clusters import SYSTEMS
                for key,(coord,_) in SYSTEMS.items():
                    ll,bb=hp.Rotator(coord=('C','G'))(*coord,lonlat=True)
                    xx=-((ll+180)%360-180)
                    ax.plot(np.deg2rad(xx),np.deg2rad(bb),'+',ms=3,color='black',lw=.5)
                    ax.annotate(key,(np.deg2rad(xx),np.deg2rad(bb)),xytext=(3,2),textcoords='offset points',fontsize=5)
                from v3_16_gd1 import ROTATION,unit as skyunit,angles,track,EDGES
                phi1=np.linspace(EDGES[0],EDGES[-1],201)
                ra,dec=angles(ROTATION.T@skyunit(phi1,track(phi1)))
                ll,bb=hp.Rotator(coord=('C','G'))(ra,dec,lonlat=True);xx=-((ll+180)%360-180)
                ax.plot(np.deg2rad(xx),np.deg2rad(bb),'--',color='.1',lw=.6)
                mid=len(xx)//2
                ax.annotate('GD-1',(np.deg2rad(xx[mid]),np.deg2rad(bb[mid])),xytext=(3,2),textcoords='offset points',fontsize=5)
            stats.append(dict(panel=label,finite_pixels=int(np.isfinite(a).sum()),
                below_color=int((a<norm.vmin).sum()),above_color=int((a>cap).sum()),
                color_lo=norm.vmin,color_hi=cap,linthresh=lin,units=unit))
        for ax in list(axes.flat)[len(maps):]:ax.set_visible(False)
        f.supxlabel('Galactic longitude',fontsize=8);f.supylabel('Galactic latitude',fontsize=8)
        f.colorbar(z,ax=list(axes.flat),location='bottom',fraction=.025,pad=.07,shrink=.8,
                   aspect=48,extend='both' if signed else 'max',label=unit,
                   ticks=np.r_[-decades[::-1],0,decades] if signed else np.r_[0,decades],
                   format=plt.FormatStrFormatter('%g'))
        figure(d,name,f,stats,dict(l=l,b=b,values=maps[:,pix].astype('f4')))
    return stats


def assemble():
    d=folder('atlas');inputs={}
    resume=(d/'cmd_filter_cube.fits').exists()
    if resume:
        h=fits.open(d/'cmd_filter_cube.fits',memmap=True)
        assert h[0].header['CONFIG']==CONFIG_HASH and h[0].header['NSIDE']==C['nside']
        assert all(x.verify_checksum()==1 for x in h), 'Raw checkpoint checksum failed'
        cube=h['COUNTS'].data;prob=h['QRF_MASS'].data;sw2=h['BROAD_SUMW2'].data
        census=[]
        for i in range(C['cache_shards']):
            p=OUT/'counts'/f'{i:02}'
            mark=json.loads((p/'complete.json').read_text());assert mark['config']==CONFIG_HASH
            inputs[str(p.relative_to(ROOT))]=sha(p/'complete.json')
            with (p/'census.csv').open() as f:census.extend(csv.DictReader(f))
        print('resumed checksum-verified raw cube; count/PDF conservation rechecked below',flush=True)
    else:
        cube=np.zeros((2,3,len(MU),NPIX),'i8');prob=np.zeros((2,7,NPIX));sw2=np.zeros((2,5,NPIX))
        census=[]
        for i in range(C['cache_shards']):
            p=OUT/'counts'/f'{i:02}'
            t=read(p,('cube.npz','counts','probability','channel_sumw2'),inputs)
            cube+=t['counts'];prob+=t['probability'];sw2+=t['channel_sumw2']
            with (p/'census.csv').open() as f:census.extend(csv.DictReader(f))
            print('assembled',i,flush=True)
    assert (cube[:,1:]<=cube[:,:1]).all()
    areas=[];depth=[];ebv=[]
    for side in ('south','north'):
        a=read(OUT/'area'/side,('randoms.npz','area','counts','depth_mag','EBV'),inputs)
        areas.append(a['area']);depth.append(a['depth_mag']);ebv.append(a['EBV'])
    areas=np.asarray(areas);depth=np.asarray(depth);ebv=np.asarray(ebv)
    area=areas[:,2].sum(axis=0)
    fraction=np.clip(area/PIXAREA,0,1)
    coverage=smooth(fraction,.3)
    masked=(fraction>=.1)&(coverage>=.7)&(area>0)
    smarea=np.maximum(smooth(area,.3),0)
    bgarea=np.maximum(smooth(area,5.),0)
    weights=channel_weights(MU,C['distance_bins_kpc'])
    # Fixed-node geometric checks precede exporting and rendering.
    ra,dec=hp.Rotator(coord=('G','C'))(0,0,lonlat=True)
    np.testing.assert_allclose([ra%360,dec],[266.4049948,-28.936174],atol=1e-4)
    np.testing.assert_allclose(weights.sum(axis=1),1)
    totals=[]
    for v,version in enumerate(('bright','deep')):
        union=sum(int(r['cmd_pdf_common']) for r in census if r['version']==version)
        np.testing.assert_allclose(prob[v].sum(),union,rtol=0,atol=.01)
        expected=sum(int(r['node_count']) for r in census if r['version']==version)
        assert cube[v,0].sum()==expected
        for j,mu in enumerate(MU):
            totals.append(dict(version=version,mu=mu,distance_kpc=DISTANCE[j],
                count=int(cube[v,0,j].sum()),score_count=int(cube[v,1,j].sum()),
                common_count=int(cube[v,2,j].sum()),
                zero_area_count=int(cube[v,0,j,area==0].sum()),
                mapped_count=int(cube[v,0,j,masked].sum()),
                eligible_area_deg2=area.sum(),display_area_deg2=area[masked].sum()))
    # Per-phase visibility is an approximate mean-depth/color-domain diagnostic,
    # not an IMF-weighted detection or PSF completeness correction.
    template=read(OUT/'native',('template.npz','south','north','phase','north_support'),inputs)
    if resume:
        visible=h['PHASE_VISIBLE'].data
    else:
        visible=np.zeros((2,2,len(MU),NPIX),'u1')
        attenuation=np.array([3.214,2.165,1.211])
        for native,key in enumerate(('south','north')):
            lim=depth[native]-attenuation[:,None]*ebv[native]
            for v,limit in enumerate((20.,23.)):
                limv=np.minimum(lim,limit-attenuation[:,None]*ebv[native])
                for bit,phase in enumerate((0,2)):
                    take=template['phase']==phase
                    if native:take&=template['north_support']
                    mags=template[key][take]
                    maxmu=np.full(NPIX,-np.inf)
                    for m in mags:maxmu=np.maximum(maxmu,np.min(limv-m[:,None],axis=0))
                    visible[v,native]|=((MU[:,None]<=maxmu[None,:])&(areas[native,2]>0)).astype('u1')*(1<<bit)
    hd=fits.Header();hd['NSIDE']=C['nside'];hd['ORDERING']='NESTED';hd['COORDSYS']='C'
    hd['CONFIG']=CONFIG_HASH;hd['SIGMA']=.3;hd['LMAX']=C['smoothing']['lmax']
    hd['COMMENT']='Counts axes: version(bright,deep),gate(all,score>=.8,PDF-common),mu,pixel.'
    hd['COMMENT']='Node sumw2 equals binary counts. Broad channels overlap, not membership shells.'
    hdus=[fits.PrimaryHDU(header=hd),fits.ImageHDU(MU,name='MU'),
        fits.ImageHDU(cube,name='COUNTS'),fits.ImageHDU(cube[:,0],name='NODE_SUMW2'),
        fits.ImageHDU(sw2,name='BROAD_SUMW2'),fits.ImageHDU(weights,name='BROAD_WEIGHTS'),
        fits.ImageHDU(prob,name='QRF_MASS'),fits.ImageHDU(areas,name='NATIVE_AREAS'),
        fits.ImageHDU(depth,name='DEPTH'),fits.ImageHDU(ebv,name='EBV'),
        fits.ImageHDU(visible,name='PHASE_VISIBLE'),fits.ImageHDU(masked.astype('u1'),name='DISPLAY_MASK')]
    if not resume:fits.HDUList(hdus).writeto(d/'cmd_filter_cube.fits',overwrite=True,checksum=True)
    csv_rows(d/'node_totals.csv',totals)
    # Smooth independent maps concurrently with single-threaded spherical transforms.
    work=[];keys=[]
    for v in (0,1):
        for gate in (0,1,2):
            broad=np.tensordot(weights,cube[v,gate],axes=(1,0))
            for b in range(5):keys.append((v,gate,b,'broad'));work.append(broad[b])
        for mu in C['fixed_mu']:
            node=int(np.argmin(abs(MU-mu)));keys.append((v,0,node,'fixed'));work.append(cube[v,0,node])
        for b in range(5):keys.append((v,3,b,'qrf'));work.append(prob[v,b+1])
    def smoothed(x):
        a=smooth(x,.3);bg=smooth(x,5.)
        rate=np.divide(np.maximum(a,0),smarea,out=np.full(NPIX,np.nan),where=smarea>0)
        back=np.divide(np.maximum(bg,0),bgarea,out=np.full(NPIX,np.nan),where=bgarea>0)
        rate[~masked]=np.nan;back[~masked]=np.nan
        return rate,back,int((a<0).sum())
    mapped={}
    with ThreadPoolExecutor(6) as pool:
        for key,val in zip(keys,pool.map(smoothed,work)):mapped[key]=val
    atomic_npz(d/'smoothed_maps.npz',keys=np.array([str(k) for k in mapped]),values=np.array([v[:2] for v in mapped.values()]).astype('f4'))
    del work,cube
    render_maps(d,inputs,mapped,areas,depth,ebv,area,masked)


def render_maps(d,inputs,mapped,areas,depth,ebv,area,masked):
    annotations=[];redraw={};labels=[f'{a:.1f}–{b:g} kpc' for a,b in C['distance_bins_kpc']]
    for v,version in enumerate(('bright','deep')):
        main=np.array([mapped[v,0,b,'broad'][0] for b in range(5)])
        background=np.array([mapped[v,0,b,'broad'][1] for b in range(5)])
        mainname='cmd_filter_aitoff_distance_slices'+('' if v==1 else '_bright')
        annotations+=panels(d,mainname,main,labels,'Mean template-selected PSFs / deg²')
        panels(d,mainname+'_annotated',main,labels,'Mean template-selected PSFs / deg²',annotated=True)
        fixed=np.array([mapped[v,0,int(np.argmin(abs(MU-mu))),'fixed'][0] for mu in C['fixed_mu']])
        panels(d,'cmd_filter_aitoff_fixed_mu_'+version,fixed,[f'μ = {mu:g} · {10**((mu-10)/5):.2f} kpc' for mu in C['fixed_mu']],
               'Template-selected PSFs / deg²')
        panels(d,'cmd_filter_background_'+version,background,labels,'Background PSFs / deg²')
        panels(d,'cmd_filter_background_residual_'+version,main-background,labels,'Density residual / deg²',signed=True)
        common=np.array([mapped[v,2,b,'broad'][0] for b in range(5)])
        qr=np.array([mapped[v,3,b,'qrf'][0] for b in range(5)])
        # Shared units scale; template means and exclusive PDF masses remain different estimands.
        compare=np.array([a for pair in zip(common,qr) for a in pair])
        panels(d,'photdis_vs_cmd_same_sources_'+version,compare,
            [q for label in labels for q in (label+' · mean CMD',label+' · QRF mass')],
            'Mean CMD count or integrated PDF mass / deg²')
        scored=np.array([mapped[v,1,b,'broad'][0] for b in range(5)])
        panels(d,'cmd_filter_score_comparison_'+version,np.array([a for pair in zip(main,scored) for a in pair]),
               [q for label in labels for q in (label+' · imaging',label+' · score ≥ 0.8')],
               'Mean template-selected PSFs / deg²')
    # Diagnostic maps use their own physical linear scales, never density normalization.
    l,b,pix=grid()
    native=np.divide(areas[1,2],area,out=np.full(NPIX,np.nan),where=area>0)
    combined_depth=np.divide(np.nansum(depth*areas[:,2,None,:],axis=0),area[None,:],
                       out=np.full((3,NPIX),np.nan),where=area[None,:]>0)
    combined_ebv=np.divide(np.nansum(ebv*areas[:,2],axis=0),area,out=np.full(NPIX,np.nan),where=area>0)
    arrays=[np.where(area>0,fraction,np.nan),*combined_depth,combined_ebv,native]
    diaglabels=['Eligible pixel coverage','g depth','r depth','z depth','E(B−V)','Northern area fraction']
    units=['Fraction','0.1-mag sensitivity (mag)','0.1-mag sensitivity (mag)','0.1-mag sensitivity (mag)','mag','Fraction']
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(3,2,figsize=(7.15,7.4),subplot_kw={'projection':'aitoff'},layout='constrained')
        for ax,a,label,unit in zip(axes.flat,arrays,diaglabels,units):
            good=a[np.isfinite(a)];lo,hi=np.quantile(good,[.01,.99])
            if unit=='Fraction':lo,hi=0,1
            z=ax.pcolormesh(l,b,np.ma.masked_invalid(a[pix]),cmap=COLORMAP,vmin=lo,vmax=hi,rasterized=True)
            ax.set_xticks(np.deg2rad([-120,-60,0,60,120]),labels=['120°','60°','0°','300°','240°'])
            ax.grid(ls=':',lw=.5,color='.49');ax.tick_params(labelsize=7);ax.set_title(label)
            f.colorbar(z,ax=ax,location='bottom',fraction=.04,pad=.08,label=unit,extend='both')
        figure(d,'cmd_filter_coverage_depth',f,[dict(map=x,unit=u) for x,u in zip(diaglabels,units)],
               dict(l=l,b=b,values=np.array(arrays)[:,pix].astype('f4')))
    save_json(d/'cmd_filter_manifest.json',dict(config=CONFIG_HASH,inputs=inputs,nside=C['nside'],
        ordering='NESTED',coordinates='C',display='Galactic Aitoff, l=0 center, longitude increases left',
        geometric_deg2=float(areas[:,0].sum()),unmasked_deg2=float(areas[:,1].sum()),
        depth20_deg2=float(area.sum()),display_deg2=float(area[masked].sum()),
        sigma_deg=.3,lmax=1024,background_sigma_deg=5,
        depth_scope='Both observed-grz<20 and <23 use the independent grz depth20 footprint; deep selection is not uniformly complete to 23.',
        visibility_scope='bit0 MS / bit1 SGB-RGB has at least one supported template point at mean pixel sensitivity and observed limit; not an IMF-weighted completeness',
        templates='DECam MIST1.0 12Gyr Z=.0001; northern same-star empirical native transform with finite color support',
        display_clipping=annotations,
        cube_path=str(d/'cmd_filter_cube.fits'),cube_sha256=sha(d/'cmd_filter_cube.fits'),
        smoothing_negative_pixels=[dict(key=str(k),n=v[-1]) for k,v in mapped.items()],
        redraw='Final panel NPZs store actual Galactic display-grid values, edges and CSV color scales; full raw counts in private cube.'))
    complete(d,inputs,required_maps='actual full-footprint density/fixed-mu/background/residual/common-QRF/score/depth maps',
        cube=str(d/'cmd_filter_cube.fits'),cube_sha256=sha(d/'cmd_filter_cube.fits'))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();require_compute()
    if args.render_only:
        import ast
        d=folder('atlas');marker=d/'complete.json';r=json.loads(marker.read_text())
        assert r['config']==CONFIG_HASH
        archive=folder('execution')/'atlas_before_colorbar_revision.json'
        if not archive.exists():archive.write_text(marker.read_text())
        inputs=r['inputs'];inputs['render_previous_complete']=sha(archive)
        with np.load(verified(d/'smoothed_maps.npz',r['outputs']['smoothed_maps.npz'])) as a:
            keys=[ast.literal_eval(k) for k in a['keys']];values=a['values']
        previous=json.loads((d/'cmd_filter_manifest.json').read_text())
        negatives={q['key']:q['n'] for q in previous['smoothing_negative_pixels']}
        mapped={k:(values[i,0],values[i,1],negatives[str(k)]) for i,k in enumerate(keys)}
        with fits.open(d/'cmd_filter_cube.fits',memmap=True) as h:
            areas=h['NATIVE_AREAS'].data;depth=h['DEPTH'].data;ebv=h['EBV'].data
            area=areas[:,2].sum(axis=0);masked=h['DISPLAY_MASK'].data.astype(bool)
            render_maps(d,inputs,mapped,areas,depth,ebv,area,masked)
    else:assemble()
