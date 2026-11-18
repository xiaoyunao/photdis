#!/usr/bin/env python3
"""Check real M13 imaging CMDs and spherical area before interpreting atlas layers."""
import healpy as hp
from matplotlib.colors import LogNorm
from photdis.stage1 import require_compute
from halo_common import *
from halo_filtering import accepted,prepare_templates
from v3_13_clusters import SYSTEMS
from v3_13_cmd import M5_CENTER


def main():
    require_compute();d=folder('known');inputs={};parts=[]
    center,ref=SYSTEMS['M13'];vec=hp.ang2vec(*center,lonlat=True)
    totalarea=np.zeros(hp.nside2npix(C['nside']))
    for side in ('south','north'):totalarea+=read(OUT/'area'/side,('randoms.npz','area'),inputs)['area'][2]
    ex=read(OUT/'parent',('exclusions.npz','TYPE','ls_id'),inputs)['ls_id']
    for path in cache_folders():
        t=read(path,('photometry.npz','TYPE','ls_id','RA','DEC','m','err','observed','native','pdf'),inputs)
        v=hp.ang2vec(t['RA'],t['DEC'],lonlat=True)
        sep=np.rad2deg(np.arccos(np.clip(v@vec,-1,1)))
        use=(sep<1.2)&~np.isin(t['ls_id'],ex)&(t['observed']<23).all(axis=1)
        if use.any():parts.append(dict(**{k:v[use] for k,v in t.items()},sep=sep[use]))
    t={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};assert (t['TYPE']=='PSF').all()
    template=read(OUT/'native',('template.npz','south','north','phase','north_support'),inputs)
    curve=read(OUT/'parent',('error_curve.npz','g_centers','sigma_g'),inputs)
    on=t['sep']<.4;off=(t['sep']>=.8)&(t['sep']<1.2)
    disc=hp.query_disc(C['nside'],vec,np.deg2rad(.4),nest=True)
    outer=hp.query_disc(C['nside'],vec,np.deg2rad(1.2),nest=True)
    inner=hp.query_disc(C['nside'],vec,np.deg2rad(.8),nest=True)
    aon=totalarea[disc].sum();aoff=totalarea[np.setdiff1d(outer,inner)].sum()
    assert aon>0 and aoff>0
    k=accepted(t['m'],t['observed'][:,0],t['native'],MU,prepare_templates(template),curve)
    stats=[dict(mu=mu,distance=DISTANCE[i],on=int(k[on,i].sum()),off=int(k[off,i].sum()),
       on_area=aon,off_area=aoff,excess_density=k[on,i].sum()/aon-k[off,i].sum()/aoff) for i,mu in enumerate(MU)]
    csv_rows(d/'M13_node_check.csv',stats)
    m5pix=hp.query_disc(C['nside'],hp.ang2vec(*M5_CENTER,lonlat=True),np.deg2rad(.4),nest=True)
    mu=10+5*np.log10(ref)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,3,figsize=(7.15,3.2),layout='constrained')
        for ax,use,label in zip(axes[:2],(on,off),('M13 · on','M13 · control')):
            h,xe,ye=np.histogram2d(t['m'][use,0]-t['m'][use,1],t['m'][use,0],(np.linspace(-.2,1.2,60),np.linspace(14,23,80)))
            image=ax.pcolormesh(xe,ye,np.ma.masked_equal(h.T,0),cmap='viridis',norm=LogNorm(1,max(2,h.max())),rasterized=True)
            mag=template['north'];take=np.isin(template['phase'],(0,2))&template['north_support']
            ax.plot(mag[take,0]-mag[take,1],mag[take,0]+mu,color=ORANGE,lw=.8)
            curves=prepare_templates(template)[1]
            for grid,lo,hi in curves:
                ax.plot(lo-.05,grid+mu,color=ORANGE,lw=.5,ls=':');ax.plot(hi+.1,grid+mu,color=ORANGE,lw=.5,ls=':')
            ax.set(xlabel=r'$(g-r)_0$',ylabel=r'$g_0$ (mag)',ylim=(23,14),xlim=(-.2,1.2),title=label)
            f.colorbar(image,ax=ax,fraction=.035,label='Stars / bin')
        selected=k[:,np.argmin(abs(MU-mu))]
        a=axes[2];a.scatter((t['RA'][selected]-center[0])*np.cos(np.deg2rad(center[1])),t['DEC'][selected]-center[1],
            s=1,alpha=.4,lw=0,color=BLUE,rasterized=True)
        a.set(xlabel=r'$\Delta\alpha\cos\delta$ (deg)',ylabel=r'$\Delta\delta$ (deg)',aspect='equal',xlim=(-1.2,1.2),ylim=(-1.2,1.2))
        figure(d,'cmd_filter_known_system_check',f,stats,dict(TYPE=t['TYPE'],ls_id=t['ls_id'],RA=t['RA'],DEC=t['DEC'],m=t['m'],on=on,off=off,accepted=k))
    atomic_npz(d/'M13_sources.npz',**t,accepted=k)
    complete(d,inputs,system='M13',reference_kpc=ref,on_count=int(on.sum()),off_count=int(off.sum()),
        on_area_deg2=float(aon),off_area_deg2=float(aoff),M5_on_area_deg2=float(totalarea[m5pix].sum()),
        area_scope='independent random area in Nside512 center-selected spherical aperture; no reopening of M5 mask',
        plot_scope='unscored strict imaging parent, observed grz<23; same native template as full atlas')


if __name__=='__main__':main()
