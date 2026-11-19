#!/usr/bin/env python3
"""Assemble full gr coverage and true template-channel RGBs without science gates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from astropy.io import fits
import healpy as hp
from scipy.spatial import cKDTree
from photdis.stage1 import require_compute,galactic
from halo_atlas import smooth,grid
from halo_filtering import channel_weights
from stream_rgb import stretch
from stream_common import track
from sky6_common import *
from sky6_figures import rgb,coverage,physical


def assemble():
    require_compute();d=folder('atlas');inputs={};z=read(OUT/'counts',('node_counts.npz','counts','parents','geometry','strict','uncorrected','mu'),inputs)
    ar=[read(OUT/'area'/side,('geometry.npz','area','counts','EBV','depth5','depth0','extinction','value_counts'),inputs) for side in ('south','north')]
    areas=np.array([a['area'] for a in ar]);masks=np.array([(areas[:,0].sum(axis=0)>0)|(z['geometry'][:,1].sum(axis=0)>0),(areas[:,1].sum(axis=0)>0)|(z['parents'][:,1].sum(axis=0)>0)])
    header=fits.Header();header['NSIDE']=C['nside'];header['ORDERING']='NESTED';header['COORDSYS']='C';header['CONFIG']=CH;header['PRODUCT']='CMD display, not distances'
    path=d/'cmd_cube.fits';assert not path.exists()
    fits.HDUList([fits.PrimaryHDU(header=header),fits.ImageHDU(z['counts'],name='COUNTS'),fits.ImageHDU(areas,name='AREA'),fits.ImageHDU(masks.astype('u1'),name='MASK'),fits.ImageHDU(z['mu'],name='MU')]).writeto(path)
    def average(key,index):
        values=np.array([a[key] for a in ar]);n=np.array([a['value_counts'][index] for a in ar]);good=np.isfinite(values);den=np.where(good,n,0).sum(axis=0)
        return np.divide(np.where(good,values*n,0).sum(axis=0),den,out=np.full_like(den,np.nan,dtype=float),where=den>0)
    dust=average('EBV',0);depth=np.array([average('depth5',slice(1,3))[j] for j in range(2)]);ext=average('extinction',slice(5,7))
    npz(d/'companions.npz',parents=z['parents'],geometry=z['geometry'],strict=z['strict'],uncorrected=z['uncorrected'],EBV=dust,depth5=depth,extinction=ext,area=areas,mask=masks)
    ident=json.loads((OUT/'identities/identity_counts.json').read_text());parts=list(csv.DictReader((OUT/'counts/part_counts.csv').open()));raw=list(csv.DictReader((OUT/'inventory/raw_to_display.csv').open()))
    domains={}
    for source in (OUT/'cache').glob('*/domains.csv'):
        for r in csv.DictReader(source.open()):
            row=domains.setdefault(r['domain'],dict(domain=r['domain'],input_owned_PSF=0,gr_geometry_PSF=0,valid_gr_before_alias=0,valid_gr=0,CMD_union=0,strict_companion=0,area_deg2=0.))
            row['input_owned_PSF']+=int(r['raw_PSF']);row['gr_geometry_PSF']+=int(r['gr_geometry']);row['valid_gr_before_alias']+=int(r['valid_gr'])
    for r in csv.DictReader((OUT/'counts/domain_counts.csv').open()):
        for key in ('valid_gr','CMD_union','strict_companion'):domains[r['domain']][key]+=int(r[key])
    for side in ('south','north'):
        for r in csv.DictReader((OUT/'area'/side/'domains.csv').open()):domains[r['domain']]['area_deg2']+=float(r['area_deg2'])
    csv_rows(d/'domain_summary.csv',list(domains.values()))
    save_json(d/'summary.json',dict(identity=ident,sweeps=len(raw),raw_records=sum(int(r['raw']) for r in raw),PSF=sum(int(r['PSF']) for r in raw),missing_TYPE=sum(int(r['missing_TYPE']) for r in raw),non_PSF=sum(int(r['non_PSF']) for r in raw),old_legacy_entrance=sum(int(r['legacy_entrance']) for r in raw),area_deg2=areas[:,0].sum(),old_domain_area_deg2=areas[:,1].sum(),CMD_union=sum(int(r['union']) for r in parts if r['view']=='0'),node_acceptances=int(z['counts'][:,0].sum()),north_scope='existing finite-support native transfer; no new stellar population distance calibration',correction='2D full-column catalog MW transmission; display only in dusty/crowded regions'))
    complete(d,inputs,views=['all available display','same display in old geographic/dust/depth domain'],cube_axes='COUNTS[native,view,mu,equatorial-NESTED-pixel]',zero_and_mask_separate=True)


def render_maps(counts,area,mask):
    sa=np.maximum(smooth(area,.3),0);ba=np.maximum(smooth(area,5),0)
    def work(c):
        density=np.divide(np.maximum(smooth(c,.3),0),sa,out=np.full_like(sa,np.nan),where=sa>0)
        bg=np.divide(np.maximum(smooth(c,5),0),ba,out=np.full_like(ba,np.nan),where=ba>0)
        # An observed pixel with zero accepted sources stays zero, not masked.
        density[(c==0)&mask]=0;density[~mask]=np.nan;bg[~mask]=np.nan
        return density,density-bg
    with ThreadPoolExecutor(6) as pool:return np.array(list(pool.map(work,counts)))


def render():
    require_compute();inputs={};rec=json.loads((OUT/'atlas/complete.json').read_text());path=verified(OUT/'atlas/cmd_cube.fits',rec['outputs']['cmd_cube.fits']);inputs['cube']=sha(path)
    for name in C['annotations']['tracks']:
        _,source=track(name);inputs[str(source.relative_to(ROOT))]=sha(source)
    l,b,pix=grid();groups=[('overview',np.array(C['rgb']['overview_BGR_kpc']))]
    for lo,hi in ((10**.8,10),(10,15),(15,26),(26,50),(50,100)):
        edge=lo*(hi/lo)**(np.arange(4)/3);groups.append((f'bin_{lo:.1f}_{hi:g}',np.column_stack((edge[:-1],edge[1:]))))
    intervals=np.concatenate([bounds for _,bounds in groups]);stats=[]
    with fits.open(path,memmap=True) as h:
        w=channel_weights(h['MU'].data,intervals);np.testing.assert_allclose(w.sum(axis=1),1)
        for view,stage in enumerate(('rgb','old_domain')):
            d=folder(stage);counts=np.tensordot(w,h['COUNTS'].data[:,view].sum(axis=0),axes=(1,0));area=h['AREA'].data[:,view].sum(axis=0);mask=h['MASK'].data[view].astype(bool);maps=render_maps(counts,area,mask)
            npz(d/'physical_channels.npz',density=maps[:,0].astype('f4'),signed_residual=maps[:,1].astype('f4'),weighted_counts=counts,area=area,mask=mask,intervals=intervals,weights=w)
            for i,(name,bounds) in enumerate(groups):
                pair=maps[3*i:3*i+3];st=[]
                for row in (0,1):
                    _,rows=stretch(pair[:,row],mask,residual=bool(row));st.append([[r['lo'],r['hi']] for r in rows]);stats.extend(dict(stage=stage,group=name,kind=('density','positive_residual')[row],**r) for r in rows)
                a=dict(l=l,b=b,density=pair[:,0,pix].astype('f4'),signed_residual=pair[:,1,pix].astype('f4'),mask=mask[pix],channel_kpc=bounds,stretch=np.array(st),scope=np.array(stage),label=np.array('CMD-selected sky' if view==0 else 'CMD-selected sky · old domain'))
                npz(d/('rgb_'+name+'.npz'),**a);rgb(a,d,'rgb_'+name)
                if name=='overview':physical(a,d)
            csv_rows(d/'node_weights.csv',[dict(channel=j,mu=float(mu),weight=float(value)) for j,row in enumerate(w) for mu,value in zip(h['MU'].data,row) if value>0]);complete(d,inputs,coordinate='Galactic l0 center, increasing left',scope=stage,annotations=C['annotations'])
    with np.load(OUT/'atlas/companions.npz') as z:a={k:z[k] for k in z.files}
    d=folder('coverage');area=a['area'][:,0].sum(axis=0);mask=a['mask'][0];sa=np.maximum(smooth(area,.3),0)
    densities=[np.divide(np.maximum(smooth(v,.3),0),sa,out=np.full_like(sa,np.nan),where=sa>0) for v in (a['geometry'][:,0].sum(axis=0),a['parents'][:,0].sum(axis=0))]
    fields=np.array([*densities,np.min(a['depth5'],axis=0),a['EBV']]);fields[:,~mask]=np.nan;cap=dict(l=l,b=b,values=fields[:,pix].astype('f4'),extinction=a['extinction'][:,pix].astype('f4'),mask=mask[pix]);npz(d/'sky_coverage.npz',**cap);coverage(cap,d);complete(d,inputs,field_definitions=['native PSF per geometricgr area','validgr unique source per area','mean native shallower observed5sigma gr depth','mean catalog E(B-V)'],correction_fields='Ag,Ar full-column maps retained in NPZ')
    raw=a['uncorrected'].sum(axis=0);maps=render_maps(raw,area,mask);d=folder('no_extinction')
    with np.load(OUT/'rgb/rgb_overview.npz') as z:st=z['stretch']
    for row in (0,1):
        values=maps[:,row] if row==0 else np.maximum(maps[:,row],0)
        for j,ch in enumerate('BGR'):stats.append(dict(stage='no_extinction',group='overview',kind=('density','positive_residual')[row],channel=ch,lo=float(st[row,j,0]),hi=float(st[row,j,1]),saturated_fraction=float((values[j,mask]>st[row,j,1]).mean())))
    r=dict(l=l,b=b,density=maps[:,0,pix].astype('f4'),signed_residual=maps[:,1,pix].astype('f4'),mask=mask[pix],channel_kpc=np.array(C['rgb']['overview_BGR_kpc']),stretch=st,scope=np.array('No extinction in CMD only; exact same admitted sources; same corrected-overview stretch'),label=np.array('No extinction · same sources'))
    npz(d/'rgb_overview.npz',**r);rgb(r,d,'rgb_overview');physical(r,d);npz(d/'physical_channels.npz',density=maps[:,0].astype('f4'),signed_residual=maps[:,1].astype('f4'),weighted_counts=raw,mask=mask);complete(d,inputs,scope='same source IDs/admission/geometry, observed colors/magnitudes in fixed CMD; diagnostic only, not a new dust solution')
    csv_rows(folder('display')/'stretch.csv',stats)
    complete(OUT/'display',inputs,style='same V5/Pal5 base and BGR channels; Galactic annotations requested separately')


def band():
    require_compute();d=folder('band');inputs={};a=read(OUT/'atlas',('companions.npz','parents','geometry','strict','EBV','depth5','extinction','area','mask'),inputs);z=read(OUT/'rgb',('physical_channels.npz','weighted_counts'),inputs);q=read(OUT/'no_extinction',('physical_channels.npz','weighted_counts'),inputs)
    ra,dec=hp.pix2ang(C['nside'],np.arange(hp.nside2npix(C['nside'])),lonlat=True,nest=True);ll,bb=hp.Rotator(coord=('C','G'))(ra,dec,lonlat=True);_,ec=hp.Rotator(coord=('C','E'))(ra,dec,lonlat=True)
    t,source=track('Sagittarius.antoja2020');vec=hp.ang2vec(t['ra'],t['dec'],lonlat=True);nearest=cKDTree(vec).query(hp.ang2vec(ra,dec,lonlat=True),workers=4)[0];sep=np.rad2deg(2*np.arcsin(np.clip(nearest/2,0,1)));area=a['area'][:,0].sum(axis=0)
    profiles=[]
    for frame,latitude in [('Galactic',bb),('Equatorial',dec),('Ecliptic',ec)]:
        for lo in np.arange(-90,90,2):
            flag=(latitude>=lo)&(latitude<lo+2)&a['mask'][0];aa=float(area[flag].sum())
            if not aa:continue
            row=dict(frame=frame,lat_lo=lo,lat_hi=lo+2,area_deg2=aa,raw_PSF_density=a['geometry'][:,0,flag].sum()/aa,valid_gr_density=a['parents'][:,0,flag].sum()/aa,strict_fraction=a['strict'][:,flag].sum()/max(a['parents'][:,0,flag].sum(),1),mean_EBV=np.nansum(a['EBV'][flag]*area[flag])/aa)
            for j,ch in enumerate('BGR'):row['corrected_'+ch]=z['weighted_counts'][j,flag].sum()/aa;row['uncorrected_'+ch]=q['weighted_counts'][j,flag].sum()/aa
            profiles.append(row)
    rows=[]
    for name,flag in [('plane_absb10',abs(bb)<10),('plane_absb20',abs(bb)<20),('plane_absb30',abs(bb)<30),('outside_absb30',abs(bb)>=30),('Sgr5_outside_absb20',(sep<5)&(abs(bb)>=20)),('Sgr5_10_outside_absb20',(sep>=5)&(sep<10)&(abs(bb)>=20)),('high_EBV',a['EBV']>=.1),('very_high_EBV',a['EBV']>=.5)]:
        flag&=a['mask'][0];aa=float(area[flag].sum());row=dict(region=name,area_deg2=aa,raw_PSF=int(a['geometry'][:,0,flag].sum()),valid_gr=int(a['parents'][:,0,flag].sum()),strict=int(a['strict'][:,flag].sum()))
        for j,ch in enumerate('BGR'):row['corrected_'+ch]=float(z['weighted_counts'][j,flag].sum());row['uncorrected_'+ch]=float(q['weighted_counts'][j,flag].sum())
        rows.append(row)
    csv_rows(d/'latitude_profiles.csv',profiles);csv_rows(d/'fixed_strips.csv',rows);complete(d,dict(**inputs,Sgr_track=sha(source)),interpretation='morphological association and same-parent extinction sensitivity, not a calibrated physical decomposition of disk/halo/Sgr membership')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['assemble','render','band']);a=p.parse_args();globals()[a.stage]()
