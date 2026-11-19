#!/usr/bin/env python3
"""Extend cached native CMD responses into the near field for RGB comparisons."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from functools import lru_cache
from astropy.io import fits
import healpy as hp
from photdis.stage1 import require_compute
from stream_local import curves,accept
from halo_filtering import channel_weights
from stream_rgb import stretch
from sky6_atlas import render_maps
from sky6_figures import rgb
from sky6_common import *

CONFIG_EXTRA=ROOT/'configs/sky_sequence_channels_20261119.json';E=json.loads(CONFIG_EXTRA.read_text());EH=sha(CONFIG_EXTRA)
MU=np.round(np.arange(0,20.001,.1),1);NEAR=MU[MU<14];BOUNDS=np.concatenate(list(E['schemes'].values()));W=channel_weights(MU,BOUNDS)


@lru_cache(maxsize=2)
def model(native):
    t=read(HALO/'native',('template.npz','south','north','phase','north_support'),{});p=t['phase'].copy()
    if native:p[~t['north_support']]=-99
    cc=curves(t[('south','north')[native]],p,'MS_TO')
    with np.load(ROOT/C['inputs']['error_curve']) as z:ec=(z['g_centers'],z['sigma_g'])
    return cc,ec


def one(source):
    source=Path(source);d=folder('near_cache')/source.name;d.mkdir(exist_ok=True);marker=d/'complete.json';receipt=json.loads((source/'complete.json').read_text())
    if marker.exists():
        r=json.loads(marker.read_text());assert r['config']==EH and r['source']==sha(source/'complete.json') and r['code']==sha(Path(__file__));return str(d)
    t=read_columns(verified(source/'photometry.npz',receipt['outputs']['photometry.npz']),('TYPE','ls_id','m','RA','DEC','old_domain'))
    with np.load(OUT/'identities/alias_exclusions.npz') as z:keep=~np.isin(t['ls_id'],z['ls_id'])
    t={k:v[keep] for k,v in t.items()};assert (t['TYPE']=='PSF').all();parent=len(t['ls_id'])
    gr=t['m'][:,0]-t['m'][:,1];candidate=(gr>=-.2)&(gr<1.2);t={k:v[candidate] for k,v in t.items()}
    pix=hp.ang2pix(C['nside'],t['RA'],t['DEC'],nest=True,lonlat=True);up,inv=np.unique(pix,return_inverse=True);count=np.zeros((2,len(BOUNDS),len(up)));totals=np.zeros((2,len(NEAR)),'i8');cc,ec=model(int(source.name.startswith('north_')))
    for start in range(0,len(pix),20000):
        stop=min(start+20000,len(pix));accepted=accept(t['m'][start:stop],NEAR,cc,ec,'MS_TO')
        for view,flag in enumerate((np.ones(stop-start,bool),t['old_domain'][start:stop])):
            selected=accepted&flag[:,None];totals[view]+=selected.sum(axis=0);weighted=selected@W[:,:len(NEAR)].T
            for j in range(len(BOUNDS)):count[view,j]+=np.bincount(inv[start:stop],weights=weighted[:,j],minlength=len(up))
    np.testing.assert_allclose(count.sum(axis=2),totals@W[:,:len(NEAR)].T,atol=1e-5,rtol=1e-10)
    npz(d/'channels.npz',pixels=up,weighted_counts=count,near_totals=totals,parent=np.array(parent));save_json(marker,dict(config=EH,source=sha(source/'complete.json'),code=sha(Path(__file__)),sha256=sha(d/'channels.npz')));return str(d)


def main():
    require_compute();d=folder('channel_trials');inputs=dict(channel_settings=EH);count=np.zeros((2,len(BOUNDS),hp.nside2npix(C['nside'])));totals=np.zeros((2,len(NEAR)),'i8');parent=0
    paths=sorted((OUT/'cache').glob('*/complete.json'))
    with ProcessPoolExecutor(E['workers'],mp_context=get_context('spawn')) as pool:
        fs=[pool.submit(one,str(p.parent)) for p in paths]
        for i,f in enumerate(as_completed(fs)):
            part=Path(f.result());r=json.loads((part/'complete.json').read_text())
            with np.load(verified(part/'channels.npz',r['sha256'])) as z:count[:,:,z['pixels']]+=z['weighted_counts'];totals+=z['near_totals'];parent+=int(z['parent'])
            if i%100==0:print('NEAR_COUNTED',i+1,len(paths),flush=True)
    assert parent==json.loads((OUT/'identities/identity_counts.json').read_text())['valid_gr_after_aliases'];np.testing.assert_allclose(count.sum(axis=2),totals@W[:,:len(NEAR)].T,atol=.001,rtol=1e-10)
    record=json.loads((OUT/'atlas/complete.json').read_text());cube=verified(OUT/'atlas/cmd_cube.fits',record['outputs']['cmd_cube.fits']);inputs['original_cube']=sha(cube)
    with fits.open(cube,memmap=True) as h:
        np.testing.assert_allclose(h['MU'].data,MU[len(NEAR):]);area=h['AREA'].data.sum(axis=0);mask=h['MASK'].data.astype(bool)
        for view in (0,1):count[view]+=np.tensordot(W[:,len(NEAR):],h['COUNTS'].data[:,view].sum(axis=0),axes=(1,0))
    npz(d/'weighted_channels.npz',counts=count,near_totals=totals,mu=MU,weights=W,area=area,mask=mask,bounds=BOUNDS)
    maps=np.array([render_maps(count[v],area[v],mask[v]) for v in (0,1)])
    npz(d/'physical_channel_trials.npz',density=maps[:,:,0].astype('f4'),signed_residual=maps[:,:,1].astype('f4'),mask=mask,area=area,weights=W,bounds=BOUNDS)
    with np.load(OUT/'rgb/rgb_overview.npz') as z:fixed=z['stretch']
    from halo_atlas import grid
    l,b,pix=grid();stats=[];channel=[]
    for i,(name,bounds) in enumerate(E['schemes'].items()):
        reference=maps[1,3*i:3*i+3];pair=maps[0,3*i:3*i+3];balanced=[]
        for row in (0,1):
            _,rows=stretch(reference[:,row],mask[1],bool(row));balanced.append([[r['lo'],r['hi']] for r in rows])
        for kind,limits in [('fixed_gain',fixed),('balanced',np.array(balanced))]:
            a=dict(l=l,b=b,density=pair[:,0,pix].astype('f4'),signed_residual=pair[:,1,pix].astype('f4'),mask=mask[0,pix],channel_kpc=np.array(bounds),stretch=limits,scope=np.array('RGB channel comparison; '+kind+'; near support10pc; fixed native PSF parent/CMD/geometry'),label=np.array('CMD-selected sky'))
            filename='rgb_'+name+'_'+kind;npz(d/(filename+'.npz'),**a);rgb(a,d,filename)
            for row in (0,1):
                for j,ch in enumerate('BGR'):stats.append(dict(scheme=name,display=kind,row=('density','positive_residual')[row],channel=ch,lo=limits[row,j,0],hi=limits[row,j,1],saturated_fraction=float((pair[j,row,mask[0]]>limits[row,j,1]).mean())))
        for j,ch in enumerate('BGR'):channel.append(dict(scheme=name,channel=ch,d_lo=bounds[j][0],d_hi=bounds[j][1],node_average_acceptances=float(count[0,3*i+j].sum()),scope='overlapping CMD responses, not inferred-distance counts'))
        print('CHANNEL_RENDERED',name,flush=True)
    csv_rows(d/'stretch.csv',stats);csv_rows(d/'channel_totals.csv',channel);csv_rows(d/'near_node_totals.csv',[dict(mu=mu,distance_kpc=10**((mu-10)/5),all_available=int(totals[0,j]),old_domain=int(totals[1,j])) for j,mu in enumerate(NEAR)])
    save_json(d/'choices.json',dict(settings=E,parent=parent,schemes=list(E['schemes']),comparison='fixed_gain isolates bin boundaries with one unchanged BGR transfer; balanced uses the same v5 percentile recipe on the same fixed old-domain reference, then applies it to all sky',limitations='Template-distance compatibility only; full-column dust is particularly problematic for near stars; gr/source/depth composition and overlapping response vary with interval. Zero is implemented as an explicit10pc finite grid edge. No production model rerun.'))
    complete(d,inputs,scope='user-requested display comparisons; cached-source near responses0<=mu<14 added, original61far nodes reused exactly',source_parent_conserved=True,production_unchanged=True)


if __name__=='__main__':main()
