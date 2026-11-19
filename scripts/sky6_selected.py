#!/usr/bin/env python3
"""Final user choice: B5–10/G10–20/R20–100, using completed responses."""
from astropy.io import fits
from photdis.stage1 import require_compute
from halo_atlas import grid
from halo_filtering import channel_weights
from stream_rgb import stretch
from sky6_common import *
from sky6_atlas import render_maps
from sky6_figures import rgb,physical


def main():
    require_compute();d=folder('selected_channels');settings=ROOT/'configs/sky_sequence_selected_20261119.json';bounds=np.array(json.loads(settings.read_text())['BGR_kpc']);inputs=dict(final_display_settings=sha(settings))
    a=read(OUT/'channel_trials',('weighted_channels.npz','counts','mu','weights','bounds'),inputs);mu=a['mu'];weights=channel_weights(mu,bounds);j=np.flatnonzero((a['bounds']==bounds[0]).all(axis=1));assert len(j)==1
    np.testing.assert_allclose(weights[0],a['weights'][j[0]],atol=1e-14)
    count=np.zeros((2,3,a['counts'].shape[-1]));count[:,0]=a['counts'][:,j[0]];del a
    record=json.loads((OUT/'atlas/complete.json').read_text());path=verified(OUT/'atlas/cmd_cube.fits',record['outputs']['cmd_cube.fits']);inputs['original_cube']=sha(path)
    with fits.open(path,memmap=True) as h:
        far=channel_weights(h['MU'].data,bounds[1:]);assert (weights[1:,:140]==0).all();np.testing.assert_allclose(far,weights[1:,140:],atol=1e-14)
        area=h['AREA'].data.sum(axis=0);mask=h['MASK'].data.astype(bool)
        for view in (0,1):count[view,1:]=np.tensordot(far,h['COUNTS'].data[:,view].sum(axis=0),axes=(1,0))
    assert (count>=0).all();maps=np.array([render_maps(count[v],area[v],mask[v]) for v in (0,1)])
    npz(d/'physical_selected.npz',counts=count,density=maps[:,:,0].astype('f4'),signed_residual=maps[:,:,1].astype('f4'),mask=mask,area=area,weights=weights,mu=mu,bounds=bounds)
    with np.load(OUT/'rgb/rgb_overview.npz') as z:fixed=z['stretch']
    balanced=[]
    for row in (0,1):
        _,r=stretch(maps[1,:,row],mask[1],bool(row));balanced.append([[x['lo'],x['hi']] for x in r])
    l,b,pix=grid();rows=[]
    for kind,limits in [('fixed_gain',fixed),('balanced',np.array(balanced))]:
        a=dict(l=l,b=b,density=maps[0][:,0,pix].astype('f4'),signed_residual=maps[0][:,1,pix].astype('f4'),mask=mask[0,pix],channel_kpc=bounds,stretch=limits,scope=np.array('Final user choice5–10/10–20/20–100kpc; '+kind+'; original native PSF parent, CMD and geometry'),label=np.array('CMD-selected sky'))
        npz(d/('rgb_selected_'+kind+'.npz'),**a);rgb(a,d,'rgb_selected_'+kind)
        if kind=='fixed_gain':physical(a,d,'physical_selected')
        for row in (0,1):
            for j,ch in enumerate('BGR'):rows.append(dict(display=kind,row=('density','positive_residual')[row],channel=ch,d_lo=bounds[j,0],d_hi=bounds[j,1],lo=limits[row,j,0],hi=limits[row,j,1],saturated_fraction=float((maps[0,j,row,mask[0]]>limits[row,j,1]).mean())))
    csv_rows(d/'stretch.csv',rows);csv_rows(d/'node_weights.csv',[dict(channel=ch,mu=float(x),weight=float(v)) for ch,row in zip('BGR',weights) for x,v in zip(mu,row) if v>0]);save_json(d/'selection.json',dict(BGR_kpc=bounds,primary='rgb_selected_fixed_gain',companion='rgb_selected_balanced',parent=493314135,weights_exactly_reuse_completed_responses=True,production_unchanged=True))
    complete(d,inputs,scope='Final user-selected overview, all available gr footprint; fixed gain is primary, balanced companion retained',weight_identity_passed=True,coordinates='Galactic, actual Sgr/Pal5 dashed, b0/±20/±30',new_distance_fit=False)


if __name__=='__main__':main()
