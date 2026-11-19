#!/usr/bin/env python3
"""Preview2/5/10degree backgrounds on unchanged final-channel density maps."""
from concurrent.futures import ThreadPoolExecutor
from photdis.stage1 import require_compute
from halo_atlas import smooth,grid
from sky6_common import *
from sky6_figures import rgb


def main():
    require_compute();d=folder('smoothing_trials');config=ROOT/'configs/sky_sequence_smoothing_20261119.json';settings=json.loads(config.read_text());main=ROOT/settings['main'];before=sha(main);inputs=dict(smoothing_settings=sha(config),unchanged_main=before)
    physical=read(OUT/'selected_channels',('physical_selected.npz','counts','density','signed_residual','mask','area'),inputs)
    a=read(OUT/'selected_channels',('rgb_selected_fixed_gain.npz','l','b','density','signed_residual','mask','channel_kpc','stretch','scope','label'),inputs);np.testing.assert_array_equal(a['channel_kpc'],settings['channels_kpc'])
    _,_,pix=grid();count=physical['counts'][0];area=physical['area'][0];mask=physical['mask'][0];rows=[]
    for sigma in settings['background_sigma_deg']:
        if sigma==5:residual=a['signed_residual'].copy()
        else:
            den=np.maximum(smooth(area,sigma),0)
            def channel(c):
                return np.divide(np.maximum(smooth(c,sigma),0),den,out=np.full_like(den,np.nan),where=den>0)
            with ThreadPoolExecutor(3) as pool:background=np.array(list(pool.map(channel,count)))
            signed=physical['density'][0]-background;signed[:,~mask]=np.nan;npz(d/f'physical_background_{sigma}deg.npz',signed_residual=signed.astype('f4'),background=background.astype('f4'),mask=mask,background_sigma_deg=np.array(sigma));residual=signed[:,pix].astype('f4')
        out=dict(a,signed_residual=residual,scope=np.array(f'Preview only: background Gaussian sigma{sigma}deg; main density0.3deg and fixed main5deg stretch unchanged'),residual_label=np.array(f'Positive residual · background σ={sigma}°'))
        np.testing.assert_array_equal(out['density'],a['density']);np.testing.assert_array_equal(out['mask'],a['mask']);np.testing.assert_array_equal(out['stretch'],a['stretch'])
        name=f'rgb_background_{sigma}deg';npz(d/(name+'.npz'),**out);rgb(out,d,name)
        rows.append(dict(background_sigma_deg=sigma,density_sigma_deg=.3,main_unchanged=True,stretch='exact same original5deg main stretch',B_kpc='5–10',G_kpc='10–20',R_kpc='20–100'))
    assert sha(main)==before;csv_rows(d/'preview_parameters.csv',rows);save_json(d/'main_preserved.json',dict(path=settings['main'],sha256_before=before,sha256_after=sha(main),unchanged=True,scope='Background2/10degree previews are separate outputs; main image and physical density are untouched'))
    complete(d,inputs,preview_only=True,main_image_unchanged=True,interpretation=settings['interpretation'])


if __name__=='__main__':main()
