#!/usr/bin/env python3
"""Use the actual track arc length for the along-coordinate Gaussian width."""
from scipy.ndimage import gaussian_filter
import shutil
from photdis.stage1 import require_compute
from stream_common import *


def smooth_maps(raw,area,xedges,yedges,roi,sigma):
    kernel=(sigma/(track_length(roi)*np.diff(xedges)[0]),sigma/np.diff(yedges)[0]);n=gaussian_filter(raw,kernel);a=gaussian_filter(area,kernel)
    density=np.divide(n,a,out=np.full_like(n,np.nan),where=(area>0)&(a>0));cfg=C['local'][roi];off=abs((yedges[:-1]+yedges[1:])/2)>=cfg['off_abs_deg'][0]
    residual=density-np.nanmedian(density[:,off],axis=1)[:,None]
    return np.stack((n,a,residual))


def main():
    require_compute();d=folder('local');ex=folder('execution');marker=d/'complete.json';rec=json.loads(marker.read_text());old=ex/'local_before_arc_metric.json'
    if not old.exists():old.write_text(marker.read_text())
    records=[]
    for p in d.glob('*_redraw.npz'):
        with np.load(p) as a:z={k:a[k] for k in a.files}
        roi=p.stem.replace('_phase_redraw','').replace('_redraw','');backup=ex/(p.stem+'_before_arc_metric.npz')
        if not backup.exists():shutil.copy2(p,backup)
        if '_phase_redraw' in p.name:
            z['signed_maps']=np.array([smooth_maps(z[f'raw_counts_{k}'],z['raw_area'],z['along_edges'],z['offset_edges'],roi,.3)[2] for k in range(3)])
        else:
            for model in ('FOS','Pal5_control') if roi.startswith('Pal') else ('FOS',):
                raw,area=z[model+'_sky_gr_complete']
                for sigma in (.1,.3):z[model+'_sky_smoothed_'+str(sigma)]=smooth_maps(raw,area,z['sky_along_edges'],z['sky_offset_edges'],roi,sigma)
        z['track_length_deg']=np.array(track_length(roi));npz(p,**z);records.append(dict(grid=p.name,track_length_deg=track_length(roi)))
    rec['map_kernel_metric']=dict(code=sha(Path(__file__)),parent_receipt=sha(old),records=records,
        definition='same registered sigmas0.1/0.3deg; along-RA fraction now uses actual reference-track arc length, not only projected RA length. Mean local tangent metric. Raw counts, independent areas, CMD widths and unsmoothed recovery intervals unchanged.')
    rec['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(marker,rec)


if __name__=='__main__':main()
