#!/usr/bin/env python3
"""Fixed three-star DR11 image/model/residual audit (same imaging, not independent)."""
from pathlib import Path
import json,sys,hashlib,time
from concurrent.futures import ThreadPoolExecutor
import requests
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm,SymLogNorm

ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/photdis_pal14_images')
FIG=ROOT/'figures/giant_parameter_audit_v1'

def main():
    CACHE.mkdir(parents=True,exist_ok=True);FIG.mkdir(parents=True,exist_ok=True)
    inv=json.loads((ROOT/'manifests/kgiant_reference_inventory_v1.json').read_text())
    r=next(r for r in inv['rows'] if r['source_id']=='4464703441466064512')
    m=r['selected_measurement']
    stars=[dict(name='Pal14_star1',matched_ra=m['ra'],matched_dec=m['dec'],quality_raw=m['quality_raw'])]
    tasks=[(r,kind) for r in stars for kind in ('image','model','resid')]
    def get(item):
        r,kind=item;path=CACHE/(r['name']+'_'+kind+'.fits');params=dict(ra=r['matched_ra'],dec=r['matched_dec'],layer='ls-dr11'+('' if kind=='image' else '-'+kind),pixscale=.262,bands='grz',size=80)
        u=requests.Request('GET','https://www.legacysurvey.org/viewer/fits-cutout',params=params).prepare().url
        try:
            if not path.exists():
                time.sleep(10)
                response=requests.get(u,timeout=60);response.raise_for_status();path.write_bytes(response.content)
            with fits.open(path) as f:assert f[0].data.shape==(3,80,80)
            return dict(name=r['name'],kind=kind,url=u,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),available=True)
        except Exception as e:return dict(name=r['name'],kind=kind,url=u,available=False,error=str(e))
    with ThreadPoolExecutor(max_workers=1) as pool:downloads=list(pool.map(get,tasks))
    stats=[]
    for r in stars:
        if not all(d['available'] for d in downloads if d['name']==r['name']):continue
        arrays={k:fits.getdata(CACHE/(r['name']+'_'+k+'.fits')) for k in ('image','model','resid')}
        header=fits.getheader(CACHE/(r['name']+'_image.fits'));cx,cy=WCS(header).celestial.world_to_pixel_values(r['matched_ra'],r['matched_dec'])
        yy,xx=np.indices((80,80));radius=np.hypot(xx-cx,yy-cy)*.262
        fig,axes=plt.subplots(3,3,figsize=(9,9),layout='constrained')
        for i,b in enumerate('grz'):
            im,mod,res=(arrays[k][i] for k in ('image','model','resid'))
            maxval=max(float(np.nanmax(im)),float(np.nanmax(mod)));floor=maxval*.002
            limits=max(float(np.nanmax(abs(res))),floor)
            st=dict(name=r['name'],band=b,rchisq=r['quality_raw']['RCHISQ_'+b.upper()],apertures=[])
            for rad in (1.5,3.):
                mask=radius<=rad;den=float(mod[mask].sum())
                st['apertures'].append(dict(radius_arcsec=rad,residual_sum_over_model=float(res[mask].sum()/den),absolute_residual_sum_over_model=float(abs(res[mask]).sum()/den)))
            stats.append(st)
            for j,(a,label) in enumerate(((im,'Image'),(mod,'Model'),(res,'Residual'))):
                norm=AsinhNorm(linear_width=floor,vmin=0,vmax=maxval) if j<2 else SymLogNorm(linthresh=floor,vmin=-limits,vmax=limits)
                axes[i,j].imshow(a,origin='lower',cmap='gray' if j<2 else 'RdBu_r',norm=norm)
                axes[i,j].add_patch(plt.Circle((cx,cy),1.5/.262,fill=False,color='#ffcc00',lw=.8));axes[i,j].set(xticks=[],yticks=[],title=f'{b}: {label}')
                if j==0:axes[i,j].set_ylabel(f'RCHISQ={st["rchisq"]:.1f}')
        fig.suptitle(f'{r["name"]}: fixed DR11 same-image audit\n21 arcsec stamps; yellow radius1.5 arcsec; residual scale differs from image/model',fontsize=11)
        fig.savefig(FIG/(f'03_image_{r["name"]}.png'),dpi=150);plt.close(fig)
    out=dict(downloads=downloads,aperture_diagnostics=stats,
      limitation='Same coadds and model, not independent photometry. Aperture residual/model ratios include PSF/neighbor/sky effects; not total-flux corrections. No quality cuts or predictions changed.')
    (ROOT/'manifests/pal14_image_audit_v1.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
