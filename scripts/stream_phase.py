#!/usr/bin/env python3
"""Sgr phase-control plate at one frozen entrance, area and reference modulus."""
from scipy.ndimage import gaussian_filter
from matplotlib.colors import SymLogNorm
from photdis.stage1 import require_compute
from stream_common import *
from stream_local import templates,curves,boundaries,accept,region_masks
from halo_common import BLUE,ORANGE,GREEN

KINDS=('MS_SGB_RGB','MS_TO','blue_box')
LABELS=('MS + SGB/RGB','MS / turnoff','Blue turnoff box')


def render(a,d,roi,rows):
    with plt.rc_context(PAPER):
        f,axs=plt.subplots(2,3,figsize=(7.15,5.3),layout='constrained')
        h=a['hess'];cap=max(float(np.quantile(abs(h),.99)),.1)
        for k,ax in enumerate(axs[0]):
            z=ax.pcolormesh(a['gr_edges'],a['g_edges'],h.T,cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
            for lo,hi in zip(a[f'lower_{k}'],a[f'upper_{k}']):ax.plot(lo,a['g_curve'],color=ORANGE,lw=.6);ax.plot(hi,a['g_curve'],color=ORANGE,lw=.6)
            ax.set(xlim=(-.2,1.2),ylim=(24,14),xlabel=r'$(g-r)_0$',ylabel=r'$g_0$ (mag)',title=LABELS[k]);tick=10**np.floor(np.log10(cap));f.colorbar(z,ax=ax,fraction=.04,label='Excess (deg⁻² bin⁻¹)',ticks=[-tick,0,tick],format=plt.FormatStrFormatter('%g'))
        ax=axs[1,0]
        for k,col in enumerate((BLUE,ORANGE,GREEN)):ax.plot(a['mu'],a['response'][k],color=col,lw=.8,label=LABELS[k])
        ax.axvline(float(a['reference_mu']),color='.3',ls=':',lw=.6);ax.axhline(0,color='.4',lw=.5);ax.legend(frameon=False,fontsize=6);ax.set(xlabel='Distance modulus (mag)',ylabel='Target − control (deg⁻²)')
        for k,ax in zip((0,1),axs[1,1:]):
            values=a['signed_maps'][k];finite=values[np.isfinite(values)];cap=max(float(np.quantile(abs(finite),.99)),.1) if len(finite) else .1;tick=10**np.floor(np.log10(cap))
            z=ax.pcolormesh(a['along_edges'],a['offset_edges'],np.ma.masked_invalid(values.T),cmap='RdBu_r',norm=SymLogNorm(cap/20,vmin=-cap,vmax=cap),rasterized=True)
            ax.axhline(0,color='.3',ls=':',lw=.5);ax.set(xlabel='Along-track fraction',ylabel='Cross-track offset (deg)',title=LABELS[k]+' · σ=0.3°');f.colorbar(z,ax=ax,fraction=.04,label='Excess (deg⁻²)',ticks=[-tick,0,tick],format=plt.FormatStrFormatter('%g'))
        figure(d,roi+'_phase_controls',f,rows)


def main():
    require_compute();d=folder('local');inputs={};tm=templates(inputs)['FOS'];mag,phase=tm
    responses=list(csv.DictReader((d/'modulus_response.csv').open()));density=json.loads((OUT/'local_area/complete.json').read_text())['density'];phase_rows=[]
    for roi in ('Sgr_north','Sgr_south'):
        cfg=C['local'][roi];mu=cfg['reference_mu'];t=read(OUT/'local_source',(roi+'.npz','TYPE','ls_id','m','offset','along','selected','DEC'),inputs)
        r=read(OUT/'local_area',(roi+'.npz','offset','along','selected'),inputs)
        take=t['selected'][4];t={k:(v[:,take] if k=='selected' else v[take]) for k,v in t.items()}
        with np.load(d/(roi+'_redraw.npz')) as z:a={k:z[k] for k in z.files}
        ec=(a['g_centers'],a['sigma_g']);sr=region_masks(t['offset'],cfg);rr=region_masks(r['offset'],cfg);g=np.linspace(14,24,600)
        rows=[];save=dict(gr_edges=a['hess_gr_edges'],g_edges=a['hess_g_edges'],hess=a['FOS_hess_gr_complete'][0]-a['FOS_hess_gr_complete'][1],g_curve=g,reference_mu=np.array(mu),along_edges=a['sky_along_edges'],offset_edges=a['sky_offset_edges']);maps=[];ccounts=[]
        area=a['FOS_sky_gr_complete'][1];width=(cfg['ra_range'][1]-cfg['ra_range'][0])*np.cos(np.deg2rad(np.median(t['DEC'])));kernel=(.3/(width*np.diff(save['along_edges'])[0]),.3/np.diff(save['offset_edges'])[0]);sa=gaussian_filter(area,kernel)
        for k,kind in enumerate(KINDS):
            curve=curves(mag,phase,kind);accepted=accept(t['m'],[mu],curve,ec,kind)[:,0];ccounts.append(accepted)
            save[f'lower_{k}'],save[f'upper_{k}']=boundaries(g,mu,curve,ec,kind)
            count=np.histogram2d(t['along'][accepted],t['offset'][accepted],(save['along_edges'],save['offset_edges']))[0];sm=gaussian_filter(count,kernel)
            value=np.divide(sm,sa,out=np.full_like(sm,np.nan),where=(area>0)&(sa>0));off=abs((save['offset_edges'][:-1]+save['offset_edges'][1:])/2)>=cfg['off_abs_deg'][0];signed=value-np.nanmedian(value[:,off],axis=1)[:,None];maps.append(signed)
            save[f'raw_counts_{k}']=count
            rr0=[x for x in responses if x['roi']==roi and x['model']=='FOS' and x['variant']=='gr_complete' and x['phase']==kind];save['mu']=np.array([float(x['mu']) for x in rr0]);save[f'response_{k}']=np.array([float(x['signed_excess']) for x in rr0])
            for label,zone,ar in zip(('target','off'),sr[:2],rr[:2]):
                rows.append(dict(roi=roi,phase=kind,zone=label,n=int((accepted&zone).sum()),area_deg2=float((r['selected'][4]&ar).sum()/density)))
        rgb=accept(t['m'],[mu],curves(mag,phase,'SGB_RGB'),ec,'SGB_RGB')[:,0]
        np.testing.assert_array_equal(ccounts[0],ccounts[1]|rgb)
        for label,zone in zip(('target','off'),sr[:2]):
            for kind,accepted in (('MS_only',ccounts[1]&~rgb),('RGB_only',rgb&~ccounts[1]),('both',rgb&ccounts[1])):rows.append(dict(roi=roi,phase=kind,zone=label,n=int((accepted&zone).sum())))
        save.update(signed_maps=np.array(maps),raw_area=area,response=np.array([save.pop(f'response_{k}') for k in range(3)]));npz(d/(roi+'_phase_redraw.npz'),**save);render(save,d,roi,rows);phase_rows.extend(rows)
    csv_rows(d/'Sgr_phase_decomposition.csv',phase_rows)
    marker=d/'complete.json';rec=json.loads(marker.read_text());rec['phase_decomposition']=dict(code=sha(Path(__file__)),inputs=inputs,scope='exact same gr_complete source entrance and area; mutually exclusive MS-only/RGB-only/overlap acceptance accounting is not a spectroscopic stellar classification');rec['outputs']={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='complete.json'};save_json(marker,rec)


if __name__=='__main__':main()
