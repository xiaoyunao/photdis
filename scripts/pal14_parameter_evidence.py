#!/usr/bin/env python3
"""Fixed-identity literature and model-consistency evidence; never changes predictions."""
from pathlib import Path
import json,hashlib,tarfile,sys
import numpy as np
from bs4 import BeautifulSoup
from scipy.interpolate import RegularGridInterpolator
from astropy.coordinates import SkyCoord
import astropy.units as u
ROOT=Path(__file__).resolve().parents[1]
CACHE=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/tmp/photdis_pal14_sources')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bc_at(teff,logg,feh):
    # Tensor interpolation on the eight enclosing AV=0 grid nodes (no extrapolation).
    arrays=[]
    with tarfile.open(CACHE/'DECam.txz') as t:
        for m in t.getmembers():
            if m.isfile():
                a=np.loadtxt(t.extractfile(m));arrays.append(a[a[:,3]==0])
    a=np.vstack(arrays);axes=[]
    for col,x in zip([0,1,2],[teff,logg,feh]):
        v=np.unique(a[:,col]);i=np.searchsorted(v,x)
        if i==0 or i==len(v):raise ValueError('Grid extrapolation requested')
        axes.append(v[i-1:i+1])
    grid=np.empty((2,2,2,3))
    for i,T in enumerate(axes[0]):
        for j,g in enumerate(axes[1]):
            for k,f in enumerate(axes[2]):
                row=a[(a[:,0]==T)&(a[:,1]==g)&(a[:,2]==f)]
                assert len(row)==1
                grid[i,j,k]=row[0,[6,7,9]]
    return RegularGridInterpolator(axes,grid)([[teff,logg,feh]])[0], [v.tolist() for v in axes]
def main():
    typed=json.load(open(ROOT/'manifests/giant_parameter_typed_v1.json'))
    spec=next(r for r in typed['rows'] if r['source_id']=='4464703441466064512')
    old=next(r for r in json.load(open(ROOT/'manifests/kgiant_reference_inventory_v1.json'))['rows'] if r['source_id']=='4464703441466064512')
    r={**old,**spec}
    # Verify the small numeric transcription against the archived primary tables.
    j=BeautifulSoup((CACHE/'jordi.html').read_text(),'html.parser')
    table=j.select('figure.ltx_table')[0]
    line=next([c.get_text(' ',strip=True) for c in tr.select('td')] for tr in table.select('tr') if tr.select('td') and tr.select('td')[0].get_text(strip=True)=='1')
    assert line[2:6]==['16:11:05.81','14:57:45.1','17.37','1.39'] and line[-3:]==['72.53','0.07','Y']
    c=BeautifulSoup((CACHE/'caliskan.html').read_text(),'html.parser')
    line2=next([v.get_text(' ',strip=True) for v in tr.select('th,td')] for tr in c.select('figure.ltx_table')[1].select('tr') if tr.select('th,td') and tr.select('th,td')[0].get_text(strip=True)=='1')
    assert line2==['1','4100','0.7','2.1']
    lit=SkyCoord('16h11m05.81s','14d57m45.1s');ga=r['gaia'];desi=r['desi_diagnostics'];m=r['selected_measurement']
    coords={'member':(r['ra'],r['dec']),'Gaia':(ga['RA'],ga['DEC']), 'DESI_target':(desi['TARGET_RA'],desi['TARGET_DEC']),'DECam':(m['ra'],m['dec'])}
    separations={name:float(lit.separation(SkyCoord(ra*u.deg,dec*u.deg)).arcsec) for name,(ra,dec) in coords.items()}
    pairsep={name:float(SkyCoord(ga['RA']*u.deg,ga['DEC']*u.deg).separation(SkyCoord(ra*u.deg,dec*u.deg)).arcsec) for name,(ra,dec) in coords.items()}
    bc,nodes=bc_at(r['teff'],r['logg'],r['feh_corrected'])
    logg_mass=[4.438+np.log10(mass)+4*np.log10(r['teff']/5772)+.4*(old['Mg_ref']+bc[0]-4.74) for mass in (.7,.9)]
    preds=json.load(open(ROOT/'manifests/kgiant_reference_predictions_v1.json'))['rows']
    cdfs=[dict(teacher=p['teacher'],model=p['model'],reference_cdf=float(np.interp(p['reference_mu'],p['distribution_grid_mu'],p['distribution_cdf']))) for p in preds if p['source_id']==r['source_id']]
    out=dict(source_id=r['source_id'],targetid=r['targetid'],jordi2009=dict(star='1',ra_hms=line[2],dec_dms=line[3],V=17.37,B_minus_V=1.39,RV_kms=72.53,RV_error_kms=.07,member='Y',table='1',url='https://arxiv.org/html/0903.4448v1'),
      caliskan2012=dict(star='1',teff=4100,logg=.7,temperature_method='VRI/2MASS JHK empirical color calibrations, assumed FeH=-1.5 and E(B-V)=.033',gravity_method='Yale-Yonsei 10/11Gyr isochrones, FeH=-1.5 alpha=.3; not ionization equilibrium',error_note='100K/.5dex are abundance-sensitivity perturbations, not an independent precise parameter posterior',url='https://arxiv.org/html/1110.5151v1'),
      desi=r,desi_program=json.load(open(ROOT/'manifests/pal14_desi_program_v1.json')),position_separations_from_literature_arcsec=separations,position_separations_from_Gaia_arcsec=pairsep,
      position_caveat='Raw published J2000 coordinates vs catalog coordinates; equinox does not specify observing epoch. No precision astrometric epoch inference.',
      RV_difference_kms=r['rv']-72.53,RV_formal_combined_sigma_kms=float(np.hypot(r['rv_error'],.07)),
      teff_difference_K=r['teff']-4100,logg_difference_dex=r['logg']-.7,
      parameter_consistency='cross_source_conflict_unresolved',DESI_in_frozen_temperature_window=bool(4200<=r['teff']<5000),literature_in_frozen_temperature_window=False,
      photometry_note='Gaia G/BP/RP, DECam AB grz, Johnson V/B-V are different passbands; no direct zero-point differences inferred.',
      MIST=dict(url='https://mist.science/BC_tables/v1/DECam.txz',AB=True,AV=0,RV=3.1,BC_grz=bc.tolist(),nodes_T_logg_feh=nodes,
          predicted_gr_rz=[float(bc[1]-bc[0]),float(bc[2]-bc[1])],observed_gr_rz=[m['photometry']['g0']-m['photometry']['r0'],m['photometry']['r0']-m['photometry']['z0']],
          conditional_logg_mass07_09=logg_mass,solar=dict(Teff=5772,logg=4.438,Mbol=4.74),
          caveat='Atmosphere-grid consistency diagnostic at DESI parameters; gravity inferred with BC held at those parameters and assumed mass, not an independent truth or self-consistent isochrone fit'),
      reference_cdf_from_public_grid=cdfs,status='Strong positional/member association; cross-source atmospheric parameters unresolved; preserve original selection and all22 predictions',
      source_sha256={p.name:digest(p) for p in [CACHE/'jordi.html',CACHE/'caliskan.html',CACHE/'DECam.txz',ROOT/'manifests/giant_parameter_typed_v1.json',ROOT/'manifests/kgiant_reference_inventory_v1.json',ROOT/'manifests/kgiant_reference_predictions_v1.json']})
    (ROOT/'manifests/pal14_parameter_evidence_v1.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    ax[0].errorbar(r['teff'],r['logg'],xerr=r['teff_error'],yerr=r['logg_error'],fmt='o',label='DESI (formal errors)')
    ax[0].scatter([4100],[.7],marker='s',label='Literature colors + isochrone')
    ax[0].set(xlabel='Teff [K]',ylabel='logg [cgs]',title='Same identity; different parameter sources');ax[0].invert_yaxis();ax[0].legend(fontsize=8)
    mist=out['MIST'];ax[1].scatter(*mist['observed_gr_rz'],label='Measured DECam',s=65)
    ax[1].scatter(*mist['predicted_gr_rz'],label='MIST at DESI parameters',marker='s',s=65)
    ax[1].set(xlabel='(g-r)0',ylabel='(r-z)0',title='Conditional atmosphere check, AV=0');ax[1].legend(fontsize=8)
    for a in ax:a.grid(alpha=.2)
    fig.suptitle('Pal14 star1: unresolved consistency; no new truth or distance correction')
    fig.savefig(ROOT/'figures/giant_parameter_audit_v1/04_pal14_parameter_sources.png',dpi=160);plt.close(fig)
    print(json.dumps({k:out[k] for k in ['position_separations_from_literature_arcsec','RV_difference_kms','RV_formal_combined_sigma_kms','teff_difference_K','logg_difference_dex','MIST','reference_cdf_from_public_grid']},indent=2))
if __name__=='__main__':main()
