#!/usr/bin/env python3
"""Registered six-object external audit; no prediction or residual selection."""
from pathlib import Path
import sys, json, zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'.deps')]
import numpy as np
import fitsio
from photdis.photometry import derive_photometry
from photdis.stage1 import CORE_COLUMNS, PHOT_COLUMNS, require_compute, write_json
from kgiant_distance_inventory import sha

OUT=Path('data/kgiant_external_v1')
SWEEP=Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0/sweep-285m035-290m030.fits')

def separation(ra,dec,r,d):
    """Great-circle separation in arcsec, safe at RA wrap."""
    a,b=np.deg2rad(ra),np.deg2rad(dec);c,e=np.deg2rad(r),np.deg2rad(d)
    return np.rad2deg(2*np.arcsin(np.sqrt(np.clip(np.sin((b-e)/2)**2+np.cos(b)*np.cos(e)*np.sin((a-c)/2)**2,0,1))))*3600

def main():
    require_compute(); OUT.mkdir(exist_ok=True)
    if (OUT/'inventory.json').exists():raise FileExistsError(OUT/'inventory.json')
    p=np.load('data/kgiant_pal12_v1/members.npz')
    i=np.flatnonzero(p['source_id']==6817950249538781952)[0]
    rows=[dict(cluster='Pal12',name='6817950249538781952',source_id=str(p['source_id'][i]),
        ra=326.625880,dec=-21.267698,grz_quality=bool(p['tier_core_grz_clean'][i]),
        w1_quality=bool(p['w1_clean'][i]),old_dust_scope=bool(p['ebv'][i]<.03),
        ebv=float(p['ebv'][i]),pmember=float(p['pmember'][i]),identity_status='literature_link_pending',
        release=int(p['release'][i]),
        photometry={k:float(p[k][i]) if np.isfinite(p[k][i]) else None for k in
          ('g','r','z','w1','g0','r0','z0','w10','sigma_g','sigma_r','sigma_z','sigma_w1','gaia_g','bp_rp')})]
    stars=json.load(open('configs/kgiant_external_v1.json'))['stars']
    coords=fitsio.read(str(SWEEP),ext=1,columns=['RA','DEC'])
    for s in stars:
        sep=separation(coords['RA'],coords['DEC'],s['ra'],s['dec']);nearest=np.argsort(sep)[:2]
        rec=dict(cluster='Terzan7',**s,nearest_arcsec=float(sep[nearest[0]]),second_arcsec=float(sep[nearest[1]]),
                 matches_within2=int((sep<2).sum()),nominal_K=4200<=s['teff']<5000,
                 temperature_boundary_sensitive=abs(s['teff']-4200)<100,
                 identity_status='unique_position_match' if (sep<2).sum()==1 else 'ambiguous_or_missing')
        if rec['matches_within2']==1:
            r=fitsio.read(str(SWEEP),ext=1,rows=nearest[:1],columns=CORE_COLUMNS+PHOT_COLUMNS)
            ph=derive_photometry(r);psf=str(r['TYPE'][0]).strip()=='PSF'
            rec.update(source_id=str(r['REF_ID'][0]),release=int(r['RELEASE'][0]),
                       matched_ra=float(r['RA'][0]),matched_dec=float(r['DEC'][0]),
                       grz_quality=bool(ph['grz_clean'][0] and psf),w1_quality=bool(ph['w1_clean'][0]),
                       old_dust_scope=bool(r['EBV'][0]<.03),ebv=float(r['EBV'][0]),psf=psf,
                       photometry={k:float(ph[k][0]) if np.isfinite(ph[k][0]) else None for k in
                           ('g','r','z','w1','g0','r0','z0','w10','sigma_g','sigma_r','sigma_z','sigma_w1')},
                       quality_raw={k:np.asarray(r[k][0]).tolist() for k in
                           ['MASKBITS']+[f'{k}_{b}' for k in ('NOBS','FRACMASKED','FRACIN','FRACFLUX','RCHISQ') for b in ('G','R','Z')]+['WISEMASK_W1','FRACFLUX_W1']})
        rows.append(rec)
    z=zipfile.ZipFile('data/reference/vasiliev_baumgardt_2021_members.zip')
    members={}
    for cluster,file in [('Pal12','Pal_12'),('Terzan7','Terzan_7')]:
        ls=[l.split() for l in z.read('catalogues/'+file+'.txt').decode().splitlines() if not l.startswith('#')]
        members[cluster]={l[0]:float(l[16]) for l in ls}
    for r in rows:
        if 'source_id' in r:r['pmember']=members[r['cluster']].get(r['source_id'])
    write_json(OUT/'inventory.json',dict(rows=rows,source_sha256={str(x):sha(x) for x in
        [Path(__file__),Path('configs/kgiant_external_v1.json'),Path('docs/kgiant_external_protocol_v1.md'),Path('data/kgiant_pal12_v1/members.npz')]}))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
