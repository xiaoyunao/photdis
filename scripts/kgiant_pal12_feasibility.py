#!/usr/bin/env python3
"""One registered more-metal-rich reference check, without distance-model fitting."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'.deps')]
import fitsio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cluster_rgb_inventory import read_members, MEMBERS
from giant_ngc2419_reference import rvs_giants
from photdis.photometry import derive_photometry
from photdis.stage1 import CORE_COLUMNS, PHOT_COLUMNS, require_compute, write_json
from kgiant_distance_inventory import sha

SWEEP = Path('/global/cfs/projectdirs/cosmo/data/legacysurvey/dr11/south/sweep/11.0/sweep-325m025-330m020.fits')
OUT = Path('data/kgiant_pal12_v1')


def main():
    require_compute()
    OUT.mkdir(exist_ok=True)
    members = read_members(MEMBERS, 'catalogues/Pal_12.txt')
    members = members[members['pmember'] >= .5]
    small = fitsio.read(str(SWEEP), ext=1, columns=['REF_ID', 'REF_CAT'])
    ix = np.flatnonzero(np.isin(small['REF_ID'], members['source_id']) &
                       (np.char.strip(small['REF_CAT'].astype('U')) == 'G3'))
    rows = fitsio.read(str(SWEEP), ext=1, columns=CORE_COLUMNS+PHOT_COLUMNS, rows=ix)
    photo = derive_photometry(rows)
    score = 1000*photo['grz_clean'] + 100*photo['grz_base'] + np.nan_to_num(photo['snr_g'], nan=0)/10000
    order = np.lexsort((-score, rows['REF_ID']))
    _, first = np.unique(rows['REF_ID'][order], return_index=True)
    keep = order[first]; raw_n = len(rows)
    rows = rows[keep]; photo = {k:v[keep] for k,v in photo.items()}
    m = members[np.searchsorted(members['source_id'], rows['REF_ID'])]
    assert np.array_equal(m['source_id'], rows['REF_ID'])
    rv, sid = rvs_giants(members['source_id'])
    k = (rv['TEFF']>=4200)&(rv['TEFF']<5000)
    spectro_k = np.isin(rows['REF_ID'], sid[k])
    core = m['pmember']>.9
    psf = np.char.strip(rows['TYPE'].astype('U')) == 'PSF'
    clean = photo['grz_clean'] & psf
    low = clean & (rows['EBV']<.03)
    w1 = low & photo['w1_clean']
    masks = {'matched':np.ones(len(rows),bool),'core':core,
             'core_gr_clean':core&photo['gr_clean']&psf,
             'core_grz_clean':core&clean, 'core_clean_low_dust':core&low,
             'core_clean_low_dust_w1':core&w1,
             'core_spectroscopic_K':core&spectro_k,
             'core_qualified_K':core&w1&spectro_k}
    result = {'member_Pge0p5':len(members), 'member_Pgt0p9':int((members['pmember']>.9).sum()),
              'raw_matches':raw_n, 'unique_matches':len(rows),
              'missing_member_ids':[int(x) for x in members['source_id'][~np.isin(members['source_id'],rows['REF_ID'])]],
              'release_counts':{str(x):int((rows['RELEASE']==x).sum()) for x in np.unique(rows['RELEASE'])},
              'spectroscopic_K_in_all_members':int(k.sum()),
              'phase':'unknown; no RGB/RC/HB assignment from color or low gravity',
              'reference':{'mu':16.3,'sigma_mu':.1,'method':'model-dependent CMD fitting; only feasibility',
                           'url':'https://arxiv.org/abs/1709.08509'}, 'tiers':{}}
    for name, use in masks.items():
        vals = photo['g0'][use]-16.3; vals = vals[np.isfinite(vals)]
        result['tiers'][name] = {'n':int(use.sum()),'Mg_lt0p5':int((vals<.5).sum()),
                                'r16_18':int((use&(photo['r']>=16)&(photo['r']<18)).sum()),
                                'Mg_min_median_max':np.quantile(vals,[0,.5,1]).tolist() if len(vals) else [],
                                'ebv_min_median_max':np.quantile(rows['EBV'][use],[0,.5,1]).tolist() if use.any() else []}
    result['core_failures_overlapping'] = {
        **{f'satur_{b}':int((core&((rows['MASKBITS']&(1<<bit))!=0)).sum()) for b,bit in [('g',2),('r',3),('z',4)]},
        **{f'fracmasked_{b.lower()}':int((core&(rows[f'FRACMASKED_{b}']>=.4)).sum()) for b in ('G','R','Z')},
        'ebv_ge0p03':int((core&(rows['EBV']>=.03)).sum())}
    np.savez_compressed(OUT/'members.npz', **photo, source_id=rows['REF_ID'], pmember=m['pmember'],
                        gaia_g=m['gaia_g'], bp_rp=m['bp_rp'], spectroscopic_K=spectro_k,
                        ebv=rows['EBV'], release=rows['RELEASE'], **{'tier_'+a:b for a,b in masks.items()})
    fig,axes = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,x,y,xlab,ylab in [(axes[0],m['bp_rp'],m['gaia_g'],'Gaia BP-RP','Gaia G (observed)'),
                            (axes[1],photo['gr0'],photo['g0']-16.3,'(g-r)0','g0 - 16.3 [mag]'),
                            (axes[2],photo['rz0'],photo['g0']-16.3,'(r-z)0','g0 - 16.3 [mag]')]:
        for name,color,size in [('core','0.65',18),('core_grz_clean','#e69f00',25),('core_clean_low_dust_w1','#0072b2',35)]:
            use=masks[name]&np.isfinite(x)&np.isfinite(y)
            ax.scatter(x[use],y[use],s=size,c=color,label=f'{name}: {use.sum()}',alpha=.8)
        use=core&spectro_k&np.isfinite(x)&np.isfinite(y)
        ax.scatter(x[use],y[use],s=90,facecolors='none',edgecolors='red',label=f'DESI K giant: {use.sum()}')
        ax.set(xlabel=xlab,ylabel=ylab);ax.invert_yaxis();ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle('Pal 12: independent-member feasibility | phase unknown | no model scoring')
    fig.savefig(OUT/'06_pal12_cmd.png',dpi=160);plt.close(fig)
    result['source_sha256']={str(p):sha(p) for p in [MEMBERS,Path(__file__),Path('docs/kgiant_pal12_feasibility_v1.md')]}
    result['figure_sha256']=sha(OUT/'06_pal12_cmd.png')
    write_json(OUT/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__':main()
