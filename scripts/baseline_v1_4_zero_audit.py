#!/usr/bin/env python3
"""Allocated geometric/epoch diagnosis of unmatched zero-ID evidence."""
from baseline_v1_4_native import *
from photdis.stage1 import galactic


def quantile(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return np.quantile(x,[0,.1,.5,.9,.99,1]).tolist() if len(x) else []


def main():
    require_compute()
    with np.load(OUT/'zero_match.npz') as z:match={k:z[k] for k in z.files}
    rows=match['source_row']
    keys=['target_target_ra','target_target_dec','target_ref_epoch','target_pmra','target_pmdec','target_ebv','rvs_target_ra','rvs_target_dec','gaia_source_id','rvs_survey','rvs_program']
    with np.load(OUT/'sources.npz') as z:s={k:z[k][rows] for k in keys}
    with fitsio.FITS(str(SOUTH/'survey-bricks-dr11-south.fits.gz')) as f:
        names=f[1].get_colnames();print('brick index columns',names,flush=True)
        bricks=uppercase_fields(f[1].read(columns=['RA','DEC','BRICKNAME']))
    q,known=epoch_positions(*[s[k] for k in ['target_target_ra','target_target_dec','target_ref_epoch','target_pmra','target_pmdec']])
    dist,ix=cKDTree(unit_vectors(bricks['RA'],bricks['DEC'])).query(q)
    separation=np.rad2deg(2*np.arcsin(np.clip(dist/2,0,1)))
    with np.load(OUT/'zero_photometry.npz') as z:c={k:z[k] for k in ['RA','DEC','REF_EPOCH','PMRA','PMDEC','REF_ID','REF_CAT','ls_id']}
    cv,ck=epoch_positions(*[c[k] for k in ['RA','DEC','REF_EPOCH','PMRA','PMDEC']])
    near,j=cKDTree(cv).query(q);sep=np.rad2deg(2*np.arcsin(np.clip(near/2,0,1)))*3600
    lat=galactic(s['target_target_ra'],s['target_target_dec'])[1]
    rawsep=np.rad2deg(2*np.arcsin(np.clip(np.linalg.norm(unit_vectors(s['rvs_target_ra'],s['rvs_target_dec'])-unit_vectors(s['target_target_ra'],s['target_target_dec']),axis=1)/2,0,1)))*3600
    report=dict(n=len(rows),native_candidates=len(cv),candidate_epoch_known=int(ck.sum()),
        candidate_ref_cat={str(k):int(v) for k,v in zip(*np.unique(c['REF_CAT'].astype('U'),return_counts=True))},
        groups={})
    groups=dict(all=np.ones(len(rows),bool),matched=match['candidate_row']>=0,unmatched=match['candidate_row']<0,
                low_dust_high_lat=(abs(lat)>30)&(s['target_ebv']>=0)&(s['target_ebv']<.03))
    for tag,m in groups.items():
        report['groups'][tag]=dict(n=int(m.sum()),nearest_south_brick_deg=quantile(separation[m]),
            within_point2deg=int((m&(separation<.2)).sum()),within_point4deg=int((m&(separation<.4)).sum()),
            candidate_within1arcsec=int((m&(sep<1)).sum()),candidate_within2arcsec=int((m&(sep<2)).sum()),
            dec=quantile(s['target_target_dec'][m]),pm_masyr=quantile(np.hypot(s['target_pmra'][m],s['target_pmdec'][m])),
            epoch=quantile(s['target_ref_epoch'][m]),rvs_fib_position_difference_arcsec=quantile(rawsep[m]),
            galactic_b=quantile(lat[m]),ebv=quantile(s['target_ebv'][m]),
            raw_gaia_id_values={str(k):int(v) for k,v in zip(*np.unique(s['gaia_source_id'][m],return_counts=True))},
            survey={str(k):int(v) for k,v in zip(*np.unique(s['rvs_survey'][m].astype('U'),return_counts=True))})
    # Inspect schema of a few actual target bricks for epoch/astrometry metadata.
    name=str(bricks['BRICKNAME'][ix[0]]).strip()
    if name.startswith("b'"):name=name[2:-1]
    path=SOUTH/'tractor'/name[:3]/('tractor-'+name+'.fits')
    with fitsio.FITS(str(path)) as f:report['native_epoch_columns']=[k for k in f[1].get_colnames() if any(x in k.lower() for x in ['mjd','epoch','primary'])]
    write_json(OUT/'zero_audit.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
