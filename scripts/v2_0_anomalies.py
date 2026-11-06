#!/usr/bin/env python3
"""Audit the predeclared 49-object Gaia region without editing predictions."""
import json
from pathlib import Path
import numpy as np
import fitsio
import joblib
import matplotlib.pyplot as plt

import v2_0_figures as f
from v2_0_revision import REV, configure, color_label, color
from v2_0_common import ROOT, OUT, DEST, load, subset, sha, save_json, csv_rows
from photdis.stage1 import require_compute, legacy_ids
from photdis.reference_v1_4 import unique_join
from photdis.gaia import join_gaia, _decode_boolean
from photdis.spectroscopy import QUALITY_REASON
from photdis.baseline_v1 import integrate
from baseline_v1_4_experiments import arrays
from baseline_v1_4_extract import atomic_npz
from photdis_figure_style import BLUE, ORANGE, GRAY


def selected_columns(path, rows, keys):
    result = {}
    good = rows >= 0
    with np.load(path) as z:
        for key in keys:
            if key not in z.files:
                continue
            values = z[key]
            assert not good.any() or rows[good].max() < len(values)
            shape = (len(rows),) + values.shape[1:]
            fill = '' if values.dtype.kind in 'SU' else np.nan if values.dtype.kind == 'f' else False if values.dtype.kind == 'b' else -1
            result[key] = np.full(shape, fill, dtype=values.dtype)
            # Missing -1 is never allowed to address the final catalog row.
            result[key][good] = values[rows[good]]
    return result


def native_rows(t):
    """Use frozen checkpoints to identify the original sweeps; read only target rows."""
    n = len(t['identity'])
    payload = dict(raw_sweep=np.full(n, '', 'U220'), raw_row=np.full(n, -1, 'i8'), checkpoint_row=np.full(n, -1, 'i8'))
    expected = json.loads((OUT / 'native_photometry.json').read_text())['sweep_hashes']
    seen = np.zeros(n, bool)
    for name, digest in sorted(expected.items()):
        path = OUT / 'photometry' / (name + '.npz')
        with np.load(path) as z:
            if 'RELEASE' not in z.files:
                continue
            lsid = legacy_ids({k:z[k] for k in ('RELEASE', 'BRICKID', 'OBJID')})
            ix = unique_join(t['ls_id'], lsid)
            use = ix >= 0
            if not use.any():
                continue
            assert sha(path) == digest
            assert not seen[use].any()
            marker = json.loads(path.with_suffix('.json').read_text())
            original = Path(marker['sweep'])
            with fitsio.FITS(str(original)) as h:
                keys = h[1].read(columns=['RELEASE', 'BRICKID', 'OBJID'])
                rows = unique_join(t['ls_id'][use], legacy_ids(keys))
                assert (rows >= 0).all()
                fields = [k for k in h[1].get_colnames() if k in z.files or k.startswith('RCHISQ_')]
                raw = h[1].read(columns=fields, rows=rows)
                assert np.array_equal(legacy_ids(raw), t['ls_id'][use])
                for key in raw.dtype.names:
                    if key not in payload:
                        a = raw[key]
                        fill = '' if a.dtype.kind in 'SU' else np.nan if a.dtype.kind == 'f' else -1
                        payload[key] = np.full((n,) + a.shape[1:], fill, dtype=a.dtype)
                    payload[key][use] = raw[key]
            payload['raw_sweep'][use] = str(original)
            payload['raw_row'][use] = rows
            payload['checkpoint_row'][use] = ix[use]
            seen[use] = True
            print('anomaly native', int(seen.sum()), n, flush=True)
            if seen.all():
                break
    assert seen.all(), 'Anomaly original-row lookup incomplete'
    return payload


def spectral_failures(t, i):
    reasons = []
    if t['source_row'][i] < 0:
        return ['no_MWS_row']
    reason = int(t['rvs_quality_reason'][i])
    for name, bit in QUALITY_REASON.items():
        if bit != 4 and reason & (1 << bit):
            reasons.append(name)
    # FeH validity is independent of the v3 class gate.
    for key in ('teff', 'teff_err', 'logg', 'logg_err'):
        value = t['rvs_' + key][i]
        if not np.isfinite(value) or (key != 'logg' and value <= 0):
            reasons.append('invalid_' + key)
    return reasons or ['no_quality_reason_recovered']


def highlight(ax, x, y, anomaly, xlabel, ylabel, title):
    valid = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[valid & ~anomaly], y[valid & ~anomaly], s=3, color=BLUE, alpha=.18, rasterized=True, label='Other Gaia references')
    ax.scatter(x[valid & anomaly], y[valid & anomaly], s=24, facecolors='none', edgecolors=ORANGE, linewidths=.8, label=f'Fixed anomaly region ({int((valid & anomaly).sum())})')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    f.panel(title, len(x), int(valid.sum()))
    for i in np.flatnonzero(valid):
        f.ROWS.append(dict(panel=title, row=int(i), x=x[i], y=y[i], anomaly=bool(anomaly[i])))


def audit_figures(t, anomaly, selected):
    dref = 10 ** ((t['ref_Gaia'] - 10) / 5)
    f.begin()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    highlight(axes[0], dref, t['distance_kpc'][:, 2], anomaly, 'Gaia distance [kpc]', 'Predicted distance [kpc]', 'All 3,207 Gaia point references')
    axes[0].set(xscale='log', yscale='log')
    edges = np.geomspace(min(dref.min(),t['distance_kpc'][:,2].min()),max(dref.max(),t['distance_kpc'][:,2].max()),2)
    axes[0].plot(edges, edges, '--', color=GRAY)
    highlight(axes[1], t['gaia_bp_rp'], t['gaia_phot_g_mean_mag'] - t['ref_Gaia'], anomaly, 'Gaia BP-RP [mag], observed', 'Gaia $M_G$ [mag], uncorrected extinction', 'Gaia observational CMD')
    axes[1].invert_yaxis()
    for ax in axes:ax.legend(fontsize=9)
    f.save(fig, 'anomaly_gaia', 'Exactly predeclared d_Gaia<0.2 kpc and d_phot,50>2 kpc. Gaia M_G uses corrected-parallax distance modulus but observed G, without extinction correction. No anomaly removed.', section='anomalies', reference='Gaia point anchors')
    for pairs, name in [([('g','r'),('r','z'),('z','w1'),('w1','w2')], 'anomaly_colors')]:
        f.begin()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
        for ax, a, b in [(axes[0,0],('g','r'),('r','z')), (axes[0,1],('g','r'),('z','w1')), (axes[1,0],('z','w1'),('w1','w2'))]:
            highlight(ax, color(t,a), color(t,b), anomaly, color_label(a), color_label(b), 'Optical model in color coordinates')
        highlight(axes[1,1], t['g0']-t['r0'], t['g0']-t['ref_Gaia'], anomaly, '$(g-r)_0$', 'Legacy $M_{g,Gaia}$ [mag]', 'Extinction-corrected Legacy CMD')
        axes[1,1].invert_yaxis()
        f.save(fig, name, 'Legacy magnitudes extinction corrected. WISE colors require their own state4. Model remains expanded_A grz. Highlight is a residual-selected diagnostic set, not a WD classification.', section='anomalies', reference='Gaia point anchors')
    f.begin()
    fig, axes = plt.subplots(2, 2, figsize=(12,10), layout='constrained')
    reasons = {}
    for i in np.flatnonzero(anomaly):
        for reason in spectral_failures(t,i):reasons[reason]=reasons.get(reason,0)+1
    labels=list(reasons)
    axes[0,0].barh(labels,[reasons[k] for k in labels],color=ORANGE)
    axes[0,0].tick_params(axis='y',labelsize=9)
    axes[0,0].set(title='Independent class-quality failures',xlabel='Objects (reasons may overlap)')
    for label,n in reasons.items():f.ROWS.append(dict(panel='quality reasons', reason=label,n=n))
    for ax,key,label in [(axes[0,1],'gaia_ruwe','Gaia RUWE'),(axes[1,0],'gaia_ipd_frac_multi_peak','Gaia IPD multi-peak [%]')]:
        values=t[key][anomaly];valid=np.isfinite(values)
        ax.hist(values[valid],bins=15,color=BLUE)
        ax.set(xlabel=label,ylabel='Objects')
        f.panel(label,int(anomaly.sum()),int(valid.sum()))
        for value in values[valid]:f.ROWS.append(dict(panel=label,value=value))
    highlight(axes[1,1],t['RA'],t['DEC'],anomaly,'RA [deg]','Dec [deg]','Gaia-reference sky')
    f.save(fig,'anomaly_quality','Class reasons follow frozen v3 quality, including Teff/logg validity independently of FeH. Finite but unqualified spectral parameters are not physical labels. Sky distribution does not imply population.',section='anomalies',reference='Gaia + RVS quality')
    # Fixed IDs are ordered independently of PDF shape; reconstruct only four cases.
    model_path=OUT/'experiments/expanded_A_distance.joblib'
    model_meta=json.loads(model_path.with_suffix('.json').read_text())
    assert sha(model_path)==model_meta['sha256']
    model=joblib.load(model_path)
    examples=np.flatnonzero(anomaly)[np.argsort(t['identity'][anomaly],kind='stable')[:4]]
    f.begin()
    fig,axes=plt.subplots(2,2,figsize=(12,9.5),layout='constrained')
    pdf_rows=[]
    for ax,i in zip(axes.ravel(),examples):
        m,e,ids=arrays(t,np.array([i]))
        prediction=integrate(model,m,e,ids,draws=128)['wide']
        assert np.allclose(prediction['mu'][0],t['mu'][i],atol=1e-6,rtol=0)
        edges=t['g0'][i]-model.edges[::-1]
        mass=prediction['pdf'][0,::-1]
        density=mass/np.diff(edges)
        near=np.abs((edges[:-1]+edges[1:])/2-t['ref_Gaia'][i])<=.5
        ax.stairs(density,edges,color=BLUE)
        ax.axvline(t['ref_Gaia'][i],color=ORANGE,label='Gaia reference')
        ax.axvline(t['mu'][i,2],color=GRAY,ls='--',label='Frozen median')
        ax.set(title=f'ID {t["identity"][i]}\nMass within ±0.5 mag reference = {mass[near].sum():.3g}',xlabel='Distance modulus [mag]',ylabel='Density [1/mag]')
        ax.legend(fontsize=9)
        pdf_rows.append(dict(identity=str(t['identity'][i]),mass_near_reference=float(mass[near].sum()),median_reproduced=True))
        for lo,hi,p in zip(edges[:-1],edges[1:],density):f.ROWS.append(dict(panel=str(t['identity'][i]),lo=lo,hi=hi,density=p))
    f.save(fig,'anomaly_pdf_examples','Four lowest Gaia IDs from the fixed 49-object list. Frozen expanded_A and shared Sobol128; all original quantiles reproduced within1e-6. Reference-window mass is descriptive, not class probability.',section='anomalies',reference='Gaia point anchors')
    csv_rows(f.TABLES/'anomaly_pdf_summary.csv',pdf_rows)


def main():
    require_compute()
    configure()
    original=load(DEST/'chosen_expanded_A.npz')
    q=subset(original,(original['split']==2)&np.isfinite(original['ref_Gaia']))
    assert len(q['identity'])==3207
    d=10**((q['ref_Gaia']-10)/5)
    anomaly=(d<.2)&(q['distance_kpc'][:,2]>2)
    assert anomaly.sum()==49
    assert (q['class_label'][anomaly]==-1).all()
    assert (q['TYPE'][anomaly]=='PSF').all()
    reference_path=OUT/'references.npz'
    assert sha(reference_path)==json.loads((OUT/'assembly.json').read_text())['reference_sha256']
    with np.load(reference_path) as z:
        rows=unique_join(q['identity'],z['identity'])
        assert (rows>=0).all()
        keys=[k for k in z.files if k.startswith(('rvs_','gaia_','target_')) or k in ('source_row','targetid','source_id','mws_source_id','source_identity','identity_conflict','zero_id_match','match_epoch_known','match_separation_arcsec','zpt_mas','specdis_row','spdist_row','specdis_source_only_join','spdist_source_only_join')]
    q.update(selected_columns(reference_path,rows,keys))
    source_path=OUT/'sources.npz'
    assert sha(source_path)==json.loads((OUT/'sources.json').read_text())['source_sha256']
    with np.load(source_path) as z:keys=[k for k in z.files if k.startswith('gaia_')]
    q.update(selected_columns(source_path,q['source_row'],keys))
    selected=subset(q,anomaly)
    # Supplement BP/RP and all available Gaia quality fields only for the fixed49.
    ga=join_gaia(selected['identity'],'/global/cfs/cdirs/cosmo/data/gaia/dr3/healpix')
    assert ga['gaia_match'].all()
    selected.update({'dr3_'+k:v for k,v in ga.items()})
    selected.update(native_rows(selected))
    native_ids=selected['REF_ID'].astype('i8')
    native_catalog=np.char.strip(selected['REF_CAT'].astype('U'))
    assert np.array_equal(native_ids,selected['identity']) and (native_catalog=='G3').all()
    selected['match_method']=np.where(selected['zero_id_match'],'epoch_aware_zero_MWS_ID_recovery','exact_Gaia_DR3_REF_ID')
    selected['source_catalog_path']=np.full(49,str(source_path),'U180')
    selected['M_g_ref']=selected['g0']-selected['ref_Gaia']
    selected['M_g_phot']=selected['g0']-selected['mu'][:,2]
    selected['M_G_observed']=selected['dr3_phot_g_mean_mag']-selected['ref_Gaia']
    # Pair separation is checked after propagating Gaia ICRS positions to the
    # actual native reference epoch, with Astropy's spherical space motion.
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u
    pos=SkyCoord(ra=ga['ra']*u.deg,dec=ga['dec']*u.deg,pm_ra_cosdec=ga['pmra']*u.mas/u.yr,pm_dec=ga['pmdec']*u.mas/u.yr,obstime=Time(ga['ref_epoch'],format='jyear'))
    epoch=selected['REF_EPOCH'].astype(float)
    valid_epoch=np.isfinite(epoch)&(epoch>1900)&(epoch<2100)
    separation=np.full(49,np.nan)
    if valid_epoch.any():
        moved=pos[valid_epoch].apply_space_motion(new_obstime=Time(epoch[valid_epoch],format='jyear'))
        native=SkyCoord(ra=selected['RA'][valid_epoch]*u.deg,dec=selected['DEC'][valid_epoch]*u.deg)
        separation[valid_epoch]=moved.separation(native).arcsec
    selected['gaia_native_epoch_separation_arcsec']=separation
    records=[]
    for i in range(49):
        row={}
        for key,values in selected.items():
            value=values[i]
            if np.ndim(value):
                for j,item in enumerate(value):row[key+'_'+str(j)]=item.item() if hasattr(item,'item') else item
            elif isinstance(value,(bytes,np.bytes_)):
                row[key]=value.decode().strip()
            else:row[key]=value.item() if hasattr(value,'item') else value
        for key in list(row):
            if key.lower() in ('identity','ls_id','source_id','targetid','mws_source_id','source_identity','gaia_source_id','ref_id','rvs_targetid','target_targetid','dr3_source_id'):
                row[key]=str(row[key])
        row['unknown_reasons']='|'.join(spectral_failures(selected,i))
        # WD remains an explicit hypothesis. Region membership is not used as
        # the classifier: require low luminosity in two independent CMD axes.
        compact_locus=np.isfinite(selected['dr3_bp_rp'][i]) and selected['dr3_bp_rp'][i]<1.5 and selected['M_G_observed'][i]>10 and selected['M_g_ref'][i]>10
        match_risk=np.isfinite(separation[i]) and separation[i]>1
        astro_risk=ga['ruwe'][i]>=1.4 or ga['duplicated_source'][i] or ga['ipd_frac_multi_peak'][i]>10
        row['evidence_category']='photometry_or_match_concern' if match_risk or astro_risk else 'low_luminosity_WD_candidate_support' if compact_locus else 'undetermined'
        row['evidence_basis']='epoch separation>1arcsec or Gaia quality anomaly' if match_risk or astro_risk else 'Gaia BP-RP<1.5, observed M_G>10 and Legacy M_g,ref>10; heuristic, no WD catalog confirmation' if compact_locus else 'No independent class confirmation'
        row['wd_confirmed']=False
        records.append(row)
    csv_rows(f.TABLES/'gaia_anomalies.csv',records)
    save_json(f.TABLES/'gaia_anomalies.json',records)
    atomic_npz(REV/'gaia_reference_audit.npz',**q)
    atomic_npz(REV/'gaia_anomaly_audit.npz',**selected)
    audit_figures(q,anomaly,selected)
    categories={k:sum(row['evidence_category']==k for row in records) for k in sorted({row['evidence_category'] for row in records})}
    save_json(f.TABLES/'anomaly_summary.json',dict(chosen=47756,gaia=3207,giants=1163,anomalies=49,all_unknown=True,all_PSF=True,categories=categories,source_sha256=sha(DEST/'chosen_expanded_A.npz'),model_sha256=sha(OUT/'experiments/expanded_A_distance.joblib'),limitations='Heuristic CMD support, no WD catalog identification. Gaia and Legacy M share Gaia distance. Missing catalog fields remain missing. No point excluded from primary metrics.'))
    print('COMPLETE anomaly audit',categories,flush=True)


if __name__=='__main__':main()
