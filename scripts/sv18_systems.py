#!/usr/bin/env python3
"""Known-scale member tests, faint phases and actual north/south final-model differences."""
from astropy.io import fits
from photdis.stage1 import require_compute
from sv18_common import *
from v3_13_clusters import SYSTEMS
from v3_18_members import refs
from v3_17_sample import INPUT, TRACK_SHA
from v3_17_hess import MODULUS_CURVE
from v3_11_layers import read_receipt


def sample_stat(q,ref):
    stat=metric(q,ref)
    if len(ref)>1:
        draws=[]
        for _ in range(P['bootstrap_draws']):
            j=RNG.integers(len(ref),size=len(ref))
            draws.append(np.median(q[j,2]/ref[j]-1))
        stat.update(interval(draws))
    else:
        stat.update(interval([]))
    stat['can_resolve_ten_percent']=bool(len(ref)>=P['noise']['minimum_effective_n'] and
        stat['ci_lo'] is not None and stat['ci_hi']-stat['ci_lo']<.20)
    return stat


def main():
    require_compute()
    inputs={};d=out('systems');rows=[];points=[]
    for name in ('M30','M5','M68','M92','M13'):
        if name=='M68':
            # The frozen M68 archive stores binned distance responses. Its
            # template-transfer test is retained separately from single-star
            # full-PDF member tests in the other four fields.
            from v3_14_structure import DEST as M68_FOLDER
            record=json.loads((M68_FOLDER/'complete.json').read_text())
            inputs['M68_structure']=sha(M68_FOLDER/'complete.json')
            with verified(M68_FOLDER/'comparison.csv',record['outputs']['comparison.csv']).open() as f:
                csv_rows(d/'M68_frozen_transfer.csv',list(csv.DictReader(f)))
            continue
        home=ROOT/f'data/v3_14/audit/{name}'
        marker=home/'complete.json'
        r=json.loads(marker.read_text())
        path=verified(home/'native_join.npz',r['outputs']['native_join.npz'])
        with np.load(path) as a:
            t={k:a[k] for k in a.files}
        assert (t['TYPE']=='PSF').all()
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        distance=7.480 if name=='M5' else 10.404 if name=='M68' else SYSTEMS[name][1]
        q=t['qrf_kpc'];ref=np.full(len(q),distance)
        member=t['target']&t['member']&~t['training_or_calibration']
        supported=np.isfinite(q).all(axis=1)&(q>0).all(axis=1)
        for quality,clean in (('all_members',member),('low_crowding',member&t['low_crowding']),
           ('unmasked_low_crowding',member&t['low_crowding']&~t['cluster_mask'])):
            for a,b in ((14,18),(18,20),(20,21),(21,22),(22,23)):
                use=clean&supported&(t['r']>=a)&(t['r']<b)
                rows.append(dict(system=name,group=quality,variable='observed_r',lo=a,hi=b,
                    member_parent=int((clean&(t['r']>=a)&(t['r']<b)).sum()),
                    reference_kpc=distance,**sample_stat(q[use],ref[use])))
            for phase in np.unique(t['phase_box']):
                use=clean&supported&(t['phase_box']==phase)
                rows.append(dict(system=name,group=quality,variable='phase',lo=phase,hi='',
                    member_parent=int((clean&(t['phase_box']==phase)).sum()),reference_kpc=distance,
                    **sample_stat(q[use],ref[use])))
        use=member&supported&t['low_crowding']
        for i in np.flatnonzero(use):
            points.append(dict(system=name,r=float(t['r'][i]),color=float(t['g0'][i]-t['r0'][i]),
                   phase=t['phase_box'][i],ref_mu=10+5*np.log10(distance),mu=10+5*np.log10(q[i,2]),
                   delta_mu=5*np.log10(q[i,2]/distance),low_crowding=True,masked=bool(t['cluster_mask'][i])))
    home=ROOT/'data/v3_18/members';record=json.loads((home/'complete.json').read_text())
    with (verified(home/'roster.csv',record['outputs']['roster.csv'])).open() as f:
        roster=list(csv.DictReader(f))
    a=read(home,('posteriors.npz','TYPE','ls_id','science','quantiles_kpc'),inputs)
    t={int(r['ls_id']):r for r in roster}
    for group in ('thin','cocoon'):
        for lo,hi in ((14,18),(18,20),(20,21),(21,22),(22,23)):
            ix=[i for i in range(len(a['ls_id'])) if a['science'][i] and t[int(a['ls_id'][i])]['group']==group
                 and lo<=float(t[int(a['ls_id'][i])]['r'])<hi]
            rr=[t[int(a['ls_id'][i])] for i in ix]
            dist=np.array([float(r['desi_kpc']) for r in rr])
            rows.append(dict(system='GD-1',group=group,variable='observed_r',lo=lo,hi=hi,
                              member_parent=len(ix),reference_kpc=None,
                              **sample_stat(a['quantiles_kpc'][ix],dist)))
    tracks=[]
    for table in ('Table1','Table3','Figure2_track_jarvis'):
        p3=INPUT/f'v3_{table}.fits';p4=INPUT/f'v4_{table}.fits'
        same=p3.read_bytes()==p4.read_bytes()
        tracks.append(dict(table=table,v3_sha256=sha(p3),v4_sha256=sha(p4),byte_identical=same,
            used_version='v3 corrected spatial track' if table=='Table3' else 'v4 archived distance reference',
            documented_record='https://zenodo.org/records/19889980'))
        assert same
    assert sha(INPUT/'v3_Table3.fits')==TRACK_SHA
    assert sha(INPUT/'v4_Figure2_track_jarvis.fits')==MODULUS_CURVE
    csv_rows(d/'track_version_audit.csv',tracks)
    csv_rows(d/'known_scale_members.csv',rows)
    csv_rows(d/'cluster_sequence_residuals.csv',points)
    # The v3.10 seam archives already contain both final native predictions,
    # including the same actual source pairs. Replay them without new inference.
    seam=ROOT/'data/v3_10/seam'
    pair={}
    for side in ('north','south'):
        mark=seam/(side+'.json');r=json.loads(mark.read_text())
        path=verified(seam/(side+'.npz'),r['outputs'][side+'.npz'])
        pair[side]=read_columns(path,('TYPE','ls_id','r','gr','state','qrf_quantiles_kpc','distance_shell_mass'))
        assert (pair[side]['TYPE']=='PSF').all()
        inputs[str(mark.relative_to(ROOT))]=sha(mark)
    ns,ss=pair['north'],pair['south']
    good=(ns['state']==3)&(ss['state']==3)
    delta=5*np.log10(ns['qrf_quantiles_kpc'][:,2]/ss['qrf_quantiles_kpc'][:,2])
    seamrows=[]
    for lo,hi in ((14,18),(18,20),(20,21),(21,22),(22,23)):
        for a,b in ((-.2,.3),(.3,.6),(.6,1.0),(1.,1.6)):
            k=good&(ss['r']>=lo)&(ss['r']<hi)&(ss['gr']>=a)&(ss['gr']<b)
            v=delta[k]
            seamrows.append(dict(r_lo=lo,r_hi=hi,gr_lo=a,gr_hi=b,n=int(k.sum()),
                  median_delta_mu=np.median(v) if len(v) else None,
                  nmad_delta_mu=1.4826*np.median(abs(v-np.median(v))) if len(v) else None,
                  mean_shell_L1=np.mean(abs(ns['distance_shell_mass'][k]-ss['distance_shell_mass'][k]).sum(axis=1)) if len(v) else None))
    csv_rows(d/'final_model_seam.csv',seamrows)
    with plt.rc_context(PAPER):
        f,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained')
        for i,name in enumerate(('M30','M5','M68','M92','M13')):
            z=[p for p in points if p['system']==name]
            axes[0].scatter([r['color'] for r in z],[r['delta_mu'] for r in z],s=5,alpha=.4,
                            label=name,edgecolors='none',rasterized=True)
        axes[0].axhline(0,color='.6',lw=.7,ls=':')
        axes[0].set(xlabel=r'$(g-r)_0$',ylabel=r'$\Delta\mu$ (mag)',xlim=(-.1,1.5),ylim=(-4,4))
        axes[0].legend(fontsize=5.5,frameon=False,ncols=2)
        for name,col in (('thin',BLUE),('cocoon',ORANGE)):
            z=[r for r in rows if r['system']=='GD-1' and r['group']==name and r['n']]
            axes[1].errorbar([(r['lo']+r['hi'])/2 for r in z],[r['bias'] for r in z],
              yerr=np.maximum(0,[[r['bias']-(r['ci_lo'] if r['ci_lo'] is not None else r['bias']) for r in z],
                       [(r['ci_hi'] if r['ci_hi'] is not None else r['bias'])-r['bias'] for r in z]]),
              fmt='o-',ms=3,color=col,label=name.capitalize())
        axes[1].axhline(0,color='.6',lw=.7,ls=':')
        axes[1].set(xlabel='Observed r (mag)',ylabel='Median relative distance residual')
        axes[1].legend(fontsize=7,frameon=False)
        figure(d,'known_system_sequences',f,rows)
    complete(d,inputs,member_intervals='stellar bootstrap conditional on fixed external shared system scale',
             seam_scope='native final production PDFs, unconditioned on independent reference, training overlap remains explicit')


if __name__=='__main__':
    main()
