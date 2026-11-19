#!/usr/bin/env python3
"""Publish compact evidence and exact replay arrays; retain member PDFs privately."""
from nat8_common import *
from seq7_publish import render_runtime
from photdis.stage1 import require_compute
import shutil


def main():
    require_compute();dest=ROOT/'manifests/native_luminosity_20261119';fig=ROOT/'figures/native_luminosity_20261119';dest.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True);outputs={}
    files={'prepare':['identities.csv','fit_design.json','actual_gray_invariance.csv','complete.json'],
           'geometry':['per_member.csv','competitive_segments.csv','color_bins.csv','complete.json'],
           'fit':['parameters.json','parameter_draws.npz','fit_draws.csv','shared_reference_identity.csv','all_solutions.json','complete.json'],
           'predict':['support_domain.csv','complete.json'],
           'evaluate':['summary.csv','paired.csv','phase_gaps.csv','group_design.json','complete.json'],
           'audit':['candidate_history.csv','candidate_inventory.csv','membership_columns.json','three_membership.json','three_raw_imaging.json','complete.json'],
           'audit_detail':['three_source_evidence.csv','reference_ancestry.json','candidate_inventory_final.csv','complete.json'],
           'decision':['phase_gap_changes.csv','width_ratios.csv','continuation_criteria.csv','science_decision.json','luminosity_scale.csv','complete.json'],
           'examples':['selected.csv','complete.json'],
           'validation':['direct_halfgrid.csv','invariants.csv','style_regression.json','old_preservation.json','complete.json']}
    for stage,names in files.items():
        marker=json.loads((OUT/stage/'complete.json').read_text());target=dest/stage;target.mkdir(exist_ok=True)
        for name in names:
            p=OUT/stage/name
            if name!='complete.json':verified(p,marker['outputs'][name])
            shutil.copy2(p,target/name);outputs[str((target/name).relative_to(ROOT))]=sha(p)
    for name in C['independent']['candidates']:
        target=dest/'candidate_inventory'/name;target.mkdir(parents=True,exist_ok=True)
        for file in ['admission.csv','identities.csv','inventory.json']:
            p=OUT/'candidate_inventory'/name/file;shutil.copy2(p,target/file);outputs[str((target/file).relative_to(ROOT))]=sha(p)
    for directory,target in [(OUT/'public_data',dest/'data'),(OUT/'figures',fig)]:
        target.mkdir(exist_ok=True)
        for p in directory.iterdir():
            if p.is_file():shutil.copy2(p,target/p.name);outputs[str((target/p.name).relative_to(ROOT))]=sha(p)
    # Preserve every method state at export, including unsupported parents.
    t=load(OUT/'prepare/roster.npz');rows=[];private={}
    for s in C['systems']:
        e=load(OUT/'prepare'/f'{s}_E.npz');p=load(OUT/'predict'/f'{s}_calibrated.npz');q=load(OLD/'curves'/f'{s}_QRF.npz')
        for k,i in enumerate(e['ii']):
            assert t['TYPE'][i]=='PSF'
            for m,qq in [('E',e['q'][k]),('CalE',p['q'][k]),('QRF',q['q'][int(np.flatnonzero(q['ii']==i)[0])] if i in q['ii'] else np.full(5,np.nan))]+([('CalE_marg',p['marginal_q'][k])] if s!='M5' else []):
                supported=bool(e['support'][k]) if m!='QRF' else bool(i in q['ii'] and q['support'][int(np.flatnonzero(q['ii']==i)[0])]);rows.append(dict(TYPE='PSF',ls_id=int(t['ls_id'][i]),source_id=int(t['source_id'][i]),system=s,native=int(t['native'][i]),phase=t['phase'][i],method=m,supported=supported,posterior_available=bool(np.isfinite(qq).all()),mu025=qq[0],mu16=qq[1],mu50=qq[2],mu84=qq[3],mu975=qq[4],post_mass=float(e['post'][k]) if m!='QRF' else np.nan,outside_color_mass=float(p['pout'][k]),fold=int(t['fold'][i])))
        for stage,name in [('prepare',f'{s}_E.npz'),('predict',f'{s}_calibrated.npz')]:private[str(OUT/stage/name)]=sha(OUT/stage/name)
    path=dest/'predict/posterior_quantiles.csv';csv_rows(path,rows);outputs[str(path.relative_to(ROOT))]=sha(path)
    metrics=json.loads((OUT/'evaluate/family_metrics.json').read_text());path=dest/'evaluate/bootstrap_metrics.npz';npz(path,**{str(k)+'_'+metric:np.array(value) for k,record in metrics.items() for metric,value in record['bootstrap'].items()});outputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(dest/'forecast_factorization.json',dict(private=private,roster=dict(path=str(OUT/'prepare/roster.npz'),sha=sha(OUT/'prepare/roster.npz')),conditional='Exact original E mean/logweight/sigma/x arrays, calibrated means=means-a0-a1*x; M5 opposite-fold parameters, external full-M5. No nodes/weights removed for log scoring.',marginal='Uniform empirical ensemble of256 full-M5 bootstrap parameter vectors, same index shared across all external members; no target reference kernel in saved predictive PDF.',scoring='M5 coupled conditional shared reference, external once-independent target Gaussian kernel; original v7 definitions unchanged.',private_score_cache=str(OUT/'score_cache'),private_external_raw_curves=str(OUT/'marginal_raw')));outputs[str((dest/'forecast_factorization.json').relative_to(ROOT))]=sha(dest/'forecast_factorization.json')
    sky=ROOT/'figures/sky_sequence_20261119/selected_channels/rgb_selected_fixed_gain.png';assert sha(sky)=='411b0086e9d4c6f9f208805f02f064e2c24bb117d61293083b9bb51dc7faaed5'
    save_json(dest/'release.json',dict(config=CH,job=os.getenv('SLURM_JOB_ID'),production=C['production'],private_root=str(OUT),render_runtime=render_runtime(),style=check_templates(),sky_unchanged=sha(sky),outputs=outputs,source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'scripts').glob('nat8_*.py')},scope='One two-parameter hypothesis, two-fold M5 cross predictions, fixed M30/M13 transfers, two inventories only; no prospective prediction after failed registered gate, no production/manuscript edits.'))
    print('Published',len(outputs),'bound files',flush=True)


if __name__=='__main__':main()
