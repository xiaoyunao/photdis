#!/usr/bin/env python3
"""Describe the fixed reliable-giant pool, with no distance-model evaluation."""
from pathlib import Path
import json,hashlib
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];FIG=ROOT/'figures/giant_parameter_audit_v1';FIG.mkdir(exist_ok=True)
def read(n):return json.load(open(ROOT/'manifests'/n))
def chosen(r,camera='all'):
    releases={'DECam':(11010,), 'BASS/MzLS':(11011,), 'all':(11010,11011)}[camera]
    for release in releases:
        mm=[m for m in r['measurements'] if m['release']==release]
        if len(mm)==1:return mm[0]
        if len(mm)>1:return None
    return None

def main():
    inv=read('giant_parameter_inventory_v1.json');rows=inv['rows'];target=[r for r in read('kgiant_selected190_evidence_v1.json')['rows'] if r['specdis_valid']]
    definitions=read('giant_parameter_coverage_definition_v1.json');bounds=definitions['bounds']
    groups={};checks={}
    old=read('kgiant_reference_inventory_v1.json');old={r['source_id']:r for r in old['rows'] if not r['historical_diagnostic']}
    assert len(rows)==len({r['source_id'] for r in rows}) and sum(r['n_qualified_spectra'] for r in rows)==120
    checks['120_spectra_complete_and_deduplicated']=True
    assert {r['source_id'] for r in rows if r['in_original_K_window']}==set(old)
    for r in rows:
        for spec in r['qualified_spectra']:
            v=spec['values'];assert v['PRIMARY'] and v['SUCCESS'] and v['RVS_WARN']==0 and v['RR_SPECTYPE'].strip()=='STAR'
            assert v['SN_R']>=10 and 0<=v['VSINI']<30 and 0<v['TEFF']<7000 and v['TEFF_ERR']>0 and v['LOGG_ERR']>0 and v['LOGG']+v['LOGG_ERR']<3.2
        if r['source_id'] in old:
            baseline=old[r['source_id']]['measurements'];assert len(r['measurements'])==len(baseline)
            for new,prior in zip(r['measurements'],baseline):assert all(new[k]==v for k,v in prior.items())
    checks['same_spectral_eligibility_and_original8_photometry']=True
    previous=read('kgiant_reference_typed_candidates_v1.json')['systems']
    for name,sys in inv['systems'].items():
        for key in ['probable_members','mws_member_spectra','reliable_giant_spectra','K_spectra']:assert sys[key]==previous[name][key]
    checks['per_system_parent_counts_unchanged']=True
    assert inv['frozen_prediction_sha256']==hashlib.sha256((ROOT/'manifests/kgiant_reference_predictions_v1.json').read_bytes()).hexdigest()
    checks['all22_predictions_byte_unchanged']=True
    for camera in ('DECam','BASS/MzLS','all'):
        for temp in ('cold','K','warm','all'):
            rr=[r for r in rows if (temp=='all' or (r['teff']<4200 if temp=='cold' else r['in_original_K_window'] if temp=='K' else r['teff']>=5000))]
            measured=[r for r in rr if chosen(r,camera) is not None and (camera=='all' or chosen(r,camera)['natural_system']==camera)]
            core=[r for r in measured if r['pmember']>.9 and chosen(r,camera)['grz_quality']]
            def count(key):return sum(bool(chosen(r,camera).get(key)) for r in core)
            groups[camera+'_'+temp]=dict(n_parent=len(rr),n_measured=len(measured),n_core_grz=len(core),n_systems=len({r['member_system'] for r in core}),
               n_W1=count('w1_quality'),
               n_low_dust=sum(0<=chosen(r,camera)['ebv']<.03 for r in core),
               n_high_latitude=sum(r['old_latitude_scope'] for r in core),
               n_both_old_scopes=sum(0<=chosen(r,camera)['ebv']<.03 and r['old_latitude_scope'] for r in core),
               historical_roles=dict(Counter(r['historical_role'] for r in core)),
               phase_counts=dict(Counter(r['phase'] for r in core)),
               n_color_luminosity=count('target_color_luminosity_overlap'),n_color_luminosity_feh=count('target_color_luminosity_feh_overlap'),
               n_each={k:sum(chosen(r,camera)['target_marginal_overlap'].get(k,False) for r in core) for k in bounds},
               systems=dict(Counter(r['member_system'] for r in core)),
               overlap_source_ids=[r['source_id'] for r in core if chosen(r,camera)['target_color_luminosity_overlap']])
    # Stronger assertion that overlap indicators came from the registered box, not residuals.
    for r in rows:
        for m in r['measurements']:
            for k,(lo,hi) in bounds.items():
                value=m['observables'][k];assert m['target_marginal_overlap'][k]==(value is not None and np.isfinite(value) and lo<=value<=hi)
    checks['registered_coverage_boxes_recomputed']=True
    summ=dict(groups=groups,n_spectra=120,n_unique=len(rows),n_systems=len({r['member_system'] for r in rows}),
       unmatched_ids=[r['source_id'] for r in rows if not r['measurements']],checks=checks,
       camera_counting='Per-camera counts include both measurements when present; all and plots prefer unique DECam, then BASS/MzLS, avoiding duplicate stars',
       dual_camera_ids=[r['source_id'] for r in rows if len({m['release'] for m in r['measurements']})>1],
       bounds=bounds,source_sha256={n:hashlib.sha256((ROOT/'manifests'/n).read_bytes()).hexdigest() for n in ['giant_parameter_inventory_v1.json','giant_parameter_coverage_definition_v1.json','kgiant_reference_predictions_v1.json']})
    (ROOT/'manifests/giant_parameter_summary_v1.json').write_text(json.dumps(summ,indent=2)+'\n')
    vals=np.array([[r['gr0'],r['rz0'],r['g0']-10-5*np.log10(r['specdis_dist_kpc']),r['r']] for r in target])
    colors={'cold':'#0072b2','K':'#d55e00','warm':'#009e73'}
    fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    for ax,(x,y) in zip(axes.flat,[(0,1),(0,2),(3,2),(0,2)]):ax.scatter(vals[:,x],vals[:,y],s=9,c='.8',label='189 selected targets (teacher Mg)')
    for r in rows:
        m=chosen(r)
        if m is None or not m['grz_quality'] or r['pmember']<=.9:continue
        v=m['observables'];col=colors['cold' if r['teff']<4200 else 'K' if r['in_original_K_window'] else 'warm'];cam=m['natural_system']
        for ax,x,y in [(axes[0,0],v['gr'],v['rz']),(axes[0,1],v['gr'],v['Mg']),(axes[1,0],v['r'],v['Mg'])]:
            ax.scatter(x,y,c=col,marker='o' if cam=='DECam' else '^',s=32,edgecolors='black',linewidths=.3)
        if cam=='DECam':axes[1,1].scatter(v['gr'],v['Mg'],c=col,s=32,edgecolors='black',linewidths=.3)
    for ax,x,y,title in [(axes[0,0],'(g-r)0','(r-z)0','Core members, fixed grz quality'),(axes[0,1],'(g-r)0','Mg reference / teacher','Both native systems (not pooled for modelling)'),(axes[1,0],'Observed r','Mg reference / teacher','Brightness and luminosity'),(axes[1,1],'(g-r)0','Mg reference / teacher','DECam only')]:
        ax.set(xlabel=x,ylabel=y,title=title);ax.grid(alpha=.2)
        if y.startswith('Mg'):ax.invert_yaxis()
    for name,col in colors.items():axes[0,0].scatter([],[],c=col,label=name+' DESI temperature')
    axes[0,0].scatter([],[],c='black',marker='o',label='DECam');axes[0,0].scatter([],[],c='black',marker='^',label='BASS/MzLS');axes[0,0].legend(fontsize=7)
    fig.suptitle('Fixed120-spectrum observational census; no new distance predictions')
    fig.savefig(FIG/'01_observable_coverage.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for name,col in colors.items():
        rr=[r for r in rows if ('cold' if r['teff']<4200 else 'K' if r['in_original_K_window'] else 'warm')==name]
        axes[0].scatter([r['teff'] for r in rr],[r['logg'] for r in rr],c=col,s=22,label=name)
    axes[0].set(xlabel='DESI Teff [K]',ylabel='DESI logg',title='Reliable-giant labels before photometric cuts');axes[0].invert_yaxis();axes[0].legend();axes[0].axvspan(4200,5000,alpha=.08,color='black')
    x=np.arange(3)
    for j,(cam,col) in enumerate([('DECam','#0072b2'),('BASS/MzLS','#d55e00')]):
        n=[groups[cam+'_'+t]['n_core_grz'] for t in ['cold','K','warm']];axes[1].bar(x+(j-.5)*.35,n,width=.35,color=col,label=cam)
        for xx,nn in zip(x+(j-.5)*.35,n):axes[1].text(xx,nn+.2,str(nn),ha='center')
    axes[1].set(xticks=x,xticklabels=['T<4200','4200<=T<5000','T>=5000'],ylabel='Unique core grz members',title='Temperature cut vs camera/photometry availability');axes[1].legend()
    fig.savefig(FIG/'02_temperature_channel.png',dpi=160);plt.close(fig)
    print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
