#!/usr/bin/env python3
"""Post-freeze counts using exactly the scientific evaluation admission masks."""
from baseline_v1_4_experiments import *


def main():
    require_compute()
    assembly=json.loads((OUT/'assembly.json').read_text())
    assert sha(OUT/'references.npz')==assembly['reference_sha256']
    with np.load(OUT/'references.npz') as z:
        selection_columns=[k for k in z.files if k.startswith('target_') and k.endswith('_target')]
        t={k:z[k] for k in KEYS+['rvs_survey','rvs_program','zero_id_match','match_epoch_known','historical_identity']}
    clean=t['photometric_eligible']&~t['identity_conflict']&~t['nonstellar_risk']
    masks=dict(photometric=clean,class_labels=t['class_eligible'],distance_training=t['fit_eligible'],
        chosen_distance=clean&(t['label_source']>0),specdis=clean&t['specdis_valid'],spdist=clean&t['spdist_valid'],
        parallax=t['parallax_eligible'],weak_parallax=t['parallax_eligible']&(t['p_corr']/t['gaia_parallax_error']<5),
        negative_parallax=t['parallax_eligible']&(t['p_corr']<0))
    result={}
    for split in range(4):
        entries={}
        for name,group in groups(t).items():
            use=(t['split']==split)&group
            values={k:int((use&v).sum()) for k,v in masks.items()}
            values['classes']={str(k):int((use&t['class_eligible']&(t['class_label']==k)).sum()) for k in range(3)}
            values['i_class_labels']=int((use&t['class_eligible']&t['i_usable']).sum())
            values['i_distance_training']=int((use&t['fit_eligible']&t['i_usable']).sum())
            values['zero_id_class_labels']=int((use&t['class_eligible']&t['zero_id_match']).sum())
            entries[name]=values
        result[str(split)]=entries
    provenance={}
    for name in ['rvs_survey','rvs_program']:
        provenance[name]={}
        for value in np.unique(t[name]):
            use=t[name]==value
            provenance[name][str(value)]={str(sp):{key:int((use&(t['split']==sp)&mask).sum()) for key,mask in masks.items()} for sp in range(4)}
    report=dict(reference_sha256=assembly['reference_sha256'],groups=result,provenance=provenance,
        source_selection_columns=selection_columns,
        scope='Scientific counts require photometric quality and identity/nonstellar admission. Source-quality-only VAC counts in assembly coverage are broader.')
    save_json(OUT/'eligibility.json',report)
    print(json.dumps({sp:v['all'] for sp,v in result.items()}),flush=True)


if __name__=='__main__':main()
