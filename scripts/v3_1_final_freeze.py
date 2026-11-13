#!/usr/bin/env python3
"""Authenticate complete fixed predictions before opening external outcomes."""
import json
import os
from datetime import datetime, timezone
import numpy as np
from v2_1_near import ROOT, load, save_json
from photdis.stage1 import require_compute
from photdis.v3_1.freeze import Inventory, verify_prediction_inventory, verify_freeze


def main():
    require_compute()
    root = ROOT/'data/v3_1'; out = root/'final_validation'; out.mkdir(exist_ok=True)
    path = out/'freeze.json'
    if path.exists():
        print('VERIFIED EXISTING FREEZE', verify_freeze(ROOT, path), flush=True)
        return
    if (out/'field/started.json').exists() or (out/'common/started.json').exists():
        raise RuntimeError('Outcome association already started; cannot invent a new freeze')
    inv = Inventory(ROOT)
    # Missing evaluator implementation intentionally blocks outcome opening.
    for name in ['scripts/v3_1_final_freeze.py', 'scripts/v3_1_validation_evidence.py',
                 'scripts/v3_1_validation_score.py', 'src/photdis/v3_1/freeze.py',
                 'src/photdis/v3_1/validation_scores.py', 'src/photdis/reference_v1_4.py',
                 'scripts/v3_1_validation_report.py', 'src/photdis/v3_1/validation_report.py',
                 'scripts/v3_1_validation_admission.py', 'src/photdis/v3_1/admission.py',
                 'docs/v3_1_validation_admission_protocol.md',
                 'src/photdis/v3/metrics.py', 'src/photdis/v3_1/comparison.py',
                 'src/photdis/gaia.py', 'src/photdis/spectroscopy.py',
                 'docs/v3_1_final_validation_protocol.md', 'docs/v3_1_validation_roles.md',
                 'docs/v3_1_common_scale_protocol.md', 'docs/v3_1_target_ensemble_protocol.md',
                 'docs/v3_1_validation_target_protocol.md']:
        inv.add(name)
    roles = inv.marker(root/'validation_roles/complete.json')
    assert roles['roles_frozen']
    admission = inv.marker(root/'validation_admission/complete.json')
    assert admission['photometry_only'] and not admission['references_read'] and admission['original_candidate_reproduced']
    pointers = json.loads((root/'validation_roles/references.json').read_text())
    for value in pointers.values():
        inv.add(value['path'], value['sha256'])
    inv.marker(root/'common_scale_native/complete.json')
    ensemble = inv.marker(root/'target_ensemble/candidate/complete.json')
    for name, digest in ensemble['view_markers'].items():
        inv.marker(root/'target_ensemble/candidate'/name, digest)
    states = json.loads((root/'target_ensemble/candidate/states.json').read_text())['states']
    assert len(states) == 34 and all(s['passed'] for s in states)
    for state in states:
        inv.add(state['population'], state['population_sha256'])
    predictions = []; all_ids = []
    for cohort, expected_n in [('field',526), ('common',3032)]:
        photo = root/f'validation_roles/{cohort}_photometry.npz'
        t = load(photo); ids = t['identity']; all_ids.append(ids)
        assert len(ids) == expected_n and (t['TYPE'] == 'PSF').all() and (t['RELEASE'] == 11010).all()
        for family, folder in [('reference','validation_reference_refinement'),
                               ('old_target','validation_old_target'),
                               ('candidate','validation_target/candidate')]:
            marker = root/folder/cohort/'complete.json'; record = inv.marker(marker)
            passed = verify_prediction_inventory(record, ids, family=family)
            for name in record['objects']:
                obj = json.loads((marker.parent/name).read_text())
                assert obj['TYPE'] == 'PSF' and not obj['references_read']
                assert name == f'object_{obj["identity"]}/complete.json'
            predictions.append(dict(cohort=cohort,family=family,n=len(ids),numerical_passed=passed,
                marker=str(marker.relative_to(ROOT)),inherited_ensemble_accepted=record.get('inherited_ensemble_accepted')))
        # Authenticate original reference curves used for the first sidecar comparison.
        original = inv.marker(root/f'validation_reference_predictions/{cohort}/complete.json')
        assert not original['references_read']
        for name, digest in original['objects'].items():
            m = root/f'validation_reference_predictions/{cohort}'/name
            obj = inv.marker(m,digest); inv.add(m.with_suffix('.npz'),obj['sha256'])
    assert not np.intersect1d(*all_ids).size
    inv.unchanged()
    save_json(path,dict(predictions_frozen=True,outcomes_opened=False,hashes=inv.hashes,
        predictions=predictions,main_identity='candidate/response12/calibration12/beta0/bootstrap_evidence',
        utc=datetime.now(timezone.utc).isoformat(),job=os.getenv('SLURM_JOB_ID'),
        scope='Complete predictions and rules frozen; numerical flags retained; no scientific acceptance or reference scoring.'))
    print('FINAL PREDICTION FREEZE',len(inv.hashes),'artifacts',flush=True)


if __name__ == '__main__':
    main()
