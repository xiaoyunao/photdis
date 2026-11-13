#!/usr/bin/env python3
"""Freeze new reference roles and imaging-ID guards without opening residuals."""
import json
import os
import fitsio
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from v3_1_external_history import HISTORY
from photdis.stage1 import require_compute


def history_columns(path):
    wanted = ['identity', 'source_id', 'ls_id']
    if path.suffix == '.npz':
        with np.load(path) as z: return {k:z[k] for k in wanted if k in z}
    with fitsio.FITS(str(path)) as f:
        names = f[1].get_colnames(); columns = [k for k in wanted if k in names]
        t = f[1].read(columns=columns)
        return {k:t[k] for k in columns}


def main():
    require_compute(); root = ROOT/'data/v3_1'; out = root/'validation_roles'; out.mkdir(exist_ok=True)
    folders = {'field':root/'external_native', 'common':root/'common_scale_native'}
    paths = [ROOT/'docs/v3_1_validation_roles.md', ROOT/'docs/v3_1_common_scale_protocol.md',
        ROOT/'scripts/v3_1_freeze_roles.py', ROOT/'scripts/v3_1_external_history.py',
        root/'external_history/complete.json', root/'external_history/audit.npz']
    paths += [ROOT/name for name, _ in HISTORY]
    for folder in folders.values(): paths += [folder/'complete.json', folder/'photometry.npz']
    paths += [folders['field']/'references.npz', folders['common']/'membership.npz', folders['common']/'scales.json']
    inputs = {str(path.relative_to(ROOT)):sha(path) for path in paths}
    freeze = out/'inputs.json'; marker = out/'complete.json'
    if freeze.exists(): assert json.loads(freeze.read_text()) == inputs
    else: save_json(freeze, inputs)
    if marker.exists():
        done = json.loads(marker.read_text()); assert done['inputs'] == inputs
        for name, digest in done['outputs'].items(): assert sha(out/name) == digest
        return
    tables = {}
    for name, folder in folders.items():
        done = json.loads((folder/'complete.json').read_text())
        assert sha(folder/'photometry.npz') == done['outputs']['photometry.npz']
        tables[name] = load(folder/'photometry.npz')
        t = tables[name]
        assert (t['TYPE'] == 'PSF').all() and (t['RELEASE'] == 11010).all() and t['photometric_entry'].all()
        assert len(np.unique(t['identity'])) == len(t['identity']) and len(np.unique(t['ls_id'])) == len(t['ls_id'])
    history = load(root/'external_history/audit.npz')
    hd = json.loads((root/'external_history/complete.json').read_text())
    assert sha(root/'external_history/audit.npz') == hd['outputs']['audit.npz']
    np.testing.assert_array_equal(history['identity'], tables['field']['identity'])
    hit = {k:np.zeros(len(t['identity']), bool) for k,t in tables.items()}; rows = []
    for name, lineage in HISTORY:
        old = history_columns(ROOT/name)
        assert 'ls_id' in old, 'History must provide explicit imaging identity: '+name
        ls = old['ls_id']; ids = old.get('identity', old.get('source_id'))
        assert ls.dtype.kind in 'iu' and ids.dtype.kind in 'iu'
        for cohort, t in tables.items():
            direct = np.isin(t['identity'], ids[ids > 0]); imaging = np.isin(t['ls_id'], ls[ls > 0])
            negative = np.isin(-t['ls_id'], ids[ids < 0]); match = direct | imaging | negative; hit[cohort] |= match
            rows.append(dict(cohort=cohort, history=name, lineage=lineage, source_overlap=int(direct.sum()),
                imaging_overlap=int(imaging.sum()), negative_identity_overlap=int(negative.sum()), any_overlap=int(match.sum())))
    field_clean = history['no_historical_identity_or_footprint'] & ~history['seen_system'] & ~history['external_any'] & ~hit['field']
    field_role = np.where(history['seen_sky'], 'new_source_seen_sky_field', 'new_sky_field')
    common = tables['common']; rejected_systems = np.unique(common['scale_system'][hit['common']])
    common_clean = ~np.isin(common['scale_system'], rejected_systems)
    masks = {'field':field_clean, 'common':common_clean}
    assert not np.intersect1d(tables['field']['identity'][field_clean], common['identity'][common_clean]).size
    assert not np.intersect1d(tables['field']['ls_id'][field_clean], common['ls_id'][common_clean]).size
    summary = []; records = []
    for cohort, t in tables.items():
        use = masks[cohort]; roles = field_role if cohort == 'field' else np.full(len(use), 'new_common_scale')
        payload = {k:v[use] for k,v in t.items()}
        payload.update(role=roles[use], field=t['cell'][use], design_weight=np.ones(int(use.sum())))
        atomic_npz(out/f'{cohort}_photometry.npz', **payload)
        for role in np.unique(roles):
            keep = use & (roles == role)
            summary.append(dict(cohort=cohort, role=str(role), admitted=int(keep.sum()), cells=len(np.unique(t['cell'][keep]))))
        for i in range(len(use)):
            records.append(dict(cohort=cohort, identity=int(t['identity'][i]), ls_id=int(t['ls_id'][i]),
                TYPE=str(t['TYPE'][i]), cell=int(t['cell'][i]), admitted=bool(use[i]), role=str(roles[i]),
                historical_identity_or_imaging=bool(hit[cohort][i])))
    csv_rows(out/'roles.csv', records); csv_rows(out/'summary.csv', summary); csv_rows(out/'history_guard.csv', rows)
    save_json(out/'references.json', dict(field=dict(path=str((folders['field']/'references.npz').relative_to(ROOT)),
        sha256=sha(folders['field']/'references.npz'), teacher='astroNN Gaia EDR3 trained; channel eligibility independent'),
        common=dict(path=str((folders['common']/'scales.json').relative_to(ROOT)), sha256=sha(folders['common']/'scales.json'),
        membership=str((folders['common']/'membership.npz').relative_to(ROOT)),
        scope='One common scale per system; membership-selected, not a field contamination experiment')))
    save_json(marker, dict(inputs=inputs, outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npz','.csv'] or p.name == 'references.json'},
        job=os.getenv('SLURM_JOB_ID'), summary=summary, rejected_common_systems=rejected_systems.tolist(),
        field_exclusions=int((~field_clean).sum()), common_exclusions=int((~common_clean).sum()),
        native_missing_TYPE=0, native_nonPSF=0, native_exclusions_recorded_in_source_manifests=True,
        roles_frozen=True, predictions_read=False, correction_fitted=False,
        scope='Unused source/block validation roles chosen by reference availability; not label-blind acquisition. No residual-based admission. Method/prediction freeze required before final scoring.'))
    print('VALIDATION ROLES FROZEN', summary, flush=True)


if __name__ == '__main__': main()
