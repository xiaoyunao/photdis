#!/usr/bin/env python3
"""Reference-availability audit of actual historical inputs and whole systems.

No photdis predictions/residuals, correction fitting or final role assignment.
Historical acquisition pools are distinguished from admitted/scored populations.
"""
import io
import json
import os
import zipfile
import fitsio
import numpy as np
from v2_1_near import ROOT, load, sha, save_json, atomic_npz, csv_rows
from photdis.stage1 import require_compute
from photdis.holdouts import load_reference_catalogs, annotate_holdouts, _id_group, GC_ARCHIVE_ALIASES
from photdis.v3.controls import optical_magnitudes
from photdis.v3_1.sky import qualified_sky


# These are upstream populations containing every downstream fitting/scoring
# subset in their lineage; pool membership itself is not claimed to be fitting.
HISTORY = [
    ('data/pilot_audit_v1/reference.fits', 'cold_v1_v3_and_pilot_reference'),
    ('data/baseline_v0/references.npz', 'old_field_reference_parent'),
    ('data/giant_information_v1/labels.npz', 'giant_information_parent'),
    ('data/baseline_v1_4/references.npz', 'v1_4_through_v2_reference_parent'),
    ('data/v2_3/training.npz', 'explicit_v2_3_fit'),
    ('data/v3_0/references.npz', 'v3_reference_roles_parent'),
    ('data/v3_0/validation/reservation/photometry.npz', 'opened_v3_reservation'),
    ('data/v3_0/validation/new_sky/photometry.npz', 'opened_v3_newsky'),
    ('data/v3_0/validation/independent_scale/photometry.npz', 'opened_v3_NGC288'),
]


def identity_position(path):
    wanted = ['identity', 'source_id', 'RA', 'DEC', 'ra', 'dec', 'cell']
    if path.suffix == '.npz':
        with np.load(path) as z:
            result = {k: z[k] for k in wanted if k in z}
    else:
        with fitsio.FITS(str(path)) as f:
            names = f[1].get_colnames()
            columns = [k for k in wanted if k in names]
            t = f[1].read(columns=columns)
            result = {k: t[k] for k in columns}
    ids = result['identity'] if 'identity' in result else result['source_id']
    assert ids.dtype.kind in 'iu'
    if 'RA' in result:
        sky = qualified_sky(result['RA'], result['DEC'])
    elif 'ra' in result:
        sky = qualified_sky(result['ra'], result['dec'])
    elif 'cell' in result:
        sky = dict(cell=result['cell'], valid_coordinates=result['cell'] >= 0)
    else:
        # Explicit training is already included in a reference parent with sky.
        sky = dict(cell=np.full(len(ids), -1), valid_coordinates=np.zeros(len(ids), bool))
    return ids, sky


def main():
    require_compute()
    native = ROOT/'data/v3_1/external_native'
    done = json.loads((native/'complete.json').read_text())
    for name in ['photometry.npz', 'references.npz']:
        assert sha(native/name) == done['outputs'][name]
    t, evidence = load(native/'photometry.npz'), load(native/'references.npz')
    np.testing.assert_array_equal(t['identity'], evidence['identity'])
    assert (t['TYPE'] == 'PSF').all() and (evidence['TYPE'] == 'PSF').all()
    ids = t['identity']; n = len(ids)
    reference_dir = ROOT/'data/reference'
    groups = load_reference_catalogs(reference_dir)
    group_names = json.loads((reference_dir/'holdout_group_names.json').read_text())
    paths = [native/'complete.json', native/'photometry.npz', native/'references.npz',
        reference_dir/'holdout_references_v1.npz', reference_dir/'holdout_group_names.json',
        reference_dir/'vasiliev_baumgardt_2021_members.zip',
        ROOT/'scripts/v3_1_external_history.py', ROOT/'src/photdis/v3_1/sky.py', ROOT/'src/photdis/holdouts.py']
    overlaps = []; seen_id = np.zeros(n, bool); seen_sky = np.zeros(n, bool); old_groups = set()
    for name, lineage in HISTORY:
        path = ROOT/name; old, sky = identity_position(path); paths.append(path)
        old_positive = np.unique(old[old > 0]); match = np.isin(ids, old_positive)
        sky_match = np.isin(t['cell'], np.unique(sky['cell'][sky['valid_coordinates']]))
        seen_id |= match; seen_sky |= sky_match
        membership = _id_group(old_positive, groups.member_source_id, groups.member_group)
        old_groups.update(int(g) for g in np.unique(membership) if g > 0)
        overlaps.append(dict(path=name, lineage=lineage, rows=len(old), positive_ids=len(old_positive),
            candidate_identity_overlap=int(match.sum()), candidate_cell_overlap=int(sky_match.sum()),
            note='Conservative prior input/inspection exposure; not every parent row was fitted.'))
    # Full raw DESI source availability was inventoried, but raw pool inclusion
    # alone is not fitting or scoring and is not a new-ID exclusion here.
    raw_path = ROOT/'data/baseline_v1_4/sources.npz'; paths.append(raw_path)
    with np.load(raw_path) as z:
        raw_pool = np.isin(ids, z['identity'])
    typed_path = ROOT/'data/giant_parameter_audit_v1/typed_candidates.json'; paths.append(typed_path)
    typed = json.loads(typed_path.read_text())
    typed_ids = np.array([int(r['source_id']) for r in typed['rows']], dtype='i8')
    typed_match = np.isin(ids, typed_ids); seen_id |= typed_match
    old_groups.update(int(g) for g in _id_group(typed_ids, groups.member_source_id, groups.member_group) if g > 0)
    # Name-level quarantine also covers previously examined systems whose
    # observed members may not lie in the retained field-reference parents.
    documented = {'NGC_5272', 'NGC_5024', 'NGC_2419', 'NGC_4147', 'NGC_7006',
        'NGC_5466', 'NGC_5904', 'NGC_6205', 'Pal_5', 'Pal_14', 'NGC_288'}
    old_groups.update(int(g) for g, record in group_names.items() if record['name'] in documented)
    annotation = annotate_holdouts(groups, ids, t['RA'], t['DEC'])
    member = annotation['cluster_member_group']; region = annotation['cluster_region_group']
    seen_system = np.isin(member, list(old_groups)) | np.isin(region, list(old_groups))
    # Recheck every recorded footprint, including the original pilot (not only
    # field*.json). Quarantine regions are not interpreted as membership.
    footprints = np.zeros(n, bool); footprint_records = []
    for path in sorted((ROOT/'configs').glob('*.json')):
        config = json.loads(path.read_text()); bounds = config.get('bounds_deg')
        if bounds is None: continue
        paths.append(path); lo, de, hi, dh = bounds
        inside = ((t['RA'] >= lo) & (t['RA'] <= hi)) if lo <= hi else ((t['RA'] >= lo) | (t['RA'] <= hi))
        inside &= (t['DEC'] >= de) & (t['DEC'] <= dh); footprints |= inside
        footprint_records.append(dict(config=str(path.relative_to(ROOT)), bounds=bounds, overlap=int(inside.sum())))
    # Probable membership in the normalized archive is not the stricter >.9
    # common-scale qualification. Read exact ID/probability from the raw archive.
    strict = np.zeros(n, bool); probability = np.full(n, np.nan)
    centers_path = reference_dir/'vasiliev_baumgardt_2021_clusters.txt'; paths.append(centers_path)
    centers = np.genfromtxt(centers_path, names=True, dtype=None, encoding='utf8')
    with zipfile.ZipFile(reference_dir/'vasiliev_baumgardt_2021_members.zip') as archive:
        lookup = {int(s): i for i, s in enumerate(ids)}
        for group in sorted(set(member[member >= 100000])):
            name = str(centers[int(group)-100000]['Name'])
            with archive.open('catalogues/'+GC_ARCHIVE_ALIASES.get(name, name)+'.txt') as stream:
                for line in io.TextIOWrapper(stream):
                    if not line.strip() or line.startswith('#'): continue
                    parts = line.split(); row = lookup.get(int(parts[0]))
                    if row is not None and member[row] == group:
                        probability[row] = float(parts[-1]); strict[row] = probability[row] > .9
    clean = ~seen_id & ~footprints
    new_sky = clean & ~seen_sky & ~seen_system
    field = new_sky & ~annotation['external_any']
    masks = dict(native_admitted=np.ones(n, bool), no_historical_identity_or_footprint=clean,
        unused_sky_and_system=new_sky, prospective_field_availability=field,
        prospective_strict_GC_availability=new_sky & strict)
    mag, _ = optical_magnitudes(t); rmag = mag[:, 1]-2.5*np.log10(t['transmission'][:, 1])
    coverage = []
    for stage, mask in masks.items():
        for kind, label in [('all', np.ones(n, bool)), ('dwarf', evidence['label_dwarf']),
                ('giant', evidence['label_giant']), ('poor_giant', evidence['label_giant'] & (evidence['feh'] < -1.5))]:
            for low, high in zip([0, 2, 5, 10, 20, 50], [2, 5, 10, 20, 50, np.inf]):
                selected = mask & label & evidence['distance_precise'] & (evidence['distance_kpc'] >= low) & (evidence['distance_kpc'] < high)
                coverage.append(dict(stage=stage, reference_class=kind, low_kpc=low, high_kpc=None if np.isinf(high) else high,
                    n=int(selected.sum()), cells=len(np.unique(t['cell'][selected])), fields=len(np.unique(evidence['apogee_field'][selected])),
                    r18_20=int((selected & (rmag >= 18) & (rmag < 20)).sum()), r20plus=int((selected & (rmag >= 20)).sum())))
    systems = []
    for group in sorted(set(member[member > 0])):
        selected = member == group
        systems.append(dict(group=int(group), name=group_names[str(group)]['name'], native_PSF=int(selected.sum()),
            strict_members=int((selected & strict).sum()), historical_system=int(group) in old_groups,
            new_sky_strict=int((selected & strict & new_sky).sum()),
            qualified_giants=int((selected & strict & new_sky & evidence['label_giant']).sum())))
    out = ROOT/'data/v3_1/external_history'; out.mkdir(exist_ok=True)
    csv_rows(out/'history_overlaps.csv', overlaps); csv_rows(out/'coverage.csv', coverage); csv_rows(out/'systems.csv', systems)
    save_json(out/'footprints.json', footprint_records)
    atomic_npz(out/'audit.npz', identity=ids, TYPE=t['TYPE'], seen_identity=seen_id, seen_sky=seen_sky,
        seen_system=seen_system, seen_footprint=footprints, raw_DESI_pool=raw_pool, strict_GC_member=strict,
        membership_probability=probability, **annotation, **masks)
    save_json(out/'complete.json', dict(inputs={str(p.relative_to(ROOT)): sha(p) for p in paths},
        outputs={p.name: sha(p) for p in out.iterdir() if p.name != 'complete.json'}, job=os.getenv('SLURM_JOB_ID'),
        native_PSF=n, counts={name: int(mask.sum()) for name, mask in masks.items()},
        typed_history_overlap=int(typed_match.sum()), raw_DESI_pool_overlap=int(raw_pool.sum()),
        documented_system_names=sorted(documented), final_roles_frozen=False, predictions_read=False,
        teacher='APOGEE astroNN Gaia EDR3 trained; distinct source/sky is not an independent absolute scale.',
        scope='Availability audit of named upstream historical reference populations, all recorded footprints, published groups and documented systems. Raw source inventory is separate from fitted/scored use; common-scale ancestry and final role freeze still pending.'))
    print('EXTERNAL HISTORY COMPLETE', {name: int(mask.sum()) for name, mask in masks.items()}, flush=True)


if __name__ == '__main__':
    main()
