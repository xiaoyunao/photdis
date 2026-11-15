#!/usr/bin/env python3
"""Reconcile each raw VAC axis with native losses and both frozen selection cuts."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from photdis.stage1 import require_compute
from v2_1_near import ROOT, atomic_npz, csv_rows, load, save_json, sha
from v3_3_sed import verified
from v3_5_dr11_area import tile_map
from v3_5_dr11_production import CUTS
from v3_6_native_losses import COUNTS, OUT as NATIVE, inputs, sweep_owners
from v3_6_reference_index import OUT as INDEX, STATUSES
from v3_6_vac_distances import ARCHIVE, DISTANCE_EDGES, OUT as VAC, read_columns
from v3_6_vac_flow import OUT as MATCH


OUT = ROOT/'data/v3_6/reference_flow_audit'
UNRESOLVED = ('no_source_position_or_native_key', 'outside_frozen_sweeps',
              'in_frozen_sweep_without_verified_native_key')
ASSOCIATION = ('ambiguous_or_duplicate_vac_match', 'vac_candidate_has_no_position',
               'vac_candidate_position_rejected', 'archived_key_conflict',
               'archived_position_mismatch', 'archived_supported_but_vac_unverified')
GATES = ('dwarf_score_unavailable', 'below_dwarf_score', 'near_probability_unavailable',
         'below_near_probability', 'selected')
STAGES = (*UNRESOLVED, *ASSOCIATION, *(f'native_{k}' for k in COUNTS), *GATES)
R_EDGES = (-np.inf, 16., 18., 20., 21., 22., 23., np.inf)
COLOR_EDGES = (-np.inf, 0.35, 0.55, 0.8, 1.1, np.inf)
CLASSES = ('unknown', 'dwarf', 'giant', 'other')
ARCHIVE_ROLES = ('unassigned','fit','calibration','development_test','excluded')
AXES = (DISTANCE_EDGES, R_EDGES, COLOR_EDGES)
CUT_NAMES = ('primary', 'strict')


def native_rows(index, receipt):
    missing = index['match_status']==STATUSES.index('no_supported_match')
    out = dict(native_status=np.full(len(missing), -1, 'i1'),
               observed_r=np.full(len(missing), np.nan, 'f4'),
               dereddened_g_r=np.full(len(missing), np.nan, 'f4'))
    for name, digest in receipt['shards'].items():
        marker = verified(ROOT/name, digest)
        part = json.loads(marker.read_text())
        if part['reference_index_sha256']!=sha(INDEX/'complete.json'):
            raise ValueError('Native exclusion shard uses a different reference index')
        data = load(verified(marker.parent/'losses.npz', part['output_sha256']))
        if not data:
            continue
        keep = data['scope']=='baseline_v1_4'
        rows = data['reference_row'][keep]
        if len(np.unique(rows))!=len(rows) or (out['native_status'][rows]>=0).any():
            raise ValueError('Repeated archived row in native exclusion shards')
        out['native_status'][rows] = data['status'][keep]
        for key in ('observed_r', 'dereddened_g_r'):
            out[key][rows] = data[key][keep]
    np.testing.assert_array_equal(out['native_status']>=0, missing)
    return out


def trace(vac, matched, archive, native, parents_in_sweep, config):
    row = vac['archive_row']
    supported = matched['vac_row']
    status = vac['match_status']
    position = (np.isfinite(vac['reference_ra']) & np.isfinite(vac['reference_dec'])
                & (abs(vac['reference_dec'])<=90))
    stage = np.select([parents_in_sweep, position], [2, 1], default=0).astype('i1')
    archived = row>=0
    lost = np.flatnonzero(archived & (status==0))
    old_status = archive['match_status'][row[lost]]
    stage[lost] = STAGES.index('archived_supported_but_vac_unverified')
    for old, label in (('conflicting_keys', 'archived_key_conflict'),
                       ('position_mismatch', 'archived_position_mismatch')):
        stage[lost[old_status==STATUSES.index(old)]] = STAGES.index(label)
    missing = lost[old_status==STATUSES.index('no_supported_match')]
    native_status = native['native_status'][row[missing]]
    if np.any((native_status<0) | (native_status>=len(COUNTS))):
        raise ValueError('Missing or unknown native disposition')
    stage[missing] = len(UNRESOLVED)+len(ASSOCIATION)+native_status
    stage[status==1] = STAGES.index('ambiguous_or_duplicate_vac_match')
    stage[status==2] = STAGES.index('vac_candidate_has_no_position')
    stage[status>=3] = STAGES.index('vac_candidate_position_rejected')
    if (len(np.unique(supported))!=len(supported)
            or np.any(status[supported]<3) or np.any(matched['TYPE']!='PSF')):
        raise ValueError('Frozen selection requires unique, verified VAC rows and native PSF')
    np.testing.assert_array_equal(vac['catalog_row'][supported],matched['catalog_row'])

    labeled = np.flatnonzero(archived)
    valid_class = archive['class_eligible'][row[labeled]]
    label = np.full(len(stage), -1, 'i1')
    label[labeled[valid_class]] = archive['class_label'][row[labeled[valid_class]]]
    split = np.full(len(stage),-1,'i1')
    split[labeled] = archive['split'][row[labeled]]
    res = dict(class_label=label, archived_split=split,
               raw_coordinates_available=position, archive_link=archived)
    for key in ('qrf_training','selector_training'):
        flag = np.zeros(len(stage),bool)
        flag[supported] = matched[key]
        res[key] = flag
    for key in ('observed_r', 'dereddened_g_r'):
        values = np.full(len(stage), np.nan, 'f4')
        values[labeled] = native[key][row[labeled]]
        values[supported] = matched[key]
        res[key] = values

    score = matched['dwarf_selection_score']
    near = matched['nominal_qnear']
    for cut in CUT_NAMES:
        fail_score = score<config[cut]['minimum_dwarf_score']
        fail_near = near<config[cut]['minimum_near_probability']
        selected = stage.copy()
        first = STAGES.index(GATES[0])
        selected[supported] = first+np.select(
            [~np.isfinite(score), fail_score, ~np.isfinite(near), fail_near], [0, 1, 2, 3], default=4)
        np.testing.assert_array_equal(selected[supported]==STAGES.index('selected'),matched[cut])
        res[f'{cut}_stage'] = selected
        overlapping = np.zeros(len(stage), bool)
        overlapping[supported] = fail_score & fail_near
        res[f'{cut}_both_below'] = overlapping
    return res


def bin_index(values, edges):
    bins = np.searchsorted(edges, values, side='right')
    return np.where(np.isfinite(values) & (bins<len(edges)), bins, 0)


def stratum_label(edges, index):
    return f'[{edges[index-1]:g}, {edges[index]:g})' if index else 'unmeasured'


def summaries(vac, data, joint):
    distance = np.where(vac['finite_distance'], vac['distance_kpc'], np.nan)
    measured = (distance, data['observed_r'], data['dereddened_g_r'])
    bins = [bin_index(x, edges) for x, edges in zip(measured, AXES)]
    if joint:
        shape = (len(AXES[0]),len(CLASSES),len(ARCHIVE_ROLES),len(AXES[1]),len(AXES[2]))
        raw = np.ravel_multi_index((bins[0],data['class_label']+1,data['archived_split']+1,*bins[1:]),
                                   shape)
    else:
        shape = (len(AXES[0]),)
        raw = bins[0]
    cell, inverse = np.unique(raw, return_inverse=True)
    n = len(cell)
    counts = np.bincount(inverse, minlength=n)
    flags = dict(raw_distance_available=vac['finite_distance'],
                 raw_distance_error_available=vac['usable_uncertainty'],
                 historical_quality=vac['historical_quality'],
                 raw_coordinates_available=data['raw_coordinates_available'],
                 archived_identity=data['archive_link'],
                 historical_class_available=data['class_label']>=0,
                 supported_qrf_training=data['qrf_training'],
                 supported_selector_training=data['selector_training'],
                 repeated_gaia_id=vac['duplicate_source_id'])
    marginal = {k:np.bincount(inverse[valid], minlength=n) for k, valid in flags.items()}
    for cut in CUT_NAMES:
        stage = np.bincount(inverse*len(STAGES)+data[f'{cut}_stage'],
                            minlength=n*len(STAGES)).reshape(n,len(STAGES))
        np.testing.assert_array_equal(stage.sum(axis=1),counts)
        overlap = np.bincount(inverse[data[f'{cut}_both_below']], minlength=n)
        for i, k in enumerate(cell):
            coords = np.unravel_index(k, shape)
            row = dict(cut=cut, distance_kpc=stratum_label(AXES[0],coords[0]))
            if joint:
                row.update(historical_class=CLASSES[coords[1]],
                           baseline_v1_4_role=ARCHIVE_ROLES[coords[2]],
                           observed_r=stratum_label(AXES[1],coords[3]),
                           dereddened_g_r=stratum_label(AXES[2],coords[4]))
            yield dict(**row, raw_rows=int(counts[i]),
                       **{key:int(value[i]) for key,value in marginal.items()},
                       below_both_thresholds=int(overlap[i]),
                       **dict(zip(STAGES,stage[i].tolist())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('specdis', 'spdist'))
    kind = parser.parse_args().kind
    require_compute()
    raw_marker, match_marker = (path/kind/'complete.json' for path in (VAC, MATCH))
    index_marker, native_marker = (path/'complete.json' for path in (INDEX, NATIVE))
    records = [json.loads(p.read_text()) for p in (raw_marker, match_marker, index_marker, native_marker)]
    raw, match, index, native_record = records
    if match['inputs'][str(raw_marker.relative_to(ROOT))]!=sha(raw_marker):
        raise ValueError('Matched VAC was generated from a different raw inventory')
    if match['inputs'][str(CUTS.relative_to(ROOT))]!=sha(CUTS):
        raise ValueError('Matched VAC was generated with different selections')
    if match['inputs'][str(index_marker.relative_to(ROOT))]!=sha(index_marker):
        raise ValueError('Matched VAC was generated from a different reference index')
    vac_keys = ('archive_row','catalog_row','match_status','reference_ra','reference_dec','distance_kpc',
                'finite_distance','usable_uncertainty','historical_quality','duplicate_source_id')
    vac = read_columns(verified(raw_marker.parent/'vac.npz',raw['outputs']['vac.npz']),vac_keys)
    matched = load(verified(match_marker.parent/'matched.npz',match['outputs']['matched.npz']))
    filename = 'baseline_v1_4.npz'
    attached = load(verified(INDEX/filename,index['outputs'][filename]))
    archive_path = ARCHIVE/'references.npz'
    archive = read_columns(verified(archive_path,index['inputs'][str(archive_path.relative_to(ROOT))]),
                           ('class_label','class_eligible','split'))
    archive.update(match_status=attached['match_status'])
    native = native_rows(attached,native_record)
    freeze, inventory, _ = inputs()
    names = {n for group in freeze['groups'] for n in group}
    lookup = tile_map([tile for tile in inventory if tile['filename'] in names])
    owners = sweep_owners(vac['reference_ra'],vac['reference_dec'],lookup)
    data = trace(vac,matched,archive,native,owners>=0,json.loads(CUTS.read_text()))
    dest = OUT/kind
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():
        raise FileExistsError(dest/'complete.json')
    part, marginal, joint = (dest/x for x in ('raw_row_audit.npz','distance.csv','strata.csv'))
    atomic_npz(part,**data)
    csv_rows(marginal,list(summaries(vac,data,joint=False)))
    csv_rows(joint,list(summaries(vac,data,joint=True)))
    save_json(dest/'complete.json',dict(job=os.getenv('SLURM_JOB_ID'),reference=kind,stages=STAGES,
              raw_rows=len(owners),selected={cut:int(matched[cut].sum()) for cut in CUT_NAMES},
              inputs={str(p.relative_to(ROOT)):sha(p) for p in
                      (Path(__file__),CUTS,raw_marker,match_marker,index_marker,native_marker)},
              interpretation=('Audit rows are the raw VAC inventory. Values before verified selection serve '
                              'only for exclusion accounting. A raw position in a sweep does not establish '
                              'a Legacy match; overlapping quality and gate flags are counted separately.'),
              outputs={p.name:sha(p) for p in (part,marginal,joint)}))
    print(kind,len(owners),{cut:int(matched[cut].sum()) for cut in CUT_NAMES},flush=True)


if __name__=='__main__':
    main()
