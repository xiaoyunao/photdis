"""Index the frozen southern and new native partitions as one survey-release view."""
import csv
import json
from pathlib import Path

import numpy as np

from photdis.stage1 import galactic
from v2_1_near import ROOT, csv_rows
from v3_3_sed import verified
from v3_5_dr11_production import PRODUCTION
from v3_5_dr11_spatial import DELIVERY
from v3_6_vac_distances import read_columns
from v3_8_candidate import read_cache
from v3_9_catalog import OUT
from v3_9_seam import BOUNDARY


def owner_rows(side, table, overrides, *, frozen=False):
    """One mask in native photo order. Apply the measured seam corrections by ID."""
    ids = table['ls_id']
    if np.any(table['TYPE']!='PSF'):
        raise ValueError('Non-PSF row in the unified view')
    if frozen:
        north_region = (table['DEC']>=BOUNDARY)&(galactic(table['RA'],table['DEC'])[1]>0)
    else:
        north_region = table['north_primary_region']
    owner = north_region==(side=='north')
    moved = overrides['ls_id']
    found = np.searchsorted(moved,ids)
    j = np.flatnonzero(found<len(moved))
    j = j[moved[found[j]]==ids[j]]
    owner[j] = overrides['primary_survey_owner'][found[j]]
    return owner,ids[j]


def build(seam, destination):
    overrides = read_columns(read_cache(seam,'ownership_overrides.npz'),
                             ('ls_id','TYPE','primary_survey_owner'))
    moves = overrides['ls_id']
    if np.any(np.diff(moves)<=0) or np.any(overrides['TYPE']!='PSF'):
        raise ValueError('Seam overrides must have distinct ordered PSF identities')
    seen = []
    rows = []
    for side in ('south','north'):
        roster = json.loads((OUT/side/'inventory.json').read_text())
        for group in roster['groups']:
            for source in group:
                sweep = Path(source['filename']).stem
                p = OUT/side/'photo'/sweep
                q = OUT/side/'predict'/sweep
                photo = json.loads((p/'complete.json').read_text())['outputs']
                prediction = json.loads((q/'complete.json').read_text())['outputs']
                for name,digest in photo.items():
                    if name=='parent_state.npz':
                        continue
                    z = read_columns(p/name,('ls_id','TYPE','north_primary_region','region'))
                    state = 'state_'+name
                    status = read_columns(q/state,('state','ls_id'))
                    np.testing.assert_array_equal(z['ls_id'],status['ls_id'])
                    owner,applied = owner_rows(side,z,overrides)
                    seen.extend(applied)
                    pdf = name in prediction
                    if pdf != bool(np.any(status['state']==3)):
                        raise ValueError('Prediction state does not match its PDF partition')
                    region = z['region'][0]
                    if not np.all(z['region']==region):
                        raise ValueError('Mixed region in a native partition')
                    rows.append(dict(region=region,sweep=sweep,side=side,release=11011 if side=='north' else 11010,
                        photometric_system='BASS_MzLS_DR11' if side=='north' else 'DECam_DR11',
                        native_rows=len(owner),owned_rows=int(owner.sum()),
                        native_pdf_rows=int(np.sum(status['state']==3)),
                        owned_pdf_rows=int(np.sum(owner&(status['state']==3))),
                        photo_file=str((p/name).relative_to(ROOT)),photo_sha256=digest,
                        state_file=str((q/state).relative_to(ROOT)),state_sha256=prediction[state],
                        pdf_file=str((q/name).relative_to(ROOT)) if pdf else '',
                        pdf_sha256=prediction[name] if pdf else ''))
    with read_cache(DELIVERY,'sweeps.csv').open() as stream:
        for source in csv.DictReader(stream):
            sweep = Path(source['sweep']).stem
            p = PRODUCTION/sweep
            meta = verified(p/'complete.json',source['receipt_sha256'])
            for name,digest in json.loads(meta.read_text())['outputs'].items():
                if not name.startswith('part_'):
                    continue
                z = read_columns(p/name,('TYPE','ls_id','RA','DEC'))
                owner,applied = owner_rows('south',z,overrides,frozen=True)
                seen.extend(applied)
                file = str((p/name).relative_to(ROOT))
                rows.append(dict(region='south_v35',sweep=sweep,side='south',release=11010,
                    photometric_system='DECam_DR11',
                    native_rows=len(owner),owned_rows=int(owner.sum()),
                    native_pdf_rows=len(owner),owned_pdf_rows=int(owner.sum()),
                    photo_file=file,photo_sha256=digest,state_file='',state_sha256='',
                    pdf_file=file,pdf_sha256=digest))
    np.testing.assert_array_equal(np.sort(seen),moves)
    if len(set(a['photo_file'] for a in rows))!=len(rows):
        raise ValueError('Unified manifest repeats a native partition')
    csv_rows(destination,rows)
    summary = {}
    metrics = ('native_rows','owned_rows','native_pdf_rows','owned_pdf_rows')
    for row in rows:
        dest = summary.setdefault(row['region'],{'parts':0,**dict.fromkeys(metrics,0)})
        dest['parts'] += 1
        for key in metrics:
            dest[key] += row[key]
    return summary
