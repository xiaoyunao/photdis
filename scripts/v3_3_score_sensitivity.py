#!/usr/bin/env python3
"""Check score sensitivity to flagged development rows and the parallax grid."""
import json
import os

import numpy as np

from photdis.stage1 import require_compute
from photdis.v3.metrics import observed_parallax
from photdis.v3_1.validation_report import paired_field_statistic
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_development import OUT as DEVELOPMENT
from v3_3_prospective import FORECAST, SCORES
from v3_3_sed import verified

OUT = ROOT/'data/v3_3/score_sensitivity'


def payload(folder, name):
    receipt = json.loads((folder/'complete.json').read_text())
    return load(verified(folder/name, receipt['outputs'][name]))


def main():
    require_compute()
    a, b = [payload(DEVELOPMENT, f'{view}.npz') for view in ('candidate', 'nominal')]
    np.testing.assert_array_equal(a['identity'], b['identity'])
    flags = np.logical_or.reduce([(t['probability_delta'] > .003) | (t['modulus_delta'] > .03) for t in (a, b)])
    rows = []
    for group, use in [('all', np.ones(len(flags), bool)), ('giants', a['reference_class'] == 1)]:
        for name in ('gaia_score', 'reference_score', 'oracle_gaia_score', 'oracle_reference_score'):
            for selection, selected in [('all_rows', use), ('unflagged_rows', use & ~flags)]:
                rows.append(dict(group=group, metric=name, selection=selection,
                    **paired_field_statistic(a[name][selected], b[name][selected], a['cell'][selected])))
    scored = [payload(SCORES, f'{view}.npz') for view in ('candidate', 'nominal')]
    forecasts = [payload(FORECAST, f'{view}.npz') for view in ('candidate', 'nominal')]
    all_coarse = []
    for result, forecast in zip(scored, forecasts):
        np.testing.assert_array_equal(result['identity'], forecast['identity'])
        selected = np.flatnonzero(result['eligible'])
        coarse = [observed_parallax(forecast['mu'][::2], forecast['logp'][i, ::2],
                                   result['corrected_parallax'][i], result['parallax_error'][i])
                  for i in selected]
        all_coarse.append(np.asarray([r['log_score'] for r in coarse]))
    use = scored[0]['eligible']
    cells = scored[0]['cell'][use]
    near = dict(fine=paired_field_statistic(*(t['log_score'][use] for t in scored), cells),
                coarse=paired_field_statistic(*all_coarse, cells),
                max_absolute_log_score_change=[float(np.max(abs(v-t['log_score'][use])))
                                               for v, t in zip(all_coarse, scored)])
    OUT.mkdir(parents=True, exist_ok=False)
    csv_rows(OUT/'opened_development.csv', rows)
    save_json(OUT/'signed_gaia.json', near)
    save_json(OUT/'complete.json', dict(job=os.getenv('SLURM_JOB_ID'), development_flag_union=int(flags.sum()),
        inputs={str(p.relative_to(ROOT)): sha(p) for p in (DEVELOPMENT/'complete.json',
                FORECAST/'complete.json', SCORES/'complete.json')},
        outputs={p.name: sha(p) for p in sorted(OUT.iterdir())}))


if __name__ == '__main__':
    main()
