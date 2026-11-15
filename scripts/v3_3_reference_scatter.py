#!/usr/bin/env python3
"""Quantify reference-distance uncertainty represented in the training clouds."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import softmax

from photdis.stage1 import require_compute
from v2_1_near import ROOT, csv_rows, load, save_json, sha
from v3_3_sed import CONFIG, OUT as FIT, verified

OUT = FIT.parent/'reference_scatter'


def reference_scatter(cloud, knots):
    f = cloud['y'][:, :knots, 0]
    if not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('Positive g flux required for this magnitude diagnostic')
    logf = np.log(f)
    observed = cloud['mask'][:, :6].sum(axis=1)
    # Undo the distance-dependent flux Jacobian, which is required in fitting
    # but is absent from the reference measurement probability.
    w = softmax(cloud['logweight'][:, :knots] - observed[:, None]*(logf-logf[:, :1]), axis=1)
    magnitude = 22.5-2.5/np.log(10)*logf
    center = np.sum(w*magnitude, axis=1)
    ref_var = np.sum(w*(magnitude-center[:, None])**2, axis=1)
    measured_var = (2.5/np.log(10))**2 * cloud['noise'][:, :knots, 0, 0] / f**2
    variance = np.sum(w*measured_var, axis=1)
    return center, ref_var, variance


def main():
    require_compute()
    config = json.loads(CONFIG.read_text())
    inputs = [CONFIG, Path(__file__).resolve()]
    rows = []
    for part in config['partitions']:
        base = FIT/part['name']
        marker = base/'cloud.json'
        metadata = json.loads(marker.read_text())
        path = verified(base/'cloud.npz', metadata['sha256'])
        inputs += [marker, path]
        t = load(path)
        if not (t['TYPE'] == 'PSF').all() or not (t['class_label'] == part['label']).all():
            raise ValueError('Mismatched PSF training cloud')
        center, ref_var, photo_var = reference_scatter(t, config['reference_mu_nodes'])
        color = t['features'][:, 0]
        for source in (0, 1, 2, 3):
            channel = t['channel'] == source if source else np.ones(len(color), bool)
            for low, high in zip([-np.inf, 0, .4, .8, 1.2], [0, .4, .8, 1.2, np.inf]):
                use = channel & (color >= low) & (color < high)
                if use.any():
                    m = center[use]
                    rows.append(dict(class_name=part['name'], source=source, g_r_lower=low, g_r_upper=high,
                      n=int(use.sum()), reference_rms_mag=np.sqrt(ref_var[use].mean()),
                      flux_and_dust_rms_mag=np.sqrt(photo_var[use].mean()),
                      center_nmad_mag=1.4826*np.median(abs(m-np.median(m)))))
    OUT.mkdir(parents=True, exist_ok=False)
    csv_rows(OUT/'by_source_color.csv', rows)
    save_json(OUT/'complete.json', dict(all_PSF=True, bins=len(rows),
        inputs={str(x.relative_to(ROOT)): sha(x) for x in inputs},
        outputs={'by_source_color.csv': sha(OUT/'by_source_color.csv')}, job=os.getenv('SLURM_JOB_ID'),
        interpretation='Single-star reference-node RMS, projected g noise/dust RMS and center NMAD. The center NMAD also contains intrinsic locus spread and within-bin color trends.'))


if __name__ == '__main__':
    main()
