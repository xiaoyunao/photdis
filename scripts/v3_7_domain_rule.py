#!/usr/bin/env python3
"""Apply paper use rules to native v3.5 catalog columns in memory."""
import json
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parents[1]/'configs/v3_7_use_domains.json'


def classify(catalog,config=CONFIG):
    protocol = json.loads(config.read_text())
    r = catalog['observed_r']
    color = catalog['dereddened_g_r']
    d = catalog['qrf_quantiles_kpc'][:,2]
    accepted = catalog['TYPE']=='PSF'
    accepted &= np.isfinite(r)&np.isfinite(color)&np.isfinite(d)&(d>0)
    accepted &= ~catalog['nominal_numerical_flag']
    accepted &= catalog['qrf_supported']
    output = np.full(len(r),protocol['unlisted'],dtype='U64')
    output[~accepted] = 'outside_supported_sample'
    for rule in protocol['rules']:
        select = accepted&(output==protocol['unlisted'])&catalog[rule['cut']]
        for key,value in (('r',r),('gr',color),('d50_kpc',d)):
            low,high = rule[key]
            low = -np.inf if low is None else low
            high = np.inf if high is None else high
            select &= (value>=low)&(value<high)
        if 'W1' in rule:
            select &= np.isin(catalog['band_state'][:,4],rule['W1'])
        output[select] = rule['name']
    return output
