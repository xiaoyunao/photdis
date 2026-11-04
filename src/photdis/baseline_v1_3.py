"""Separate usable class evidence from distance-label eligibility."""
import numpy as np
from .baseline_v1_2 import single_mask

CLASS_NAMES={-1:'unknown',0:'dwarf',1:'evolved',2:'other qualified stars / transition'}


def class_eligible(t):
    return ((t['class_label']>=0)&~t['identity_conflict']&~t['nonstellar_risk']
            &np.isfinite(t['g0'])&np.isfinite(t['r0'])&np.isfinite(t['z0']))


def evolved_candidate(p):
    return single_mask(p['weights'],p['branch_support'],p['supported'],.9,.4)&(p['main_index']==1)


def cells(t):
    r=np.searchsorted([18,20,22,25],t['r'],side='right')
    e=np.searchsorted([.03,.1],t['optical_error'],side='right')
    w=t['B'].astype(int)+t['C'].astype(int)
    c=np.searchsorted([.5,1.],t['g0']-t['r0'],side='right')
    return ((r*3+e)*3+w)*3+c
