"""Coverage of actual expanded-A training and evaluation reference identities."""
import json

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from baseline_v1_4_experiments import ROOT, OUT, load, sha
from photdis.reference_v1_4 import unique_join
from photdis_figure_style import BLUE, GREEN, ORANGE, STYLE


def application_coverage():
    with plt.rc_context(STYLE):
        return render_coverage()


def render_coverage():
    ex = OUT/'experiments'
    new_path = OUT/'references.npz'
    old_path = ROOT/'data/baseline_v1_3/references.npz'
    new_ids = ex/'expanded_A_identities.npz'
    old_ids = ex/'expanded_A_legacy_training.npz'
    parent_path = OUT/'diagnostics/parent.npz'
    paths = [new_path, old_path, new_ids, old_ids, parent_path]
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in paths}
    assert hashes[str(new_path.relative_to(ROOT))] == json.loads((OUT/'assembly.json').read_text())['reference_sha256']
    ids, legacy, parent = load(new_ids), load(old_ids), load(parent_path)
    keys = ['identity', 'g0', 'r0', 'r']
    with np.load(new_path) as z:
        t = {k: z[k] for k in keys+['split','class_eligible','class_label','label_source',
                                     'parallax_eligible','specdis_valid','spdist_valid',
                                     'class_train_eligible','fit_eligible','photometric_eligible',
                                     'identity_conflict','nonstellar_risk']}
    with np.load(old_path) as z:
        old = {k: z[k] for k in keys}
    clean = t['photometric_eligible'] & ~t['identity_conflict'] & ~t['nonstellar_risk']
    evaluation = clean & np.isin(t['split'], [1,2])
    any_distance = (t['label_source']>0) | t['specdis_valid'] | t['spdist_valid'] | t['parallax_eligible']
    assert np.array_equal(t['identity'][evaluation], ids['class_eval_identity'])
    assert np.array_equal(t['identity'][evaluation & any_distance], ids['distance_eval_identity'])
    cls = evaluation & t['class_eligible'] & (t['class_label']>=0)
    dist = evaluation & any_distance
    xe, ye = np.linspace(-1,3,33), np.arange(14.,26.)

    def histogram(data, weight=None):
        h = np.histogram2d(data['g0']-data['r0'], data['r'], bins=[xe,ye], weights=weight)[0]
        total = len(data['r']) if weight is None else weight.sum()
        return h, total

    def select(data, rows):
        return {k: data[k][rows] for k in keys}

    parent_hist, parent_total = histogram(parent, parent['sample_weight'])
    panels, bins, summary = {}, {}, {}
    for family, qualified in [('Classification', cls), ('Distance evidence', dist)]:
        key = 'class_train_identity' if family == 'Classification' else 'distance_train_identity'
        train_new = unique_join(ids[key], t['identity'])
        train_old = unique_join(legacy[key], old['identity'])
        assert (train_new>=0).all() and (train_old>=0).all()
        assert not np.intersect1d(ids[key], legacy[key]).size
        training = {k: np.r_[old[k][train_old],t[k][train_new]] for k in keys}
        panels[family] = [histogram(training)]
        for split, name in [(1,'Calibration'),(2,'Development test')]:
            panels[family].append(histogram(select(t, qualified & (t['split']==split))))
        for name, (h,n) in zip(['Training','Calibration','Development test'], panels[family]):
            bins[f'{family}: {name}'] = dict(weight=h, n=int(n), outside_fraction=1-h.sum()/n)
    for split, label in [(1,'cal'),(2,'test')]:
        q = t['split']==split
        candidates = {'classification': cls, 'any_distance': dist,
                      'chosen_distance':dist & (t['label_source']>0),
                      'parallax_only':dist & t['parallax_eligible'] & (t['label_source']==0)
                                      & ~t['specdis_valid'] & ~t['spdist_valid'],
                      'class_or_distance':cls|dist}
        summary[label] = {}
        for name, mask in candidates.items():
            take = q & mask
            h, n = histogram(select(t,take))
            summary[label][name] = dict(n=int(n), r22_25=int((take & (t['r']>=22) & (t['r']<25)).sum()),
                                  parent_weight_fraction_in_empty_bins=parent_hist[h==0].sum()/parent_total)
    fig, axes = plt.subplots(2,4,figsize=(20.8,9.75),layout='constrained',sharex=False)
    norm = LogNorm(1e-5,.3)
    for i, (family, data) in enumerate(panels.items()):
        for j, (name,(h,n)) in enumerate(zip(['Training','Calibration','Development test'],data)):
            ax=axes[i,j]
            mesh=ax.pcolormesh(xe,ye,np.ma.masked_equal(h.T/n,0),cmap='Blues',norm=norm)
            short = 'Class' if family == 'Classification' else 'Distance'
            ax.set(title=f'{name}\n{short} · N={n:,}',xlabel=r'$(g-r)_0$',ylabel=r'$r$ [mag]',ylim=(25,14))
    ax=axes[0,3]
    ax.pcolormesh(xe,ye,np.ma.masked_equal(parent_hist.T/parent_total,0),cmap='Blues',norm=norm)
    ax.set(title='Weighted DECam parent\nN=26,000',xlabel=r'$(g-r)_0$',ylabel=r'$r$ [mag]',ylim=(25,14))
    fig.colorbar(mesh,ax=[*axes[:,:3].flat,ax],label='Fraction of each sample per bin',shrink=.76)
    ax=axes[1,3]
    names=['classification','chosen_distance','any_distance','class_or_distance']
    values=[summary['test'][k]['parent_weight_fraction_in_empty_bins'] for k in names]
    ax.barh(range(4),values,color=[BLUE,ORANGE,GREEN,'#7f7f7f'])
    ax.set(xlim=(0,1),yticks=range(4),yticklabels=['Class','Chosen distance','Distance evidence','Class or distance'],
           xlabel='Weighted parent fraction',title='Bins without test reference')
    for j,v in enumerate(values):
        ax.text(v+.02,j,f'{v:.1%}',va='center')
    fig.suptitle('Expanded A reference coverage',fontsize=20)
    assert all(sha(ROOT/path)==h for path,h in hashes.items())
    return fig, dict(xedges=xe,yedges=ye,panels=bins,validation=summary,inputs=hashes)
