#!/usr/bin/env python3
"""Render complete authenticated mode summaries, never extrapolate the pilot."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE


def main():
    source=ROOT/'manifests/v3_1/candidate_modes';marker=source/'complete.json';done=json.loads(marker.read_text())
    assert done['n']==3540 and done['fixed_main_highq']==3003 and done['all_PSF'] and not done['references_read']
    tables={};paths=[marker]
    for name in ['full_summary.csv','full_overview.csv','full_histograms.csv']:
        path=source/name;assert sha(path)==done['outputs'][name];paths.append(path)
        with path.open() as stream:tables[name]=list(csv.DictReader(stream))
    rows=tables['full_summary.csv'];overview=tables['full_overview.csv'];hist=tables['full_histograms.csv']
    assert len(rows)==3540 and len({r['identity'] for r in rows})==3540 and all(r['TYPE']=='PSF' for r in rows)
    istrue=lambda value:str(value).lower()=='true'
    assert sum(istrue(r['fixed_main_highq']) for r in rows)==3003
    assert sum(not istrue(r['numerical_passed']) for r in rows)==done['failed']
    groups=[('all','All candidates',3540,BLUE),('fixed_main_highq','Fixed high-q sample',3003,ORANGE)]
    for key,_,n,_ in groups:
        assert int(next(r['n'] for r in overview if r['group']==key))==n
        for metric in ['significant_modes','binary_probability']:
            assert sum(int(r['n']) for r in hist if r['group']==key and r['metric']==metric)==n
    out=ROOT/'figures/v3_1/candidate_modes';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/candidate_mode_figures';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE);fig,axes=plt.subplots(2,2,figsize=(15,11))
    fig.subplots_adjust(left=.09,right=.97,top=.79,bottom=.17,hspace=.56,wspace=.3)
    plotted=[]
    for j,(key,label,n,color) in enumerate(groups):
        selected=sorted([r for r in hist if r['group']==key and r['metric']=='significant_modes'],key=lambda r:float(r['lower']))
        assert len(selected)==6
        axes[0,0].bar(np.arange(6)+(j-.5)*.34,[100*int(r['n'])/n for r in selected],width=.32,color=color,label=f'{label} (n={n})')
        plotted.extend(dict(panel='A',source='full_histograms.csv',**r) for r in selected)
        value=next(r for r in overview if r['group']==key)
        keys=['different_class','same_class_different_binary','same_class_different_parent']
        axes[0,1].bar(np.arange(3)+(j-.5)*.34,[100*int(value[k])/n for k in keys],width=.32,color=color)
        plotted.extend(dict(panel='B',source='full_overview.csv',group=key,metric=k,n=int(value[k]),denominator=n) for k in keys)
        selected=sorted([r for r in hist if r['group']==key and r['metric']=='binary_probability'],key=lambda r:float(r['lower']))
        assert len(selected)==10
        edges=[float(r['lower']) for r in selected]+[float(selected[-1]['upper'])]
        axes[1,0].stairs([100*int(r['n'])/n for r in selected],edges,color=color,lw=2,label=label)
        plotted.extend(dict(panel='C',source='full_histograms.csv',**r) for r in selected)
    ax=axes[0,0];ax.set_title('A | Significant distance modes',loc='left',fontsize=14)
    ax.set_xticks(range(6),['0','1','2','3','4','5+']);ax.set(xlabel='Number of basins with probability > 0.05',ylabel='Objects (%)',ylim=(0,105));ax.legend(frameon=False,fontsize=10)
    ax=axes[0,1];ax.set_title('B | Contrasts between significant modes',loc='left',fontsize=14)
    ax.set_xticks(range(3),['Different\ndominant class','Same class, different\nsingle/binary state','Same class, different\nparent kernel'],fontsize=10)
    ax.set_ylabel('Objects / full group (%)');ax.set_ylim(bottom=0)
    ax=axes[1,0];ax.set_title('C | Marginal binary-proxy responsibility',loc='left',fontsize=14)
    prior=float(overview[0]['original_binary_prior']);ax.axvline(prior,color='.3',ls='--',lw=1.5,label=f'Original proxy prior: {prior:g}')
    ax.set(xlabel='Posterior binary-proxy probability',ylabel='Objects per bin (%)',xlim=(0,1));ax.legend(frameon=False,fontsize=10)
    ax=axes[1,1];ax.set_title('D | Actual saved-grid numerical audit',loc='left',fontsize=14)
    x=np.array([float(r['original_probability_delta'])/.003 for r in rows]);y=np.array([float(r['same_basin_joint_mass_delta'])/.003 for r in rows])
    failed=np.array([not istrue(r['numerical_passed']) for r in rows])
    assert np.isfinite(x).all() and np.isfinite(y).all() and (x>=0).all() and (y>=0).all()
    ax.scatter(x[~failed],y[~failed],s=8,c=BLUE,alpha=.45,label='All numerical gates passed',rasterized=True)
    if failed.any():ax.scatter(x[failed],y[failed],s=18,c=ORANGE,marker='x',label=f'Any numerical failure (n={failed.sum()})',rasterized=True)
    ax.axhline(1,color='.3',ls='--',lw=1);ax.axvline(1,color='.3',ls='--',lw=1)
    ax.set(xlabel='Original joint-probability difference / 0.003',ylabel='Same-basin component difference / 0.003')
    ax.set_xscale('symlog',linthresh=.01);ax.set_yscale('symlog',linthresh=.01);ax.legend(frameon=False,fontsize=9)
    plotted.extend(dict(panel='D',source='full_summary.csv',identity=r['identity'],original_probability_delta=r['original_probability_delta'],
        same_basin_joint_mass_delta=r['same_basin_joint_mass_delta'],numerical_passed=r['numerical_passed']) for r in rows)
    fig.suptitle('Distance-mode responsibilities of the frozen main candidate',y=.975,fontsize=19)
    fig.text(.5,.935,'Response12 / calibration12 | 32 evidence-weighted correction states | each population draw normalized separately',ha='center',fontsize=11)
    status='passed' if done['inherited_ensemble_accepted'] else 'FAILED'
    fig.text(.5,.895,f'Inherited ensemble numerical acceptance: {status}. Per-object mode audit failures: {done["failed"]} / 3540.',ha='center',fontsize=11,color='#8b4e14')
    fig.text(.5,.85,'All objects retained. High-q identities are fixed from the original candidate probabilities; q = P(distance < 10 kpc) > 0.95.',ha='center',fontsize=10)
    fig.text(.5,.11,'Panel B categories can overlap. Weak basins remain in saved products; small mass alone does not identify numerical artifacts.',ha='center',fontsize=10)
    fig.text(.5,.075,'Responsibilities describe this model: they do not confirm stellar classes, physical binaries or unique causes of multiple peaks.',ha='center',fontsize=10)
    fig.text(.5,.04,'Panel D uses symlog axes (linear below 0.01). Other acceptance gates include probability, quantile, mode-count and mass conservation.',ha='center',fontsize=9)
    fig.canvas.draw();labels=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in labels)
    outputs={}
    for ext in ['png','pdf']:
        path=out/f'mode_responsibilities.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
    plt.close(fig);csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    save_json(meta/'complete.json',dict(inputs={str(path.relative_to(ROOT)):sha(path) for path in paths+[ROOT/'scripts/v3_1_candidate_mode_figures.py',ROOT/'scripts/photdis_figure_style.py']},
        outputs=outputs,figures=1,panels=4,n=3540,highq_n=3003,plotted_rows=len(plotted),english_text_verified=True,
        visual_review_pending=True,scope='Authenticated full candidate mode tables, all failures retained. No pilot extrapolation or new scientific acceptance.'))


if __name__=='__main__':main()
