#!/usr/bin/env python3
"""Complete external comparisons with source, sky and shared-scale limits visible."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE

MAIN='candidate/bootstrap_evidence'
VIEWS=[('reference/nominal','Nominal + reference'),('reference/response12','Response12 + reference'),
       ('reference/corrected_point','Point correction + reference'),('reference/corrected_bootstrap','Bootstrap correction + reference'),
       ('old_target/old_target','Original target prior'),('candidate/zero','Candidate, zero correction'),
       ('candidate/point','Candidate, point correction'),(MAIN,'Main candidate, evidence bootstrap'),
       ('candidate/bootstrap_cut','Candidate, equal-weight cut')]


def number(value):
    return float(value) if value not in (None,'') else None


def main():
    source=ROOT/'manifests/v3_1/final_validation/report';marker=source/'complete.json';done=json.loads(marker.read_text())
    assert done['all_PSF'] and done['all_frozen_identities_retained'] and not done['independent_fullcandidate_contamination_measured']
    assert sorted((r['cohort'],r['n'],r['views']) for r in done['inventory'])==[('common',3032,9),('field',526,9)]
    inputs={str(marker.relative_to(ROOT)):sha(marker)};plotted=[];outputs={}
    def read(name):
        if name not in done['outputs']:return []
        path=source/name;assert sha(path)==done['outputs'][name];inputs[str(path.relative_to(ROOT))]=sha(path)
        with path.open() as stream:return list(csv.DictReader(stream))
    out=ROOT/'figures/v3_1/final_validation';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/final_validation_figures';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE)
    def finish(fig,name):
        fig.canvas.draw();labels=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in labels)
        for ext in ['png','pdf']:
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    roles=[('new_sky_field','New sky'),('new_source_seen_sky_field','New sources in seen sky')]
    baselines=[(v,label) for v,label in VIEWS if v!=MAIN]
    specs=[('gaia','gaia/log_score','available','Signed Gaia log score'),
           ('astronn','astronn/log_score','precise','Precise AstroNN log score'),
           ('class','class/log_score','available','APOGEE class log score')]
    fig,axes=plt.subplots(3,2,figsize=(18,15),sharey=True)
    fig.subplots_adjust(left=.23,right=.88,top=.84,bottom=.14,wspace=.6,hspace=.45)
    for col,(role,title) in enumerate(roles):
        for row,(channel,metric,qualification,label) in enumerate(specs):
            ax=axes[row,col];name=f'{role}_paired_{channel}_all_objects.csv';table=read(name)
            for i,(view,_) in enumerate(baselines):
                selected=[r for r in table if r['stratum']=='candidate/all' and r['baseline']==view and r['metric']==metric and r['qualification']==qualification]
                assert len(selected)<=1
                if selected:
                    r=selected[0];assert r['main']==MAIN and r['admission']=='all_objects'
                    mean,lo,hi=[number(r.get(k)) for k in ['mean','lo95','hi95']];n=int(r['n']);cells=int(r['cells'])
                    plotted.append(dict(figure='field_score_gains',panel=f'{row}_{col}',source=name,**r))
                else:r={};mean=lo=hi=None;n=cells=0
                if mean is not None:
                    assert np.isfinite(mean)
                    ax.plot(mean,i,'o',color=BLUE,ms=5)
                    if lo is not None and hi is not None:ax.plot([lo,hi],[i,i],color=BLUE,lw=1.8)
                annotation=f'n={n}, sky={cells}'
                if mean is None:annotation=('N/F; ' if n else 'NA; ')+annotation
                elif lo is None or hi is None:annotation+='; no CI'
                ax.text(1.02,i,annotation,transform=ax.get_yaxis_transform(),ha='left',va='center',fontsize=8)
            ax.axvline(0,color='.3',ls='--',lw=1);ax.set_yticks(range(8),[label for _,label in baselines],fontsize=10)
            ax.set_ylim(7.65,-.65);ax.set_title(f'{title} | {label}',loc='left',fontsize=12)
            ax.set_xlabel('Main minus baseline mean log score (nat)',fontsize=10);ax.grid(axis='y',visible=False)
    fig.suptitle('Product-matched external field comparisons',y=.975,fontsize=21)
    fig.text(.5,.94,'Primary candidate-admission subset | all objects retained, including inherited numerical failures',ha='center',fontsize=12)
    fig.text(.5,.905,'Main candidate = response12 / calibration12 + evidence-weighted luminosity corrections + candidate population prior',ha='center',fontsize=11)
    fig.text(.5,.875,'Positive score differences favor the main candidate. Its inherited ensemble evidence gate remains failed.',ha='center',fontsize=11,color='#8b4e14')
    fig.text(.5,.095,'Intervals resample whole sky cells. NA = no qualified evidence; N/F = retained nonfinite score. Single-cell results have no sky interval.',ha='center',fontsize=10)
    fig.text(.5,.06,'AstroNN has Gaia ancestry; reference channels are not independent absolute-distance truths. New-source seen-sky evidence is not new-sky validation.',ha='center',fontsize=10)
    fig.text(.5,.025,'Sparse qualified samples do not validate empty distance, metallicity or photometric domains; source and numerical denominators remain in the tables.',ha='center',fontsize=10)
    finish(fig,'field_score_gains')
    name='common_scale_summary.csv';table=read(name);assert table
    specs=[('median_relative_residual','Median relative distance residual (%)',100,0),
           ('severe_fraction','Fraction with |distance-modulus residual| > 1 (%)',100,None),
           ('reference_agreement95','Reference point inside 95% interval (%)',100,95)]
    fig,axes=plt.subplots(1,3,figsize=(18,9),sharey=True)
    fig.subplots_adjust(left=.24,right=.97,top=.73,bottom=.23,wspace=.35)
    for ax,(metric,label,scale,reference) in zip(axes,specs):
        for i,(view,_) in enumerate(VIEWS):
            selected=[r for r in table if r['system']=='NGC_1261' and r['subset']=='candidate_members' and r['view']==view and r['admission']=='all_objects' and r['metric']==metric and float(r['method_factor'])==1.]
            assert len(selected)==1;r=selected[0];assert int(r['n'])==1732 and int(r['independent_scales'])==1
            lo,median,hi=[float(r[k])*scale for k in ['lower95','median','upper95']]
            color=ORANGE if view==MAIN else BLUE
            ax.plot([lo,hi],[i,i],color=color,lw=2);ax.plot(median,i,'D' if view==MAIN else 'o',color=color,ms=6)
            plotted.append(dict(figure='common_scale_agreement',panel=metric,source=name,**r))
        if reference is not None:ax.axvline(reference,color='.3',ls='--',lw=1)
        ax.set_yticks(range(9),[label for _,label in VIEWS],fontsize=10);ax.set_ylim(8.65,-.65)
        ax.set_xlabel(label,fontsize=10);ax.grid(axis='y',visible=False)
        if metric!='median_relative_residual':ax.set_xlim(-3,103)
    fig.suptitle('NGC 1261: one shared external distance scale',y=.975,fontsize=21)
    fig.text(.5,.925,'1732 frozen candidate-admission members | one HST kinematic scale | 4096 common reference-scale draws',ha='center',fontsize=12)
    fig.text(.5,.875,'Intervals reflect the shared reference-distance uncertainty, not 1732 independent distance measurements.',ha='center',fontsize=12)
    fig.text(.5,.81,'All objects and the main candidate evidence failure are retained. Orange: exact main candidate.',ha='center',fontsize=11,color='#8b4e14')
    fig.text(.5,.15,'Reference-point agreement is not independently measured posterior coverage; the common-scale uncertainty is correlated across every member.',ha='center',fontsize=10)
    fig.text(.5,.105,'Shown: nominal method factor 1.00. Registered 0.97 / 1.03 sensitivities, all 3032 members and outside-domain objects remain in source tables.',ha='center',fontsize=10)
    fig.text(.5,.06,'This is validation of frozen photometric predictions. It does not refit a structure, resolve physical depth or establish fullcandidate contamination.',ha='center',fontsize=10)
    finish(fig,'common_scale_agreement')
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    for path in [ROOT/'scripts/v3_1_validation_figures.py',ROOT/'scripts/photdis_figure_style.py']:inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(meta/'complete.json',dict(inputs=inputs,outputs=outputs,figures=2,panels=9,plotted_rows=len(plotted),
        english_text_verified=True,visual_review_pending=True,scope='Actual source-qualified complete external report only; all failures/empty groups retained; no new scientific acceptance.'))


if __name__=='__main__':main()
