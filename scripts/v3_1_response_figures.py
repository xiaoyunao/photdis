#!/usr/bin/env python3
"""Complete matched-response evidence and seen-source transfer, with all flags."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v2_1_near import ROOT,sha,save_json,csv_rows
from photdis_figure_style import STYLE,BLUE,ORANGE


def main():
    source=ROOT/'manifests/v3_1/response_probes/report';marker=source/'complete.json'
    done=json.loads(marker.read_text())
    assert (done['states'],done['target_rows'],done['development_rows'])==(64,128,256) and done['all_PSF']
    inputs={str(marker.relative_to(ROOT)):sha(marker)};outputs={};plotted=[]
    def read(name):
        path=source/name;assert sha(path)==done['outputs'][name];inputs[str(path.relative_to(ROOT))]=sha(path)
        with path.open() as stream:return list(csv.DictReader(stream))
    inventory=read('inventory.csv');weights=read('response_weights.csv');numeric=read('weight_numerics.csv')
    count_numeric=read('target_count_numerics.csv');all_pairs=read('paired_all.csv');class_pairs=read('paired_reference_class.csv')
    ids=list(range(32))+list(range(1000,1032));records={int(r['draw']):r for r in inventory}
    assert len(records)==len(inventory)==64 and set(records)==set(ids)
    sets={r['set']:r for r in done['sets']};assert set(sets)=={'original','new','pooled'}
    evidence=np.array([float(records[i]['target_conditional_log_evidence']) for i in ids]);delta=evidence-evidence[12]
    assert np.isfinite(delta).all()
    out=ROOT/'figures/v3_1/response_probes';out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/'manifests/v3_1/response_figures';meta.mkdir(exist_ok=True)
    plt.rcParams.update(STYLE);colors={'original':BLUE,'new':ORANGE,'pooled':'#54856b'}
    def finish(fig,name):
        fig.canvas.draw();labels=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in labels)
        for ext in ['png','pdf']:
            path=out/f'{name}.{ext}';fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(17,12))
    fig.subplots_adjust(left=.08,right=.96,bottom=.16,top=.79,hspace=.62,wspace=.34)
    ax=axes[0,0]
    for part,color,label in [(slice(0,32),BLUE,'Original 32 responses'),(slice(32,64),ORANGE,'Regenerated 32 responses')]:
        x=np.arange(64)[part];ax.scatter(x,delta[part],color=color,s=25,label=label)
    ax.axhline(0,color='.3',ls='--',lw=1);ax.axvline(31.5,color='.6',lw=1)
    ax.set(title='A | Fixed-prior conditional evidence',xlabel='Response index (new draws 1000-1031 at 32-63)',ylabel='Log evidence minus response12 (nat)')
    ax.legend(frameon=False,fontsize=9)
    plotted.extend(dict(figure='response_evidence',panel='A',source='inventory.csv',**r) for r in inventory)
    ax=axes[0,1]
    for label,color in colors.items():
        item=sets[label];rows=sorted([r for r in weights if r['set']==label],key=lambda r:-float(r['log_posterior_weight']))
        if not rows:
            assert not item['target_inputs_passed'];ax.plot([],[],color=color,label=f'{label}: inputs failed, no weights');continue
        assert len(rows)==item['states']
        logw=np.array([float(r['log_posterior_weight']) for r in rows]);assert np.isfinite(logw).all()
        assert abs(np.exp(logw).sum()-1)<1e-8
        status='pass' if item['accepted'] else 'FAILED'
        ax.plot(np.arange(1,len(rows)+1),logw/np.log(10),color=color,label=f'{label}: ESS={item["ESS"]:.2f}/{len(rows)}, {status}')
        plotted.extend(dict(figure='response_evidence',panel='B',source='response_weights.csv',**r) for r in rows)
    ax.set(title='B | Evidence-weight concentration',xlabel='Response rank within each fixed set',ylabel='Log10 posterior weight (no clipping)');ax.legend(frameon=False,fontsize=9)
    labels=list(colors);alternatives=['native_coarse','entry_second_seed','entry_previous_level']
    ax=axes[1,0]
    for j,(label,color) in enumerate(colors.items()):
        for k,alternative in enumerate(alternatives):
            rows=[r for r in numeric if r['set']==label and r['alternative']==alternative]
            assert len(rows)<=1
            if rows:
                value=abs(float(rows[0]['log_evidence_delta']))/.05
                ax.scatter(k+(j-1)*.2,value,color=color,s=35,label=label if k==0 else None)
                plotted.append(dict(figure='response_evidence',panel='C',source='weight_numerics.csv',**rows[0]))
    ax.axhline(1,color='.3',ls='--');ax.set_yscale('symlog',linthresh=.1)
    ax.set_xticks(range(3),['Coarse native grid','Independent entry seed','Previous entry level'],rotation=15,ha='right',fontsize=9)
    ax.set(title='C | Absolute evidence sensitivity',ylabel='Absolute log-evidence change / 0.05');ax.legend(frameon=False,fontsize=9)
    ax=axes[1,1]
    for j,(label,color) in enumerate(colors.items()):
        for k,(table,column,gate) in enumerate([(numeric,'maximum_weight_delta',.01),(count_numeric,'maximum_probability_delta',.003),(count_numeric,'maximum_mean_count_delta',.5)]):
            rows=[r for r in table if r['set']==label]
            if rows:
                assert len(rows)==3;value=max(float(r[column])/gate for r in rows)
                ax.scatter(k+(j-1)*.2,value,color=color,s=35)
                plotted.extend(dict(figure='response_evidence',panel='D',source='weight_numerics.csv' if table is numeric else 'target_count_numerics.csv',display_metric=column,**r) for r in rows)
    ax.axhline(1,color='.3',ls='--');ax.set_yscale('symlog',linthresh=.1)
    ax.set_xticks(range(3),['State weight / 0.01','Object probability / 0.003','Expected count / 0.5'],rotation=15,ha='right',fontsize=9)
    ax.set(title='D | Maximum registered numerical changes',ylabel='Maximum across three alternatives / gate')
    fig.suptitle('Regenerated response support on the same fixed-prior probes',y=.975,fontsize=20)
    fig.text(.5,.93,'All 64 response states | same 128 native-PSF target objects | same nominal class / N(15, 5) prior',ha='center',fontsize=12)
    fig.text(.5,.88,'These are conditional probe evidences, not 26000-object population-marginal evidences or final-candidate response weights.',ha='center',fontsize=11)
    fig.text(.5,.835,f'Original per-object flags: {done["original_numerical_flags"]}; unresolved flags: {done["numerical_flags"]}. All responses remain in the inventory.',ha='center',fontsize=11)
    fig.text(.5,.09,'A numerical pass does not establish response-library completeness. Regenerated draws still share the empirical teacher and model family.',ha='center',fontsize=10)
    fig.text(.5,.045,'Panels C-D retain failed gates; values at or above 1 fail their strict threshold. Failed input sets have no normalized weight curve.',ha='center',fontsize=10)
    finish(fig,'response_evidence')
    fig,axes=plt.subplots(1,3,figsize=(20,8))
    fig.subplots_adjust(left=.07,right=.97,bottom=.23,top=.72,wspace=.38)
    specs=[(all_pairs,'all','signed_gaia','signed_gaia_logscore_gain',1,'All qualified signed-Gaia scores','Mean log-score gain (nat)'),
        (class_pairs,'1','primary','reference_logscore_gain',1,'Evolved primary-reference scores','Mean log-score gain (nat)'),
        (class_pairs,'1','primary','severe_mu_error_change',100,'Evolved severe distance errors','Severe fraction change (percentage points)')]
    for panel,(ax,(table,group,source_name,metric,scale,title,ylabel)) in enumerate(zip(axes,specs)):
        selected={int(r['draw']):r for r in table if r['group']==group and r['source']==source_name and r['metric']==metric}
        denominators=set()
        for j,draw in enumerate(ids):
            if draw not in selected:continue
            r=selected[draw];denominators.add((int(r['n']),int(r['cells'])))
            color=BLUE if j<32 else ORANGE;passed=r['all_numerically_passed']=='True'
            if r['mean']!='':
                y=scale*float(r['mean']);ax.scatter(delta[j],y,s=26,color=color,marker='o' if passed else 'x',zorder=3)
                if r['lo95']!='' and r['hi95']!='':ax.plot([delta[j]]*2,scale*np.array([float(r['lo95']),float(r['hi95'])]),color=color,alpha=.3,lw=.7)
            plotted.append(dict(figure='response_source_transfer',panel=str(panel),source_table='paired_all.csv' if table is all_pairs else 'paired_reference_class.csv',**r))
        assert len(denominators)<=1
        annotation='No qualified reference' if not denominators else f'n={next(iter(denominators))[0]}, sky cells={next(iter(denominators))[1]}; responses={len(selected)}/64'
        ax.text(.02,.97,annotation,transform=ax.transAxes,va='top',fontsize=9)
        ax.axhline(0,color='.4',ls='--',lw=1);ax.axvline(0,color='.4',ls='--',lw=1)
        ax.set(title=title,xlabel='Target conditional evidence gain over response12 (nat)',ylabel=ylabel)
        ax.xaxis.label.set_fontsize(9);ax.yaxis.label.set_fontsize(10);ax.title.set_fontsize(12)
    axes[0].plot([],[],'o',color=BLUE,label='Original response set');axes[0].plot([],[],'o',color=ORANGE,label='Regenerated response set')
    axes[0].plot([],[],'x',color='.4',label='Any per-object numerical failure');axes[0].legend(frameon=False,fontsize=9,loc='lower left')
    fig.suptitle('Target evidence versus source-qualified transfer on seen development stars',y=.97,fontsize=19)
    fig.text(.5,.9,'Same frozen 256-star development design; each panel uses its qualified subset and original sampling weights.',ha='center',fontsize=12)
    fig.text(.5,.83,'Thin intervals resample whole sky cells. Higher scores are better; lower severe-error changes are better.',ha='center',fontsize=11)
    fig.text(.5,.13,'All matched response points and numerical failures are retained. These comparisons are diagnostics, not new independent validation.',ha='center',fontsize=11)
    fig.text(.5,.075,'Common teachers, overlapping references and the shared response family remain; target evidence gains do not imply calibrated real distances.',ha='center',fontsize=10)
    finish(fig,'response_source_transfer')
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    for path in [ROOT/'scripts/v3_1_response_figures.py',ROOT/'scripts/photdis_figure_style.py']:inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(meta/'complete.json',dict(inputs=inputs,outputs=outputs,figures=2,panels=7,plotted_rows=len(plotted),
        english_text_verified=True,visual_review_pending=True,scientific_acceptance=False,
        scope='All64actual matched-response summaries. Conditional evidence and seen-source comparisons only; no synthetic fallback or finalcandidate calibration claim.'))


if __name__=='__main__':main()
