#!/usr/bin/env python3
"""Render small scientific summary manifests locally; never read large catalogs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    d=json.loads((ROOT/'manifests/baseline_v1_4_result.json').read_text());m=d['models']
    out=ROOT/'figures/baseline_v1_4';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['frozen_A','expanded_A','frozen_B','expanded_B','frozen_C','expanded_C','matched_grz','matched_griz']
    labels=['old A','new A','old B','new B','old C','new C','grz matched','griz matched']
    colors=['#888888','#2878b5']*4
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    fig.suptitle('Compare within each pair; A/B/C and matched samples differ',fontsize=11)
    for ax,key,title in zip(axes,['average_precision','recall','known_contamination'],['Evolved average precision','Evolved recall at cal 2% dwarf budget','Known-label contamination']):
        values=[m[n]['classification']['marginalized' if n.startswith('matched') else 'nominal']['groups']['all'][key] for n in names]
        ax.bar(np.arange(len(names)),[np.nan if v is None else v for v in values],color=colors)
        ax.set_xticks(np.arange(len(names)),labels,rotation=55,ha='right');ax.set_title(title);ax.set_ylim(0,1)
    fig.savefig(out/'classification.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    fig.suptitle('Chosen-distance agreement; compare within pairs, not across WISE tiers\nOrange bars absent in B/C: these tiers require W1',fontsize=10)
    for ax,key,title in zip(axes,['tail_1','coverage68','coverage95'],['Distance |error| > 1 mag','Distance 68% coverage','Distance 95% coverage']):
        for group,offset,col in [('all',-.18,'#2878b5'),('no_W1',.18,'#e1812c')]:
            vals=[m[n]['distance'][group]['chosen'].get(key,np.nan) for n in names]
            ax.bar(np.arange(len(names))+offset,vals,width=.34,label='all references' if group=='all' else 'no usable W1',color=col)
        ax.set_xticks(np.arange(len(names)),labels,rotation=55,ha='right');ax.set_title(title);ax.set_ylim(0,1)
    axes[0].legend();fig.savefig(out/'distance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    fig.suptitle('Gaia likelihood comparisons use identical stars within each pair',fontsize=11)
    for name,label in [('matched_grz','grz'),('matched_griz','griz')]:
        ga=m[name]['distance']['all']['full_Gaia'];h=np.array(ga['pit_histogram']);axes[0].stairs(h/max(1,h.sum()),np.linspace(0,1,11),label=label,color='#888888' if name=='matched_grz' else '#2878b5')
    axes[0].axhline(.1,color='gray',ls='--');axes[0].set(xlabel='Full observed-parallax PIT',ylabel='Fraction',title='Matched Gaia evidence; negative parallaxes retained');axes[0].legend()
    v=[m[n]['distance']['all']['full_Gaia']['mean_log_density'] for n in names]
    axes[1].bar(np.arange(len(names)),v,color=colors);axes[1].set_xticks(np.arange(len(names)),labels,rotation=55,ha='right');axes[1].set_title('Mean full Gaia log predictive density (higher better)')
    fig.savefig(out/'gaia.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    coverage=d['parent_coverage']['groups'];bins=['r:0:18','r:18:20','r:20:22','r:22:25']
    for suffix,label in [('', 'DECam parent'),(':no_W1','No usable W1')]:
        axes[0].plot(range(4),[coverage[k+suffix]['weighted_i_fraction'] for k in bins],'o-',label=label)
    axes[0].set(xticks=range(4),xticklabels=['<18','18–20','20–22','22–25'],ylim=(0,1),ylabel='Original-weight i availability',xlabel='r magnitude');axes[0].legend()
    cov=d['assembly']['coverage']['2'];bins=['bright:all','18_20:all','20_22:all','22_25:all']
    for key,label in [('classes','Reliable classes'),('distance','Chosen distance'),('gaia','Gaia likelihood')]:
        y=[sum(cov[b][key].values()) if key=='classes' else cov[b][key] for b in bins];axes[1].plot(range(4),y,'o-',label=label)
    axes[1].set(xticks=range(4),xticklabels=['<18','18–20','20–22','22–25'],ylabel='New test reference count',xlabel='r magnitude')
    axes[1].set_yscale('symlog',linthresh=1)
    top=max(max(sum(cov[b]['classes'].values()),cov[b]['distance'],cov[b]['gaia']) for b in bins)
    axes[1].set_ylim(0,top*1.8);axes[1].legend()
    fig.savefig(out/'coverage.png',dpi=170);plt.close(fig)
    focus=['frozen_A','expanded_A','matched_grz','matched_griz']
    focus_labels=['old grz','new-data grz','matched grz','matched griz']
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    for ax,group,kind,title in zip(axes,['r18_20:no_W1','r18_20:no_W1','r20_22:no_W1'],
            ['class','distance','distance'],['18–20, no W1: evolved AP','18–20, no W1: distance tails','20–22, no W1: distance tails']):
        vals=[];counts=[]
        for n in focus:
            if kind=='class':
                variant='marginalized' if n.startswith('matched') else 'nominal'
                v=m[n]['classification'][variant]['groups'][group];vals.append(v['average_precision']);counts.append(f"{v['known']} known\n{v['evolved']} evolved")
            else:
                v=m[n]['distance'][group]['chosen'];vals.append(v.get('tail_1'));counts.append(f"N={v['n']}")
        numeric=[np.nan if v is None else v for v in vals]
        ax.bar(range(4),numeric,color=['#888888','#2878b5']*2)
        for i,(value,count) in enumerate(zip(numeric,counts)):
            ax.text(i,(value if np.isfinite(value) else 0)+.025,count,ha='center',fontsize=8)
        ax.set_xticks(range(4),focus_labels,rotation=35,ha='right');ax.set_title(title);ax.set_ylim(0,1)
    fig.suptitle('Each adjacent pair uses the same stars; 20–22 has no held-out evolved labels',fontsize=11)
    fig.savefig(out/'priority.png',dpi=170);plt.close(fig)
    lines=['# Baseline v1.4 results','', '## Scope and evidence','',
           f"All1600 sweeps extracted. Assembled {d['assembly']['unique_reference_rows']:,} unique reference rows before scientific eligibility; raw counts are not qualified labels.",
           f"Actual artifact checks: {d['validation']['n_checks']}; all passed. Deployed v1 remains unchanged.",'',
           '| Model | Test AP | Evolved recall | Known contamination | Chosen distance N | Tail >1mag | Coverage68 | Full Gaia N |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'absent' if x is None else f'{x:.4f}'
    for n in names:
        c=m[n]['classification']['marginalized' if n.startswith('matched') else 'nominal']['groups']['all'];r=m[n]['distance']['all'];v=r['chosen']
        lines.append(f"| {n} | {fmt(c['average_precision'])} | {fmt(c['recall'])} | {fmt(c['known_contamination'])} | {v['n']} | {fmt(v.get('tail_1'))} | {fmt(v.get('coverage68'))} | {r['full_Gaia']['n']} |")
    lines+=['','Classification: old/new comparisons are nominal scores; matched grz/griz uses shared four-band Sobol128 marginalization. Each workpoint is fixed by cal at the same2% dwarf budget; no test tuning.',
            '','Expanded training adds only new train identities to historical training. Each old/new pair has identical new evaluation identities. Matched grz/griz uses only the new i-usable train identities, with identical training/evaluation identities and hyperparameters between the two controls.',
            '', 'A/B/C use different WISE-availability subsets: compare scores within each old/new pair, not across these subsets. The matched grz/griz pair uses exactly the same stars.',
            '', 'The original26000 probability sample covers the southern DECam parent; northern rows are outside this i-band coverage estimate.',
            '', '## Validation limits','',d['validation_limits'],
            '', 'Gaia, SpecDis and SPDist scores are retained separately in the result manifest. The chosen training-distance recipe remains Gaia anchors then SpecDis; SPDist is a separate check.',
            '', 'SpecDis/SPDist error and interval-coverage scores measure agreement with published point estimates; their retained reference uncertainties are not convolved into these point-reference metrics. These are not universal absolute-accuracy claims. The full Gaia likelihood instead convolves the predicted distribution with the signed observed-parallax uncertainty.',
            '', 'Reference-selected population fractions do not estimate parent purity. Zero-ID associations with unknown epochs are flagged, use a0.5arcsec threshold and2arcsec ambiguity veto; systematic matching risk remains.',
            '', 'Figures: classification.png, distance.png, gaia.png, coverage.png, priority.png under figures/baseline_v1_4. Scientific interpretation and visual inspection must be completed before final delivery.','']
    lines+=['## Actual faint reference support','', '| Split | r / WISE | Class labels | Evolved | Chosen distance | SpecDis | SPDist | Gaia likelihood |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,label in [('1','cal'),('2','test')]:
        for group in ['r18_20:no_W1','r20_22','r22_25']:
            c=d['eligibility']['groups'][split][group]
            lines.append(f"| {label} | {group} | {sum(c['classes'].values())} | {c['classes']['1']} | {c['chosen_distance']} | {c['specdis']} | {c['spdist']} | {c['parallax']} |")
    lines+=['','Zero-count cells are unvalidated. Sparse counts are descriptive evidence, not a general accuracy guarantee.','']
    lines+=['## Actual training composition','',
            '| Model | Class training N | Distance training N | Composition |',
            '|---|---:|---:|---|']
    for n in names:
        v=m[n]['training'];c=v['class_train']+v.get('legacy_class_train',0);r=v['distance_train']+v.get('legacy_distance_train',0)
        composition='frozen historical' if n.startswith('frozen') else 'historical plus new' if n.startswith('expanded') else 'new only, matched i-usable'
        lines.append(f"| {n} | {c} | {r} | {composition} |")
    lines+=['','## Priority same-star comparisons','',
            '| Model | Group | Known class N | Evolved N | AP | Recall | Known contamination | Distance N | Tail >1mag | Coverage68 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['r18_20:no_W1','r20_22','r20_22:no_W1','r22_25']:
        for n in ['frozen_A','expanded_A','matched_grz','matched_griz']:
            variant='marginalized' if n.startswith('matched') else 'nominal'
            c=m[n]['classification'][variant]['groups'][group];r=m[n]['distance'][group]['chosen']
            lines.append(f"| {n} | {group} | {c['known']} | {c['evolved']} | {fmt(c['average_precision'])} | {fmt(c['recall'])} | {fmt(c['known_contamination'])} | {r['n']} | {fmt(r.get('tail_1'))} | {fmt(r.get('coverage68'))} |")
    lines+=['','## Separate distance evidence','',
            '| Model | Evidence | N | Bias (mag) | Scatter (mag) | Tail >1mag | Coverage68 | Coverage95 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        for kind in ['Gaia','specdis','spdist']:
            v=m[n]['distance']['all'][kind]
            lines.append(f"| {n} | {kind} | {v['n']} | {fmt(v.get('bias_mag'))} | {fmt(v.get('nmad_mag'))} | {fmt(v.get('tail_1'))} | {fmt(v.get('coverage68'))} | {fmt(v.get('coverage95'))} |")
    lines+=['','## Gaia numerical convergence','',
            'Full Gaia log density and PIT use separately frozen convergence-controlled numerical scores of the unchanged PDFs. The original8-order results remain on NERSC.16/32/64/128-order checks escalate unresolved cases to adaptive integration; target absolute precision1e-5.','',
            '| Model | Gaia cal+test N | Max log-density correction | Mean correction | Max PIT correction |',
            '|---|---:|---:|---:|---:|']
    for n in names:
        v=m[n]['gaia_refinement']
        lines.append(f"| {n} | {v['n']} | {fmt(v['log_abs_difference_max'])} | {fmt(v['log_difference_mean'])} | {fmt(v['pit_abs_difference_max'])} |")
    lines+=['','## Distance tails by known class','',
            'Chosen-distance agreement; unknown classes are retained separately and are not counted as dwarfs.','',
            '| Model | Dwarf N / tail | Evolved N / tail | Other N / tail | Unknown N / tail |',
            '|---|---|---|---|---|']
    for n in names:
        values=[m[n]['distance']['class'+str(k)]['chosen'] for k in [0,1,2,-1]]
        lines.append('| '+n+' | '+' | '.join(f"{v['n']} / {fmt(v.get('tail_1'))}" for v in values)+' |')
    lines+=['','## Weighted WISE measurement states','',
            'Original26000 DECam probability sample; all original weights retained. Low-SNR fluxes were preserved but not passed to Gaussian magnitude models.',
            'States: -1 unmatched; 0 no observation; 1 invalid measurement; 2 quality failure; 3 low S/N including negative flux; 4 usable S/N>=5 color.','',
            '| Band | State | N | Weighted fraction |','|---|---|---:|---:|']
    for band in ['w1','w2']:
        for state,v in coverage['all']['wise'][band].items():
            lines.append(f"| {band} | {state} | {v['n']} | {fmt(v['weighted_fraction'])} |")
    lines+=['','## Identity and coverage exclusions','',
            f"Zero-ID recovery: {d['assembly']['matching']['matched']} accepted matches from {d['assembly']['matching']['identities']} source identities; ambiguous and unmatched sources remain excluded. Raw fluxes and candidate metadata are retained separately.",
            f"Historical identity exclusions: {d['assembly']['historical_rows']}; native collision rows: {d['assembly']['native_collision_rows']}; source/native geometry disagreements: {d['assembly']['geometry_disagreement_rows']}.",'']
    for kind,v in d['zero_vac_quality'].items():
        lines.append(f"{kind}: {v['zero_id_rows']:,} raw rows without valid Gaia IDs; {v['quality_passing']} pass the fixed distance-quality recipe. These raw rows are retained; none supplies a lost qualified distance label.")
    lines.append('')
    lines+=['## Full observed-parallax checks','',
            '| Model | Test N | Negative N | Mean log density | PIT <.05 | PIT >.95 | S/N-bin Brier |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for n in names:
        v=m[n]['distance']['all']['full_Gaia']
        lines.append(f"| {n} | {v['n']} | {v['negative']} | {fmt(v['mean_log_density'])} | {fmt(v['low05'])} | {fmt(v['high95'])} | {fmt(v['snr_brier'])} |")
    lines+=['','Signed observed parallaxes are never inverted into weak-distance labels. Observed and predicted S/N fractions (negative,0–5,5–10,10–20,>=20), selection-conditioned PIT, full predictive mean and per-object scores remain in the NERSC artifacts. Full PIT and likelihood are diagnostics on reference-selected sky; they do not establish an unbiased population calibration.','']
    a=m['matched_grz'];b=m['matched_griz']
    ac=a['classification']['marginalized']['groups'];bc=b['classification']['marginalized']['groups']
    ad=a['distance'];bd=b['distance'];priority='r18_20:no_W1';faint='r20_22:no_W1'
    def pct(x):return f'{100*x:.2f}%'
    ca=ac['all'];cb=bc['all'];da=ad['all']['chosen'];db=bd['all']['chosen']
    old_ap='/'.join(fmt(m['frozen_'+n]['classification']['nominal']['groups']['all']['average_precision']) for n in 'ABC')
    new_ap='/'.join(fmt(m['expanded_'+n]['classification']['nominal']['groups']['all']['average_precision']) for n in 'ABC')
    old_tail='/'.join(pct(m['frozen_'+n]['distance']['all']['chosen']['tail_1']) for n in 'ABC')
    new_tail='/'.join(pct(m['expanded_'+n]['distance']['all']['chosen']['tail_1']) for n in 'ABC')
    faintest=d['eligibility']['groups']['2']['r22_25']
    summary=['## Main findings','',
        '**Retain deployed v1.** The registered broader-reference and optional-i experiments are complete. The fixed tests establish modest and mixed changes, with severe unresolved evolved-star distance failures; they do not justify a general reliability claim or an automatic deployment change.','',
        f"The new geometry supplies {d['eligibility']['groups']['0']['all']['class_labels']:,} class-training labels and {d['eligibility']['groups']['0']['all']['distance_training']:,} distance-training objects. New test has {d['eligibility']['groups']['2']['all']['class_labels']:,} class labels in frozen sky outside the historical fields.",'',
        f'Adding the new training data with fixed algorithms raises A/B/C classification AP from {old_ap} to {new_ap}; chosen-distance severe tails decrease only modestly, from {old_tail} to {new_tail}. The priority18–20/no-W1 interval coverage does not improve in this data-only control.','',
        f"On the matched i-usable stars, adding i changes marginalized class AP from{fmt(ca['average_precision'])} to{fmt(cb['average_precision'])}, evolved recall from{pct(ca['recall'])} to{pct(cb['recall'])}, and known-label contamination from{pct(ca['known_contamination'])} to{pct(cb['known_contamination'])}. On the same {da['n']:,} distance references, severe tails change from{pct(da['tail_1'])} to{pct(db['tail_1'])};68% coverage is {pct(da['coverage68'])} versus {pct(db['coverage68'])}.",'',
        f"The matched full Gaia comparison retains {ad['all']['full_Gaia']['n']:,} test objects, including {ad['all']['full_Gaia']['negative']:,} negative parallaxes. Mean log predictive density rises from{fmt(ad['all']['full_Gaia']['mean_log_density'])} to{fmt(bd['all']['full_Gaia']['mean_log_density'])}, but the combined PIT<.05/>.95 fraction rises slightly from{pct(ad['all']['full_Gaia']['low05']+ad['all']['full_Gaia']['high95'])} to{pct(bd['all']['full_Gaia']['low05']+bd['all']['full_Gaia']['high95'])}. The likelihood and tail diagnostics therefore do not show a uniform improvement.",'',
        f"The priority18–20/no-W1 comparison has {ac[priority]['evolved']} evolved class labels: AP changes from{fmt(ac[priority]['average_precision'])} to{fmt(bc[priority]['average_precision'])}, while distance tails change from{pct(ad[priority]['chosen']['tail_1'])} to{pct(bd[priority]['chosen']['tail_1'])}. On the {ad[faint]['chosen']['n']} matched20–22/no-W1 distance references, the severe-error count changes from{round(ad[faint]['chosen']['tail_1']*ad[faint]['chosen']['n'])} to{round(bd[faint]['chosen']['tail_1']*bd[faint]['chosen']['n'])}; this sparse descriptive improvement is not a broad faint-star accuracy guarantee.",'',
        f"Most importantly, on {ad['class1']['chosen']['n']} same-star evolved distance references, grz/griz severe tails remain {pct(ad['class1']['chosen']['tail_1'])}/{pct(bd['class1']['chosen']['tail_1'])}, with68% coverage only {pct(ad['class1']['chosen']['coverage68'])}/{pct(bd['class1']['chosen']['coverage68'])}. Overall averages obscure this failure. Adding one optical color has not resolved the luminosity-branch problem in the fixed algorithm.",'',
        f"Weighted i availability is {pct(coverage['all']['weighted_i_fraction'])} overall and {pct(coverage['no_W1']['weighted_i_fraction'])} without usable W1. Coverage is sufficient for the experiment, but is not evidence of distance accuracy. New20–22 cal/test has no evolved class labels;22–25 test has {faintest['class_labels']} reliable class labels and only{faintest['chosen_distance']} chosen-distance/{faintest['parallax']} Gaia references. Broad22–25 validation remains absent.",'',
        'Stop this registered experiment here. Preserve the new frozen references, selection/provenance fields, raw weak fluxes and all negative results for the next scientific review. Do not launch another model family, tune candidate cuts, rerun the parent catalog or replace v1 automatically.','']
    lines[2:2]=summary
    delivery_path=ROOT/'manifests/baseline_v1_4_delivery.json'
    if delivery_path.exists():
        import hashlib
        delivery=json.loads(delivery_path.read_text())
        verified=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in delivery['inspected_figure_sha256'].items())
        if verified:
            lines=[line.replace('Scientific interpretation and visual inspection must be completed before final delivery.',
                                'All five figures were visually inspected; exact image hashes and validation evidence are recorded in manifests/baseline_v1_4_delivery.json.') for line in lines]
    lines+=['## Figures','']
    for name,caption in [('coverage','Weighted parent i availability and actual new-test evidence'),
                         ('classification','Classification comparisons within each fixed pair'),
                         ('distance','Same-star chosen-distance errors and interval coverage'),
                         ('gaia','Signed-parallax PIT and full predictive likelihood'),
                         ('priority','Priority faint/no-WISE comparisons with actual sample counts')]:
        lines += [f'![{caption}](../figures/baseline_v1_4/{name}.png)','',caption+'.','']
    import re
    text='\n'.join(lines)
    text=re.sub(r'\b(from|to|with|only|same|New|new|matched|priority|All|all)(?=\d)',r'\1 ',text)
    text=re.sub(r';\s*','; ',text)
    (ROOT/'docs/baseline_v1_4_result.md').write_text(text)

if __name__=='__main__':main()
