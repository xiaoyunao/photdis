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
    for ax,key,title in zip(axes,['average_precision','recall','known_contamination'],['Evolved average precision','Evolved recall at cal 2% dwarf budget','Known-label contamination']):
        values=[m[n]['classification']['marginalized' if n.startswith('matched') else 'nominal']['groups']['all'][key] for n in names]
        ax.bar(np.arange(len(names)),[np.nan if v is None else v for v in values],color=colors)
        ax.set_xticks(np.arange(len(names)),labels,rotation=55,ha='right');ax.set_title(title);ax.set_ylim(0,1)
    fig.savefig(out/'classification.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes,['tail_1','coverage68','coverage95'],['Distance |error| > 1 mag','Distance 68% coverage','Distance 95% coverage']):
        for group,offset,col in [('all',-.18,'#2878b5'),('no_W1',.18,'#e1812c')]:
            vals=[m[n]['distance'][group]['chosen'].get(key,np.nan) for n in names]
            ax.bar(np.arange(len(names))+offset,vals,width=.34,label=group,color=col)
        ax.set_xticks(np.arange(len(names)),labels,rotation=55,ha='right');ax.set_title(title);ax.set_ylim(0,1)
    axes[0].legend();fig.savefig(out/'distance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for name,label in [('matched_grz','grz'),('matched_griz','griz')]:
        ga=m[name]['distance']['all']['full_Gaia'];h=np.array(ga['pit_histogram']);axes[0].stairs(h/max(1,h.sum()),np.linspace(0,1,11),label=label)
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
    axes[1].set(xticks=range(4),xticklabels=['<18','18–20','20–22','22–25'],yscale='symlog',ylabel='New test reference count',xlabel='r magnitude');axes[1].legend()
    fig.savefig(out/'coverage.png',dpi=170);plt.close(fig)
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
            '','Expanded training adds only new train identities to historical training. Each old/new pair has identical new evaluation identities. Matched grz/griz has identical training/evaluation identities and hyperparameters.',
            '', 'A/B/C use different WISE-availability subsets: compare scores within each old/new pair, not across these subsets. The matched grz/griz pair uses exactly the same stars.',
            '', 'The original26000 probability sample covers the southern DECam parent; northern rows are outside this i-band coverage estimate.',
            '', '## Validation limits','',d['validation_limits'],
            '', 'Gaia, SpecDis and SPDist scores are retained separately in the result manifest. The chosen training-distance recipe remains Gaia anchors then SpecDis; SPDist is a separate check.',
            '', 'Reference-selected population fractions do not estimate parent purity. Zero-ID associations with unknown epochs are flagged, use a0.5arcsec threshold and2arcsec ambiguity veto; systematic matching risk remains.',
            '', 'Figures: classification.png, distance.png, gaia.png, coverage.png under figures/baseline_v1_4. Scientific interpretation and visual inspection must be completed before final delivery.','']
    lines+=['## Actual faint reference support','', '| Split | r / WISE | Class labels | Evolved | Chosen distance | SpecDis | SPDist | Gaia likelihood |', '|---|---|---:|---:|---:|---:|---:|---:|']
    for split,label in [('1','cal'),('2','test')]:
        for group in ['r18_20:no_W1','r20_22','r22_25']:
            c=d['eligibility']['groups'][split][group]
            lines.append(f"| {label} | {group} | {sum(c['classes'].values())} | {c['classes']['1']} | {c['chosen_distance']} | {c['specdis']} | {c['spdist']} | {c['parallax']} |")
    lines+=['','Zero-count cells are unvalidated. Sparse counts are descriptive evidence, not a general accuracy guarantee.','']
    lines+=['## Priority same-star comparisons','',
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
    (ROOT/'docs/baseline_v1_4_result.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
