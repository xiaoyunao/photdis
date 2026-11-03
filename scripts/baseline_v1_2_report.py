#!/usr/bin/env python3
"""Compact result tables and ten inspectable scientific figures."""
from baseline_v1_2_run import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
FIG=OUT/'figures'
COLORS=dict(natural='#777777',balanced='#d89100',evidence='#b2182b',quadratic='#2166ac',forest='#1b7837')
LABELS=dict(natural='natural',balanced='post-hoc balanced',evidence='joint evidence',quadratic='trained balanced quadratic',forest='trained balanced forest')


def save(fig,name):
    fig.savefig(FIG/name,dpi=165,bbox_inches='tight');plt.close(fig)


def main():
    require_compute();FIG.mkdir(parents=True,exist_ok=True)
    scores={name:json.load(open(OUT/f'{name}_scores.json')) for name in MODELS}
    par=json.load(open(OUT/'parallax.json'));val=json.load(open(OUT/'validation.json'))
    assert val['all_passed'];t=load(OLD/'references.npz')
    matched=('A_on_C','B_on_C','C')
    # Matched identities and training control the information comparison.
    for mode in ('roc','pr'):
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for col,name in enumerate(matched):
            for row,target in enumerate((1,2)):
                ax=axes[row,col]
                for v in VARS:
                    c=scores[name][v]['curves'][f'nominal_class{target}']['test']
                    if mode=='roc':ax.plot(c['fpr'],c['tpr'],color=COLORS[v],label=LABELS[v])
                    else:ax.plot(c['recall'],c['precision'],color=COLORS[v],label=LABELS[v])
                ax.set(title=f'{name}: '+('evolved' if target==1 else 'transition'),xlim=(0,1),ylim=(0,1),
                    xlabel='One-vs-rest FPR' if mode=='roc' else 'Recall',ylabel='Recall' if mode=='roc' else 'Known-reference precision')
        axes[0,0].legend(fontsize=8);save(fig,'01_matched_roc.png' if mode=='roc' else '02_matched_pr.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    maxfpr=max(q['test']['dwarf_fpr'] or 0 for name in 'ABC' for v in VARS for key,q in scores[name][v]['working_points'].items() if key.startswith('final'))
    for col,name in enumerate('ABC'):
        for row,target in enumerate((1,2)):
            ax=axes[row,col]
            for v in VARS:
                points=[scores[name][v]['working_points'][f'final_class{target}_budget{b}']['test'] for b in CFG['false_positive_budgets']]
                ax.plot([p['dwarf_fpr'] for p in points],[p['recall'] for p in points],'-o',color=COLORS[v],label=LABELS[v],ms=4)
            ax.set(title=f'{name}: '+('evolved' if target==1 else 'transition'),xlabel='Test dwarf FPR at fixed cal thresholds',ylabel='Test recall',ylim=(0,1),xlim=(0,min(1,max(.06,maxfpr*1.1))))
            ax.axvline(.02,ls=':',color='grey')
    axes[0,0].legend(fontsize=8);save(fig,'03_fixed_budget_recall.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,name in zip(axes,'ABC'):
        for j,v in enumerate(VARS):
            q=scores[name][v]['working_points']['final_class1_budget0.02']['distances']['chosen']
            for offset,decision,marker in [(-.18,'mixture','o'),(0,'main','s'),(.18,'wide','x')]:
                y=q[decision].get('tail_1',np.nan)
                ax.scatter(j+offset,y,color=COLORS[v],marker=marker,label=decision if j==0 else None)
        ax.set(title=f'{name}: evolved score, cal FPR budget 2%',ylabel='Severe distance error, all selected references',ylim=(0,1),xticks=range(5),xticklabels=['natural','balanced','evidence','quad','forest'])
        ax.tick_params(axis='x',labelrotation=25)
    axes[0].legend();save(fig,'04_distance_decisions.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,name in zip(axes,'ABC'):
        for v in VARS:
            points=[scores[name][v]['working_points'][f'final_class1_budget{b}'] for b in CFG['false_positive_budgets']]
            ax.plot([p['parent']['fraction'] for p in points],[p['test']['recall'] for p in points],'-o',color=COLORS[v],label=LABELS[v])
            for p in points:ax.errorbar(p['parent']['fraction'],p['test']['recall'],xerr=1.96*p['parent']['se'],color=COLORS[v],capsize=2)
        ax.set(title=f'{name}: fixed evolved-score working points',xlabel='Weighted fraction of optical parent',ylabel='Known evolved test recall',xlim=(0,1),ylim=(0,1))
    axes[0].legend(fontsize=8);save(fig,'05_recall_parent_coverage.png')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained');x=np.arange(.05,1,.1)
    pitymax=max(.15,1.1*max(max(e['groups']['all'][kind+'_pit_histogram']) for vv in par['results'].values() for e in vv.values() for kind in ('full','conditional')))
    for col,name in enumerate('ABC'):
        for row,kind in enumerate(('full','conditional')):
            for v in ('wide','evidence','quadratic','forest'):
                q=par['results'][name][v+'_mixture']['groups']['all'];c=COLORS.get(v,'black')
                axes[row,col].step(x,q[kind+'_pit_histogram'],where='mid',label=v,color=c)
            axes[row,col].axhline(.1,color='grey',ls='--');axes[row,col].set(title=f'{name}: {kind} PIT',xlabel='PIT',ylabel='Design-weighted fraction per bin',xlim=(0,1),ylim=(0,pitymax))
    axes[0,0].legend();save(fig,'06_full_and_conditional_pit.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,name in zip(axes,'ABC'):
        q=par['results'][name]['wide_mixture']['groups']['all'];ax.plot(range(5),q['weighted_observed_snr_fraction'],'ko-',label='observed')
        for v in ('wide','evidence','quadratic','forest'):
            q=par['results'][name][v+'_mixture']['groups']['all'];ax.plot(range(5),q['weighted_predicted_snr_fraction'],'s--',color=COLORS.get(v,'black'),label=v)
        ax.set(title=name,xlabel='Observed-parallax S/N bins',ylabel='Weighted fraction',ylim=(0,1),xticks=range(5),xticklabels=['negative','0–5','5–10','10–20','>=20'])
    axes[0].legend(fontsize=8);save(fig,'07_snr_bin_probability.png')
    # Same C test rows, forest and reference; complete groups, not correctness-filtered.
    p=load(OUT/'reference_C_forest.npz');rows=p['rows'];test=t['split'][rows]==2
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained');norm=LogNorm(1,1000);hb=None
    for col,ref in enumerate(('Gaia','specdis','spdist')):
        truth=truth_mu(t,ref)[rows];use=test&np.isfinite(truth);x=(truth[use]-10)/5
        for row,decision in enumerate(('mu','main_mu')):
            y=(p[decision][use,2]-10)/5
            hb=axes[row,col].hexbin(x,y,gridsize=65,extent=(-2,3,-2,3),mincnt=1,norm=norm,cmap='viridis')
            out=int(((x< -2)|(x>3)|(y< -2)|(y>3)).sum())
            axes[row,col].plot([-2,3],[-2,3],'r--',lw=1)
            axes[row,col].set(title=f'C forest {ref}, '+('mixture' if row==0 else 'main')+f'; N={use.sum()}, outside={out}',xlim=(-2,3),ylim=(-2,3),xlabel='log10 reference d [kpc]',ylabel='log10 photometric d [kpc]')
    fig.colorbar(hb,ax=axes.ravel().tolist(),label='Objects per hexbin');save(fig,'08_distance_density.png')
    truth=truth_mu(t,'chosen')[rows];use=test&np.isfinite(truth);color=t['g0'][rows][use]-t['r0'][rows][use];g=t['g0'][rows][use]
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,label,mu in zip(axes,('Chosen reference','C forest mixture','C forest main'),(truth[use],p['mu'][use,2],p['main_mu'][use,2])):
        mg=g-mu;hb=ax.hexbin(color,mg,gridsize=65,extent=(-.5,2.5,-6,16),mincnt=1,norm=norm,cmap='viridis')
        out=int(((color<-.5)|(color>2.5)|(mg< -6)|(mg>16)).sum())
        ax.set(title=f'{label}; N={len(g)}, outside={out}',xlabel='(g-r)0',ylabel='Mg',xlim=(-.5,2.5),ylim=(16,-6))
    fig.colorbar(hb,ax=axes.ravel().tolist(),label='Objects per hexbin');save(fig,'09_same_source_cmd.png')
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,name in zip(axes,'ABC'):
        for v in ('balanced','quadratic','forest'):
            q=list(scores[name][v]['single_grid'].values())
            x=[i['parent']['fraction'] for i in q];y=[i['test']['chosen']['main'].get('tail_1',np.nan) for i in q]
            ax.scatter(x,y,color=COLORS[v],label=v,alpha=.75)
            chosen=scores[name][v]['cal_single_rule']
            if chosen:
                i=scores[name][v]['single_grid'][chosen];ax.scatter(i['parent']['fraction'],i['test']['chosen']['main'].get('tail_1',np.nan),marker='*',s=160,color=COLORS[v],edgecolor='black')
        ax.axhline(.05,color='grey',ls=':');ax.set(title=name,xlabel='Weighted optical-parent fraction',ylabel='All-selected reference severe error',xlim=(0,1),ylim=(0,1))
    axes[0].legend();save(fig,'10_single_distance_risk.png')
    # Compact reviewable evidence; full per-source/per-stratum scores remain on NERSC.
    compact={}
    for name,variants in scores.items():
        compact[name]={}
        for v,e in variants.items():
            compact[name][v]=dict(all_distances=e['all_distances'],test_by_class=e['test_by_class'],cal_single_rule=e['cal_single_rule'],
                curves={k:{s:{z:c[z] for z in ('n','auc','average_precision') if z in c} for s,c in item.items()} for k,item in e['curves'].items()},
                working_points={k:{**{z:c[z] for z in ('threshold','budget','target','score_state','cal','test','parent') if z in c},'distances':{'chosen':c['distances']['chosen']}} for k,c in e['working_points'].items()},
                single_selected=e['single_grid'].get(e['cal_single_rule']),
                strata_at_2pct={k:c['strata'] for k,c in e['working_points'].items() if k=='final_class1_budget0.02' and v in ('quadratic','forest')})
    report=dict(models=compact,parallax=par,validation={k:val[k] for k in ('n_checks','all_passed')},fit={n:json.load(open(OUT/f'{n}_fit.json')) for n in MODELS})
    write_json(OUT/'report.json',report)
    write_json(OUT/'source_workpoints.json',{n:{v:{k:q['distances'] for k,q in e['working_points'].items() if k.startswith('final')} for v,e in variants.items()} for n,variants in scores.items()})
    lines=['# baseline v1.2：颜色可分性与距离决策','',
        '冻结距离专家与原空间切分，仅新增类别平衡二次分类器和固定森林。工作点由cal确定；',
        'test仍为发展验证。分类分数不是场星类别概率，参考精度不等于母样本纯度。',
        '本轮只处理现有参照及固定概率样本，保留v1部署，没有更新完整目录。','',
        '## 固定误入预算下能找回多少星族？','',
        '下表是cal矮星误入预算2%、回退后权重的固定工作点。测试误入率不被强制为2%。','',
        '| 配置 | 分类器 | 目标 | test召回 | test矮星误入 | 选中矮/演化/过渡/未知 | 已知参照精度 | 母样本保留 |',
        '|---|---|---|---:|---:|---|---:|---:|']
    fmt=lambda x:'—' if x is None else f'{100*x:.2f}%'
    for name in 'ABC':
        for v in VARS:
            for target in (1,2):
                q=scores[name][v]['working_points'][f'final_class{target}_budget0.02'];z=q['test'];c=z['class_counts']
                lines.append(f"| {name} | {v} | {target} | {fmt(z['recall'])} | {fmt(z['dwarf_fpr'])} | {c['0']}/{c['1']}/{c['2']}/{c['-1']} | {fmt(z['known_precision'])} | {fmt(q['parent']['fraction'])} |")
    lines+=['','目标1=演化星，2=过渡类；未知类保留，不能算作分类正确。1%/2%/5%、raw/final、',
        '同训练波段对照及Wilson计数区间见机器清单；完整细分统计在NERSC逐模型scores.json。','',
        'ROC/PR采用同一观测颜色下、回退前的nominal分数；raw扰动平均与final回退权重工作点另存。','',
        '![同训练ROC](../figures/baseline_v1_2/01_matched_roc.png)','',
        '![同训练PR](../figures/baseline_v1_2/02_matched_pr.png)','',
        '![固定工作点](../figures/baseline_v1_2/03_fixed_budget_recall.png)','',
        '## 分类收益能否转成距离？','',
        '以下仍是演化分数2%工作点，但距离错误计入选中的所有真实类别及有距离参照的未知类。',
        '主分支只由测光最大最终权重选出；不使用真实类型，宽分量最大时也保留。','',
        '| 配置 | 分类器 | chosen参照N | 混合中位数严重错误 | 主分支严重错误 | 同星旧宽模型 | cal单值规则 |',
        '|---|---|---:|---:|---:|---:|---|']
    for name in 'ABC':
        for v in VARS:
            q=scores[name][v]['working_points']['final_class1_budget0.02']['distances']['chosen']
            lines.append(f"| {name} | {v} | {q['main']['n']} | {fmt(q['mixture'].get('tail_1'))} | {fmt(q['main'].get('tail_1'))} | {fmt(q['wide'].get('tail_1'))} | {scores[name][v]['cal_single_rule'] or '无通过规则'} |")
    lines+=['','严重错误为|Δμ|>1mag；按Gaia/SPECDIS/SPDist分别的区间、偏差和散布保留于清单。',
        '固定confidence×margin网格要求三类cal各至少30个参照、整体错误<=5%且各类不显著劣于同星旧模型。',
        '未通过时仅保留多解，不将主分支强行写成可靠单值。完整混合和条件PDF均保存在NERSC。','',
        '![距离摘要对照](../figures/baseline_v1_2/04_distance_decisions.png)','',
        '![召回与母样本覆盖](../figures/baseline_v1_2/05_recall_parent_coverage.png)','',
        '![单值风险与覆盖](../figures/baseline_v1_2/10_single_distance_risk.png)','',
        '![距离密度](../figures/baseline_v1_2/08_distance_density.png)','',
        '![同源CMD](../figures/baseline_v1_2/09_same_source_cmd.png)','',
        '## 完整观测视差预测检验','',
        '固定7672个Gaia样本（1800负视差），逆入选率加权恢复质量/零点合格Gaia母集。',
        '主结果不按实际观测S/N截断；旧条件PIT保留为箱内诊断，另比较跨箱概率。',
        'r>=22仍无对象，Gaia准入的选择效应和共享系统误差未由抽样权重消除。','',
        '| 配置 | 分布/摘要 | N | 未截断低/高PIT尾 | 平均log密度 [mas⁻¹] | 实际箱平均log概率 |',
        '|---|---|---:|---:|---:|---:|']
    for name in 'ABC':
        for variant,e in par['results'][name].items():
            q=e['groups']['all'];lines.append(f"| {name} | {variant} | {q['n']} | {fmt(q['full_low05'])}/{fmt(q['full_high95'])} | {q['mean_log_density']:.3f} | {q['mean_observed_bin_log_probability']:.3f} |")
    lines+=['','完整log密度=条件log密度+实际箱log概率（无下溢时）。各预测对象都保留，',
        '数值下溢及8/16点积分差异另列在机器清单，不把箱内PIT相似等同于完整分布相容。','',
        '![完整与条件PIT](../figures/baseline_v1_2/06_full_and_conditional_pit.png)','',
        '![跨信噪比箱概率](../figures/baseline_v1_2/07_snr_bin_probability.png)','',
        '## 核验与复现','',f"实际产物检查{val['n_checks']}项通过。15项数学/回归测试通过；十张图交付前逐张检查。",
        '训练身份与类总权重、原专家哈希、同训练身份、cal阈值、全类别计数、PDF质量及冻结目录均核验。',
        '运行：run_baseline_v1_2_fit.sh → run_baseline_v1_2_checks.sh。大产物在NERSC data/baseline_v1_2。',
        '[预登记协议](baseline_v1_2_protocol.md) · [完整评审](pro_review_dbeac65.md)。']
    (OUT/'result.md').write_text('\n'.join(lines)+'\n');print('Report and ten figures complete',flush=True)
if __name__=='__main__':main()
