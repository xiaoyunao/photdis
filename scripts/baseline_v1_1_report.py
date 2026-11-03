#!/usr/bin/env python3
"""Render bounded v1.1 actual results, including failures and sample scope."""
from baseline_v1_1_common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from photdis.stage1 import require_compute
from baseline_v1_1_risk import estimate
FIG=OUT/'figures'


def save(fig,name):
    fig.tight_layout();fig.savefig(FIG/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)

def fmt(x,n=3):return '—' if x is None else f'{x:.{n}f}'

def main():
    require_compute();FIG.mkdir(exist_ok=True)
    report=json.load(open(OUT/'reference_scores.json'));selection=json.load(open(OUT/'selection.json'))
    risk=json.load(open(OUT/'risk.json'));par=json.load(open(OUT/'parallax.json'));samples=json.load(open(OUT/'samples.json'))
    routed=json.load(open(OUT/'routed_scores.json'));valid=json.load(open(OUT/'validation.json'))
    t=load(OLD/'references.npz')
    variants=('wide','oracle','oracle_support','oracle_full','natural','balanced','evidence')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,cl in zip(axes,('class_dwarf','class_evolved','class_transition')):
        for v in variants:
            val=[report[n]['scores'][v]['chosen']['test'][cl].get('tail_1',np.nan) for n in 'ABC']
            ax.plot(range(3),val,'o-',label=v)
        ax.set(xticks=range(3),xticklabels=list('ABC'),ylim=(0,1),ylabel='Fraction |delta mu| > 1 mag',title=cl)
    axes[0].legend(fontsize=7);save(fig,'01_fallback_population')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,source in zip(axes,('Gaia','specdis')):
        for v in ('wide','evidence'):
            vals=[report[n]['scores'][v][source]['test']['all'].get('nmad_mag',np.nan) for n in ('A_on_C','B_on_C','C')]
            ax.plot(range(3),vals,'o-',label=v)
        ax.set(xticks=range(3),xticklabels=['grz','grzW1','grzW1W2'],ylabel='Delta-mu NMAD [mag]',title=source+'; identical training/test objects');ax.legend()
    save(fig,'02_matched_colors')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,'ABC'):
        for key,entry in risk[name]['groups'].items():
            if not key.startswith('cut_'):continue
            cal=entry['chosen']['cal'];test=entry['chosen']['test'];parent=entry['parent']
            if cal['n']<30:continue
            x=parent['fraction'];ax.errorbar(x,cal['tail_1'],xerr=1.96*parent['se'],fmt='o',color='tab:blue')
            if test['n']:ax.plot([x,x],[cal['tail_1'],test['tail_1']],c='gray',alpha=.5);ax.scatter(x,test['tail_1'],marker='x',color='tab:orange')
        ax.plot([],[],'o',color='tab:blue',label='cal');ax.plot([],[],'x',color='tab:orange',label='development test')
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Estimated fraction of optical parent',ylabel='All-reference severe-error rate',title=name+'; fixed prediction cuts');ax.legend(fontsize=8)
    save(fig,'03_risk_retention')

    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for j,source in enumerate(('Gaia','specdis','spdist')):
        for i,v in enumerate(('wide','evidence')):
            p=load(OUT/f'routed_{v}_evaluation.npz');rows=p['rows'];truth=truth_mu(t,source)[rows]
            use=(t['split'][rows]==2)&np.isfinite(truth);x=(truth[use]-10)/5;y=(p['mu'][use,2]-10)/5
            ax=axes[i,j];im=ax.hexbin(x,y,gridsize=65,extent=(-2,2,-2,2),mincnt=1,norm=LogNorm(vmin=1,vmax=1000),cmap='viridis')
            ax.plot([-2,2],[-2,2],'r--',lw=1);ax.set(xlim=(-2,2),ylim=(-2,2),xlabel='log10 reference distance [kpc]',ylabel='log10 photometric distance [kpc]',title=f'{source} / {v}, N={use.sum()}')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Sources per bin',shrink=.7);fig.subplots_adjust(right=.85,wspace=.25,hspace=.35)
    fig.savefig(FIG/'04_distance_density.png',dpi=150);plt.close(fig)

    p=load(OUT/'routed_wide_evaluation.npz');new=load(OUT/'routed_evidence_evaluation.npz');rows=p['rows'];truth=truth_mu(t,'chosen')[rows]
    use=(t['split'][rows]==2)&np.isfinite(truth);color=t['g0'][rows][use]-t['r0'][rows][use]
    mg=[t['g0'][rows][use]-a for a in (truth[use],p['mu'][use,2],new['mu'][use,2])]
    fig,axes=plt.subplots(1,3,figsize=(14,4),sharex=True,sharey=True)
    for ax,y,title in zip(axes,mg,('Reference','Wide QRF','Evidence gate')):
        im=ax.hexbin(color,y,gridsize=65,extent=(-.5,2.5,-5,15),mincnt=1,norm=LogNorm(vmin=1,vmax=1000),cmap='viridis')
        ax.set(xlim=(-.5,2.5),ylim=(15,-5),xlabel='(g-r)0',ylabel='Mg [mag]',title=f'{title}; same {use.sum()} sources')
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Sources per bin',shrink=.75);fig.subplots_adjust(right=.86,wspace=.25)
    fig.savefig(FIG/'05_same_source_cmd.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,'ABC'):
        p=load(OUT/f'{name}_evidence_evaluation.npz');rows=p['rows'];use=t['split'][rows]==2
        im=ax.hexbin(p['unknown_fraction'][use],p['weights'][use,3],gridsize=35,extent=(0,1,0,1),mincnt=1,bins='log')
        ax.plot([0,1],[0,1],'r--');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Leaf missing-class fraction',ylabel='New final broad-component weight',title=name+'; different quantities')
    save(fig,'06_missing_labels_vs_residual')

    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,name in zip(axes,'ABC'):
        for v in ('wide','evidence'):
            d=par['results'][name][v]['all'];ax.step(np.linspace(.05,.95,10),d['pit_histogram'],where='mid',label=f"{v}, N={d['n']}")
        ax.axhline(.1,c='gray',ls='--');ax.set(xlim=(0,1),xlabel='Observed-SNR-conditioned PIT',ylabel='Design-weighted fraction per bin',title=name+' / quality Gaia test sample');ax.legend(fontsize=8)
    save(fig,'07_observed_parallax')

    sample=load(OUT/'parent_sample.npz');fig,axes=plt.subplots(1,2,figsize=(11,4))
    mids=np.arange(13,25,2)
    for name in 'ABC':
        ps=load(OUT/f'parent_{name}_evidence.npz');use=np.zeros(len(sample['identity']),bool);use[ps['rows']]=True
        supported=np.zeros_like(use);supported[ps['rows']]=ps['supported']
        for ax,mask,label in [(axes[0],use,name+' band available'),(axes[1],supported,name+' available + nominal support')]:
            values=[];errors=[]
            for lo in range(12,24,2):
                d=estimate(sample,mask,(sample['r']>=lo)&(sample['r']<lo+2));values.append(np.nan if d['fraction'] is None else d['fraction']);errors.append(np.nan if d['se'] is None else 1.96*d['se'])
            ax.errorbar(mids,values,yerr=errors,marker='o',label=label)
    for ax in axes:ax.set(ylim=(0,1.05),xlabel='Observed r magnitude',ylabel='Estimated optical-parent fraction');ax.legend(fontsize=8)
    save(fig,'08_parent_coverage')

    lines=['# baseline v1.1：分支回退与联合监督实验','',
        '本轮是冻结v1专家、原空间切分和Sobol128后的有限比较。没有重算完整目录；',
        'test是已使用过的发展验证，区间和得分主要检查带噪教师相容性，不代表独立绝对距离校准。','',
        '## 采用决定','', '| 配置 | cal采用新门控 | 三类宏平均严重错误改善 | 未通过的条件 |','|---|---|---:|---|']
    for n,d in selection['models'].items():lines.append(f"| {n} | {d['adopt']} | {fmt(d['macro_tail_gain'])} | {', '.join(k for k,v in d['checks'].items() if not v) or '无'} |")
    adopted=[k for k,v in selection['models'].items() if v['adopt']]
    lines+=['',f"通过预登记cal门槛的配置：{', '.join(adopted) or '无'}。本轮仅给出采用建议，v1内部目录保持原样。",'',
        '## 同对象回退分解','', 'C配置、chosen参照的演化星发展测试；oracle只用于诊断。',
        '| 分布 | N | 中位Δμ [mag] | 严重错误率 | 95%宽度 [mag] |','|---|---:|---:|---:|---:|']
    for v in variants:
        d=report['C']['scores'][v]['chosen']['test']['class_evolved'];lines.append(f"| {v} | {d['n']} | {fmt(d.get('bias_mag'))} | {fmt(d.get('tail_1'))} | {fmt(d.get('width95'))} |")
    lines+=['','直接oracle→支持回退→完整回退使用完全相同的对象和测光扰动。差值用于定位机制，',
        '不把它们强行解释为可相加的因果百分比。unknown_fraction是缺类别标签份额；',
        '新门控第四分量是宽经验分布权重，两者均不是第四种恒星类别概率。','',
        '![分星族严重错误与回退对照](../figures/baseline_v1_1/01_fallback_population.png)','',
        '![同训练、同测试对象的波段增益](../figures/baseline_v1_1/02_matched_colors.png)','',
        '![缺类别标签与宽分布权重的差异](../figures/baseline_v1_1/06_missing_labels_vs_residual.png)','',
        '## 固定v1波段路由下的实际距离对照','',
        '| 参照 | 分布 | N | 中位Δμ | NMAD | 严重错误率 | 95%覆盖 |','|---|---|---:|---:|---:|---:|---:|']
    for source in ('Gaia','specdis','spdist'):
        for v in ('wide','evidence','selected'):
            d=routed['scores'][v][source]['test']['all'];lines.append(f"| {source} | {v} | {d['n']} | {fmt(d.get('bias_mag'))} | {fmt(d.get('nmad_mag'))} | {fmt(d.get('tail_1'))} | {fmt(d.get('coverage95'))} |")
    lines+=['','![光度距离与Gaia/光谱距离](../figures/baseline_v1_1/04_distance_density.png)','',
        '![相同测试源的参照和预测CMD](../figures/baseline_v1_1/05_same_source_cmd.png)','',
        '## 预测状态、污染与覆盖','',
        f"固定抽样{samples['parent']['n']:,}颗，来自13个DECam光学母样本场；按场规模加权。",
        '横轴置信区间仅反映有限母样本抽样误差，不包含模型或天区迁移系统误差。',
        '固定筛选组包含其混入的所有真实星族；cal少于30参照者不作可靠性判定。',
        '没有参照的母样本对象不能被计为预测正确。详细联合状态、来源和污染计数见risk清单。','',
        '![风险与母样本保留率](../figures/baseline_v1_1/03_risk_retention.png)','',
        '![母样本波段覆盖与名义支持](../figures/baseline_v1_1/08_parent_coverage.png)','',
        '## 真实观测视差检查','',
        f"质量与零点有效的Gaia测试母集{samples['parallax']['eligible']:,}颗，分层抽样{samples['parallax']['n']:,}颗，其中负视差{samples['parallax']['negative']:,}颗。",
        '不使用正视差或高S/N准入；在每个观测S/N区间条件化预测CDF和256次模拟。',
        '积分保留g与颜色的共享误差；不再次叠加g误差。每箱8点积分并以16点复核。',
        '模拟针对固定逐星经验PDF，没有加入共享Gaia/模型系统误差，其p值不是最终科学校准证明。','',
        '| 配置 | 分布 | N | 低PIT<0.05 | 高PIT>0.95 | 条件中心残差 [mas] |','|---|---|---:|---:|---:|---:|']
    for n in 'ABC':
        for v in ('wide','evidence'):
            d=par['results'][n][v]['all'];lines.append(f"| {n} | {v} | {d['n']} | {fmt(d.get('low05'))} | {fmt(d.get('high95'))} | {fmt(d.get('mean_center_residual_mas'))} |")
    lines+=['','![条件化观测视差PIT](../figures/baseline_v1_1/07_observed_parallax.png)','',
        '## 验证与复现','',
        f"实际产物核验{valid['n_checks']}项通过。训练证据来自四折空间留出，不含自身训练标签；",
        '同训练波段比较共用证据身份与权重，cal/test未进入门控拟合。旧预测复现、',
        '权重/PDF归一化、身份隔离、负视差保留和v1目录哈希均已核验。',
        '代码入口：run_baseline_v1_1_fit.sh → run_baseline_v1_1_checks.sh；',
        '固定样本由run_baseline_v1_1_samples.sh单独准备。大产物保留在NERSC data/baseline_v1_1。',
        '预登记方案见[协议](baseline_v1_1_protocol.md)；完整来源见[Pro评审](pro_review_97c2775.md)。',
        '新门控是本轮唯一新增方案。后续应按本轮实测的回退损失和星族代价确定工作，',
        '不自动重新跑九百万行目录、扩展星团或搜索模型族。','']
    (OUT/'result.md').write_text('\n'.join(lines))
    # Full fine-grained scores stay on NERSC; retain inspectable key populations
    # and every oracle decomposition in the public bounded manifest.
    kept={'all','class_unknown','class_dwarf','class_evolved','class_transition','no_W1','train_precision'}
    for model in report.values():
        for variant in model['scores'].values():
            for source in variant.values():
                for split,layers in source.items():source[split]={k:v for k,v in layers.items() if k in kept}
    compact=dict(selection=selection,samples=samples,routed=routed,parallax=par,validation=dict(n_checks=valid['n_checks'],all_passed=valid['all_passed']),
                 references=report,product_hashes=valid['product_hashes'])
    write_json(OUT/'report.json',compact)
    print('Report and eight figures complete',flush=True)
if __name__=='__main__':main()
