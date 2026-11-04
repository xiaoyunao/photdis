#!/usr/bin/env python3
"""Render compact v1.3 evidence without loading catalogs or model objects."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.load(open(ROOT/'manifests/baseline_v1_3_result.json'));out=ROOT/'figures/baseline_v1_3';out.mkdir(exist_ok=True,parents=True)
    parent=np.array(r['parent_cells']).reshape(5,3,3,3).sum(axis=(1,3))
    old=sum(np.array(v['old_class_training']['cells']) for v in r['inventory'].values()).reshape(5,3,3,3).sum(axis=(1,3))
    new=sum(np.array(v['class_labels']['cells']) for v in r['inventory'].values()).reshape(5,3,3,3).sum(axis=(1,3))
    # Northern references are a separate unsupported domain, excluded here.
    north=r['inventory']['3'];old-=np.array(north['old_class_training']['cells']).reshape(5,3,3,3).sum(axis=(1,3));new-=np.array(north['class_labels']['cells']).reshape(5,3,3,3).sum(axis=(1,3))
    fig,axes=plt.subplots(1,3,figsize=(13,5),layout='constrained')
    for ax,a,title in zip(axes,[parent,old,new],['DECam parent','Old class training-quality evidence','Independent class evidence']):
        ax.imshow(np.log10(a+1),cmap='Blues',aspect='auto')
        for i in range(5):
            for j in range(3):ax.text(j,i,str(int(a[i,j])),ha='center',va='center',color='black',fontsize=9,bbox=dict(facecolor='white',alpha=.7,edgecolor='none'))
        ax.set(title=title,xticks=range(3),xticklabels=['no W1','W1 only','W1W2'],yticks=range(5),yticklabels=['r<18','18–20','20–22','22–25','>=25'])
    fig.savefig(out/'01_label_coverage.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');names=list(r['models'])
    for j,key in enumerate(['old','expanded']):
        values=[r['models'][n]['comparisons'].get(key,{}).get('1',{}).get('expanded_test',{}).get('average_precision',np.nan) for n in names]
        ax.bar(np.arange(len(names))+(j-.5)*.35,values,width=.35,label=key)
    ax.set(xticks=range(len(names)),xticklabels=names,ylabel='Evolved AP, same expanded test identities',ylim=(0,1));ax.legend();fig.savefig(out/'02_fixed_forest_ap.png',dpi=150);plt.close(fig)
    c=r['candidate'];d=c['reference']['test'];fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for j,key in enumerate(['wide','mixture','main']):
        y=[d['by_class'][str(k)]['chosen'][key].get('tail_1',np.nan) for k in [-1,0,1,2]]
        ax.scatter(np.arange(4)+(j-1)*.15,y,label=key)
    ax.set(xticks=range(4),xticklabels=['unknown','dwarf','evolved','other / transition'],ylim=(-.02,1.05),ylabel='Severe distance error in predicted-evolved group');ax.legend();fig.savefig(out/'03_candidate_risk.png',dpi=150);plt.close(fig)
    lines=['# v1.3：独立标签入口与固定演化星候选','',
      '基线ebf37f8；原空间切分、距离专家和v1部署保持冻结。类别2统一称“其他合格恒星／过渡类”。',
      '本轮标签入口使用已有DESI匹配对象；分类不再要求距离或Mg。新增类别训练仍保持原0.03mag测光门槛；验证不要求距离有效。','',
      '## 实际新增监督','', '|配置|原训练数|新训练数|新增|剔除冲突/污染等|','|---|---:|---:|---:|---:|']
    for n,e in r['models'].items():
        f=e['fit'];lines.append(f"|{n}|{f['n_old']}|{f['n_new']}|{f['added']}|{f['removed']}|")
    lines+=['','|配置|原森林演化AP|扩展森林演化AP|原/新2%预算召回|','|---|---:|---:|---:|']
    for n,e in r['models'].items():
        a=e['comparisons']['old']['1'];b=e['comparisons'].get('expanded',{}).get('1')
        if b:lines.append(f"|{n}|{a['expanded_test'].get('average_precision',0):.4f}|{b['expanded_test'].get('average_precision',0):.4f}|{100*a['workpoints']['0.02']['test']['recall']:.2f}% / {100*b['workpoints']['0.02']['test']['recall']:.2f}%|")
    lines+=['','A新增6287个训练对象：矮星4648、演化星395、其他合格恒星／过渡类1244；新增无W1训练对象508个。',
      '新增证据主要在亮端。22–25等全体类别标签仅2个且都在train，cal/test为0；这不能建立暗端验证。',
      '固定森林的演化AP仅小幅变化，不能声称已解决人群覆盖问题。']
    lines+=['','固定森林与原森林在相同扩展cal/test身份上比较，阈值仅由cal设定；另保留原参照子集结果。',
      '分数为nominal类别排序分数，与v1.2回退后工作点不是同一量。类别平衡分数不解释为母样本概率。',
      '![覆盖](../figures/baseline_v1_3/01_label_coverage.png)','![AP](../figures/baseline_v1_3/02_fixed_forest_ap.png)','',
      '## 仍缺什么证据','', '|r层|无W1母样本/类别标签|仅W1母样本/类别标签|W1W2母样本/类别标签|','|---|---|---|---|']
    for i,tag in enumerate(['<18','18–20','20–22','22–25','>=25']):lines.append('|'+tag+'|'+'|'.join(f'{int(parent[i,j])}/{int(new[i,j])}' for j in range(3))+'|')
    lines+=['','完整135格按r、光学误差、WISE和g-r拆分，分别保存类别、距离和弱/负视差证据以及空间分区；零证据格保留。',
      '下一批数据优先针对母样本非零而类别证据为零的暗端、无W1格；需合格光谱类别和按来源有效距离。',
      '此次没有把已有数据释放等同于新增外部观测。若暗端仍空缺，需要定向扩大匹配或取得新参照，并预留未查看的新天区。','',
      '## 固定预测候选的完整风险','',f"定义：{c['definition']}。只使用测光预测及支持域，不用真实类别选样。",
      f"母样本加权保留率：{100*c['parent']['fraction']:.4f}%；test选中{d['counts']['n_selected']}个参照入口对象，类别计数{d['counts']['class_counts']}。",
      f"候选组相对比较条件通过：{c['relative_comparison_pass']}；绝对可靠性未定义，状态为candidate_only。原父组cal通过不等于子组通过。",'',
      '|真实类别（仅评价）|chosen N|wide严重错误|混合严重错误|演化主分支严重错误|','|---|---:|---:|---:|---:|']
    fmt=lambda z:'无参照' if z is None else f'{100*z:.2f}%'
    for k in [-1,0,1,2]:
        e=d['by_class'][str(k)]['chosen'];lines.append(f"|{r['class_names'][str(k)]}|{e['main']['n']}|{fmt(e['wide'].get('tail_1'))}|{fmt(e['mixture'].get('tail_1'))}|{fmt(e['main'].get('tail_1'))}|")
    lines+=['','完整chosen参照147颗：主分支/混合严重错误59.18%，同星wide49.66%。',
      '该候选未通过相对比较条件；目前不推荐替代wide，保留完整PDF与状态，结束这个固定阈值尝试。',
      '这与真实演化星子集中的局部改善可以同时成立，因为应用筛选仍包含误入对象。']
    lines+=['','|距离参照|N|wide严重错误|混合严重错误|主分支严重错误|','|---|---:|---:|---:|---:|']
    for ref in ['Gaia','specdis','spdist']:
        e=d['distances'][ref];lines.append(f"|{ref}|{e['main']['n']}|{fmt(e['wide'].get('tail_1'))}|{fmt(e['mixture'].get('tail_1'))}|{fmt(e['main'].get('tail_1'))}|")
    lines+=['','同一预测组包含误入的矮星、其他合格恒星／过渡类、未知对象；Gaia/SPECDIS/SPDist分别评分。',
      '完整PDF与多解继续保留，混合分布的区间覆盖见机器结果；本轮没有后验调阈值。',
      '![候选风险](../figures/baseline_v1_3/03_candidate_risk.png)','', '### 同组完整Gaia检验','',
      '|预测|N|平均log密度|高PIT尾|','|---|---:|---:|---:|']
    for key,e in c['Gaia'].items():lines.append(f"|{key}|{e['n']}|{e.get('mean_log_density','无样本')}|{fmt(e.get('full_high95'))}|")
    lines+=['','无样本时不能作性能推断。Gaia质量/零点准入限制与r>=22无验证边界保留。','',
      '## W2表述修正','',
      'v1.2同星森林AP从B_on_C的0.359提高到C的0.407；1%预算召回从62/287提高到80/287。',
      '“收益小”仅指2%工作点（114到115颗），不能概括W2无用。','',
      f"本轮实际检查{r['validation']['n_checks']}项通过；本地18项聚焦测试通过。模型、身份与新参照表哈希记录于机器清单。",
      '[协议](baseline_v1_3_protocol.md) · [完整Pro建议](pro_review_ebf37f8.md) · [机器结果](../manifests/baseline_v1_3_result.json)']
    lines+=['','补充权重与冻结模型审计24项通过：逐对象训练权重另存NERSC，旧六套距离专家与旧森林哈希均未改变。',
      '[权重与冻结审计](../manifests/baseline_v1_3_audit.json)。']
    (ROOT/'docs/baseline_v1_3_result.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
