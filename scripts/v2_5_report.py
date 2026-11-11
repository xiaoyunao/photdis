#!/usr/bin/env python3
"""Measured v2.5 report; preserve failed scientific and numerical criteria."""
import csv
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manifests/v2_5'


def read(name):
    with open(M/(name+'.csv')) as f:return list(csv.DictReader(f))


def main():
    red=read('red_migration');metal=read('metal_gaia_availability');rec=read('recovery_summary');diag=read('identifiability');templates=json.loads((M/'execution/templates.json').read_text())
    out=['# Photdis v2.5 — 目标总体适应原型与有符号视差诊断','',
         '本轮执行 de4904b 的完整 Pro 审查。原 QRF、修正 XD、已有目录均保持冻结；新程序是有限经验 SED 响应的总体推断原型，不能与已部署目录合并。',
         '', '## 红端：不按 PIT 选择的应用对照', '',
         '对完整711历史对象匹配颜色、r、光学误差和Gaia误差，控制池排除拟合身份与已知非恒星风险。最大标准化距离2，失败对象保留。下表为匹配身份口径；父样本权重和逐场区间另存CSV。历史条件PIT与应用原始PIT具有不同选择含义，不能称作模型性能改进。','',
         '| 分组 | 对照匹配/历史分母 | 唯一对照 | 历史原始低尾 | 历史条件低尾 | 应用原始低尾 | 应用控制身份区间 |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for group in ['all','red','blue']:
        q=[r for r in red if r['group']==group and r['weighting']=='matched_identity' and r['tail']=='low' and r['cluster']=='reused_control'];d={r['method']:r for r in q};a=d['application_raw']
        out.append(f"| {group} | {a['matched']}/{a['denominator']} | {a['unique_controls']} | {100*float(d['historical_raw']['value']):.2f}% | {100*float(d['historical_conditional']['value']):.2f}% | {100*float(a['value']):.2f}% | {100*float(a['low']):.2f}–{100*float(a['high']):.2f}% |")
    high=[r for r in red if r['group']=='red' and r['method']=='application_raw' and r['tail']=='high' and r['weighting']=='matched_identity' and r['cluster']=='reused_control'][0]
    out += ['',f"应用红端高PIT尾部仍为{100*float(high['value']):.2f}%，控制身份区间[{100*float(high['low']):.2f},{100*float(high['high']):.2f}]%。未重现历史低尾异常不等于全面校准；不能仅报告低尾而隐藏高尾。"]
    out += ['', '该设计约束迁移解释，而非证明Gaia或光谱距离哪一方正确。匹配重复降低有效证据量；不能用匹配行数当独立对象数。','', '## 贫金属：全有符号Gaia观测', '',
            '| 侧别 | 原配对数 | 合格观测行 | 唯一身份 | 负视差行 | SNR<5行 | 拟合重叠行 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in metal:out.append('| '+' | '.join(r[k] for k in ['side','original_pairs','eligible','unique_eligible','negative','weak','fit_overlap'])+' |')
    pair=[r for r in read('metal_gaia_pairs') if r['model']=='control' and r['source']=='all' and r['role']=='neither_fitted' and r['metric']=='high_tail' and r['cluster']=='control'][0]
    out += ['',f"双侧合格配对{pair['n']}，普通侧独立控制{pair['unique_controls']}。贫金属侧相对普通侧的QRF高PIT尾部差为{100*float(pair['mean']):.2f}个百分点，重复控制身份区间[{100*float(pair['low']):.2f},{100*float(pair['high']):.2f}]。高PIT方向对应观测视差比模型更大，与贫金属距离偏远相符；不能据此给所有对象统一乘修正因子。不同误差、原配对和参考选择仍需保留。"]
    out += ['', '不对弱/负视差取倒数。两侧均合格才形成配对差；单侧观测保留在逐对象表。按参考来源、训练身份、重复普通对照与天空配对分别统计完整预测视差log score与PIT尾部。共享教师、原配对选择和误差分布仍限制绝对几何尺度判断。', '',
            '## 总体推断：明确选择归一化与身份隔离', '',
            f"校准模板{templates['cal']}，独立测试模板{templates['test']}；其中校准模板{templates['missing_labels']}缺类别标签，仍以颜色/Mg核的软成员贡献光度信息。缺标签不是物理类别，该软归属尚未独立校准。训练/测试/模型拟合身份隔离。",
            '', '四参数控制三个星族相对权重和共同距离倾斜，20个固定星族×距离组分；日心距离节点1、3.5、7.5、15、30kpc。包含远端贡献，没有在10kpc截断。ordinary/poor矮星模板分别要求[Fe/H]>=−0.5/<−1.5，中间丰度未构成独立组分；这是受限实验。', '',
            '前向响应由独立校准经验SED加原生grz噪声构建，不从QRF后验造真值，也不把QRF除以全局先验当似然。条件似然显式除以组分加权入选概率。选择为14≤带噪消光校正r<20且赋予的三波段误差<0.1mag；未模拟探测、饱和、拥挤、形态变化。补集单独作为支持外压力试验。', '',
            '192套独立观测目录、每套128对象：8种组成×2种选择域×12次。每套重新抽源身份及原生波段噪声；身份角色不改变。同目录点距离、冻结PDF积分、总体后验比较。实际应用3250身份没有参与这些实验。', '',
            '## 距离分布恢复结果', '',
            '下表是六距离格（含>50kpc QRF泄漏）的总体比例L1误差；不是单星距离误差。总体真值由独立测试模板的前向Monte Carlo给出。', '',
            '| 设计 | 选择域 | 点距离L1 | PDF L1 | 总体后验L1 |','|---|---|---:|---:|---:|']
    for design in dict.fromkeys(r['design'] for r in rec):
        for regime in ['0','1']:
            q={r['method']:r for r in rec if r['design']==design and r['regime']==regime and r['method'] in ['point','pdf','population']}
            out.append(f"| {design} | {'预定测光入口' if regime=='0' else '补集压力'} | "+' | '.join(f"{100*float(q[k]['mean_L1']):.2f}%" for k in ['point','pdf','population'])+' |')
    composition=read('group_comparison')
    out += ['', '### 星族组成：距离总量改善不能替代类别验收', '',
            '| 设计/星族（预定测光入口） | 设计真比例 | 新总体后验均值 | QRF参考贡献 |', '|---|---:|---:|---:|']
    for design in ['reference_like','giant_rich','metal_poor_rich']:
        for channel in ['dwarf','evolved','other']:
            q=[r for r in composition if r['design']==design and r['regime']=='0' and r['channel']==channel]
            out.append('| '+design+'/'+channel+' | '+' | '.join(f"{100*np.mean([float(r[k]) for r in q]):.2f}%" for k in ['truth','population','QRF_reference'])+' |')
    out += ['', 'QRF的无标签贡献单独保留在表与图中，不能强行称作一类真实恒星。新原型虽能提高巨星富集设计的巨星比例恢复，但接近参考组成时other与dwarf之间仍有明显错配；不能把总体人数接近理解成每类人数可靠。']
    out += ['', '### 近端各距离层的参数区间覆盖', '',
            '下表为每个设计12套独立目录中，名义95%参数区间覆盖设计总体比例的次数。它不是单星距离覆盖；每格Wilson区间在CSV保留。', '',
            '| 设计（预定测光入口） | 0–2 kpc | 2–5 kpc | 5–10 kpc |', '|---|---:|---:|---:|']
    for design in dict.fromkeys(r['design'] for r in rec):
        q={int(r['bin']):r for r in rec if r['design']==design and r['regime']=='0' and r['method']=='parameter'}
        out.append('| '+design+' | '+' | '.join(f"{round(12*float(q[b]['coverage']))}/12" for b in range(3))+' |')
    out += ['', '**科学验收：均值恢复有进展，但不能通过可靠人数区间的验收。** 部分2–5kpc区间仍系统性漏掉真值；全距离平均覆盖会掩盖这一点。当前只表达总体参数的不确定度，没有把所有模板、教师和选择模型误差纳入总区间。因此保留原型及正负结果，不据此部署新目录或声称已恢复真实银河系密度。']
    support=read('actual_parent_support')
    out += ['', '### 父样本观测支持审计', '', '| 预定选择 | 总模拟行 | 颜色/星等邻居距离≤1 |', '|---|---:|---:|']
    for regime in ['0','1']:
        q=[r for r in support if r['regime']==regime];out.append(f"| {regime} | {sum(int(r['n']) for r in q)} | {sum(int(r['parent_neighbor_within1']) for r in q)} |")
    out += ['', '邻居距离以(r0,g−r,r−z)/(1,0.3,0.3)度量，按星族和场景的表保留。固定r/误差入口只是一个选择域，不等于完整观测支持；14–15等的噪声标定仅3个父样本对象，14等以下无噪声对象。支持外压力结果不能外推真实目录人数。']
    out += ['', '## 区间、可辨识性与数值可信度', '',
            f"双倍Sobol积分网格的最大组分比例变化为{max(float(r['grid_max_delta']) for r in diag):.5f}；最小积分有效点数{min(float(r['ess']) for r in diag):.2f}；最大参数相关绝对值{max(float(r['max_correlation']) for r in diag):.4f}。边界质量最大为{100*max(float(r['boundary']) for r in diag):.2f}%（任一log-tilt绝对值>2.8）；逐目录保留。积分未稳定或边界堆积的情况不得认作精确人口测量。",
            '', '数值加密遵循[显式修订](v2_5_numerical_amendment.md)：主积分2^18，对照2^17；初始结果与全部预测保留。总体参数区间与重复目录人数预测区间分别保存。前者对设计条件比例评分，后者在每个独立目录的实现人数上作诊断；后者不是逐对象潜在类别后验计数区间。每格只有12次独立目录覆盖试验，Wilson区间必须同时阅读，不能据点估计宣布95%校准。', '',
            '观测随机性由独立目录表示；有限校准模板用16次按源身份的Bayesian bootstrap；参数后验用有界均匀log-tilt先验；教师尺度用整体±1sigma扰动，另有仅带标签模板敏感性。这些不确定度分别展示，尚未形成包含所有系统误差的总区间。经验SED教师误差、模板选择及离散距离假设限制迁移。', '',
            '## 波段收益与预留验证', '',
            '波段收益按dwarf/evolved/other逐类给出同身份差值和该类自己的天空簇区间。固定探针和距离分区训练支持不同，不能把它们的差别都解释成架构效应；匹配波段的暗端演化星证据仍缺失。', '',
            '3250个原生PSF新身份继续保持未打开。已固定equal-field、此前未使用子总体的目标口径，每对象1/3250；不是全父样本面积或数量权重。QRF/修正XD、主要指标、颜色/星等/误差/距离/天区层及历史参考排除条件已冻结。历史排除可能依赖Gaia SNR，未量化前不能声称无条件PIT应均匀。新原型仍是模拟验收阶段，不自动部署或打开前瞻结果。', '',
            '完整数值见[统计目录](../manifests/v2_5/recovery_summary.csv)，图见[英文图集](v2_5_atlas.md)，范围见[冻结协议](v2_5_protocol.md)。']
    out += ['', '## 执行与交付核验', '', '最终数值加密/核验作业61127630完成，1966项实际产物检查通过；4项独立数学测试通过。8组英文PNG300dpi/PDF/CSV/JSON图全部检查，代表PDF渲染核对；初始结果和失败/恢复过程均保留。全部阶段作业结束。']
    (ROOT/'docs/v2_5_result.md').write_text('\n'.join(out)+'\n')
    lines=['# Photdis v2.5 figure atlas','', 'English-only, source-linked PNG300dpi/PDF plates; scientific denominators remain in CSV.','']
    for path in sorted((M/'figures').glob('*.json')):
        q=json.loads(path.read_text());n=q['name'];lines += ['## '+n,'',q['caption'],'',f'![{n}](../figures/v2_5/{n}.png)','',f'[PDF](../figures/v2_5/{n}.pdf) · [CSV](../manifests/v2_5/figures/{n}.csv)','']
    (ROOT/'docs/v2_5_atlas.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
