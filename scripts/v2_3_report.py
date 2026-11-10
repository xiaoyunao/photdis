#!/usr/bin/env python3
"""Assemble source-backed stage report only after production validation."""
import csv
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manifests/v2_3';D=ROOT/'docs'


def read(name):
    with open(M/(name+'.csv')) as f:return list(csv.DictReader(f))


def select(rows,**kw):return [r for r in rows if all(r.get(k)==str(v) for k,v in kw.items())]


def number(value,scale=1):
    if value is None or value=='':return '—'
    if scale==1 and str(value).lstrip('-').isdigit():return f'{int(value):,}'
    try:return f'{float(value)*scale:.4g}'
    except (ValueError,TypeError):return str(value)


def table(rows,columns):
    return '\n'.join(['| '+' | '.join(label for key,label,scale in columns)+' |','|'+'|'.join('---' for _ in columns)+'|']+['| '+' | '.join(number(r.get(key),scale) for key,label,scale in columns)+' |' for r in rows])


def main():
    valid=json.loads((M/'execution/validation.json').read_text());assert valid['all_passed']
    decision=json.loads((M/'execution/decision.json').read_text());fit=json.loads((M/'execution/fit_convergence.json').read_text());prep=json.loads((M/'execution/prepare_complete.json').read_text())
    dist=read('development_distance');ga=read('reserved_gaia');app=read('application_scope');far=read('far_conditional')
    parts=['# Photdis v2.3 — full-support error-aware density experiment\n',
        '**阶段实验已完成；现有部署保持不变。** 候选是否通过预先固定的cal条件：**'+('通过' if decision['cal_pass'] else '不通过，拒绝采用')+'**。新身份成绩没有用于改训练、参数或阈值。',
        '总体严重距离错误率略降（9.65%→9.45%），但cal完整Gaia视差评分及近端污染条件均失败。新身份视差检验也未提供改善证据，因此结束这一固定候选实验；近端范围仍是待验证范围。',
        '本轮按[完整Pro建议](pro_review_83e1fe8.md)和[冻结协议](v2_3_protocol.md)执行。所有新样本为已核实的原生DR11 PSF。',
        '## 方法、资格与实际范围',
        f'同资格QRF与唯一XD32候选均使用{prep["training"]:,}个原训练身份，其中{prep["unlabeled_training"]:,}个没有可靠类别标签。保留原测光owner及TYPE，参考误差从对应旧/新源表精确连接。排除23个非PSF和50206个未核实TYPE身份；没有光度或可靠类别删样。',
        f'候选在(g-r,r-z,Mg)中进行异方差误差反卷积，包含共享g/r协方差和参考距离误差；允许多峰，类别仅为辅助贡献，不先硬分类。固定32成分、单一初始化，无模型族/超参数搜索。{fit["iterations"]}次迭代收敛。QRF使用原128树、leaf32、depth20与Sobol128。',
        '推断仅使用grz测光及误差；Gaia、光谱、真实类别、FeH不作为应用输入。学习到的密度仍包含参考选样先验，不能作为独立于先验的银河系似然。参考距离误差的高斯近似及共享距离尺度不是独立真值。',
        'qnear定义为日心距离小于10kpc（距离模数小于15）的概率。首个候选近端范围固定为：既有DECam DR11原生clean grz、高银纬/低消光入口，PSF，观测r<20，qnear>0.95，最大光学误差<0.1mag，并满足同一PSF QRF训练池的颜色边界及近邻支持。WISE可选。应用只检查已有13场的26000加权抽样，不生成新全目录。观测格参考数仅表示证据可用性。',
        '## 两项已完成的缓存修正',
        '42076个保留身份有173个chosen距离评价参考（4Gaia、169SpecDis）；旧表把fit_eligible误称为距离评价资格。3596个有合格有符号视差，其中2952个S/N<5、644个S/N>=5；860个为负视差。r<20只有1个chosen距离参考，且参考误差很大，亮端距离验收主要仍受证据限制。所有这些对象均未参与新训练或校准。',
        '22–25噪声模板的误差平均分类log score约−1.27275；完全无颜色的均匀基准为−1.09861，设计一致基准为−1.09681（越高越好）。因此相对旧先验的正增益不能证明恢复暗端颜色信息。按类标准化及原配对噪声变化均保留，未重做模拟。',
        '[噪声基准](../manifests/v2_3/noise_baselines.csv) · [参考资格](../manifests/v2_3/reference_inventory.csv)',
        '## 同一开发身份上的完整距离对照',
        '以下为development、chosen、未按预测结果删选。百分数均相对当前参考，严重错误为|Δμ|>1mag。']
    chosen=[]
    for model in ('control','candidate'):
        for group in ('all','dwarf','evolved','other','unknown','metal_poor_dwarf','low_luminosity'):
            chosen+=select(dist,role='development_test',model=model,group=group,selection='all',reference='chosen',low_kpc=0,high_kpc='inf')
    cols=[('model','模型',1),('group','人群',1),('n','N',1),('median_relative','偏差%',100),('nmad_relative','NMAD%',100),('tail_1','严重错误%',100),('coverage68','68%覆盖%',100),('interval68','区间分数↓',1)]
    parts.append(table(chosen,cols))
    parts+=['候选对参考误差进行反卷积，因此相对参考点值的68%覆盖从68.54%降至43.17%，不能单独解释为真实距离后验必然失准。应同时看含参考误差的完整评分及实际观测视差检验。下表为同一开发人群的参考误差卷积log score（越高越好）；结果随参考来源不同。']
    scored=[]
    for model in ('control','candidate'):
        for reference in ('chosen','Gaia','specdis','spdist'):
            scored+=select(dist,role='development_test',model=model,group='all',selection='all',reference=reference,low_kpc=0,high_kpc='inf')
    parts.append(table(scored,[('model','模型',1),('reference','参考',1),('n','N',1),('reference_convolved_log','卷积log score↑',1)]))
    parts+=['[完整分层](../manifests/v2_3/development_distance.csv) · [配对空间不确定度](../manifests/v2_3/paired_spatial_uncertainty.csv) · [历史模型与PSF入口效应](../manifests/v2_3/admission_effect.csv)',
        '## 近端选中之后的距离质量',
        '每个模型按同一预定规则选择，但选中身份可不同；下表不能被称为同星改进。保留率分母与距离/星族分层见CSV。']
    risk=read('development_risk');risk_rows=[]
    for role in ('cal','development_test'):
        for model in ('control','candidate'):
            risk_rows+=select(risk,role=role,model=model,group='all',selection='near_scope',reference='chosen')
    parts.append(table(risk_rows,[('role','样本',1),('model','模型',1),('near_selected','选中参考近端',1),('far_selected','选中参考远端',1),('completeness','近端完整率%',100),('far_contamination','远端混入率%',100)]))
    parts+=['开发参考上，完整率77.11%→85.04%，但远端混入率0.839%→1.828%；这些比例不能直接转移到实际应用人口。以下继续检查选中后的逐星距离质量。']
    rr=[]
    for model in ('control','candidate'):
        for group in ('all','dwarf','evolved','other','unknown','metal_poor_dwarf'):
            rr+=select(dist,role='development_test',model=model,group=group,selection='near_scope',reference='chosen',low_kpc=0,high_kpc='inf')
    parts.append(table(rr,cols));parts+=['## 新身份的完整有符号视差检验','保留实际误差与负视差；这是同天空新对象的一致性检验，不能替代独立距离尺度、新天空或真实人口组成验证。']
    rr=[]
    for model in ('control','candidate'):
        for group in ('all','no_W1','r_18_20','r_20_22'):
            rr+=select(ga,model=model,group=group,selection='all')
    parts.append(table(rr,[('model','模型',1),('group','观测层',1),('n','N',1),('negative','负视差',1),('log_mean','平均log score↑',1),('log_median','中位log score',1),('pit_low','PIT<.05%',100),('pit_high','PIT>.95%',100)]))
    parts+=['[新身份距离](../manifests/v2_3/reserved_distance.csv) · [视差](../manifests/v2_3/reserved_gaia.csv) · [PIT](../manifests/v2_3/reserved_pit.csv)',
        '## 应用候选规模与证据缺口','加权计数仅为模型预测选中规模，不能当成已验证的真实近端恒星人数。']
    rr=[]
    for model in ('control','candidate'):rr+=select(app,model=model,selection='near_scope',r_low='-inf',r_high=20,WISE='all')
    parts.append(table(rr,[('model','模型',1),('n','抽样选中N',1),('weight','加权规模',1),('zero_dev_reference_weight','无开发距离证据权重',1),('no_new_parallax_weight','无新视差证据权重',1)]))
    parts+=['[应用分层](../manifests/v2_3/application_scope.csv) · [逐格证据](../manifests/v2_3/application_coverage.csv) · [PSF风险控制](../manifests/v2_3/psf_nonstellar_controls.csv)',
        '## 远端形状与贫金属条件偏差','下面是真实参考类别条件化后的oracle诊断，不是应用时可以获得的条件；SpecDis/SPDist仍是已有共享教师。']
    rr=[r for r in far if r['role']=='development_test' and r['reference'] in ('specdis','spdist')]
    parts.append(table(rr,[('model','模型',1),('reference','参考',1),('low_kpc','距离下界kpc',1),('high_kpc','距离上界kpc',1),('n','N',1),('median_relative','条件偏差%',100),('tail_1','条件严重错误%',100),('coverage68','条件覆盖%',100)]))
    parts+=['20–50kpc的SpecDis条件巨星严重错误由7.29%升至27.86%，SPDist由6.40%升至25.60%；这次候选并未保住原有远端条件分支质量。两源距离差异和不同模型的类条件化近似仍需分别保留，不能据此统一放大所有巨星权重。',
        '[参考距离差异](../manifests/v2_3/far_reference_disagreement.csv) · [观测格内分支频率](../manifests/v2_3/branch_frequency.csv)',
        '贫金属对照保持同一贫金属身份，匹配颜色、温度、参考来源及参考/测光误差；允许对照对象重复，不能把重复对照当独立样本，也不能作因果归因。应用未按真实FeH删样。',
        '[匹配统计](../manifests/v2_3/metal_poor_matching.csv)',
        '匹配的2518个合格贫金属对象中2055个匹配成功，对照仅354个唯一身份。匹配限制为两颜色各0.1mag、Teff500K、光学误差0.03mag、参考误差0.2mag且来源相同。仍存在贫金属条件偏差，不能把小幅总体改善当作已修复银晕尺度。',
        '## 固定cal验收与数值验证',
        table([dict(guard=k,passed='PASS' if v else 'FAIL') for k,v in decision['guards'].items()],[('guard','条件',1),('passed','结果',1)]),
        f'生产验证共{valid["actual_checks"]}项通过；7项数学/评分测试通过。原Gauss CDF右尾舍入与0.1mag分箱精度问题均保留诊断和显式迁移记录；拟合模型文件不变。最终候选保存float64、0.01mag PDF及完整解析混合参数，近端积分一致性容差1e-3。控制模型保留其原生直方图网格。粗结果已归档，不与最终结果混用。',
        '完整Gaia cal平均log score为0.083769→0.039926（下降0.043842）；新身份为−0.875888→−0.924684。16/32阶数检查中控制模型单对象最大差0.015342、按对象数加权的分块平均差约1.46e−7，候选最大差约8e−15；不能声称每个对象积分差均小于1e−3。cal下降超过控制模型最坏单对象差界，因此这一有限阶数对照不改变拒绝结论；它不是对无限阶积分误差的严格证明。',
        '[生产检查](../manifests/v2_3/execution/validation.json) · [模型冻结](../manifests/v2_3/execution/models_frozen.json) · [积分阶数检查](../manifests/v2_3/development_quadrature.csv)',
        '## 保留的科学限制',
        '本轮没有获得新的独立距离尺度、系统成员归属证明或代表性非恒星人口样本。18–22无WISE、近端贫金属尺度、20–50巨星独立分支验证仍需补证；10–20参考差异与分支形状必须分别处理。风险标志控制通过率不能外推目录污染人数。未创建定时监控、额外候选、全目录预测或自动部署。',
        '[九组图集](v2_3_atlas.md) · [逐项要求](v2_3_requirements_result.md)']
    (D/'v2_3_result.md').write_text('\n\n'.join(parts)+'\n')
    names=['01_noise_benchmarks','02_reference_inventory','03_model_comparison','04_reserved_parallax','05_near_scope','06_far_branches','07_metal_poor','08_selected_quality','09_posterior_examples']
    atlas=['# v2.3 English figure atlas\n','All figures use the established Pal5 font/palette. Scientific statistics are untrimmed; posterior examples state displayed mass.']
    for name in names:
        for suffix in ('png','pdf'):assert (ROOT/'figures/v2_3'/f'{name}.{suffix}').exists()
        meta=json.loads((M/'figures'/f'{name}.json').read_text())
        atlas+=['## '+name,meta['caption'],f'[PNG](../figures/v2_3/{name}.png) · [PDF](../figures/v2_3/{name}.pdf) · [CSV](../manifests/v2_3/figures/{name}.csv) · [JSON](../manifests/v2_3/figures/{name}.json)']
    (D/'v2_3_atlas.md').write_text('\n\n'.join(atlas)+'\n')
    print('REPORT',decision['cal_pass'],valid['actual_checks'])


if __name__=='__main__':main()
