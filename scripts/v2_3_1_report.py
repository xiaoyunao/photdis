#!/usr/bin/env python3
"""Source-backed correction report; requires successful actual computations."""
import json
from pathlib import Path
import v2_3_report as fmt
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manifests/v2_3_1';D=ROOT/'docs'
fmt.M=M
read,select,table=fmt.read,fmt.select,fmt.table


def main():
    valid=json.loads((M/'execution/validation.json').read_text());extra=json.loads((M/'execution/correction_validation.json').read_text())
    assert valid['all_passed'] and extra['all_passed']
    decision=json.loads((M/'execution/decision.json').read_text());replay=json.loads((M/'execution/selection_replay.json').read_text())
    dist=read('development_distance');reserved=read('reserved_distance');ss=read('reserved_selection_scores');far=read('far_full_pdf_mass')
    parts=['# Photdis v2.3.1 — corrected frozen-model inference and selection audit',
        '**修正实验已完成。原预定cal规则结果：'+('通过' if decision['cal_pass'] else '不通过，仍不采用候选')+'；现有部署不变。** 这是对同一冻结实验的公式纠错，不是新的盲测。',
        '按[完整Pro审查](pro_review_569d6ff.md)及[冻结修正协议](v2_3_1_protocol.md)执行。原v2.3候选输出含共享g符号错误，已明确标为历史审计；本报告为正确推断后的重评。',
        '## 实际修改与独立验证',
        '定义观测误差为观测减真值：g_obs=Mg+mu+epsilon_g，故有效变量T=Mg+epsilon_g；Cov(T,observed(gr,rz))=V_Mc+(sigma_g²,0)。修复条件均值与方差所用交叉项的减号，并删除复制错误公式的测试。',
        '独立测试直接从三波段生成模型m=M+mu·1+epsilon完成平方求解。已知SED例恢复mu=13.0666667和sigma_mu=0.057735；非对角一般高斯混合检查成分权重、均值、方差、整张PDF及qnear。训练协方差、266774原生PSF身份（含43914无类别距离成员）、32成分、179次EM和两个模型文件均不变。',
        '显式归档旧候选PDF、汇总和派生统计；新缓存同时检查模型哈希、对象身份、推断版本和源码哈希。只重算候选，所有QRF逐星汇总与分块文件哈希不变。完整PDF及参数留在NERSC，未重训、重调阈值或更换模型。',
        '[迁移与字节哈希](../manifests/v2_3_1/execution/inference_plus_g_migration.json)',
        '## 修正前后：错误究竟影响了多少结果',
        '以下为相同身份配对，按最大grz测光误差分层。选样变化与预测数值变化分开，不把不同入选对象称为同星改善。']
    delta=select(read('inference_correction_by_error'),role='development_test')
    parts.append(table(delta,[('error_low','误差下界mag',1),('error_high','上界mag',1),('version','推断',1),('n','N',1),('distance_tail_1','距离严重错误%',100),('gaia_log','原始Gaia log↑',1),('changed_scope','入选变化N',1),('median_abs_mu_change','中位绝对mu变化',1),('mean_abs_q_change','平均绝对q变化',1)]))
    parts+=['这批开发对象主要处于小测光误差区，因此总体数值变化小；符号错误仍必须纠正。误差≥0.1mag层仅6个开发对象，不能把这层修正效果外推成暗端人口成绩。',
        '[全部cal/开发/保留/应用误差分层](../manifests/v2_3_1/inference_correction_by_error.csv)',
        '## 正确推断后的主要科学结果','chosen开发参考，未按预测删选；所有值相对有误差参考。qnear指日心d<10kpc。']
    rr=[]
    for model in ('control','candidate'):
        for group in ('all','dwarf','evolved','other','unknown','metal_poor_dwarf'):
            rr+=select(dist,role='development_test',model=model,group=group,selection='all',reference='chosen',low_kpc=0,high_kpc='inf')
    cols=[('model','模型',1),('group','人群',1),('n','N',1),('median_relative','中位距离残差%',100),('nmad_relative','NMAD%',100),('tail_1','严重错误%',100),('coverage68','点参考68%覆盖%',100),('reference_convolved_log','参考卷积log↑',1)]
    parts.append(table(rr,cols));parts+=['XD反卷积参考误差而QRF保留带噪标签分散，点参考覆盖不能单独判定本征距离概率校准，也不能自动用反卷积解释全部差异。保留每来源卷积评分、原始观测评分与选择条件化分析。',
        '## 固定近端范围：选入与距离准确分开',
        '范围仍为原DECam clean grz、高银纬低消光、核实PSF、观测r<20、qnear>.95、最大光学误差<.1mag及共同QRF颜色/邻域支持；WISE可选。']
    risk=read('development_risk');rr=[]
    for role in ('cal','development_test'):
        for model in ('control','candidate'):rr+=select(risk,role=role,model=model,group='all',selection='near_scope',reference='chosen')
    parts.append(table(rr,[('role','角色',1),('model','模型',1),('near_selected','选中参考近端',1),('far_selected','误入参考远端',1),('completeness','完整率%',100),('far_contamination','远端混入%',100)]))
    rr=[]
    for model in ('control','candidate'):
        for group in ('all','dwarf','evolved','other','unknown','metal_poor_dwarf'):rr+=select(dist,role='development_test',model=model,group=group,selection='near_scope',reference='chosen',low_kpc=0,high_kpc='inf')
    parts.append(table(rr,cols));parts+=['## 新身份的SPDist证据必须保留',
        'chosen不是全部评价资源。下表是保留身份中通过非恒星风险、身份、参考质量筛选的各来源距离；all与near_scope分别列出。SPDist仍是共享教师，不是独立无误差真值；没有合格类别标签，不据此宣称真实巨矮组成。']
    rr=[]
    for model in ('control','candidate'):
        for reference in ('chosen','specdis','spdist'):
            for selection in ('all','near_scope'):rr+=select(reserved,model=model,group='all',selection=selection,reference=reference,low_kpc=0,high_kpc='inf')
    parts.append(table(rr,[('model','模型',1),('reference','参考',1),('selection','选样',1),('n','N',1),('median_relative','偏差%',100),('nmad_relative','NMAD%',100),('tail_1','严重错误%',100),('coverage68','68%覆盖%',100),('coverage95','95%覆盖%',100)]))
    rr=[]
    for model in ('control','candidate'):rr+=select(read('reserved_risk'),model=model,group='all',selection='near_scope',reference='spdist')
    parts.append(table(rr,[('model','模型',1),('near_selected','选中参考近端',1),('far_selected','误入参考远端',1),('far_contamination','参考混入%',100)]))
    parts+=['少量参考远端误入不能外推目录纯度；仅1个r<20chosen参考不代表没有SPDist近端证据。',
        '## 保留资格与视差选择：原始和条件评分同时保留',
        f'逐项读取历史身份清单并重放：{replay["reserved"]}个保留身份完全一致；历史distance-fit资格也逐对象复现。PSF/测光/空间/历史排除后的clean unknown对象有{replay["clean_unknown"]}个，其中观测时used状态不能由distance-fit规则解释的有{replay["unexplained_clean_unknown"]}个。',
        '条件化不是全局SNR<20：在clean、未知类别且光学误差<.03时，合格SpecDis会直接触发历史距离入训；否则有效Gaia锚点在校正观测SNR≥20触发入训。光学误差≥.03的对象不受这项距离训练上限约束。其他身份/类别/源资格固定，保留其排除审计。',
        '这只重建固定协变量下的历史距离入训选择；不等于已知完整光谱目标选择函数，也未模拟参考来源与视差之间的所有相关误差。原始PIT不能自动要求在被选择样本上均匀；条件PIT也不能自动推广到实际人口。',
        '[逐原因历史计数](../manifests/v2_3_1/selection_history.csv) · [解析条件资格](../manifests/v2_3_1/selection_qualification.csv)']
    rr=[]
    for model in ('control','candidate'):
        for selection in ('all','near_scope'):
            rr+=select(ss,model=model,group='all',selection=selection)
    parts.append(table(rr,[('model','模型',1),('selection','选样',1),('score','评分口径',1),('n','N',1),('negative','负视差',1),('log_mean','平均log↑',1),('pit_low','PIT低尾%',100),('pit_high','PIT高尾%',100),('mean_selection_probability','平均保留概率',1)]))
    parts+=['直接从完整距离PDF采样并加入实际Gaia误差，再按源资格/误差/锚点规则重新选择；每对象2048次，独立两半用于CDF估计和检验。模拟检验是否重现选择效应，是方法检查，不是新的天体真值或真实校准证据。条件化评分为raw log−log P(selection)，不能拿它与另一条件下的raw数值直接判模型优劣。',
        table(read('selection_simulation_checks'),[('model','模型',1),('n','N',1),('mean_probability_difference','模拟−解析平均保留概率',1),('rms_probability_difference','保留概率差RMS',1),('median_abs_observed_pit_difference','观测条件CDF差中位绝对值',1)]),
        '补充分析的raw与conditional PIT统一使用32阶积分；另记录与原8阶raw PIT的最大差。模拟尾率按可模拟对象等权平均，另存按模拟样本数加权的值和实际模拟对象数。',
        '原来的cal raw-Gaia验收规则保持不动；新增条件分析仅作补充。保留身份已在v2.3被看过，本次不重新包装成新盲测。',
        '在这一固定协变量选择模型下，条件化后近端PIT低尾仍明显高于5%；历史锚点截断只能解释一部分现象，剩余偏斜不能据此视为已经解决。其他源选样/相关误差尚未建模，不能直接全归因于距离模型。',
        '## 远端：完整PDF与辅助类别标记分开']
    parts.append(table(far,[('model','模型/版本',1),('reference','参考',1),('low_kpc','下界kpc',1),('high_kpc','上界kpc',1),('n','同星N',1),('mean_mass_within1','平均P(|Δmu|<1)',1),('median_mass_within1','中位概率质量',1),('reference_convolved_log','卷积log↑',1)]))
    parts+=['上述积分来自完整PDF，不用成分级类别频率。QRF叶内类别—光度联合分布与XD每成分固定类别比例并不等价；后者假设成分内类别不随颜色/光度变。辅助巨星条件结果见far_conditional，不能单独解释成完整PDF丢掉远端支。两参考仍非独立尺度。',
        '## 贫金属配对与应用证据分母',
        '保留原2055对（354唯一普通对照），新增both/poor-only/ordinary-only/neither计数。both-in-stratum在两侧同时满足距离/选样时才保留配对；poor-side-defined用贫金属侧条件选整对。原每侧独立分层仍保留，但不冒称配对结果。总体配对的重复对照不能视为独立样本，也不能作因果归因。']
    rr=[]
    for model in ('control','candidate'):rr+=select(read('metal_poor_pair_conserving'),model=model,rule='both_in_stratum',selected=False,low_kpc=0,high_kpc='inf')
    parts.append(table(rr,[('model','模型',1),('group','配对侧',1),('n','N',1),('median_relative','偏差%',100),('tail_1','严重错误%',100),('coverage68','覆盖%',100)]))
    rr=select(read('application_coverage_definitions'),selection='near_scope')
    parts+=['同一近端选择的dev-only与cal+dev证据口径如下。v2.2用前者、v2.3用后者；不能把全部覆盖变化归功于质量规则。加权人数是13场的抽样预测估计，非确认近端恒星。']
    parts.append(table(rr,[('model','模型',1),('evidence','证据来源',1),('weight','选中权重',1),('zero_weight','零证据权重',1),('sparse1_19_weight','1–19证据权重',1),('reference_n','参考N',1)]))
    parts+=['## 原预定cal条件、验证与结论',table([dict(guard=k,result='PASS' if v else 'FAIL') for k,v in decision['guards'].items()],[('guard','原条件',1),('result','修正后结果',1)]),
        f'实际生产检查{valid["actual_checks"]}项、纠错/字节保持/选择与配对检查{extra["actual_checks"]}项通过；10项独立数学/评分/选择测试通过。图件与格式/source检查另见delivery_validation.json。完整概率、源身份和原始负结果均保留。',
        '本轮结论仅针对这一个正确推断后的固定XD32。'+('原cal条件通过仍不自动部署，也不构成独立人口验证。' if decision['cal_pass'] else '修正后仍未通过原cal条件，故仍不采用；不能把这一个实现的失败扩大成所有显式误差密度方法不可行。'),
        '近端仍是有规模、有部分参考支持的候选适用范围；贫金属尺度、远端权重及代表性应用验证仍是科学缺口。未新建模型、监控、子代理或全目录预测。',
        '[全部图集](v2_3_1_atlas.md) · [逐项要求](v2_3_1_requirements_result.md)']
    (D/'v2_3_1_result.md').write_text('\n\n'.join(parts)+'\n')
    atlas=['# v2.3.1 English correction atlas','All scientific statistics retain their full denominators. Pal5 font/palette; PNG300dpi/PDF/CSV/JSON. Figures07 and15 distinguish independent-side and pair-conserving strata.']
    for path in sorted((M/'figures').glob('*.json')):
        x=json.loads(path.read_text());n=x['name'];atlas+=['## '+n,x['caption'],f'[PNG](../figures/v2_3_1/{n}.png) · [PDF](../figures/v2_3_1/{n}.pdf) · [CSV](../manifests/v2_3_1/figures/{n}.csv) · [JSON](../manifests/v2_3_1/figures/{n}.json)']
    (D/'v2_3_1_atlas.md').write_text('\n\n'.join(atlas)+'\n')
    print('CORRECTION REPORT',decision['cal_pass'])


if __name__=='__main__':main()
