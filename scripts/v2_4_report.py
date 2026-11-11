#!/usr/bin/env python3
"""Source-backed account of frozen-model scientific-use tests and their limits."""
import json
from pathlib import Path
import v2_3_report as fmt
ROOT=Path(__file__).resolve().parents[1];M=ROOT/'manifests/v2_4';D=ROOT/'docs';fmt.M=M
read,select,table=fmt.read,fmt.select,fmt.table


def main():
    check=json.loads((M/'execution/validation.json').read_text());assert check['all_passed']
    app=json.loads((M/'execution/application_complete.json').read_text());reserve=json.loads((M/'execution/reservation.json').read_text());design=json.loads((M/'execution/recovery_design.json').read_text())
    overlap=read('anomaly_overlap');pairs=read('metal_source_pairs');recovery=read('population_recovery')
    parts=['# Photdis v2.4 — frozen-posterior scientific-use audit',
        '**本轮完成了冻结模型下的科学用途审计；没有拟合新模型，也没有更换部署。** 首要新发现是：低PIT与SPDist交集中的60个对象，SPDist总体贴近测光距离而Gaia视差偏小，两种参考并未共同支持“远距离分支被压低”的单一解释。贫金属尺度偏差在SpecDis/SPDist配对中仍然存在；新的人数恢复实验用于测量这些概率在明确人口假设下的用途与失败。',
        '按[完整Pro审查](pro_review_0f9c412.md)及[冻结协议](v2_4_protocol.md)执行。v2.3.1的共享g修正和XD32拒绝结论保留；本轮所有科学样本核实原生DR11 TYPE=PSF。',
        '## 1. 固定711个对象，逐身份追查147个低PIT异常',
        '固定QRF near_scope与Gaia条件评分资格的交集711个；147个条件PIT<.05。表中control=同PSF QRF，candidate=修正XD32。XD始终在这同一711个身份上比较，不使用其自行选出的952个对象替代分母。保留全部负/低SNR视差，不对弱/负视差取倒数生成距离。',
        table(select(overlap,model='control'),[('group','对象组',1),('denominator','总分母',1),('SPDist_intersection','SPDist交集',1),('SPDist_farther','SPDist比QRF更远',1),('SPDist_farther1mag','更远超过1mag',1),('median_SPDist_minus_phot','SPDist−QRF mu中位数',1)]),
        table(read('anomaly_reference_consistency'),[('low_pit','低PIT组',1),('n','交集N',1),('median_Gaia_SPDist_standardized','Gaia−SPDist/合并误差中位数',1),('below_minus3','低于−3',1),('RUWE_available','RUWE可用N',1),('median_RUWE','RUWE中位数',1),('same_direction_XD','同身份XD也低PIT',1)]),
        '合并误差采用SPDist距离模数误差向视差的一阶传播，再与Gaia观测误差平方相加；未建模跨参考相关性，因此这个标准化差是诊断量，不是独立显著性检验。原始Gaia RUWE/visibility/IPD等通过MWS GAIA原始行号和SOURCE_ID复核取得。不能凭普通RUWE就宣布天体测量不存在系统误差。',
        table(select(read('anomaly_diagnosis'),group='low'),[('model','模型',1),('category','PDF描述',1),('n','N',1),('denominator','分母',1)]),
        'PDF描述使用预设0.1mag平滑、最大密度1%显著度的局部峰、参考mu±1mag质量及主峰偏移。它们描述后验形状，不把“有峰”等同于正确校准，也不把“无合格SPDist”的87个异常对象强行分类为远巨星。完整711行及颜色、星等、误差、天区、W1与Gaia质量分层均保留。',
        '[逐对象证据](../manifests/v2_4/anomaly_objects.csv) · [全部分层](../manifests/v2_4/anomaly_strata.csv)',
        '## 2. 贫金属尺度：按参考来源保持原配对',
        '沿用2055对、354个唯一普通对照；每个来源要求配对两侧同时合格，每个距离层要求两侧都落在该参考距离范围。以下为未施加模型近端选择的全距离配对。',
        table([r for r in pairs if r['low_kpc']=='0' and r['high_kpc']=='inf'],[('model','模型',1),('reference','参考',1),('n','配对N',1),('unique_controls','唯一普通对照',1),('poor_median_relative','贫金属偏差%',100),('ordinary_median_relative','普通偏差%',100),('median_reference_Mg_delta','参考Mg差中位数',1),('control_mean','平均配对残差差mag',1),('control_low','控制身份95%下限',1),('control_high','上限',1)]),
        'QRF相对SpecDis的贫金属中位偏差约+25%，相对SPDist约+35.5%；普通对照分别约−5.4%和+3.3%。这说明条件尺度差并未消失，但两来源使用不同配对子集、存在共享光谱/教师信息，不能据此确认完全独立的距离尺度或丰度因果效应。双侧Gaia锚点只有10对，独立标尺仍不足。',
        '不确定度分别按重复普通对照身份和两侧天区组合进行1000次整组bootstrap；两套区间均输出，不用2055行当作2055个独立对照。图中误差条是配对残差差的均值区间，中位数另外报告。0–2、2–5、5–10kpc和全距离均保留，包括零/稀少证据层。',
        '[来源与距离分层](../manifests/v2_4/metal_source_pairs.csv) · [逐配对Mg与PDF支持](../manifests/v2_4/metal_pair_objects.csv)',
        '## 3. 先固定光测目标，再关联参考与核算人数',
        f'母样本仍是已有13个DECam场的26000个加权抽样对象，权重代表8899334个既有应用对象，并非全部DR11。可观测入口保持原clean grz、高银纬、低消光、核实PSF，r<20、最大grz误差<.1及冻结QRF颜色/近邻支持；尚不要求qnear。实际留下{app["observable_n"]}个抽样对象，代表权重{app["observable_weight"]:,.1f}；随后qnear>.95得到{app["near_n"]}个预测入选对象。',
        '原样本身份、原生TYPE和可观测入口先冻结，然后关联光谱参考；原光测父样本中的有符号Gaia观测也按原owner行精确恢复，不以是否有光谱或SNR正负决定目标入选。已知非恒星风险标记的Gaia评分单列。已有参考/父样本并非新的盲测。',
        table(select(read('application_target_coverage'),group='all'),[('selection','选择',1),('n','抽样N',1),('weight','代表权重',1),('direct_distance_weight','直接合格距离参考权重',1),('Gaia_observed_weight','Gaia观测权重',1),('none_weight','格内零参考权重',1),('sparse_weight','格内1–19参考权重',1),('dense_weight','格内20+参考权重',1)]),
        '参考格沿用r、gr、rz、光学误差与WISE条件，合并cal+development；它没有天区维，不能把“格内20个以上”写成跨天空代表性验证。直接参考覆盖与格内参考覆盖是不同量；按field/颜色/星等/误差/W1的抽样人数及权重完整输出。',
        '联合分布重建使用原PSF训练身份、原树、叶节点class×Mg和source×Mg，以及共享g扰动/Sobol128。每个新分块的PDF与类别边缘逐对象复现缓存QRF。下表是积分d<10kpc后的模型参考贡献；unlabeled只表示训练类别证据缺失。',
        table(select(read('application_joint_counts'),group='all',low_kpc=0,high_kpc=10),[('selection','选择',1),('channel','参考贡献',1),('n','抽样N',1),('expected_count','联合期望贡献',1),('product_marginals_count','边缘相乘错误口径',1)]),
        '这些期望不是已确认的真实巨矮人数，不能把缺标签质量分配给任意星族来凑数。条件加权人数方差也不是抽样设计、教师尺度及人口先验全部不确定度的置信区间。',
        table([r for r in read('application_gaia') if r['group']=='all'],[('selection','选择',1),('risk_flag','非恒星风险标记',1),('n','Gaia N',1),('negative','负视差N',1),('log_mean','加权观测log',1),('low','PIT低尾%',100),('high','PIT高尾%',100)]),
        '这批Gaia是从光测目标关联，保留负值，不套用“历史未训练保留身份”的SNR<20条件；但Gaia可获得性/质量门槛及部分身份历史训练重叠仍限制独立评价。',
        table(read('application_training_overlap'),[('selection','选择',1),('in_fitted_training','属于拟合训练身份',1),('n','目标N',1),('weight','目标权重',1),('Gaia_n','Gaia N',1),('Gaia_log','Gaia加权log',1),('Gaia_low','PIT低尾%',100),('Gaia_high','PIT高尾%',100)]),
        '上述训练身份交集按原始精确ID核对，包含风险对象而不据参考类别再选样；已在拟合中的对象不能算独立验证，不在拟合中的对象也仍有先前查看和同天区选择限制。[完整身份重叠分层](../manifests/v2_4/application_training_overlap.csv)。',
        f'另从原13场光测父样本预留{reserve["n"]}个新身份，先排除全部已知历史参考/训练及已查看应用身份，再按身份排序与可观测条件选取；逐原生sweep核实TYPE。只保存测光，不打开新参考结果、不生成距离预测。这是同既有天空范围的前瞻身份预留；系统成员身份尚未核实，新天空/独立标尺的最终验证仍需后续获取，不能将这个预留直接称为已完成最终验证。',
        '[预留逐场审计](../manifests/v2_4/reservation_counts.csv) · [预留记录](../manifests/v2_4/execution/reservation.json)',
        '## 4. 波段信息与模型保留能力',
        '复用已冻结的四波段探针/分区概率缓存，在相同33251个development身份上比较。无颜色基准来自cal类别频率，另用cal-only r频率，以及固定60度RA×南北半球×r频率加30个对象强度的r层收缩；未使用development标签拟合先验。最初细天区基准因cal/dev天区隔离而全部退回全局，已保留旧结果并按[固定支持修正](v2_4_information_amendment.md)补上r-only与粗天区基准，没有按成绩搜索天区尺度。自然参考组成与等类人口的显式label-shift压力实验分别评分（所有方法和基准同时乘以uniform/global-cal先验比，再归一化），保留Brier、可靠性和天区bootstrap。',
        table(select(read('information_population'),group='all',population='natural_reference'),[('bands','波段',1),('method','方法',1),('n','N',1),('log_score','log score',1),('gain_r_sky','相对r/sky增益',1),('brier','Brier',1),('ece','分箱校准差',1)]),
        table(read('information_paired_bands'),[('left','原波段',1),('right','加入后',1),('method','方法',1),('mean','同星平均log增益',1),('low','天区95%下限',1),('high','上限',1),('clusters','天区数',1)]),
        '粗天区基准有4.21%的development对象回退到r层；全局、r-only、r/sky三个无颜色基准的log score依次为−0.528509、−0.528831、−0.531245，r/sky本身未改善这个留出人口。grz分类探针相对全局仍有0.134081nats增益，相对r/sky为0.136817；不能把较弱位置基准产生的额外差额称为颜色信息。',
        '固定分类探针的i/W1/W2增量与冻结距离分区的增量不完全相同，提示信息保留方式值得研究；但冻结分区原来在不同较大距离样本上拟合，不能把差别全部归因于某一模型结构。该matched i+WISE样本偏向较好测光/参考覆盖，只有9个dev天区；缓存误差处理固定，尚未测得grz物理信息上限。',
        '## 5. 独立经验模板的人口恢复实验',
        f'使用{design["templates"]}个development留出经验SED模板，与被冻结QRF训练身份交集为0；不是从QRF自身PDF采样生成真值。类别/贫金属分层按身份排序取样，平移到1、3.5、7.5、15、30kpc；一份clean和三份具有实际光测父样本误差的重复，共{design["rows"]}行。模板标签和绝对SED仍依赖chosen参考，因而这是一项条件实验，不是新暗端天体真值。',
        '四组预设人口：普通主导、巨星富集、贫金属富集、三类均衡；距离层权重固定(.25,.30,.25,.12,.08)。相同误差结构和模板用于所有人口，通过设计权重改变组成。比较点距离直方图、完整PDF和联合类别距离贡献，分别在未选样、可观测入口和near_scope后与相同选择后的注入真值比较。实际误差来自r/颜色最近邻，超出父样本支持的平移模板只能取得最近可用误差，不能当作真实观测噪声验证；[逐星族/距离噪声支持](../manifests/v2_4/population_noise_support.csv)给出匹配距离及r、误差范围，template_group=0/1/2/3依次表示普通矮星、贫金属矮星、演化星、other。',
        '**噪声外推主要出现在亮端巨星。** 巨星模板移到1/3.5kpc后，r中位数约10.16/12.88；97.9%/91.0%的噪声匹配距离超过一个预设尺度单位。该压力实验没有重新模拟饱和、探测、拥挤或完整影像选样，observable_parent只重施r/误差/支持门槛；这些近端巨星人数不能直接当作真实光测母样本的误差预报。clean分支仍保留模板测量误差，且巨星富集与均衡人口的PDF人数L1也分别为395.04和405.94，因此大偏差并非仅在噪声外推后才出现。',
        '区间使用N=1000条件下的Bernoulli预测方差；500次按模板身份整组的Bayesian bootstrap保持各人口组成，检查条件区间对注入人数的覆盖。该覆盖衡量固定经验SED与教师假设下的恢复能力，不包含真实银河系人口和教师系统误差的全部不确定度。额外输出共同±1sigma教师距离尺度变化后的真值人数。']
    summary=[]
    for pop in sorted({r['population'] for r in recovery}):
        for condition in ('clean','target_noise'):
            q=select(recovery,population=pop,condition=condition,selection='all',channel='all')
            near=[r for r in q if float(r['high_kpc'])<=10]
            summary.append(dict(population=pop,condition=condition,point_L1=sum(abs(float(r['point_count'])-float(r['expected_truth'])) for r in q),pdf_L1=sum(abs(float(r['bias'])) for r in q),
                true_near=sum(float(r['expected_truth']) for r in near),pdf_near=sum(float(r['pdf_count']) for r in near),mean_conditional_coverage=sum(float(r['template_bootstrap_coverage']) for r in q)/len(q)))
    parts.append(table(summary,[('population','人口设计',1),('condition','观测',1),('point_L1','点值人数L1误差',1),('pdf_L1','PDF人数L1误差',1),('true_near','注入近端人数',1),('pdf_near','积分近端人数',1),('mean_conditional_coverage','跨距离格条件覆盖均值%',100)]))
    parts+=['**在这组独立模板实验中，完整PDF并未使群体恢复普遍可靠。** 施加目标误差后，普通主导人口的距离格L1误差从点值89.74增至PDF122.36；贫金属富集人口从126.73增至148.79。巨星富集和均衡人口虽分别从462.55/470.86降至424.17/428.50，剩余误差仍大；两组各距离格的条件区间覆盖均为0。这里的0是固定模板压力实验的结果，不是对真实银河系覆盖率的测量。',
        '近端总数还会掩盖类别之间的抵消：巨星富集人口通过near_scope后，实际注入近端总数598.44，模型积分608.25，看起来只差9.81；但近端巨星注入243.91，演化星参考联合贡献只有31.93，矮星参考贡献则为410.59而注入269.06，另有88.26缺标签贡献。不能以总人数接近宣布巨矮边缘化已经校准。普通主导人口同样需逐距离格验收。',
        '所有数字以1000个设计对象为单位；L1是各距离格绝对计数误差之和，不是单星距离误差。表中覆盖均值包含六个距离格，其中50+kpc无注入对象；逐格覆盖和500次模板bootstrap区间完整保留，可由固定种子与逐对象缓存复算。不能仅以总近端人数看似接近就忽略各距离格与星族的相互抵消。完整后验能够保留信息，但当前恢复结果未通过普适人口统计的科学验收。',
        '[全部恢复与覆盖](../manifests/v2_4/population_recovery.csv) · [模板组不确定度](../manifests/v2_4/population_recovery_uncertainty.csv) · [教师尺度敏感性](../manifests/v2_4/population_teacher_sensitivity.csv)',
        '## 验证与当前决策',
        f'本轮{check["actual_checks"]}项实际产物检查通过；三个独立数学测试和一个原生FITS读取测试通过。核对原模型与推断源码冻结哈希、每块PSF/联合概率守恒、旧预测边缘复现、新预留身份隔离及已知注入人数。英文图件、源表哈希、PDF渲染和交付检查另有记录。',
        '没有新增模型候选。有SPDist的60个近端异常呈现具体观测/参考冲突，其余87个缺少该参考；现有证据不能证明统一放大远端权重即可解决。贫金属条件尺度在多个教师来源下持续，但独立Gaia配对稀少。现阶段的联合人数是模型参考贡献，经验后验不能除以一个巨矮比例就变成可自由换银河先验的似然。',
        '最终独立验证尚未完成：新身份结果保持未打开，缺独立新天空/系统成员核验；实际人口代表性、贫金属尺度和远端分支权重仍需要外部证据。本轮交付为诊断及科学恢复实验，不据此扩大应用目录或更换模型。',
        '[全部英文图件](v2_4_atlas.md) · [逐项要求及剩余限制](v2_4_requirements_result.md)']
    (D/'v2_4_result.md').write_text('\n\n'.join(parts)+'\n')
    atlas=['# v2.4 scientific-use atlas','English-only; Pal5 styling. PNG300dpi, PDF and source CSV/JSON. Scientific samples and denominators are untrimmed.']
    for path in sorted((M/'figures').glob('*.json')):
        q=json.loads(path.read_text());n=q['name'];atlas+=['## '+n,q['caption'],f'![{n}](../figures/v2_4/{n}.png)',f'[PNG](../figures/v2_4/{n}.png) · [PDF](../figures/v2_4/{n}.pdf) · [CSV](../manifests/v2_4/figures/{n}.csv) · [JSON](../manifests/v2_4/figures/{n}.json)']
    (D/'v2_4_atlas.md').write_text('\n\n'.join(atlas)+'\n');print('V2.4 REPORT COMPLETE')


if __name__=='__main__':main()
