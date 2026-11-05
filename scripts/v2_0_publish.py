#!/usr/bin/env python3
"""Build the inspectable v2.0 narrative and artifact audit from collected tables."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manifests/v2_0'
D=ROOT/'docs'


def read_csv(name):
    with open(M/name) as f:return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str,r))+' |' for r in rows]])


def main():
    snapshot=json.loads((M/'snapshot.json').read_text())
    config=json.loads((ROOT/'configs/v2_0_snapshot.json').read_text())
    app=read_csv('C_application_fields.csv')
    total=sum(int(r['unique_rows']) for r in app)
    predicted=sum(int(r['predicted']) for r in app)
    supported=sum(int(r['supported']) for r in app)
    ratios=[]
    for numerator,n in [('应用目录对象',total),('实际距离预测',predicted)]:
        ratios.append([numerator,f'{n:,}',f'{100*n/config["dr11_denominators"]["unique_sources"]:.4f}%',
                       f'{100*n/config["dr11_denominators"]["unique_psf"]:.4f}%'])
    scope="""# Photdis v2.0 阶段成果冻结范围

报告基准为提交 `8b6f744`，本轮统一统计以冻结 `expanded_A` 为主诊断模型。应用目录来自 `baseline_v1`。其后的 frozen/expanded ABC、matched grz/griz、r₀ 和弱 WISE 具有各自的训练、评价身份及模型标识。本次 v2.0 命名用于阶段报告。

## 规模与分母

"""
    scope+=table(['项目','数量','解释'],[
        ['应用天区','14','13个DECam、1个北方BASS/MzLS'],
        ['唯一应用目录对象',f'{total:,}','包含有距离及明确未预测的对象'],
        ['实际DECam预测',f'{predicted:,}','已运行的baseline_v1'],
        ['北方未预测',f'{total-predicted:,}','route=-1，距离NaN'],
        ['提取扫描','1,600 south sweep','参考源寻找的输入扫描'],
        ['扫描输入行','3,561,828,221','输入读取规模'],
        ['参考组装对象','3,565,499','资格筛选前的唯一身份'],
        ['新分类／距离训练','138,080 / 153,403','资格独立'],
        ['expanded_A分类／距离训练','237,784 / 317,003','历史+新训练'],
        ['新cal分类／chosen','38,408 / 43,112','不同证据入口'],
        ['新test分类／chosen','42,964 / 47,756','空间留出开发测试'],
        ['test Gaia／负视差','49,327 / 4,139','观测视差概率评价'],
        ['matched距离训练／测试','95,101 / 43,706','相同身份波段对照']])
    scope+='\n\n'+table(['分子','数量','占DR11全部唯一源','占DR11唯一PSF源'],ratios)
    scope+="""

分母为 DR11 官方按南北区域消重后的 3,920,661,806 个唯一源及 2,198,481,904 个 PSF 形态源。PSF 是形态分类。以上为数量尺度比，Photdis 同口径全 DR11 选样分母及其完成率保留 null。掩膜后的有效面积未统计，场边界也仅表示几何配置。官方表与南北解析规则于本轮核对。[DR11 官方说明](https://www.legacysurvey.org/dr11/description/#morphological-classification)

"""
    scope+=table(['路由','数量','占DECam预测比例'],[
        [name,f'{sum(int(r["route_"+str(i)]) for r in app):,}',f'{100*sum(int(r["route_"+str(i)]) for r in app)/predicted:.2f}%']
        for i,name in enumerate(['A grz','B grzW1','C grzW1W2'])])
    scope+='\n\n[完整14场边界和状态表](../manifests/v2_0/C_application_fields.csv) · [机器快照](../manifests/v2_0/snapshot.json)\n'
    (D/'v2_0_scope.md').write_text(scope)

    methods="""# Photdis v2.0 方法、资格与实际范围

## 数据及流程

应用使用 Legacy DR11 DECam grz 测光，WISE 为可选测量。参考标签分别来自 Gaia DR3 高精度锚、DESI DR1 MWS RVS v1.0、SpecDis v2.1 与 SPDist v1.0。冻结配置和输入 SHA256 见快照。native LS ID、Gaia source ID 和 DESI target ID 分开保存，原始配对、唯一ID回退和位置／历元恢复保留为字段。与历史训练身份、历史天区及冲突身份重合的对象隔离。粗类别与距离资格各自判断。

## 光学和形态入口

| 条件 | 应用母样本 | v1.4扩展参考 |
|---|---|---|
| TYPE | PSF | 不要求PSF，已审计构成 |
| 银纬和消光 | abs(b)>30°，0≤EBV<0.03 | 相同，南天原生RELEASE=11010 |
| g/r/z测量 | flux、ivar、MW transmission有限且正，NOBS≥1，S/N≥5 | 相同 |
| 光学质量 | FRACMASKED<0.4，FRACIN>0.7，FRACFLUX<0.5 | 相同 |
| MASKBITS | 排除2、3、4、5、6、7、10 | 相同 |
| 距离训练 | max(sigma_g,sigma_r,sigma_z)<0.03 mag | 相同 |
| 分类标签和评价 | 独立资格 | 不要求距离或0.03 mag训练门槛 |

星等消光修正使用各波段 MW_TRANSMISSION。主要亮度分层是观测 r，r₀ 对照输入是消光改正值。负 WISE 通量保留在原始测量和状态字段中，无法定义的星等列保留缺失。WISE颜色只对合格 S/N≥5 测量生成。

## Gaia 和光谱距离

Gaia 锚要求精确 DR3 身份匹配、有限视差和正误差、31/95解、0<RUWE<1.4、VISIBILITY_PERIODS_USED≥9、非重复源、有效零点。零点域为6<G<21，31解 nu_eff∈[1.1,1.9]，95解 pseudocolour∈[1.24,1.72]。再要求校正视差正且 S/N≥20，排除已标记非恒星风险。距离为1/parallax_corr，mas对应kpc。弱和负视差仅进入观测视差评分。

SpecDis 使用唯一匹配、BINARY_FLAG=1、DIST和DISTERR有限正、DISTERR<DIST。SPDist 使用 FLAG_GOOD、QFLAG_INPUT、有效星等/误差/分位数，10<G₀<30、−8<MG50<15、MG84≥MG16且差<4 mag。SPDist 距离来自其G₀−MG50，误差含0.167 mag项。Chosen 规则是高精度Gaia优先、否则SpecDis。分来源表允许同星多行，独立星数按身份消重。SpecDis/SPDist可能共享上游数据，参考点区间覆盖只表示与目录估计一致。

## 类别和丰度

可靠 RVS 参数入口核对 desi_quality_reasons 和 desi_class_evidence_v3。要求STAR、RVS_WARN=0、SUCCESS/PRIMARY、SN_R≥10、0≤vsini<30 km/s，温度与重力及其正误差有效。分类不要求有效丰度。dwarf 为 Teff<7000 K 且 logg−sigma_logg>4.2，evolved 为 Teff<7000 K 且 logg+sigma_logg<3.2，其余合格恒星为 other/transition。Unknown 显式保留。当前样本包含宽温度范围，粗类别并未细分RGB/RC/HB/AGB。

贫金属诊断分别用原始RVS、限制校正域后的原始RVS、同域校正RVS。校正域为4200≤Teff<6600 K、1.8≤logg<5.5，贫金属门槛为FeH<−1.5。域外校正值保持缺失。

## 切分、模型和概率表示

旧参考按整场切分；新参考按10°×10°格，seed=20261104、70/15/15哈希分配。比例定义格的分配规则。九个有chosen证据的test格已反复用于开发，记录为空间留出开发测试。

| 模型部分 | 固定设置 |
|---|---|
| A/B/C | grz两色 / 加z−W1 / 再加W1−W2 |
| QRF | 128trees，leaf≥32，depth≤20，max_features0.75，bootstrap=False |
| 目标与表示 | Legacy Mg，0.1mag压缩直方图 |
| 测光传播 | 主QRF使用Sobol128，共享波段误差 |
| matched grz/griz | 相同训练/测试身份，共享四波段扰动 |
| r₀ | expanded_A增加r₀，317003训练；候选特征数从1变2的维数效应保留 |
| 弱WISE | 122217联合模板、32次光学扰动、96邻模板、固定5%通量散布 |

WISE质量使用有效观测/ivar/transmission、WISEMASK=0、FRACFLUX<1；五种状态分别为缺测、无效、质量失败、合格弱测量、S/N≥5。i使用自己的掩膜和污染条件，见代码。应用mu和mu_cal分别表示原始QRF分位数与cal扩展区间，后者没有产生新的PDF。expanded模型和后续实验的分位数保持raw标识。

## 统一统计和逐星视图

epsilon_d=100×(10^(delta_mu/5)−1)。百分比NMAD直接由逐星epsilon计算。主表含N、中位epsilon、NMAD、绝对误差中位数、P16/P84、20/50%尾率、1mag尾率及68/95%覆盖和全宽。1mag门槛等价于低估超过36.90%或高估超过58.49%。

NERSC data/v2_0/chosen_<model>.npz保存每模型每身份一行。sources_<model>.npz通过chosen_view_row链接到同一行的输入与预测，每个来源一条证据。视图保留身份、具体模型训练身份排除标志、切分、TYPE、输入质量、距离/视差、参考误差、原始区间、残差和支持状态。缺少某种模型标记时带_available=False。弱WISE full-PDF定位由chunk_start、chunk_row和arm组成。其他未缓存完整逐星PDF的模型，pdf_source_row定位冻结模型的可复现输入；cached_full_pdf=False明确其缓存状态。固定四个示例另存完整PDF。

## 实际范围和显示范围

范围表B包含min/P1/P5/P50/P95/P99/max、有限值与缺失数。parent保留原权重；训练范围使用各自真实身份。旧数据未保存的i或观测波段值以missing计数，不能从缺失列推断该波段覆盖。光学S/N由已有误差恢复的历史值只适用于正通量测量。绘图范围和图外数量保存在每图元数据/CSV，选样阈值独立于显示范围。

"""
    ranges_rows=read_csv('B_sample_ranges.csv')
    methods+=table(['样本','变量','N有限','P1','P50','P99','min—max'],[
        [r['sample'],r['variable'],r['finite'],f"{float(r['p1']):.3f}" if r['p1'] else '缺失',
         f"{float(r['p50']):.3f}" if r['p50'] else '缺失',f"{float(r['p99']):.3f}" if r['p99'] else '缺失',
         f"{float(r['min']):.3f}—{float(r['max']):.3f}" if r['min'] else '缺失']
        for r in ranges_rows if r['sample'] in ('parent_weighted','distance_train','class_train','test_chosen') and r['variable'] in ('r','optical_error','rvs_teff')])
    methods+='\n\n[完整范围表B](../manifests/v2_0/B_sample_ranges.csv) · [选样表A](../manifests/v2_0/A_selection_flow.csv) · [独立代码和数据哈希](../manifests/v2_0/snapshot.json)\n'
    methods += '''
## 报告复现入口

在NERSC规范checkout提交下列阶段，后一步依赖前一步成功。报告阶段只读取冻结产物、构造派生表和图，固定PDF个例的小量前向复现由render阶段执行。

```bash
sbatch scripts/run_v2_0.sh collect
sbatch --dependency=afterok:COLLECT_JOB scripts/run_v2_0.sh render
sbatch --dependency=afterok:RENDER_JOB scripts/run_v2_0.sh finalize
```

将NERSC `data/v2_0/manifests/` 与 `data/v2_0/figures/` 分别同步到本地 `manifests/v2_0/` 与 `figures/v2_0/`，然后执行下列发布入口。它从已核对的紧凑表生成scope、methods、status、atlas。

```bash
python3 scripts/v2_0_publish.py
```

每张图的CSV与JSON位于 `manifests/v2_0/figures/`；PNG/PDF位于 `figures/v2_0/main/` 或 `appendix/`。统一逐星表及完整PDF保留在NERSC的 `data/v2_0/` 或原冻结目录。
'''
    (D/'v2_0_methods.md').write_text(methods)

    field_configs={row['field']:json.loads((ROOT/row['config']).read_text()) for row in app}
    (M/'field_configurations.json').write_text(json.dumps(field_configs,ensure_ascii=False,indent=2)+chr(10))
    keys=[];index=[]
    for file in sorted(M.glob('performance_*.csv')):
        values=read_csv(file.name)
        index.append(dict(model=file.stem.removeprefix('performance_'),file=file.name,rows=len(values)))
        keys.extend(row for row in values if row['split']=='development_test' and row['reference']=='chosen'
                    and row['group'] in ('all','dwarf','evolved','other','unknown','TYPE_PSF'))
    for filename,rows in [('D_performance_index.csv',index),('D_performance_key.csv',keys)]:
        columns=list(dict.fromkeys(k for row in rows for k in row))
        with open(M/filename,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    performance=read_csv('performance_expanded_A.csv')
    main_rows=[r for r in performance if r['split']=='development_test' and r['reference']=='chosen']
    by={r['group']:r for r in main_rows}
    status="""# Photdis v2.0 阶段状态和科学结果

流程已覆盖从DR11原生测光到距离分布及目录输出。多星族参考、身份审计、空间切分、独立分类/距离/弱视差资格及一系列固定信息对照均已完成。应用目录保留baseline_v1身份；下表主诊断来自expanded_A。

## 主要性能

"""
    status+=table(['人群','N','中位距离偏差','百分比NMAD','中位绝对误差','1mag严重尾率','68%覆盖'],[
        [label,by[key]['n'],f"{float(by[key]['median_pct']):+.2f}%",f"{float(by[key]['nmad_pct']):.2f}%",
         f"{float(by[key]['median_abs_pct']):.2f}%",f"{float(by[key]['tail1mag']):.2%}",f"{float(by[key]['coverage68']):.2%}"]
        for key,label in [('all','全体'),('dwarf','矮星'),('evolved','演化星'),('other','其他/过渡'),('unknown','未知')]])
    status+="""

总体偏差接近零时，演化星仍有−91.30%的中位距离偏差。其6.49% NMAD是围绕这一巨大偏差的散布。可靠矮星的1.95%严重错误率是对象比例，中位绝对百分比误差约7.41%。未知类中位偏差接近零但约五分之一严重失败。

## 新增形态审计

"""
    morph=read_csv('morphology.csv')
    kinds=sorted({r['TYPE'] for r in morph})
    index={(r['sample'],r['TYPE']):r['n'] for r in morph}
    status+=table(['TYPE','全组装参考','新距离训练','cal chosen','test chosen'],[
        [k,index.get(('assembled',k),'0'),index.get(('train:fit_eligible',k),'0'),
         index.get(('cal:chosen',k),'0'),index.get(('test:chosen',k),'0')] for k in kinds])
    psf=by['TYPE_PSF']
    status+=f"\n\nPSF子集chosen测试有{int(psf['n']):,}颗，严重尾率{float(psf['tail1mag']):.3%}，中位距离偏差{float(psf['median_pct']):+.3f}%。与全体的差别很小，扩展参考入口的非PSF少数对象不能解释演化星的大规模失败。形态audit保留全部旧成绩和训练身份。\n"
    status+="""
## 固定实验结论

- 扩充训练数据，A总体尾率9.82%→9.68%，演化星95.44%→96.56%。
- matched grz/griz同星总体9.71%→9.57%，演化星仍97.46%→96.52%。
- r₀使演化星尾率96.56%→89.85%，但距离仍约十一分之一；矮星尾率1.95%→2.16%。18–20无W1的47颗演化星无净收益。
- 真实弱WISE固定原型在主目标层净损伤，弱W1 39.13%→52.17%，弱W2与合格W1 20.60%→20.93%；后者28颗演化星仍全部严重错误。
- 既有高S/N WISE匹配训练对照保留于主图10b。信息增益与测量覆盖分别统计，每个实验使用自己的匹配身份。
- 贫金属矮星在校正域中仍有约+23%偏差、约43%的68%覆盖。原始/同域原始/校正丰度三种定义分别显示。

## 限制和未完成范围

"""
    status+=f"- 当前v1应用的名义颜色支持域比例为{supported/predicted:.2%}，域内也存在演化星失败。预设数值层级和全星族研究质量条件未全部通过。\n"
    status+="""
- 新chosen测试只有九个天区格，最大格占45.7%，这些是开发证据。天区bootstrap反映当前选择下的波动。
- 20–22等只有618个chosen参考且无可靠演化星标签；22–25等只有7个chosen、3个合格Gaia视差、0个可靠类别参考。
- 目录参考点覆盖、潜在真实距离覆盖与数值积分收敛各自保留不同名称。光谱距离目录存在选择效应和相关上游信息。
- 全DR11同口径完成率和掩膜有效面积未统计，北方491184行具有明确未预测状态。
- 旧星团与星流证据来自历次特定模型/开发核查，索引保留在原文档。它们不足以充当当前expanded_A全星族独立检验。本次整理不补做局部目标实验。
- 后续模型方向留待阶段汇报讨论，本轮冻结现有证据、表格和图件。
"""
    status+='\n[性能表D](../manifests/v2_0/performance_expanded_A.csv) · [完整形态表](../manifests/v2_0/morphology.csv) · [原阶段目录说明](baseline_v1_catalog.md)\n'
    (D/'v2_0_status.md').write_text(status)

    registry=json.loads((M/'figure_registry.json').read_text())
    atlas=['# Photdis v2.0 图集','', '主诊断expanded_A，应用范围baseline_v1。每图附300dpi PNG、PDF、CSV和元数据。PDF可用于投影或排版。对象数量、分箱、缺失和图外数量以实际输出表为准。','']
    for section,title in [('main','主图'),('appendix','附图')]:
        atlas+=['## '+title,'']
        for name,record in registry.items():
            if record['section']!=section:continue
            atlas += ['### '+name,'',f'![{name}](../{record["png"]})','',record['caption'],'',
                      f'[PDF](../{record["pdf"]}) · [CSV](../{record["csv"]}) · [元数据](../manifests/v2_0/figures/{name}.json)','']
    atlas += ['## 已有外部证据索引','',
              '- [早期统一基线](baseline_v0_result.md)、[当前应用目录范围](baseline_v1_catalog.md)和历次cluster_rgb文档保留其开发身份。外部目标未被纳入当前主模型的独立绝对精度证明。',
              '- 固定PDF附图包含普通矮星、校正域贫金属矮星、演化星失败和宽/多峰，身份由固定哈希规则选取。','']
    (D/'v2_0_figure_atlas.md').write_text('\n'.join(atlas))
    print('Published scope, methods, status and atlas',len(registry))


if __name__=='__main__':
    main()
