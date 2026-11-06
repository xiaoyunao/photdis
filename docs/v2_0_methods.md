# Photdis v2.0 方法、资格与实际范围

## 数据及流程

主模型按颜色标定绝对星等分布 p(M_g | (g-r)_0,(r-z)_0)，再用目标的视星等求 μ=g₀−M_g 和 d/kpc=10^((μ−10)/5)。在无噪声且消光修正正确时，各波段同时变暗5等会使距离增大10倍，颜色和预测绝对星等分布相同。亮端参考能够标定这一关系。暗端限制指目标误差、缺测、路由和星族混合条件下的验证不足，不能仅凭视星等较暗判定A模型在颜色特征空间外推。现有共享波段误差传播已纳入测光误差，历史降深实验使用原模板星族。

后续所有新建科学样本入口和输出必须核实 Legacy DR11 TYPE=PSF，缺失或无法核实不通过。新样本记录非PSF和TYPE缺失排除数。冻结v2.0的历史例外仅作基线审计。

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

可靠 RVS 参数入口核对 desi_quality_reasons 和 desi_class_evidence_v3。要求STAR、RVS_WARN=0、SUCCESS/PRIMARY、SN_R≥10、0≤vsini<30 km/s，温度与重力及其正误差有效。分类不要求有效丰度。dwarf 为 Teff<7000 K 且 logg−sigma_logg>4.2，巨星参考子样本（内部字段 evolved）为 Teff<7000 K 且 logg+sigma_logg<3.2，其余合格恒星为其他合格光谱星。Unknown 表示未获得可靠光谱类别，可能已有未通过质量条件的光谱。当前样本包含宽温度范围，粗类别并未细分RGB/RC/HB/AGB。

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

带符号图统计每格 median(ε_d)，绝对值图统计 median(|ε_d|)，单位都为百分比。后者衡量误差幅度，两者不能互相替代。参考CMD使用M=m₀−μ_ref，预测CMD使用M=m₀−μ_phot；两张残差图均在参考坐标上。CMD与残差共享距离变量，其关联不能独自确立物理原因。

epsilon_d=100×(10^(delta_mu/5)−1)。百分比NMAD直接由逐星epsilon计算。主表含N、中位epsilon、NMAD、绝对误差中位数、P16/P84、20/50%尾率、1mag尾率及68/95%覆盖和全宽。1mag门槛等价于低估超过36.90%或高估超过58.49%。

NERSC data/v2_0/chosen_<model>.npz保存每模型每身份一行。sources_<model>.npz通过chosen_view_row链接到同一行的输入与预测，每个来源一条证据。视图保留身份、具体模型训练身份排除标志、切分、TYPE、输入质量、距离/视差、参考误差、原始区间、残差和支持状态。缺少某种模型标记时带_available=False。弱WISE full-PDF定位由chunk_start、chunk_row和arm组成。其他未缓存完整逐星PDF的模型，pdf_source_row定位冻结模型的可复现输入；cached_full_pdf=False明确其缓存状态。固定四个示例另存完整PDF。

## 实际范围和显示范围

范围表B包含min/P1/P5/P50/P95/P99/max、有限值与缺失数。parent保留原权重；训练范围使用各自真实身份。旧数据未保存的i或观测波段值以missing计数，不能从缺失列推断该波段覆盖。光学S/N由已有误差恢复的历史值只适用于正通量测量。绘图范围和图外数量保存在每图元数据/CSV，选样阈值独立于显示范围。

| 样本 | 变量 | N有限 | P1 | P50 | P99 | min—max |
| --- | --- | --- | --- | --- | --- | --- |
| parent_weighted | r | 26000 | 16.287 | 22.789 | 24.793 | 14.939—25.355 |
| parent_weighted | optical_error | 26000 | 0.001 | 0.101 | 0.215 | 0.000—0.217 |
| parent_weighted | rvs_teff | 0 | 缺失 | 缺失 | 缺失 | 缺失 |
| class_train | r | 237784 | 15.366 | 17.679 | 19.156 | 14.440—23.721 |
| class_train | optical_error | 237784 | 0.000 | 0.002 | 0.007 | 0.000—0.195 |
| class_train | rvs_teff | 237784 | 3237.699 | 5300.521 | 6707.430 | 2767.894—14977.801 |
| distance_train | r | 317003 | 15.257 | 17.763 | 19.968 | 14.390—21.336 |
| distance_train | optical_error | 317003 | 0.000 | 0.002 | 0.011 | 0.000—0.030 |
| distance_train | rvs_teff | 266672 | 3221.201 | 5315.296 | 7406.821 | 2700.049—15000.021 |
| test_chosen | r | 47756 | 15.693 | 17.991 | 20.048 | 14.376—22.565 |
| test_chosen | optical_error | 47756 | 0.000 | 0.002 | 0.011 | 0.000—0.137 |
| test_chosen | rvs_teff | 47756 | 3215.155 | 5230.771 | 7601.264 | 2708.311—15000.021 |

[完整范围表B](../manifests/v2_0/B_sample_ranges.csv) · [选样表A](../manifests/v2_0/A_selection_flow.csv) · [独立代码和数据哈希](../manifests/v2_0/snapshot.json)

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
