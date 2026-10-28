# 第一阶段：样本定义、信息质量与测试集保留

本阶段建立可审计的样本角色、检查覆盖范围并固定测试分区。**不训练距离模型或恒星分类器，不报告距离预测精度。** 数值阈值由 [stage1.json](../configs/stage1.json) 与版本锁定的 [gaia.py](../src/photdis/gaia.py)、[spectroscopy.py](../src/photdis/spectroscopy.py) 等模块共同定义；本文记录当前配置和物理依据。实际数量只引用运行产生的 `summary.json`，不在这里预填。

## 1. 本阶段的适用范围

- 使用 Legacy DR11 south sweep `11.0`，具体路径及原始星表版本见 [data_sources.md](data_sources.md)。网上当前 DR11 文档可能描述更新的小版本，实际 FITS 字段和 `REF_CAT=G3` 已另行检查。
- 起始域为 `TYPE=PSF`、`|b|>30°`、有限且 `0<=EBV<0.03`。这是 Pro 方案要求的高纬、低消光起点；尚不能代表低银纬银盘、拥挤场或高消光区。
- 不预先施加 K/M 矮星、巨星、BHB 等颜色框，也不施加逐星金属丰度或运动学先验。保留颜色空间中的不同分支，供后续判断哪些分支有足够标定。

**需要形态和测光质量预筛；颜色—颜色选择不能替代污染控制。** `PSF` 是图像模型类别，不是恒星身份证明，未分辨星系和类星体仍可进入。DR11 在 BRIGHT、GALAXY、CLUSTER 区域还会强制点源模型，因此这些区域尤其不能仅凭 PSF 断言纯度。Gaia 保留的部分亮星没有有效成像拟合，其光学流量来自 Gaia 颜色变换，逆方差为零；不能把它们当作独立的 Legacy 实测测光。[DR11 测量流程](https://www.legacysurvey.org/dr11/description/)

## 2. 测光母样本与 clean 子集

`grz_base` 对 g、r、z 每个波段要求：

1. `FLUX`、`FLUX_IVAR` 和 `MW_TRANSMISSION` 有限且大于零；`NOBS>=1`。
2. `SNR=FLUX*sqrt(FLUX_IVAR)>=5`。

这一步不要求 W1/W2，不要求视差为正，不排除任何金属丰度或颜色分支。负流量和缺测不会被伪造为星等；其原始观测字段在保留对象中仍可检查。

`grz_clean` 在 base 上增加下列**项目起始阈值**，不是巡天保证最佳的通用标准：

| 检查 | 当前条件 | 作用 |
|---|---|---|
| 光学掩膜影响 | 每个 grz `FRACMASKED<0.4` | 控制被掩膜像素影响 |
| 模型位于图像内的比例 | 每个 grz `FRACIN>0.7` | 控制边界／截断测量 |
| 邻源贡献 | 每个 grz `FRACFLUX<0.5` | 控制混叠污染 |
| DR11 像素标志 | MASKBITS 的 2、3、4、5、6、7、10 位均未置位 | 排除对应 grz 饱和、全曝光坏像素及拟合放弃区域 |

分数阈值继承旧项目的保守设置，但本次将它们与更宽的 base 层明确分开。`grz_precise` 是 base 加每个 grz `sigma_m<0.03 mag`；`grz_high_quality` 是 clean 与该精度条件的交集，便于单独统计质量掩膜造成的损失。

不采用 `MASKBITS==0`、`ANYMASK==0` 或统一 `RCHISQ<10` 作为母样本条件。BRIGHT（位1，Gaia G<13 邻域）、MEDIUM（位11，G<16 邻域）、GALAXY（位12）、CLUSTER（位13）保留作诊断。ANYMASK 表示任一曝光中被掩膜，不等于所有曝光都不可用。对这些字段一刀切会改变亮星与星团覆盖；程序记录旧 ANYMASK/RCHISQ 条件对高视差 SNR 样本的影响。[DR11 位掩码定义](https://www.legacysurvey.org/dr11/bitmasks/)

采用 sweep 已有的纳米 maggy、AB 星等体系，包括 WISE，不再次增加 Vega→AB 偏移。计算

`m0 = 22.5 - 2.5*log10(FLUX) + 2.5*log10(MW_TRANSMISSION)`，

`sigma_m = (2.5/ln(10))/SNR`。

这里的误差只是流量误差的一阶传播，未加入零点误差、消光误差或颜色之间共享波段的协方差；后续推断必须处理这些项。SFD 的总柱消光对近星是近似，本阶段通过低 EBV 限制其影响。[DR11 星表单位与字段](https://www.legacysurvey.org/dr11/catalogs/)

## 3. 红外信息分层

W1、W2 各自要求有效测光、`NOBS>=1`、`FRACFLUX<1`、对应 `WISEMASK=0`；**当前运行配置两者均要求 SNR>=5**。模块的通用 W2 默认值为3，但不覆盖本次配置。

| `photometry_tier` | 定义 |
|---|---|
| 0 | 有 grz base，但没有可用 W1 层 |
| 1 | grz base + 可用 W1，但不满足 W2 层 |
| 2 | grz base + 可用 W1 + 可用 W2 |

另提供高精度标志：`grz_w1_high_quality` 要求光学 high-quality 且 `sigma_W1<0.05 mag`；`full_high_quality` 再要求 `sigma_W2<0.10 mag`。Gaia 锚点本身不要求红外完整，避免 W2 深度造成样本偏置。

## 4. Gaia 锚点与弱／负视差扩展

使用整数 Gaia DR3 `source_id` 精确匹配，保存原始视差及误差。零点采用 Lindegren 等人的五／六参数修正：`p_corr = parallax - zpt`，单位均为 mas。模块预检查要求 `6<G<21`；五参数解（31）使用 `1.1<=nu_eff<=1.9`，六参数解（95）使用 `1.24<=pseudocolour<=1.72`，并以原作者函数返回有限零点作为最终 `zpt_valid` 判据。原函数对精确端点返回 NaN 时，该对象不会成为有效校准锚点。越界保留原测量和标志，不外推校准。[零点论文](https://arxiv.org/abs/2012.01742)，[原作者实现](https://gitlab.com/icc-ub/public/gaiadr3_zeropoint/-/raw/master/zero_point/zpt.py)

| 样本／标志 | 条件 |
|---|---|
| `finite_parallax`／`extended` | Gaia 已匹配，原始视差有限、误差有限且大于零；不限制视差符号或 SNR |
| `astro_quality` | 有限视差测量，五／六参数解，`0<RUWE<1.4`，`visibility_periods_used>=9`，非 `duplicated_source` |
| `anchor_astrometry` | astro_quality + 有效零点 + `p_corr/parallax_error>=20` |
| `anchor` | anchor_astrometry + grz_high_quality + 当前恒星核心污染防护 |
| `extended_quality` | extended + astro_quality + 有效零点 + grz_clean + 恒星核心污染防护；仍不限制视差符号或 SNR |

RUWE、可见期、重复源和 SNR 阈值是本阶段的质量选择，不能证明选择无偏。高 SNR Gaia 样本偏向附近、较亮、特定星族；覆盖诊断必须保留这一限制。弱／负视差仍携带似然信息，不转换为逐星 `1/parallax` 距离或绝对星等标签。[Gaia 团队的视差使用指南](https://arxiv.org/abs/1804.09376)

高质量正视差可得到 `M_g=g0-10+5*log10(p_corr)`，其保存的 `sigma_mg` 只包含形式测光误差和形式视差误差。它不是未来距离模型的总预测不确定性。

Gaia `IN_QSO_CANDIDATES`／`IN_GALAXY_CANDIDATES` 保留为**候选污染标志**。当前严格 anchor 和 extended_quality 暂时排除这些标志，也排除无歧义关联的 Redrock QSO/GALAXY 分类；宽扩展表仍保留它们。该光谱分类没有另加 SNR 门槛，尤其低 SNR 时不能视为绝对真值。Gaia 候选标志也不等同于确认的非恒星，此保守排除会损失真恒星，必须在统计中单独显示，不能宣称已获得完整或无污染恒星样本。

## 5. 光谱辅助样本

光谱提供参数和分支诊断，不作为独立绝对距离真值。DESI 内嵌成像属于旧 Legacy release，不能直接用它替代当前 DR11 测光。DESI 的 RVTAB／FIBERMAP／GAIA 只在**同一原始 MWS 文件内**按行对应，并检查 TARGETID 顺序；不同 VAC 不按行号拼接。[MWS 官方数据模型](https://desi-mws-dr1-datamodel.readthedocs.io/en/latest/mwsall.html)

DESI 审计池保留全部 PRIMARY，包括 Redrock 的 QSO/GALAXY 分类。参数质量要求：`RR_SPECTYPE=STAR`、`RVS_WARN=0`、`SUCCESS=True`、`SN_R>=10`、`0<=VSINI<30 km/s`，TEFF/LOGG/FEH 有限且非占位值，三个误差有限且大于零。未通过的观测仍保留。其 logg、Teff 相关金属丰度偏差和部分低重力／冷星参数问题需要单独验证，不能把原始 FEH 当作精确真值；低 SNR 的 Redrock 分类也不是完美分类器。[DESI DR1 恒星星表论文](https://arxiv.org/html/2505.14787v4)

APOGEE 使用 DR17 校准后的命名列 TEFF、LOGG、FE_H 及其误差；要求 `SNR>=50`、`FE_H_FLAG=0`、有效参数与正误差。[APOGEE 参数说明](https://www.sdss4.org/dr17/irspec/parameters/)

当前 APOGEE 质量层排除以下已定义的位，保留完整原始标志和未通过对象：

| 字段 | 排除位 | 含义 |
|---|---|---|
| ASPCAPFLAG | 23、24 | STAR_BAD、CHI2_BAD |
| ASPCAPFLAG | 31、32、33、36 | 无 ASPCAP 结果、缺少 apStar、无模型网格、FERRE 失败 |
| ASPCAPFLAG | 40、41 | 不适于标准分析的目标、疑似复合光谱 |
| STARFLAG | 0、3、18、22 | 大量坏像素、极亮邻星、错误 RV 合并、RV 失败 |

这是单星大气参数质量层的保守选择，不是要求所有位为零；例如 persistence 警告本身不删除对象。疑似复合光谱仍在审计池中，避免把其参数当作可靠单星标签。`EXTRATARG` 位4的原始重复标志保留，并按正 Gaia ID、否则 APOGEE_ID 全局选择最高 SNR 观测。[APOGEE DR17 官方位定义](https://www.sdss4.org/dr17/irspec/apogee-bitmasks/)

DESI 与 APOGEE 参数分别保存，不平均成统一真值。临时冷星分支标志要求 Teff<7000 K、质量合格，并使一倍 logg 误差区间完全位于 `logg>4.2`（矮星）或 `<3.2`（巨星）一侧；这是诊断标签，不是完备的演化阶段分类。BHB、蓝离散星、白矮星和 RC 需要独立定义。

SpecDis/SPDist 目前只登记为潜在比较资源，不参与训练或本阶段距离标签。尤其 Gaia 标定的 SpecDis 不能被当作独立于 Gaia 的绝对距离检验。

## 6. 身份、分区与输出解释

优先精确 Gaia ID；缺失时使用1角秒的保守坐标关联，并记录方法、歧义及身份冲突。不能用近邻距离覆盖两个已知 Gaia ID 的冲突。只有无歧义且跨巡天一致的 ID 才可恢复到主身份，之后重新检查 Gaia、成员目录及分区。未知 ID、不同观测历元、高自行和拥挤场仍可能降低匹配完整性；坐标近邻不是身份正确率的保证。

固定 NSIDE=8 的空间 HEALPix 分区，种子20261028，约70%训练、15%验证、15%空间测试。已登记星团成员、星团背景隔离区域、宽双星两端及身份歧义隔离优先于空间分区。`partition=3` 同时包含外部参考和保守隔离对象，**不能把该整张表都称为有独立距离真值的测试集**。`cluster_region` 不是成员资格。尚未登记的星团／双星不能被宣称已全部隔离。

Gaia 锚点是扩展样本的子集，光谱和外部角色又可与两者重叠，不应相加。持久化的 `parent` 包含 Gaia 匹配、光谱匹配、外部保留对象及约1/128的确定性覆盖参考抽样；它不是所有通过 grz_base 的纯测光对象全表。完整输入选择计数保存在审计中。角色表使用最终去重后的对象，每次使用都必须同时选择角色与 partition。

图只展示真实输出的计数与分布：天空散点最多每面板200,000行，按整数 ID 确定性选择；直方图使用全表中合格有限测量。各图明确有限数、显示范围和重叠关系。正式科学结论必须在实际运行和身份／分区检查完成后给出。
