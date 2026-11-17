# Photdis DR11 · 发布准备与论文定位

本轮依据 GPT Pro 对 Final Science Analysis 的完整评审
`6c0ffd81`，以提交 `ef329cc6` 为冻结基线。包含相对已发布距离
产品的增量、光谱参考约束下的选样表现，以及可执行的目录和后验
访问路径。

## 同星比较口径

分母固定为原生北天和南天低纬低消光窗的 1,150,705 个唯一 DR11
PSF。源属于 v3.10 矮星工作样本，Photdis 中位距离在 2–5 kpc，
Gaia 关联经位置核验，视差或天体测量不足以满足既定几何距要求。
使用原生 DR3 源号与发布表精确连接，每个源都进入查找。
Gaia DR3 使用了 EDR3 的源列表，三个查找都保留 64 位整数键。
[冻结样本与 Gaia 状态](final_science.md) ·
[Gaia DR3 发布说明](https://arxiv.org/abs/2208.00211)

| 产品 | 发布表 | 数值口径 |
| --- | --- | --- |
| Bailer-Jones EDR3 | GAVO `gedr3dist.main` | geometric 和 photogeometric 各自的 16、50、84 百分位，pc 转 kpc |
| Gaia DR3 GSP-Phot | ESA `gaiadr3.astrophysical_parameters` | distance 与上下 68% 界限，pc 转 kpc |
| StarHorse 2022 | Gaia@AIP `gaiaedr3_contrib.starhorse` | dist16、dist50、dist84，原生 kpc |

产品表有行、公布距离为正、区间有限且排序有效分别统计。
区间比较以各产品公布 68% 区间和落入区间的参考中心为准。距离
偏差、NMAD 和 >1 mag 错距率在同一对照样本上计算。使用冻结
留出星的合格 SpecDis 和 SPDist 独立分层，并分别给出两两
共同样本与所有方法均可用的共同样本。参考产品之间存在标定相关性，配对数量不能在不同参考来源之间相加。Photdis 预测距离参与
定义目标样本，因此即使参考距离约束相同，配对结果也属于选样后的
验证。附表保留全部参考距离，以及 0–2、2–5、5–10、10 kpc
以上的拆分。
[Bailer-Jones EDR3 数据](https://bailer-jones.www3.mpia.de/gedr3_distances.html) ·
[GSP-Phot 论文](https://arxiv.org/abs/2206.06138) ·
[StarHorse 发布](https://gaia.aip.de/metadata/gaiaedr3_contrib/starhorse/)

## 已发布距离的覆盖与实际增量

对全部 1,150,705 个源完成查找。Bailer-Jones 几何目录已经给出
几乎所有对象的距离估计。Photdis 在这个样本上的主要增量，
是独立的原生 LS 测光距离和完整后验。同一工作样本里有 3,094
个源在这四种公开距离估计中均无有效中位数，其中 2,361 个为
18 ≤ r < 20。这个数只对应本轮查询的产品和样本。

| 方法 | 全样本有有效距离 | 18 ≤ r < 20 有有效距离 |
| --- | ---: | ---: |
| Photdis | 1,150,705 / 1,150,705 | 815,822 / 815,822 |
| Bailer-Jones geometric | 1,147,611 (99.7%) | 813,461 (99.7%) |
| Bailer-Jones photogeometric | 1,146,841 (99.7%) | 812,895 (99.6%) |
| Gaia GSP-Phot | 674,497 (58.6%) | 344,143 (42.2%) |
| StarHorse 2022 | 471,713 (41.0%) | 145,389 (17.8%) |

共有 469,696 个源同时具备四种对照估计，说明共同样本的
入选本身会受到公开目录深度影响。距离表有行、有效数值、可用区间和
各种产品的并集均可在底表审查。
[匹配与有效距离覆盖](../manifests/release_readiness/comparison/coverage.csv) ·
[产品重叠](../manifests/release_readiness/comparison/method_overlap.csv)

### 使用同样的光谱星衡量距离表现

下表为 18 ≤ r < 20、光谱确认矮星、参考 2–5 kpc 且所有方法
距离均可用的留出对象。SpecDis 有 4,749 颗，SPDist 有 4,456 颗。

| 方法 | SpecDis 偏差 / NMAD | SPDist 偏差 / NMAD | SpecDis >1 mag | SpecDis 68% 覆盖 |
| --- | ---: | ---: | ---: | ---: |
| Photdis | −3.1% / 7.6% | +0.7% / 7.5% | 0.02% | 78.5% |
| Bailer-Jones geometric | −14.3% / 24.3% | −10.8% / 25.4% | 16.6% | 65.2% |
| Bailer-Jones photogeometric | −2.5% / 10.85% | +1.4% / 11.04% | 0.23% | 78.9% |
| Gaia GSP-Phot | −31.1% / 10.0% | −28.6% / 10.6% | 28.5% | 2.8% |
| StarHorse 2022 | −7.7% / 14.3% | −3.8% / 15.1% | 3.8% | 51.3% |

最有意义的基准是 Bailer-Jones photogeometric。它与 Photdis
的距离尺度在这个样本上接近。将共同条件放宽为只需这两个
方法可用，SpecDis 参考的 10,904 个暗星矮星上，散布为
Photdis 7.9% 和 BJ 11.3%。这个结果量出了两种方法在这一
选样里的区别。由于工作样本先按 Photdis 的预测距离截断，
同星对比的优劣仍须和参考距离未限定的行、保留率及星族误差一起读。
图中采用 1–12 kpc 视域，统计包含视域外对象。前五个面板用
共同 SpecDis 星，右下角单列 Photdis 对 SPDist 的对照。

只用分类器完全留出的天区，SpecDis 和 SPDist 共同样本分别
为 3,008 和 2,806 颗，Photdis 散布约 7.8%–7.9%，
BJ photogeometric 为 11.36%–11.48%。在原工作选样上放宽
参考距离限制，5,097 个 SpecDis 对象的散布为 8.12% 与 11.49%，
严重错距率升为 1.45% 与 1.57%，两种方法的错距率差距缩小。配对表中 `all_role4` 标记额外的分类器留出天区，原表
保留 Final Science Analysis 的距离模型留出口径。
[F12 同星直接比较](../figures/release_readiness/F12_published_distances.png) ·
[配对底表](../manifests/release_readiness/comparison/paired_distances.csv)

## 星族使用表的分母

北天和南天低纬窗保留原工作索引、分类阈值、参考匹配和留出
天区。普通矮星这里指有可靠光谱金属丰度且 [Fe/H] ≥ −1.5
的矮星。贫金属矮星单列。Fe/H 缺失或误差超过 0.3 dex 的
矮星也单列。中间重力层要求原标签为其他恒星，光谱
3.2 ≤ log g < 4.2，并且误差不超过 0.3 dex。非恒星须有
Redrock 质量确认，标签冲突单列。

保留率以合格参考且有原生 QRF 支持的对象为分母。逐参考距离
计算 2–5、5–10 kpc 的去向，包括 2 kpc 内、10 kpc 外和
2–10 kpc 的完整 PDF 概率质量。观测框架的星族混入只描述有
相关光谱证据的样本。光谱未分类对象和不均匀的靶选同时限制了
外推。[原工作索引方法](v3_10_protocol.md) ·
[主序与转折星的同星对照](v3_12_result.md)

### 使用条件的实测结果

SpecDis 以参考距离分层。下表的保留分子包括通过工作索引
门槛的对象，保留对象尚未按预测中位距离切到同一层。

| 天区与参考距离 | 光谱星族 | QRF 支持 → 通过门槛 | 通过者中位偏差 | NMAD | 错距 >1 mag | 68% 覆盖 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 北天，2–5 kpc | 普通矮星 | 17,399 → 11,919 | −3.7% | 8.2% | 0.5% | 80.9% |
| 北天，2–5 kpc | 贫金属矮星 | 2,347 → 224 | +25.4% | 14.4% | 2.2% | 37.9% |
| 北天，2–5 kpc | 中间重力 | 1,331 → 168 | −35.3% | 13.3% | 45.8% | 19.6% |
| 南天低纬窗，2–5 kpc | 普通矮星 | 5,998 → 3,946 | −2.4% | 8.6% | 0.4% | 79.2% |
| 南天低纬窗，2–5 kpc | 贫金属矮星 | 164 → 6 | +22.3% | 20.4% | 0.0% | 66.7% |
| 南天低纬窗，2–5 kpc | 中间重力 | 431 → 58 | −29.6% | 15.3% | 31.0% | 27.6% |

北天普通矮星保留约 69%，贫金属矮星仅约 10%，且后者
的偏远程度在选择后增大。参考在 5–10 kpc 的北天中间重力
星中，有 230 个通过门槛，全部被放入预测 2–5 kpc 层。
南天低纬窗相应为 60/63。北天近层还检出来自参考 5–10 kpc
的 4 个已标巨星。错误分支会改变工作样本的空间分布。

在全部有光谱标签或合格 Redrock 类型的预测 2–5 kpc 参考
重叠中，北天有 14,772/16,035 个矮星、584 个中间重力对象、
127 个巨星、11 个非恒星。南天低纬窗相应为 4,268/4,530、
151、2、9。另有 119,011 和 33,702 个参考框架对象未分类。
这些构成统计描述具体光谱选源。整体目录仍缺对星族混入率的
无偏直接测量。
[按参考距离的保留、误差、硬迁移与 PDF 权重](../manifests/release_readiness/population/retention_migration.csv) ·
[按原生天区的参考构成](../manifests/release_readiness/population/reference_composition.csv)

## 距离产品与数据访问

[机器可读字段和适用条件](../configs/photdis_dr11_release.json)
固定了归属掩码、PDF 状态、模型版本、区间概率、距离层、科学
选择和面积。大数据在 NERSC 项目根目录下，入口路径为
`data/v3_9/delivery/unified_parts.csv`。原始分区的
`photo_row` 与 `ls_id` 一起核验。冻结南天的 PDF 和测光同
行。面积用南北两套原生随机点统计，按归属区、银纬和深度查询。
[原始目录结构](v3_9_protocol.md) · [有效面积和科学样本](final_science.md)

**冻结原始 PDF**适用于大目录，68% 区间的参考中心包含率依次为
亮红近端 Gaia 61.2%、北天 2–5 kpc SpecDis 77.4%、南天低纬
窗 75.3%。使用 Gaia 作参考的两个工作样本值约为 89.4% 和
91.7%，光谱与 Gaia 框架的选择与距离误差有差异。

**已校准亮红近端伴随产品**仅用于符合其四个颜色/W1 分组的对象。
它的原生归档共有 1,142,280 行，合并时继承原发布归属。
校准调整距离模数分布宽度并保留中位数。此前按参考误差积分
的 68% 区间相符质量从 47.7%–60.0% 提高到 67.0%–68.5%，
不同于上段的参考中心包含率。两个产品均可访问，2–5 kpc
工作样本使用冻结原始 PDF。
[四组校准和独立天区复核](../manifests/v3_8/reliability/interval_calibration.csv) ·
[伴随产品归档](../manifests/v3_9/near_calibration/summary.json)

在 NERSC 提交 Slurm 作业，可按统一清单的 `photo_file` 和
源号提取原始 PDF。去掉 `--partition` 和 `--ls-id` 会重放
四个原始域和已校准伴随产品的访问验收。

```bash
sbatch -q regular -t 00:05:00 -c 2 scripts/run_release_readiness.sh access \
  --partition "$photo_file" --ls-id "$ls_id"
```

完整研究归档目前的已验证访问入口为 NERSC，GitHub 提供
可审阅清单、代码、表和图。
[已核验的分区与例子](../manifests/release_readiness/access/partition_examples.csv)

## 论文定位与使用边界

**结果支持继续推进 AJ 目录论文。** 论文可以说明，经过清楚
定义的选择，LS 原生 grz 为缺少高质量视差的暗端普通矮星提供
独立距离信息，能在同星谱距核验中区分已有测距方法。与
Bailer-Jones photogeometric 的接近零点和更小的配对散布是最
重要的对照。115 万对应几何视差门槛，三套发布产品之外的
目录覆盖增量为本轮测到的 3,094 个。完整后验和清楚的适用域
构成其余主要贡献。

亮红近端提供较直接的零点参照。正式科学使用集中于 2–5 kpc
的矮星增强层，应保留原生巡天和观测条件，并以光谱结果限定
所需星族。5–10 kpc、暗端与低纬的扩展对象保留为研究层。
普通矮星的精度、贫金属矮星的尺度偏差、中间重力星的远端误入
应在同一组使用条件中并列。纬度密度对照和 GD-1 作为应用与
失效实例，现有结果的物理解释以选择条件内的观测分布为边界。
[银河系空间和 GD-1 结果](final_science.md)

发布字段和分区访问已用真实归档核验，论文数据可用性说明应
区分当前 NERSC 大文件入口与 GitHub 的可复核材料。
[完成回执](../manifests/release_readiness/delivery/complete.json)
