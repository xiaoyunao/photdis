# v1.4 科学诊断：图集与可复查统计

本图集读取冻结 v1.4 的逐星结果。分类器与 wide QRF 距离模型独立评分；分类分数没有传入这些距离预测。没有重新训练，也没有更新部署或重跑母星表。

## 关键逐星结果

距离残差定义为预测中位距离模数减参考距离模数；严重错误为绝对残差 >1 mag。Gaia、SpecDis 和 SPDist 分列；后两者的覆盖率是与发布点估计的一致性，未卷积参考误差。

| 配置 | 类别 | N | 中位残差 / mag | NMAD / mag | 严重错误率 | 68% / 95% 覆盖 | 中位距离比 |
|---|---|---:|---:|---:|---:|---:|---:|
| expanded_A | dwarf | 32,719 | +0.032 | 0.239 | 1.95% | 75.09% / 97.72% | 1.015 |
| expanded_A | evolved | 1,163 | -5.301 | 1.727 | 96.56% | 19.35% / 65.78% | 0.087 |
| expanded_A | other_transition | 6,408 | -0.314 | 0.535 | 21.36% | 62.38% / 96.22% | 0.865 |
| expanded_A | unknown | 7,466 | +0.003 | 0.535 | 19.98% | 55.72% / 87.92% | 1.002 |
| matched_grz | dwarf | 29,865 | +0.048 | 0.237 | 1.97% | 74.45% / 97.49% | 1.022 |
| matched_grz | evolved | 1,064 | -5.335 | 1.731 | 97.46% | 14.94% / 60.81% | 0.086 |
| matched_grz | other_transition | 5,873 | -0.305 | 0.534 | 20.91% | 62.10% / 95.91% | 0.869 |
| matched_grz | unknown | 6,904 | +0.015 | 0.531 | 20.19% | 55.62% / 87.57% | 1.007 |
| matched_griz | dwarf | 29,865 | +0.048 | 0.236 | 1.93% | 74.29% / 97.45% | 1.022 |
| matched_griz | evolved | 1,064 | -5.340 | 1.730 | 96.52% | 15.88% / 61.84% | 0.086 |
| matched_griz | other_transition | 5,873 | -0.306 | 0.530 | 20.48% | 61.88% / 95.66% | 0.868 |
| matched_griz | unknown | 6,904 | +0.011 | 0.526 | 19.94% | 56.01% / 87.57% | 1.005 |

## 改善了谁、损伤了谁

下表是 chosen 参考下的同星配对。置信区间来自 2000 次天区 cell 整块 bootstrap，重新计算星数加权比例；负差表示改善。它反映当前参考选择与天区样本的波动，不能外推为全母样本精度。

| 新配置（对原配置） | 类别 | N / cells | 错→对 | 对→错 | 新−旧尾率 / 百分点 | 95% 区间 / 百分点 |
|---|---|---:|---:|---:|---:|---:|
| expanded_A | all | 47,756 / 9 | 230 | 161 | -0.144 | [-0.289, -0.103] |
| expanded_A | dwarf | 32,719 / 9 | 104 | 41 | -0.193 | [-0.263, -0.118] |
| expanded_A | evolved | 1,163 / 8 | 11 | 24 | +1.118 | [+0.083, +1.960] |
| expanded_A | other_transition | 6,408 / 9 | 56 | 51 | -0.078 | [-0.636, +0.073] |
| expanded_A | unknown | 7,466 / 9 | 59 | 45 | -0.188 | [-0.580, +0.000] |
| expanded_B | all | 42,786 / 9 | 211 | 147 | -0.150 | [-0.235, -0.132] |
| expanded_B | dwarf | 30,266 / 9 | 76 | 51 | -0.083 | [-0.236, -0.012] |
| expanded_B | evolved | 1,075 / 8 | 21 | 33 | +1.116 | [+0.345, +1.994] |
| expanded_B | other_transition | 5,717 / 9 | 65 | 36 | -0.507 | [-0.865, -0.277] |
| expanded_B | unknown | 5,728 / 9 | 49 | 27 | -0.384 | [-0.684, -0.140] |
| expanded_C | all | 37,752 / 9 | 207 | 135 | -0.191 | [-0.336, -0.128] |
| expanded_C | dwarf | 28,687 / 9 | 95 | 45 | -0.174 | [-0.263, -0.066] |
| expanded_C | evolved | 1,019 / 8 | 16 | 44 | +2.748 | [+1.332, +5.251] |
| expanded_C | other_transition | 4,644 / 9 | 62 | 27 | -0.754 | [-1.526, -0.484] |
| expanded_C | unknown | 3,402 / 9 | 34 | 19 | -0.441 | [-1.298, -0.181] |
| matched_griz | all | 43,706 / 9 | 282 | 219 | -0.144 | [-0.337, -0.043] |
| matched_griz | dwarf | 29,865 / 9 | 90 | 79 | -0.037 | [-0.137, -0.005] |
| matched_griz | evolved | 1,064 / 8 | 18 | 8 | -0.940 | [-3.167, +0.000] |
| matched_griz | other_transition | 5,873 / 9 | 91 | 66 | -0.426 | [-0.943, +0.119] |
| matched_griz | unknown | 6,904 / 9 | 83 | 66 | -0.246 | [-0.578, +0.017] |

## WISE：质量与深度损失有交集

原 26,000 颗概率样本保留原权重，只代表原南天 DECam 母样本。以下独立统计 S/N、WISEMASK 和 FRACFLUX；不是顺序互斥筛选比例。

| 波段 | 所有低 S/N 的加权比例 | 质量失败比例 | 同时低 S/N 且质量失败 | 质量失败组内低 S/N 比例 |
|---|---:|---:|---:|---:|
| W1 | 56.92% | 60.99% | 49.42% | 81.03% |
| W2 | 77.59% | 57.28% | 53.87% | 94.03% |

## 固定 PDF 个例

每组按预先固定的身份哈希顺序抽取两颗、不重复选星。下表概率为预测 PDF 在参考距离模数 ±1 mag 内的积分。个例仅用于解释，不能据十颗星估计总体失败机制比例。

| 组 | 身份 | 模型 | 中位残差 / mag | 68% 宽度 / mag | 参考 ±1 mag 概率 | 支持域内 | 歧义标记 |
|---|---|---|---:|---:|---:|---|---|
| ordinary_success | 2493528475844383488 | matched_grz | +0.10 | 0.27 | 99.10% | True | False |
| ordinary_success | 3673682345145843712 | matched_grz | -0.13 | 0.88 | 96.13% | True | False |
| evolved_failure | 2489018077283984128 | matched_grz | -5.96 | 6.24 | 20.73% | True | False |
| evolved_failure | 3673732682162574848 | matched_grz | -4.80 | 6.41 | 39.50% | True | True |
| ambiguity_flagged | 3671427998416231552 | matched_grz | -0.50 | 4.40 | 36.43% | True | True |
| ambiguity_flagged | 3672631555627203456 | matched_grz | -0.68 | 3.85 | 33.11% | True | True |
| no_W1 | 3672919863896448896 | matched_grz | -0.17 | 0.95 | 95.06% | True | False |
| no_W1 | 2496485268770069632 | matched_grz | +0.08 | 0.86 | 95.18% | True | False |
| unknown | 2500352216804651008 | matched_grz | -0.15 | 1.01 | 89.96% | False | False |
| unknown | 2493423335044891264 | matched_grz | -0.03 | 0.86 | 93.34% | True | False |
| ordinary_success | 2493528475844383488 | matched_griz | +0.10 | 0.29 | 99.81% | True | False |
| ordinary_success | 3673682345145843712 | matched_griz | -0.12 | 0.79 | 96.29% | True | False |
| evolved_failure | 2489018077283984128 | matched_griz | -5.95 | 6.31 | 22.66% | True | False |
| evolved_failure | 3673732682162574848 | matched_griz | -5.51 | 6.29 | 36.31% | True | True |
| ambiguity_flagged | 3671427998416231552 | matched_griz | -1.31 | 4.51 | 34.16% | True | True |
| ambiguity_flagged | 3672631555627203456 | matched_griz | -0.35 | 3.88 | 33.86% | True | True |
| no_W1 | 3672919863896448896 | matched_griz | -0.17 | 0.93 | 95.67% | True | False |
| no_W1 | 2496485268770069632 | matched_griz | +0.09 | 0.89 | 94.57% | True | False |
| unknown | 2500352216804651008 | matched_griz | -0.14 | 0.84 | 96.01% | True | False |
| unknown | 2493423335044891264 | matched_griz | -0.08 | 0.82 | 92.18% | True | False |

## 图集

距离和 CMD 同时展示广覆盖 expanded_A 与含 i 的 matched_griz。其他物理细分主要使用 matched_griz；八种配置的全部分天区统计均已导出。A/B/C 使用不同 WISE 子样本，不能直接跨配置归因波段收益。

天区性能格要求 N≥20；颜色与趋势格要求 N≥10。稀疏格留白，完整计数保留。残差尾部不裁到 ±1 mag；固定颜色/参数范围的排除数写在图中。

### distance_expanded_A_dwarf

![distance_expanded_A_dwarf](../figures/baseline_v1_4_science/distance_expanded_A_dwarf.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_expanded_A_evolved

![distance_expanded_A_evolved](../figures/baseline_v1_4_science/distance_expanded_A_evolved.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_expanded_A_other_transition

![distance_expanded_A_other_transition](../figures/baseline_v1_4_science/distance_expanded_A_other_transition.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_expanded_A_unknown

![distance_expanded_A_unknown](../figures/baseline_v1_4_science/distance_expanded_A_unknown.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### cmd_expanded_A

![cmd_expanded_A](../figures/baseline_v1_4_science/cmd_expanded_A.png)

Same Legacy g definition and same objects in each column; chosen Gaia-then-SpecDis reference. Four lowest-ID objects per class have within-panel arrows from reference to prediction in the top row; open endpoints mark predicted positions. Color scales match vertically within each class. A plausible predicted sequence does not establish correct distances.

### distance_matched_griz_dwarf

![distance_matched_griz_dwarf](../figures/baseline_v1_4_science/distance_matched_griz_dwarf.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_matched_griz_evolved

![distance_matched_griz_evolved](../figures/baseline_v1_4_science/distance_matched_griz_evolved.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_matched_griz_other_transition

![distance_matched_griz_other_transition](../figures/baseline_v1_4_science/distance_matched_griz_other_transition.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### distance_matched_griz_unknown

![distance_matched_griz_unknown](../figures/baseline_v1_4_science/distance_matched_griz_unknown.png)

Separate reference sources; distance in kpc, full residual extent. Red horizontal errors on up to16 identity-ordered stars show reference sigma_mu transformed to distance. VAC coverage is agreement with point estimates, not error-free truth; catalogs overlap and are not independent.

### cmd_matched_griz

![cmd_matched_griz](../figures/baseline_v1_4_science/cmd_matched_griz.png)

Same Legacy g definition and same objects in each column; chosen Gaia-then-SpecDis reference. Four lowest-ID objects per class have within-panel arrows from reference to prediction in the top row; open endpoints mark predicted positions. Color scales match vertically within each class. A plausible predicted sequence does not establish correct distances.

### colors_gr_rz

![colors_gr_rz](../figures/baseline_v1_4_science/colors_gr_rz.png)

Fixed color limits; excluded counts disclosed per row and saved. All residuals contribute inside color range, including catastrophic tails. Empty/low-count performance cells are blank. WISE plane requires usable W1; i plane uses matched i-usable stars. Color overlap alone cannot prove absence of useful information.

### colors_gr_zW1

![colors_gr_zW1](../figures/baseline_v1_4_science/colors_gr_zW1.png)

Fixed color limits; excluded counts disclosed per row and saved. All residuals contribute inside color range, including catastrophic tails. Empty/low-count performance cells are blank. WISE plane requires usable W1; i plane uses matched i-usable stars. Color overlap alone cannot prove absence of useful information.

### colors_ri_iz

![colors_ri_iz](../figures/baseline_v1_4_science/colors_ri_iz.png)

Fixed color limits; excluded counts disclosed per row and saved. All residuals contribute inside color range, including catastrophic tails. Empty/low-count performance cells are blank. WISE plane requires usable W1; i plane uses matched i-usable stars. Color overlap alone cannot prove absence of useful information.

### sky_class

![sky_class](../figures/baseline_v1_4_science/sky_class.png)

Existing new-sky test only; performance needs20 references per cell. Reference counts shown without a minimum; raw count is not sky density. Stratification exposes composition differences; residual patterns alone do not establish a spatial calibration error.

### sky_source

![sky_source](../figures/baseline_v1_4_science/sky_source.png)

Existing new-sky test only; performance needs20 references per cell. Reference counts shown without a minimum; raw count is not sky density. Stratification exposes composition differences; residual patterns alone do not establish a spatial calibration error.

### sky_brightness

![sky_brightness](../figures/baseline_v1_4_science/sky_brightness.png)

Existing new-sky test only; performance needs20 references per cell. Reference counts shown without a minimum; raw count is not sky density. Stratification exposes composition differences; residual patterns alone do not establish a spatial calibration error.

### footprint

![footprint](../figures/baseline_v1_4_science/footprint.png)

Train/cal/test geometry and quarantined identities; cells may contain quarantined historical identities alongside a nonhistorical role. Counts concern assembled references, not the full optical sky. No unobserved cell is filled.

### sky_density

![sky_density](../figures/baseline_v1_4_science/sky_density.png)

N divided by exact spherical cell area; this is reference selection density, not stellar population density.

### kiel

![kiel](../figures/baseline_v1_4_science/kiel.png)

Published RVS parameters with original parameter-quality qualification. Fixed displayed Teff/logg limits and excluded counts. Other/transition is not assigned a definite evolutionary phase.

### metallicity

![metallicity](../figures/baseline_v1_4_science/metallicity.png)

Continuous spectral trends use reliable RVS parameters, finite metallicity and positive formal metallicity error. Full scatter extents retained; fixed-bin medians/16–84% spread show occupied ranges, not causal metallicity corrections.

### observing_conditions

![observing_conditions](../figures/baseline_v1_4_science/observing_conditions.png)

Fixed bins with N≥10; class labels retained separately. Bottom legends give actual reference N in every bin. Trends mix selection and measurement effects; sparse20–25 bins do not establish faint-star accuracy.

### intervals

![intervals](../figures/baseline_v1_4_science/intervals.png)

All occupied cells show N; N<10 estimates blank. Width is in distance-modulus magnitudes. Large evolved-star misses remain visible despite the overall dwarf-dominated coverage.

### gaia_flags

![gaia_flags](../figures/baseline_v1_4_science/gaia_flags.png)

Gaia PIT and log density read only from gaia_refined/matched_griz/evaluation.npz. Negative/weak parallaxes retained in observation space. Flat PIT is not proof of unbiased target-population calibration; support/ambiguity flags are assessed against actual errors.

### pdf_examples_1

![pdf_examples_1](../figures/baseline_v1_4_science/pdf_examples_1.png)

Two deterministic, nonoverlapping identity-hash examples from each registered category. Full modulus support shown; densities integrate to1. Frozen grz/griz quantiles independently reproduced within1e-6. Reference uncertainty is shown; an ambiguity flag does not itself prove a distinct astrophysical branch.

### pdf_examples_2

![pdf_examples_2](../figures/baseline_v1_4_science/pdf_examples_2.png)

Two deterministic, nonoverlapping identity-hash examples from each registered category. Full modulus support shown; densities integrate to1. Frozen grz/griz quantiles independently reproduced within1e-6. Reference uncertainty is shown; an ambiguity flag does not itself prove a distinct astrophysical branch.

### paired_physics

![paired_physics](../figures/baseline_v1_4_science/paired_physics.png)

Exact same reference and identity in each pair. Symmetric-log y scale resolves small changes while retaining full extrema. Bands show dispersion of star-level changes, not bootstrap uncertainty. Validated bin counts and summaries saved separately.

### paired_uncertainty

![paired_uncertainty](../figures/baseline_v1_4_science/paired_uncertainty.png)

2000 paired bootstrap resamples of entire occupied10° test cells; star-weighted rate recomputed in every draw, percentile95% interval. Negative rate difference improves. Small cell counts and reference selection limit generalization; this is exploratory comparison, not an adoption test.

### application_coverage

![application_coverage](../figures/baseline_v1_4_science/application_coverage.png)

Original26000 probability sample represents the old southern DECam parent only. Historical training is joined to the saved expanded_A legacy training identities; old/new unions are disjoint. New cal/test panels use the union of qualified class/distance rows. Each panel normalized separately. Fixed color-magnitude-bin absence is distinct from model support, and nonempty bins do not prove accuracy. W2 availability is independent of W1.

### wise_joint

![wise_joint](../figures/baseline_v1_4_science/wise_joint.png)

Independent axes: WISEMASK!=0, FRACFLUX>=1 (or invalid), measured S/N<5. Finite positive IVAR, observations and transmission required. Negative flux belongs to low S/N. Fractions use original weights and entire-parent denominator; unmeasured fraction retained separately in statistics.

## 数据来源、可复现性与边界

- [诊断协议](baseline_v1_4_diagnostics_protocol.md)；[完整 Pro 建议](pro_review_46cb68a.md)。
- [紧凑统计与输入哈希](../manifests/baseline_v1_4_science.json)；[分箱数值与逐图说明](../manifests/baseline_v1_4_science_plot_data.json)；[实际检查](../manifests/baseline_v1_4_science_validation.json)。
- 本轮 1943 项实际身份/数值/哈希检查通过。逐星表和 PDF 留在 NERSC `data/baseline_v1_4/diagnostics`；程序为 `scripts/baseline_v1_4_diagnostics.py` 和 `scripts/baseline_v1_4_science_report.py`。
- 逐星关联以精确身份校验，primary 与 refined 评分分别绑定 SHA256；计算开始与结束复核读过的输入未变。
- 完整 Gaia PIT/log density 使用 refined sidecars；弱与负视差从未倒数成距离散点。
- 参考目录之间存在重叠、选择效应和相关信息；参考间一致不等于独立绝对真值。金属贫乏不是银河晕成员分类。
- 本轮按 test 结果作物理诊断。若据此修改模型，这些天区成为开发证据；新的最终结论须另留独立检验。22–25 等仍广泛未验证。
- 原 v1.4 结果、模型及已部署 v1 保持不变；没有自动开启下一轮训练或定时监控。
