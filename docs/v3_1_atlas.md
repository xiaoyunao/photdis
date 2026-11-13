# v3.1 进行中图集

以下两张图是已见开发样本的机制诊断，不是最终独立验证。2026-11-13
逐张目检PNG；62项哈希、分辨率、PDF、英文文字及源表对应检查通过。
源表均保留原设计权重和天空块bootstrap区间。

## 目标先验迁移

![Target-prior transfer](../figures/v3_1/paired_figures/target_prior_transfer.png)

保持response12不变时，目标先验对演化星的开发诊断明显退化。
相对名义模型的对照同时改变响应与先验，不能解释为纯先验效应。
1542星都是新格子的共享超均值迁移；原1个数值标记保留，另有通过的精化侧表。
[PDF](../figures/v3_1/paired_figures/target_prior_transfer.pdf)；
[源表](../manifests/v3_1/paired_figures/target_prior_transfer.csv)。

## 类别和FeH的同身份条件对照

![Class and metallicity](../figures/v3_1/paired_figures/class_metallicity_contrasts.png)

每个对照使用完全相同的合格对象。演化星中，给定FeH后加入类别信息仍有
明显效应；给定类别后加入FeH并不保证改善。这些真实参考条件只用于诊断，
不能进入生产推断。参考点一致率不是实测真实覆盖率。
[PDF](../figures/v3_1/paired_figures/class_metallicity_contrasts.pdf)；
[源表](../manifests/v3_1/paired_figures/class_metallicity_contrasts.csv)。

[实际检查记录](../manifests/v3_1/paired_figures/delivery_validation.json)。
其余工作包图件与最终适用域图仍在推进。

## Constrained luminosity correction

![Calibration correction](../figures/v3_1/calibration/luminosity_correction.png)

Six panels show class-specific shift/scatter and held-out sky changes in
proper score, reference-point agreement and signed-Gaia PIT tails. All15
bootstrap guards and96numerical checks pass. Evolved scatter remains poorly
constrained by only3fit/2guard sky blocks; this is not independent calibration.
Source rows and actual image/PDF checks: `manifests/v3_1/calibration_figure/`.
PNG300dpi/PDF and all six panels visually inspected.
