# v3.1 进行中图集

前两张图是已见开发样本的机制诊断，不是最终独立验证。2026-11-13
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

## Latent-flux support and failure association

![Latent support](../figures/v3_1/latent/latent_support_failure.png)

Four panels distinguish weighted latent-negative bounds, individual W1/W2
probabilities, evolved-star reference failures and raw missing-reference counts.
Among297evolved primary-reference stars with negative-probability upper bound
<0.001, the weighted severe |delta_mu|>1mag fraction remains87.6635%; this does
not support negativity as the sole giant-failure explanation. The four
unknown-class objects above0.1lack primary distance references. No causal or
new-validation claim follows. All four panels visually inspected after label
and legend repairs;1574actual image/PDF/hash/source checks pass.
[PDF](../figures/v3_1/latent/latent_support_failure.pdf);
[source rows and acceptance](../manifests/v3_1/latent_figure/acceptance.json).

## Candidate joint counts and finite-state numerical sensitivity

![Candidate conditional counts](../figures/v3_1/target_candidate/conditional_counts.png)

Four registered views share3540nativePSF objects. Points are conditional expected
class–distance counts; intervals are shared realized-count95%intervals. All three
classes use the same scale, linear below1and logarithmic above. Count-CDF half
sample discrepancies are explicitly printed. These are not measured physical
populations or densities; independent external validation remains pending.

![Candidate weights and numerics](../figures/v3_1/target_candidate/weights_numerics.png)

All34conditional states pass their gates, but the main evidence-weighted ensemble
retains its failed absolute-evidence seed gate: delta logZ=.248538 versus.05.
Probability/count/state-weight changes pass their own gates. Finite luminosity
bootstrap ESS21.17325/32and maxweight7.01524%do not measure64-SED response support.
Nativefine sensitivity covers36candidate objects, not all3540.

Both PNG300dpi/PDF exports and all seven panels were individually inspected;
138actual hash/PNG/PDF/source-row checks pass. [Plotted rows](../manifests/v3_1/target_candidate_figures/plotted.csv),
[acceptance record](../manifests/v3_1/target_candidate_figures/acceptance.json),
[count PDF](../figures/v3_1/target_candidate/conditional_counts.pdf),
[sensitivity PDF](../figures/v3_1/target_candidate/weights_numerics.pdf).

## Fixed high-q joint count table

The main qnear=P(d<10kpc)>.95 subset contains3003of3540nativePSF objects.
Each cell below is the conditional expectation [95%shared realized-count interval].
All four controls use exactly these same identities; this table shows the
registered evidence-weighted main view, including its inherited numerical flag.

| Class | 0–2 kpc | 2–5 kpc | 5–10 kpc | 10–20 kpc | 20–50 kpc | >50 kpc |
|---|---:|---:|---:|---:|---:|---:|
| Dwarf | 1824.740 [1803, 1847] | 991.948 [964, 1020] | 150.145 [133, 167] | 1.488 [0, 4] | 0.112 [0, 1] | 0.049 [0, 1] |
| Evolved | 0.430 [0, 2] | 0.184 [0, 1] | 0.173 [0, 1] | 0.283 [0, 2] | 0.450 [0, 2] | 0.478 [0, 2] |
| Other | 2.958 [0, 7] | 9.460 [4, 17] | 14.960 [6, 26] | 3.425 [0, 8] | 1.573 [0, 5] | 0.146 [0, 1] |

Summed model-expected d>=10kpc count is8.00304, leakage0.266502%; the far-count
interval[3,15] comes from the jointly summed draws, not summed bin endpoints.
Its split-half CDF discrepancy is0.01378. Neither the leakage nor the intervals
establish independent contamination or complete model uncertainty.
[All views, numerical alternatives and count moments](../manifests/v3_1/highq_counts/counts.csv);
[joint far-count source](../manifests/v3_1/highq_counts/model_leakage.csv).
