# 同一天区的测光审计

天区：RA [0°,10°)、Dec [−10°,0°)。本轮审计使用原试验的四个 DR11
sweep、[原样本清单](../manifests/pilot_10deg.json)和
[审计清单](../manifests/pilot_audit.json)。参考集按 Gaia source ID 去重，
同一 Gaia 星优先保留通过严格测光条件的 Legacy 记录，共 6,942 颗。

## 光学条件实际移除了什么

| 互斥组 | 颗数 | Gaia G 中位数 | Gaia 绝对 G 中位数 |
|---|---:|---:|---:|
| 严格光学条件通过 | 1,804 | 17.08 | 9.20 |
| 仅 SATUR | 462 | 16.02 | 7.03 |
| 仅 FRACMASKED | 16 | 17.27 | 11.44 |
| SATUR 与 FRACMASKED | 1,492 | 15.38 | 6.32 |
| ALLMASK 或 BAILOUT | 3,057 | 13.92 | 4.86 |
| 没有真实有效 grz 测量 | 31 | 9.19 | 1.45 |
| 其他光学条件 | 80 | 16.35 | 8.13 |

严格组中再剔除两个 Gaia 候选污染源，得到原先的 1,802 个锚点。
有 28 个合格 APOGEE 巨星落在 ALLMASK/BAILOUT 组；严格组中没有
合格光谱巨星。[联合失败图](../figures/pilot_audit/01_joint_failures.png)
显示被移除对象的亮度与 CMD 位置。SATUR 是接触过至少一次饱和像素的
标记；ALLMASK 和 BAILOUT 的定义见
[Legacy DR11 bitmasks](https://www.legacysurvey.org/dr11/bitmasks/)。

Gaia 目录有 22 个 `BP_RP=0` 值，22 个对象都没有有效 BP/RP 通量与观测
次数。由有效 BP、RP 星等重新计算后，6,920 个参考对象有可用颜色，
严格锚点中有 1,801 个；有效颜色与目录值偏差大于 0.001 mag 的对象为 0。
[Gaia 颜色核查图](../figures/pilot_audit/02_gaia_color_check.png)显示原零颜色竖列消失。

## 与 PS1 的测光对照

使用 NERSC 的 PS1 qz-star-v3 星表中位星等，按每颗对象的观测历元传播
Gaia 自行，再作唯一的 1″ 匹配。比较限制为 PS1 每波段至少三次合格观测、
星等 15–20、离散度 <0.03 mag，且 0.4<PS1 g−i<2.7。用
[Legacy 发布的 PS1→DECam 颜色变换](https://www.legacysurvey.org/dr11/description/#photometry)
计算预测 grz；差值是 Legacy 实测星等减预测星等，两侧都使用观测星等。
PS1 也参与 Legacy 零点标定；这里检验对象级相对光度和颜色一致性。

| 三波段共同满足 PS1 对照质量 | 颗数 | 三波段差值均 <0.03 mag | 均 <0.05 mag |
|---|---:|---:|---:|
| 严格光学通过 | 725 | 679 | 720 |
| 仅 SATUR | 256 |  | 253 |
| 仅 FRACMASKED | 4 |  | 4 |
| SATUR 与 FRACMASKED | 357 |  | 353 |
| 上述三类恢复候选合计 | **617** | **580** | **610** |

单波段对照中，严格组 g 的中位差值为 +0.0017 mag、NMAD 0.0128 mag；
仅 SATUR 组分别为 −0.0002、0.0112 mag，SATUR+FRACMASKED 组分别为
+0.0004、0.0102 mag。[残差图](../figures/pilot_audit/03_ps1_residuals.png)
展示差值随亮度和颜色的变化。独立测光支持恢复其中一部分较亮的主序星；
这些恢复对象保留为基线的外部敏感性检查。严格样本的数量已经足够开展
单场冷主序基线，因此主基线暂用严格测光口径。

## 输入颜色层级

| 输入 | 天区目标 PSF 记录 | 严格 Gaia 锚点 |
|---|---:|---:|
| grz | 312,975 | 1,802 |
| grzW1 | 73,044 | 1,699 |
| grzW1W2 | 44,806 | 1,688 |

三个平面分别见 [grz](../figures/pilot_audit/04_colors_grz.png)、
[grzW1](../figures/pilot_audit/05_colors_grzW1.png)、
[grzW1W2](../figures/pilot_audit/06_colors_grzW1W2.png)。图中比较目标点源、
严格锚点、合格 DESI 恒星与已知 DESI QSO/星系。

DESI RVS 丰度保留原始值和[作者校准函数](https://github.com/desimilkyway/dr1_vac_code/blob/main/feh_correct.py)
的适用域内结果。1,065 个合格光谱对象中有 772 个在校准域内；
55 个合格光谱严格锚点中仅 7 个在域内。冷主序基线仅用 Gaia 几何
标签训练，不把光谱丰度作为预测输入。
