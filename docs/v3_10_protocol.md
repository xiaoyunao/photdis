# Photdis v3.10 验证协议

本轮以 v3.9 `a4b2bcb` 的预测和物理配对为对照。科学样本须用原生
DR11 `TYPE=PSF`，应用仅使用测光。Gaia 和光谱用于建立参考、
校准和检验。新候选与冻结目录分开归档。

## 图件

v3.8 `c935235` 是唯一绘图样式基准。距离对照、同星 CMD、
逐源 Aitoff 和日心投影沿用共同的 v3.8 绘图函数。字体、配色、
点样式、网格、尺寸、布局和导出分辨率通过
[`v3_10_figure_style.json`](../configs/v3_10_figure_style.json)
的校验值固定。基准回归直接比较发布版的 F02 和 F03 PNG
像素。纠正后的 v3.9 扩展图单独写入 `data/v3_10/figures`，
原图和科学数据继续留档。

| 适用域 | Aitoff 预览 | 对照图 |
| --- | --- | --- |
| 南天暗端 | [逐源 Aitoff](../figures/v3_10/redraw/south_faint_aitoff.png) | [距离和残差](../figures/v3_10/redraw/south_faint_distances.png) |
| 南天低纬 | [逐源 Aitoff](../figures/v3_10/redraw/south_low_aitoff.png) | [同星 CMD](../figures/v3_10/redraw/south_low_cmd.png) |
| 北天 | [逐源 Aitoff](../figures/v3_10/redraw/north_aitoff.png) | [日心空间投影](../figures/v3_10/redraw/north_xyz.png) |

三个域各有全体、矮星、巨星及其他类别的距离和 CMD 图。逐源
地图按目录预测、合格参考、光谱矮星、光谱巨星各单独绘制。
[原图像素回归记录](../manifests/v3_10/figures/regression/complete.json)

## 尘埃系数

以两套标度在同一个 PS1 `g-r` 颜色上的红化量作为转换基准。
Green et al. 的 Bayestar19 系数差是 3.518 − 2.617 = 0.901，
Schlafly & Finkbeiner 的 SFD 标度系数差在 `R_V=3.1` 下是
3.172 − 2.271 = 0.901。用两者之比把 Bayestar19 地图数值
映射到 SFD 标度，再乘 Legacy DR11 的原生 DECam 三带系数
3.214、2.165 和 1.211。这是一个按颜色锚定的标度转换。
Bayestar19 和 SFD 所用的消光律并不完全相同，高消光区域
仍须用实测样本检验其适用性。
[Green et al. §2.1 与表 1](https://arxiv.org/html/1905.02734) ·
[Schlafly & Finkbeiner 表 6](https://arxiv.org/html/1012.4804) ·
[Legacy DR11 透射率定义](https://www.legacysurvey.org/dr11/catalogs/#galactic-extinction-coefficients)

同星复算读取 v3.9 留存的 311 个源、同一组尘埃曲线与有效距离，
撤销存档中 0.883 的数值换算以恢复地图原单位。原始 SFD 三带
透射率、名义 SED、有效距离掩码和 Gaia 参考自检一致后，才比较
旧、新三维消光 PDF。

## 矮星工作样本与 M92 对照

工作样本在北天和南天低纬低消光原生域内，要求巡天归属、
观测 `r<20`、每带误差小于 0.1 等及实际 QRF 支持。扩大标签
后的分类器使用原本的算法，以固定天区切分拟合和校准。在
已有门槛网格中取 0.80 作为候选工作点，再要求完整 QRF PDF
在 2–10 kpc 内至少有一半概率。原南天主选和已发布预测各自
保留。该工作点与距离层定义用于明确科学示范的观测选择。

逐行索引的状态 0 至 4 分别表示通过明亮域入表、通过严格
光学质量、有完整 PDF、通过矮星分数、通过距离概率。每个
分区 CSV 指向原预测文件及校验值，NPZ 记录原生键、PDF 行号、
选择状态和质量。DR11 随机点提供天区覆盖、掩膜和近似三带
深度的有效面积。样本保留比例还按颜色统计。光谱控制测得的
通过率只适用于这些控制样本。

M92 示范使用固定中心 (259.28076°, 43.13594°)，目标环带
0.08–0.35°，邻近控制环带 0.65–1.20°。两环带均限制在北天
第 4 天区角色，按随机点估计面积。比较全距离和 7.5–9.5 kpc
的面积归一化信号、5–10 kpc 最大正差的距离与颜色依赖。
外部 8.50 kpc 标尺用于展示和对照。选择和距离 PDF 只使用
测光。[M92 外部距离与坐标](https://people.smp.uq.edu.au/HolgerBaumgardt/globular/newdata/parameter.html)
