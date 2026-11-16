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

## 尘埃系数

以两套标度在同一个 PS1 (g-r) 颜色上的红化量作为转换基准。
Green et al. 的 Bayestar19 系数差是 3.518 − 2.617 = 0.901，
Schlafly & Finkbeiner 的 SFD 标度系数差在 (R_V=3.1) 下是
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
