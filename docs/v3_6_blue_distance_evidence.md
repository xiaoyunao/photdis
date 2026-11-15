# 两组蓝星为什么会测到相反的方向

在同一段 Legacy 光学颜色（去消光后 $0.5\le g-r<0.8$）里，Gaia
揭示了两组光度差异很大的星。当前 QRF 却给两组星分配了相近的绝对星等。
这是距离偏差方向相反的直接表现。两组来自 v3.5 原有的主选样 Gaia 抽样，
本次按冻结名单和 Gaia DR3 ID 逐星复查。

116 颗暗蓝近邻星的 Gaia 距离中位数是 99 pc，QRF 距离中位数是 5.61
kpc。按逐星距离比计算，典型高估约 58 倍。根据 Gaia 距离和 Legacy
测光推得的 $M_g$ 中位数是 15.24 mag，QRF 距离所对应的 $M_g$
中位数是 6.41 mag。这些星在 Gaia 的颜色和约化自行图上也形成一条低光度候选序列。
116 颗星在本次关联的档案中都缺少合格的光谱分类。它们的测光和自行支持
白矮星候选解释，单星类别仍需独立证据。一颗星也出现在历史的 49 颗异常星中。
严格选样剩下 17 颗，逐星距离比的中位数仍约 57 倍。

189 颗蓝色 2–5 kpc 参考星的 Gaia 距离实际分布在 2.00–3.46 kpc，
中位数 2.41 kpc。QRF 距离中位数是 1.16 kpc，有 183 颗被估得更近，
逐星距离比的中位数为 0.45。它们根据 Gaia 距离得到的 $M_g$ 中位数
是 4.28 mag，根据 QRF 距离是 6.01 mag。严格选样剩下 78 颗，其中
74 颗仍被估得更近。关联档案里只有 3 颗有合格光谱分类，这组星的恒星
演化阶段尚不明确。两组都通过冻结的 Gaia 质量和位置规则。位置核对使用
Legacy 保存的 Gaia 关联坐标，图中的 Legacy 与 Gaia 颜色和星等有测光上的
独立信息。

接回四份外部光谱库存后，暗蓝近邻仍没有匹配。189 星组中有一颗通过了
Legacy 和 Gaia 两个键的 APOGEE 匹配。它的表面重力是
$\log g=3.35\pm0.04$，有效温度约 5,030 K，金属丰度约 −0.46 dex。
光谱支持这个对象处于巨星和主序矮星之间的重力范围。Gaia 距离为 3.00
kpc，astroNN 距离为 $3.29\pm0.58$ kpc，QRF 只给出 0.70 kpc。
这个例子表明中间光度分支在低估组中确实存在。这四个光谱库只给出一颗
直接证据，尚不能衡量它在其余 188 颗中的比例。
[四份外部光谱逐星核对](../manifests/v3_6/blue_replay/external_spectroscopy.csv)

全量 DESI DR1 MWS/RVS 关联为这组另提供三颗具有合格重力测量的星，
$\log g$ 分别为 3.83、4.74 和 4.87。第一颗的 QRF 距离是 0.86 kpc，
Gaia 距离是 2.13 kpc。后两颗的重力支持矮星类别，距离同样偏低，说明低估
也出现在有光谱支持的矮星中。暗蓝近邻里有 16 颗匹配到 DESI，但这些光谱
记录都缺合格类别与重力。光谱匹配使用 Legacy 和 Gaia 两套身份。
[逐星光谱与质量](../manifests/v3_6/distance_evidence/blue_spectroscopy.csv)

两组在光学颜色平面明显重叠。可用 W1 的暗蓝近邻只有 29 颗，W1 和 W2
同时可用的只有 8 颗。蓝色 2–5 kpc 参考星中，两带同时可用的有 106 颗。
现有 grz QRF 没有使用这两条红外颜色，对可用的少数源可以看出额外分辨
光度分支的线索。两组 QRF 在 10 kpc 内的概率中位数仍为 97.2% 和 99.7%，
因此两种误差都能保留在近端样本。

两组均没有 QRF 中位距离大于 10 kpc 或 QRF 近端概率低于 0.5 的对象。
按名义 SED 和 QRF 使用相同的选样门槛，QRF 近端概率低于 0.90 的有
2 颗暗蓝近邻和 1 颗蓝色 2–5 kpc 参考星。严格档把门槛提高到 0.97，
档内仍有 4 颗和 2 颗低于各自的 QRF 门槛。误差主要发生在 10 kpc 内部。

完整后验的对照中，有一颗星的 Gaia 和 QRF 距离都约为 2.17 kpc。另一颗
在 1.18 和 4.28 kpc 附近都有明显峰，Gaia 距离为 3.20 kpc，而 QRF
中位距离为 1.34 kpc。W2 弱信号与两条 WISE 波段都不可用的例子分别展示了
高估和低估。标记的 WISE 状态来自测光记录，QRF 预测使用 $grz$。
[完整后验示例](../figures/v3_6/F03_qrf_pdf_examples.pdf)

另选了四颗星查看 DR11 South 的 $r$ 波段叠加图。四个局部光心与目录点
的偏差在 0.01–0.15 角秒。典型低估源旁有一个分得开的近邻，低估最重的例子
在该分辨率下没有同程度的近邻。两颗亮的低估例子都具有较高的 $r$ 波段
拟合残差。整个 189 星组的原生 `RCHISQ_R` 中位数约 34。光度分支、
亮星测光误差和个别混光的贡献，需要用亮度及颜色匹配的参考对照来分辨。
叠加图本身也无法提供各次曝光的历元。
[Legacy Survey DR11 图像说明](https://www.legacysurvey.org/dr11/description/)

全量 Gaia 关联现在找到 582 颗主选和 85 颗严格档的暗蓝几何参考，参考距离
全部在 2 kpc 内。排除当前 QRF 训练星后，剩 577 颗和 83 颗，
QRF 中位距离分别高估约 57 倍和 56 倍，68% 区间均未覆盖参考中心。按
QRF 中位距离分组，主选中有 374 颗、严格档中有 48 颗近邻被放到 5–10 kpc。
这些计数的适用范围受 Gaia 视差信噪比筛选限制。
[按参考距离](../manifests/v3_6/distance_evidence/distance_cells.csv) ·
[按分配距离](../manifests/v3_6/distance_evidence/photometric_reference_coverage.csv)

- [F03 两组星的逐星图](../figures/v3_6/F03_blue_distance_failures.pdf)、
  [图表数据](../manifests/v3_6/figures/F03_blue_distance_failures.csv)、
  [完整逐星复查表](../manifests/v3_6/blue_replay/blue_objects.csv)
- [F04 按星等、颜色和参考距离分层](../figures/v3_6/F04_gaia_magnitude_color.pdf)、
  [各格的计数和误差](../manifests/v3_6/figures/F04_gaia_magnitude_color.csv)
- [四个例子的完整距离后验和两组的筛选得分](../figures/v3_6/F03_blue_posteriors_and_scores.pdf)、
  [同星叠加图](../figures/v3_6/F03_selected_coadd_cutouts.pdf)
- [F05 光学、i 和 WISE 颜色对照](../figures/v3_6/F05_blue_color_planes.pdf)
