# Photdis v3.9 扩展域验证图

每个天区有六组图，提供 300 dpi PNG 和单页 PDF。NERSC 图件根目录为
`/pscratch/sd/y/yunao/photdis_pipeline/data/v3_9/expansion/figures_labeled`，
下设 `south_faint`、`south_low` 和 `north` 三个目录。

| 文件名 | 图中内容 |
| --- | --- |
| `distance_comparisons` | 分 Gaia、SpecDis、SPDist 显示测光距离与参考距离、一一对应线、相对残差和参考距离分布。 |
| `same_star_cmd` | 同一批星分别使用参考距离和测光距离计算绝对星等。 |
| `reference_controls` | Gaia 带符号视差与测光 PDF 前向预测均值、光谱类别对照、颜色及误差与距离残差的关系。 |
| `aitoff_source_positions` | 按 0–1、1–5、5–10 和 0–10 kpc 分层，分别画测距源、合格距离参考、光谱矮星和光谱巨星的逐源银道坐标。 |
| `heliocentric_xyz` | 用小于 10 kpc 的距离中位数绘制三个日心空间投影。 |
| `photometry_environment` | 双色图、SFD 消光与邻源通量比例、源所在位置的 $r$ 波段深度。 |

距离比较、CMD 和参考对照使用评估样本。北天使用留出的天区，
南天排除模型训练对象。Aitoff 中，测距样本用 QRF 中位距离入层，
其他三行用合格参考距离入层。空间图采用接缝小范围修正前的巡天
归属。完整 PDF 与接缝修正后的目录入口见[目录协议](v3_9_protocol.md)。

每个图件目录保存 `distance_by_source.csv`、`sweep_counts.csv`、
`aitoff_counts.csv`、`signed_gaia_and_spectral_controls.csv` 和
`figure_reference_rows.npz`。逐星参考底表保留评估样本标志。
`complete.json` 提供输入和输出校验值。逐星参考表另留历史分类
与补接的原始 DESI RVS 分类。原始分类须满足既有光谱质量规则和
唯一身份、位置一致性检查。分类冲突从矮星和巨星分组中排除。
