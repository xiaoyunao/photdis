# Photdis v3.8 论文主图

| 图组 | 读图方式 | 图件 |
|---|---|---|
| 1. 目录概况 | 实际银道坐标、全部模型支持源的观测 CMD，以及合格参考的颜色—颜色图。灰底表示模型支持源覆盖的近似范围。 | [总体分布](../figures/v3_8/F01_catalog_overview.pdf) |
| 2. 距离直接比较 | Gaia、SpecDis、SPDist 分开。主选中质量合格且排除当前 QRF 训练星的同一批对象，用于一一线、距离模数残差和统计量。 | [直接距离与残差](../figures/v3_8/F02_direct_distances.pdf) |
| 3. 同星绝对 CMD | 各来源左右两列使用相同恒星和去消光颜色，纵轴分别由参考距离、测光中位距离计算。 | [参考与 QRF CMD](../figures/v3_8/F03_same_star_cmd.pdf) |
| 4. 逐星 Aitoff | 从原生坐标绘出 0–1、1–5、5–10 和 0–10 kpc。末面板为前三面板的并集。参考优先采用合格 Gaia，其次 SpecDis、SPDist 和其他谱距。 | [完整测距目录](../figures/v3_8/F04_catalog.pdf) · [唯一参考](../figures/v3_8/F04_references.pdf) · [可靠矮星](../figures/v3_8/F04_dwarfs_reference.pdf) · [可靠巨星](../figures/v3_8/F04_giants_reference.pdf) |
| 5. 日心空间分布 | 主选 0–10 kpc 的三个笛卡尔投影，下排为同一主选匹配到合格参考的参考距离投影。X 指向银心，Y 指向银经 90°，Z 指向北银极。 | [XY、XZ、YZ](../figures/v3_8/F05_heliocentric_projection.pdf) |
| 6. 远端选样 | 固定单星测光距离，展示选回的远矮星、落入正确层的人数和新增的远巨星误入。 | [实际人数与偏差](../figures/v3_8/F06_selection_outcomes.pdf) |
| 7. 弱 Gaia | 预测中位距离 5–10 kpc 的带符号视差实测及同一入选条件下的前向分布。 | [原主选与候选](../figures/v3_8/F07_signed_parallax_distribution.pdf) |
| 8. 光度分支 | 四类固定个例的原生 grz 似然、先验和距离后验。 | [个例分解](../figures/v3_8/F08_luminosity_branches.pdf) |

Aitoff 的补充视图包括 [原主选](../figures/v3_8/F04_primary.pdf)、
[候选门槛已核实子集](../figures/v3_8/F04_candidate_known.pdf)、
[同一矮星按 QRF 距离分层](../figures/v3_8/F04_dwarfs_qrf.pdf)、
[同一巨星按 QRF 距离分层](../figures/v3_8/F04_giants_qrf.pdf)。
候选层仅包含已有或补算了门槛的对象。Aitoff 每颗星使用真实天球位置，
只在写文件时栅格化点层。图中显示的是足迹内计数；定量密度需按实际球面面积计算。
F04 提供 6,600 像素宽的 PNG 和对应 PDF。

[图和底表](../manifests/v3_8)保留输入及校验记录。详细分区误差、阈值变体、
LF 情景、历史对照和全套图件继续见 [v3.6](v3_6_figure_index.md) 和
[v3.7](v3_7_paper_figures.md) 图件索引。完整分箱数组和逐星大件在 NERSC。
[新增星的物理参数与参考 CMD](../figures/v3_8/A01_recovered_stars.pdf)列入附图。
