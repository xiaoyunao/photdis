# v3.6 科学图件索引

图内采用 Pal 5 New Step 5 的英文样式。每张已保存的图配有 300 dpi PNG、
PDF、统计表和包含输入校验值的 JSON。下表列出当前可读图件和仍由计算作业
生成的部分。各图中的星数、后验计数与定向参考样本分别使用其原有分母。

| 图组 | 当前图件 | 科学含义与证据范围 |
|---|---|---|
| F01 样本进入与流失 | [筛选流程](../figures/v3_6/F01_catalog_admission.pdf) | 核对原生成像、预测支持和两档筛选。[全量参考覆盖程序](../scripts/v3_6_plot_reference_admission.py)按来源与距离追踪历史参考流失，并去重比较母样本、参考星和两档选样的观测 CMD。 |
| F02 测距比较 | [绘图程序](../scripts/v3_6_plot_distance_atlas.py) | 同一来源和星族下比较可预测、主选和严格样本。中央残差图另列超出范围的对象，低数量组保留逐星误差条。全量图等待参考关联。 |
| F03 蓝星异常 | [距离失效](../figures/v3_6/F03_blue_distance_failures.pdf) · [后验和评分](../figures/v3_6/F03_blue_posteriors_and_scores.pdf) · [同星图像](../figures/v3_6/F03_selected_coadd_cutouts.pdf) | 116 颗暗蓝近邻和 189 颗偏亮蓝星来自冻结的 Gaia 抽样。颜色、绝对星等和图像说明两种偏差的证据与局限。 |
| F04 条件误差 | [Gaia 星等和颜色分层](../figures/v3_6/F04_gaia_magnitude_color.pdf) | 显示偏差及各参考分箱的实际数量。[全量条件误差图](../scripts/v3_6_plot_condition_errors.py)同时显示两档选样的区间宽度、参考中心相符率及积分参考误差后的相符率。 |
| F05 测光信息 | [蓝星颜色平面](../figures/v3_6/F05_blue_color_planes.pdf) | 同星比较光学和可用的红外颜色。每个平面分别应用必要波段的有效性条件。全量[同星 CMD 与颜色图](../scripts/v3_6_plot_reference_colors.py)另外按来源给出观测星等、参考绝对星等、QRF 绝对星等及距离残差。 |
| F06 天图与有效面积 | [面积及深度](../figures/v3_6/F06_coverage_and_near_counts.pdf) · [赤道图](../figures/v3_6/F06_shell_density_equatorial.pdf) · [银道图](../figures/v3_6/F06_shell_density_galactic.pdf) | 用同一套成像随机点面积比较两档后验密度。6,616.12 平方度描述可观测覆盖。全量参考的覆盖与残差图等待匹配。 |
| F07 选样的径向代价 | [六距离层组成](../figures/v3_6/F07_radial_photometric_composition.pdf) · [保留率](../figures/v3_6/F07_shell_retention.pdf) | 暗蓝样本在两档 5–10 kpc 后验计数中都超过一半。远端三个距离层均保留。硬分箱比较等待目录投影。 |
| F08 带符号视差 | [冻结弱 Gaia 评分和区间约束](../figures/v3_6/F08_frozen_weak_parallax.pdf) | QRF 的视差预测优于宽距离模数先验。暗蓝距离层仍有大范围的区间歧义。条件评分将加入名义 SED 与前向尺度检验。 |
| F09 远端混入 | [定向光谱巨星](../figures/v3_6/F09_frozen_spectroscopic_leakage.pdf) | 严格筛选仍有明确超过 10 kpc 的巨星。图示是定向匹配数。光度函数情景估计使用另一个具有统计权重的抽样。 |
| F10 方法对照 | [历史波段、巨星和星团测试](../figures/v3_6/F10_historical_method_controls.pdf) | 整理历史波段贡献、巨星条件距离和 NGC 288 完整似然，保留各自的样本与误差定义。 |

现有的主文候选图覆盖 F01、F03、F04、F05、F06、F07、F08、F09 和 F10。
最终选图以全量匹配、独立证据的数量及各模型比较的可解释性为依据。
