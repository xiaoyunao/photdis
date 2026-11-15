# v3.6 科学图件索引

图内采用 Pal 5 New Step 5 的英文样式。每张已保存的图配有 300 dpi PNG、
PDF、统计表和包含输入校验值的 JSON。下表列出当前可读图件和仍由计算作业
生成的部分。各图中的星数、后验计数与定向参考样本分别使用其原有分母。

| 图组 | 当前图件 | 科学含义与证据范围 |
|---|---|---|
| F01 样本进入与流失 | [目录筛选](../figures/v3_6/F01_catalog_admission.pdf) · [两份原始距离库的流失](../figures/v3_6/F01_near_reference_flow.pdf) · [去重参考与目录观测 CMD](../figures/v3_6/reference_admission/observed_cmd.pdf) | 从原始谱距记录到预测支持、矮星评分和最终入选。CMD 显示 2,674,275 颗有可用正参考距离中心的原生对象，以及整个支持目录和两档选样。 |
| F02 测距比较 | [相同测光的旧、新 QRF](../figures/v3_6/F02_paired_qrf_same_grz.pdf) · [Gaia](../figures/v3_6/distance/Gaia_all_classes.pdf) · [SpecDis 矮星](../figures/v3_6/distance/SpecDis_dwarfs.pdf) · [SPDist](../figures/v3_6/distance/SPDist_all_classes.pdf) · [DR20 南站矮星](../figures/v3_6/distance/DR20_MINESweeper_dwarfs.pdf) | 七个参考来源分别比较可预测、主选和严格样本。低数量组保留逐星误差条，残差窗口以外的对象保留在统计中。SPDist 图注明主图轴域之外的参考中心数。 |
| F03 蓝星异常 | [距离失效](../figures/v3_6/F03_blue_distance_failures.pdf) · [后验和评分](../figures/v3_6/F03_blue_posteriors_and_scores.pdf) · [完整后验示例](../figures/v3_6/F03_qrf_pdf_examples.pdf) · [同星图像](../figures/v3_6/F03_selected_coadd_cutouts.pdf) | 冻结的 Gaia 抽样中有 116 颗暗蓝近邻和 189 颗偏亮蓝星。完整后验区分吻合、多峰、弱红外和无可用红外的例子。 |
| F04 条件误差 | [Gaia 星等和颜色分层](../figures/v3_6/F04_gaia_magnitude_color.pdf) | 显示偏差及各参考分箱的实际数量。[全量条件误差图](../scripts/v3_6_plot_condition_errors.py)同时显示两档选样的区间宽度、参考中心相符率及积分参考误差后的相符率。 |
| F05 测光信息 | [蓝星颜色平面](../figures/v3_6/F05_blue_color_planes.pdf) | 同星比较光学和可用的红外颜色。每个平面分别应用必要波段的有效性条件。全量[同星 CMD 与颜色图](../scripts/v3_6_plot_reference_colors.py)另外按来源给出观测星等、参考绝对星等、QRF 绝对星等及距离残差。 |
| F06 天图与有效面积 | [面积及深度](../figures/v3_6/F06_coverage_and_near_counts.pdf) · [赤道系目录密度](../figures/v3_6/F06_catalog_density_equatorial.pdf) · [银道系目录密度](../figures/v3_6/F06_catalog_density_galactic.pdf) · [赤道系后验壳层](../figures/v3_6/F06_shell_density_equatorial.pdf) · [银道系后验壳层](../figures/v3_6/F06_shell_density_galactic.pdf) · [Gaia 参考图](../figures/v3_6/reference_sky/F06_Gaia_galactic.pdf) · [SpecDis 参考图](../figures/v3_6/reference_sky/F06_SpecDis_equatorial.pdf) | 6,616.12 平方度来自成像随机点。各来源的参考密度和残差分别绘制。每个像素和选样有至少十个参考时才显示误差颜色，计数表保留低数量像素。 |
| F07 选样的径向代价 | [六距离层组成](../figures/v3_6/F07_radial_photometric_composition.pdf) · [硬分箱与后验和](../figures/v3_6/F07_hard_and_posterior_counts.pdf) · [概率与矮星评分](../figures/v3_6/F07_catalog_probability_conflicts.pdf) · [银纬与深度](../figures/v3_6/F07_radial_sensitivity.pdf) | 暗蓝样本在两档 5–10 kpc 后验计数中都超过一半。严格档的 10 kpc 外中位数计数为零，但尾部后验权重约为 16,103 颗。 |
| F08 带符号视差 | [冻结弱 Gaia 评分和区间约束](../figures/v3_6/F08_frozen_weak_parallax.pdf) · [主选条件比较](../figures/v3_6/F08_conditional_weak_gaia_primary.pdf) · [严格档条件比较](../figures/v3_6/F08_conditional_weak_gaia_strict.pdf) | 用同一分层抽样的带符号视差比较 QRF、名义 SED 和宽距离先验。前向图显示距离尺度、远端权重及视差共同零点的敏感性。 |
| F09 远端混入 | [定向光谱巨星](../figures/v3_6/F09_frozen_spectroscopic_leakage.pdf) · [按星族保留率](../figures/v3_6/F09_class_retention.pdf) · [SpecDis 迁移](../figures/v3_6/F09_direct_migration_SpecDis.pdf) · [DR20 南站迁移](../figures/v3_6/F09_direct_migration_DR20_MINESweeper.pdf) · [五种光度函数和星族情景](../figures/v3_6/F09_conditional_lf_contamination.pdf) | 两档都有超过 10 kpc 的光谱巨星。各来源分别保留距离迁移，情景图使用分层抽样估计各距离层的混入。 |
| F10 科学展示与方法对照 | [亮红近端样本的日心投影](../figures/v3_6/F10_near_cartesian_catalog.pdf) · [各预测层的 Gaia 与光谱覆盖](../figures/v3_6/F10_photometric_reference_coverage.pdf) · [历史波段、巨星和星团测试](../figures/v3_6/F10_historical_method_controls.pdf) | 日心投影使用 QRF 中位距离。亮红近端和暗蓝远端的直接证据情况不同。历史对照沿用各自的样本和模型。 |

现有的主文候选图覆盖 F01–F10。
最终选图以全量匹配、独立证据的数量及各模型比较的可解释性为依据。
