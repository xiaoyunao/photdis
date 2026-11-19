# 科学图件索引

每组都有PNG/PDF和对应表/网格。目录中保留36组图；主线先看下面七类，其余用于解释选样、分支和失败控制。所有数字结论见[科学决定](science_decisions.md)，重画命令见[README](README.md)。

## 先看的图

| 科学问题 | 图件 | 数值来源 |
|---|---|---|
| 同一固定gr入口、MS/TO筛选在全足迹呈现什么？ | [总览RGB](../../figures/stream_response_20261119/full_rgb/rgb_overview.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_overview.pdf) | `full_rgb/rgb_overview.npz`，`full_validation/full_summary.json` |
| 此前Pal 5为何空白，放宽后出现什么？ | [尾段联合图](../../figures/stream_response_20261119/local/Pal5_tail_FOS_recovery.png) · [PDF](../../figures/stream_response_20261119/local/Pal5_tail_FOS_recovery.pdf) | `local/Pal5_tail_redraw.npz`、`segment_excess.csv`及source/area cutflow |
| 年龄和金属丰度更接近Pal 5的模板怎样改变响应？ | [物理对照](../../figures/stream_response_20261119/local/Pal5_tail_Pal5_control_recovery.png) · [PDF](../../figures/stream_response_20261119/local/Pal5_tail_Pal5_control_recovery.pdf) | 同一源、同一面积；对照MIST1.2、11.5Gyr、FeH−1.4，不能与MIST1.0主模板作纯FeH因果对比 |
| Sgr北段的RGB兼容分支和MS/TO各接受什么？ | [北段相位图](../../figures/stream_response_20261119/local/Sgr_north_phase_controls.png) · [PDF](../../figures/stream_response_20261119/local/Sgr_north_phase_controls.pdf) | `local/Sgr_north_phase_redraw.npz`、`Sgr_phase_decomposition.csv` |
| 近一些的Sgr南段结论是否相同？ | [南段相位图](../../figures/stream_response_20261119/local/Sgr_south_phase_controls.png) · [PDF](../../figures/stream_response_20261119/local/Sgr_south_phase_controls.pdf) | `local/Sgr_south_phase_redraw.npz`与同一phase表 |
| 一个M13零点能否迁移到其他系统和相位？ | [同星尺度图](../../figures/stream_response_20261119/sequence/known_system_branch_calibration.png) · [PDF](../../figures/stream_response_20261119/sequence/known_system_branch_calibration.pdf) | `sequence/predictions.npz`、`distance_scale.csv`、`shared_bootstrap.npz` |
| 控制参考距离本身会怎样改变“金属丰度差”？ | [条件化图](../../figures/stream_response_20261119/conditioning/metallicity_conditioning.png) · [PDF](../../figures/stream_response_20261119/conditioning/metallicity_conditioning.pdf) | `conditioning/estimands.csv`、`balance.csv`、`sky_draws.csv`和`effect_definitions.csv` |

## 六种RGB：两个完整版本

总览统一为B10–15/G15–26/R26–100kpc。五个宽bin内部各自等模数三分；B/G/R分别为该bin内的近/中/远响应，实际边界直接写在图中。每个RGB文件同时保留密度RGB和positive-residual RGB两行。

| 距离响应 | 原v4 cube重绘（本轮必须交付） | 新统一入口/MS_TO全足迹 |
|---|---|---|
| 总览 | [PNG](../../figures/stream_response_20261119/rgb/rgb_overview.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_overview.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_overview.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_overview.pdf) |
| 6.3096–10kpc内部 | [PNG](../../figures/stream_response_20261119/rgb/rgb_bin_6.3_10.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_bin_6.3_10.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_bin_6.3_10.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_bin_6.3_10.pdf) |
| 10–15kpc内部 | [PNG](../../figures/stream_response_20261119/rgb/rgb_bin_10.0_15.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_bin_10.0_15.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_bin_10.0_15.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_bin_10.0_15.pdf) |
| 15–26kpc内部 | [PNG](../../figures/stream_response_20261119/rgb/rgb_bin_15.0_26.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_bin_15.0_26.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_bin_15.0_26.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_bin_15.0_26.pdf) |
| 26–50kpc内部 | [PNG](../../figures/stream_response_20261119/rgb/rgb_bin_26.0_50.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_bin_26.0_50.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_bin_26.0_50.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_bin_26.0_50.pdf) |
| 50–100kpc内部 | [PNG](../../figures/stream_response_20261119/rgb/rgb_bin_50.0_100.png) · [PDF](../../figures/stream_response_20261119/rgb/rgb_bin_50.0_100.pdf) | [PNG](../../figures/stream_response_20261119/full_rgb/rgb_bin_50.0_100.png) · [PDF](../../figures/stream_response_20261119/full_rgb/rgb_bin_50.0_100.pdf) |

两套的入口、相位和面积不同，不能把图面差异全部归因于测距改进。RGB无单一物理密度色条；每种RGB旁都有`physical_*`标量图，三列是对应B/G/R密度，上/下行为密度/signed残差，各有物理色条。例如新增[总览物理密度](../../figures/stream_response_20261119/full_rgb/physical_overview.png)与[PDF](../../figures/stream_response_20261119/full_rgb/physical_overview.pdf)。这些图的负值没有裁掉。三色图和标量图共享同一个NPZ中的科学显示网格。

## 读图时的科学口径

局部图的Hess为target-minus-control；橙线包括计数实际使用的2sigma_g加宽。星等、源质量和mask入口另见逐项cutflow。没有独立面积的面板掩膜，不解释为零数密度。角向图的0.1°/0.3°平滑只用于展示，段超额和区间来自原始计数与独立面积。

相位图三列顶图使用同一parent Hess，只改变叠加的接受边界；底图比较相同入口和背景下的相位响应。RGB-only、MS-only和both是接受区的互斥分解，不是独立光谱分类。

同星尺度图顶行为参考绝对星等与固定序列之间的差；底行为同一批成员的QRF/grid/固定序列/标定序列，空心点为N<30。灰色区间标明10%尺度范围；转折曲线灰带为DECam折返颜色范围，实际南北范围和噪声频率在`native_fold_regions.json`逐native记录。误差条的共同尺度与成员draw跨方法、星等和相位共享。M13低主序是标定组。

条件化图是贫金属减普通组的加权mean，三个面板的样本域不同；每个域内两种设计的源身份与支持集相同。没有可比样本的参考/距离组保持空白。mean/median、ESS、支持流失和所有控制变量balance另见表，不能把小的条件化对比读成物理FeH效应为零。

Pal 5 leading、相邻轨迹条带、Sgr额外入口/窄核对照均在`local/`完整保存。leading的一个角段无覆盖，未被抹去。`core/actual_core_counts.csv`才描述巡天特殊检测核心。
