# Photdis v3.5 目录读取说明

目录生产记录见[当前结果](v3_5_result.md)。源表和大型产品位于 NERSC
`/pscratch/sd/y/yunao/photdis_pipeline/data/v3_5`。每个已验收的
`catalog/sweep-*/complete.json` 列出本分区全部文件及校验值。最后的总清单
和天区统计由 `delivery` 作业生成。

`part_*.npz` 已包含通过原生测光与 QRF 支持要求的对象。每个分区可单独读取。
`ls_id` 是目录唯一键。`source_id` 是原生 Gaia DR3 别名，0 表示没有可用
原生别名。重复 Gaia 别名在总核验中单独标记，Gaia 验证不会用这些别名
进行一对一关联。`identity` 是冻结预测的内部身份，复用 v3.4 时沿用原值。
`parent_row` 是该 sweep 中从零开始的行号。

| 字段 | 含义 |
|---|---|
| `TYPE`, `photometric_system`, `RA`, `DEC`, `EBV` | 原生 PSF、测光系统、赤道坐标与尘埃 |
| `observed_r`, `dereddened_g_r`, `band_state`, `band_mask` | 星等、消光校正颜色及原生波段状态 |
| `qrf_quantiles_kpc` | 第 2.5、16、50、84、97.5 距离百分位，单位 kpc |
| `qrf_mu_edges`, `qrf_mu_mass` | 完整距离模数后验的分格边界与每格质量，格内按均匀模数密度解释 |
| `qrf_qnear`, `qrf_bins` | 10 kpc 以内质量与 0–2、2–5、5–10、10–20、20–50、50 kpc 以上的质量 |
| `primary`, `strict` | 冻结主选样和严格子样本 |
| `dwarf_selection_score`, `nominal_qnear`, `nominal_evaluated` | 矮星评分、名义 SED 近端量及计算状态。评分低于主阈值的源只需要 QRF 后验 |
| `qrf_supported`, `nominal_numerical_flag` | QRF 支持与名义 SED 计算质量 |
| `analog_v34_observable`, `reused_v34` | 落在 v3.4 观测条件内、或精确复用了 v3.4 目录行。前者允许新天空，参考覆盖需要另查 |

`delivery/v34_footprint.json` 提供 v3.4 使用的 13 个赤道坐标矩形及其来源。
每个矩形的 `bounds_deg` 按 `RA 下界、DEC 下界、RA 上界、DEC 上界` 排列，
采用左闭右开边界。按目录坐标可判断新预测是否位于原天区。`sweeps.csv`
逐分区汇总旧范围内的预测数量和范围内外的亮红对象数量。Gaia 和光谱验证的
对象名册同时给出原天区标记，参考历史仍按对象身份单独追踪。

`parent_status.npz` 按原始 sweep 顺序保存每条记录的准入状态。`complete.json`
中的 `exclusive_counts` 依次对应周期边界、测光版本、缺失形态、非 PSF、
银纬与消光、r 星等、光学质量、QRF 支持和最终有预测。目录做限星等距离研究，
星数与面积要结合当次观测选择解释。远端、暗蓝和弱视差对象的可用精度见
实际参考验证。
