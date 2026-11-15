# Photdis v3.4 近端矮星候选目录

定义、验证分母和适用边界见[主结果](v3_4_result.md)。目录使用既有
13 个 DECam 天区的原生 Legacy DR11 PSF。生产输入沿用已经过原生波段、
高银纬和低消光约束的 1,183,990 行母样本，仅在固定配置指定的
(16\le r<18)、(0.8\le (g-r)_0<1.4) 和两种 W1 状态中计算准入。
状态 2 表示有测量但未通过质量要求，状态 4 表示显著测量。
光谱类别和 Gaia 数据没有进入准入及 QRF 测距。

## 读取

最终目录含 117,127 个主选样、96,824 个严格选样和 37 个数据分区。
[全量核验清单](../manifests/v3_4/catalog/delivery_validated.json)记录了逐源
核对和实际后验汇总。

大文件在 NERSC
`/pscratch/sd/y/yunao/photdis_pipeline/data/v3_4/near_catalog`。
`field_00` 至 `field_12` 的 `part_*.npz` 是入选对象。任意一个分区
都可以独立读取。整体结果由 `complete.json`、`field_counts.csv` 和
`delivery_validated.json` 绑定。

在 NERSC 目录下可用以下方式查看一个小分区的距离和后验字段。

```python
from pathlib import Path
import numpy as np

root = Path("/pscratch/sd/y/yunao/photdis_pipeline/data/v3_4/near_catalog")
part = next(iter(sorted(root.glob("field_*/part_*.npz"))))
with np.load(part) as data:
    print(data["ls_id"][:5])
    print(data["qrf_quantiles_kpc"][:5])
    print(data["qrf_qnear"][:5])
```

## 字段

| 字段 | 含义 |
|---|---|
| `ls_id`, `identity`, `parent_row` | 原生 Legacy ID、项目固定身份和已核验的父表行 |
| `TYPE`, `photometric_system`, `field`, `RA`, `DEC` | 原生形态、测光系统、天区和赤道坐标，角度单位为度 |
| `observed_r`, `dereddened_g_r`, `stratum` | 观测 r、消光校正 g−r、原应用抽样层 |
| `band_state`, `band_mask` | 原生 grizW1W2 数据状态与可用掩码 |
| `dwarf_selection_score`, `nominal_qnear` | 冻结 grz 矮星评分及用于选样的名义 SED 近端概率 |
| `primary`, `strict`, `qrf_supported` | 已通过主选样、是否通过严格选样、是否在 QRF 测光支持内 |
| `nominal_numerical_flag` | 名义 SED 概率积分的网格、区间或边缘质量标记 |
| `qrf_quantiles_kpc` | QRF 距离的第 2.5、16、50、84、97.5 百分位，单位 kpc |
| `qrf_qnear`, `qrf_bins` | QRF 在 10 kpc 内的概率及六个区间概率，区间为 0–2、2–5、5–10、10–20、20–50、50 kpc 以上 |
| `qrf_mu_edges`, `qrf_mu_mass` | 完整距离模数网格边界和每格概率质量。每格采用模数空间内的均匀密度，质量和为 1 |

每个天区的 `selection_audit.npz` 记录整个归档 PSF 输入的身份和
首个准入结果。状态 0 表示超出生产域，1 表示矮星评分未达阈值，
2 表示在 QRF 支持外，3 表示名义 SED 近端概率未达阈值，4 表示进入目录。
严格选样是主选样中的子集。条件后验可计入远端质量，目录受星等范围限制，
不能用作 10 kpc 体积完备计数。弱 Gaia 子样本的 QRF 距离精度仍待验证，
当前定向光谱与历史标签也不足以测定全目录的巨星混入比例。
