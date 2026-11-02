# baseline v0 目录、模型与复现说明

完整定义见[实验协议](baseline_v0_protocol.md)。这是内部、未完成绝对标定的多星族测光距离基线；未知类型、域外和多解对象仍有记录，不能将全部点估计当成同等可靠的距离。

## 数据入口和训练

母表沿用已处理 DR11 PSF、grz S/N≥5、clean、高银纬低消光条件；没有 Gaia、光谱或 WISE 必需条件。A 使用 `(g-r)0,(r-z)0`，B 加 `(z-W1)0`，C 再加 `(W1-W2)0`。B/C 要求所用 WISE 波段通过原质量条件、S/N≥5。

宽人群标签由高质量 S/N≥20 的 Gaia 校正视差优先，合格且唯一身份关联的 SPECDIS 补充。训练核心要求各光学误差<0.03mag，但应用和完整测试没有这个额外限制。训练目标统一为 Legacy `Mg = g0 - mu_reference`。不向预测器输入真实温度、重力、金属丰度或 Gaia 距离。

模型学习带噪参考标签的经验光度分布。它不是去卷积后的真实光度分布，也没有纠正 Gaia／DESI 参照样本的选择函数。DESI 粗类别只是叶内经验标签比例；未知标签质量占比保留，不能把其余比例当成银河恒星的校准类别概率。训练对象不按粗类别拆开，因而能保存相隔较远的光度解。

## NERSC 产物

根目录：`/pscratch/sd/y/yunao/photdis_pipeline/data/baseline_v0/`。

| 文件 | 内容 |
|---|---|
| `parent_00.npz`–`parent_13.npz` | 完整光学母表，含逐波段误差、质量、原始类别和参考诊断 |
| `owner_XX.npy` | 跨场身份归属，保证应用目录唯一对象 |
| `references.npz` | 宽人群参照候选、标签来源/误差、整场分区、训练资格 |
| `A/B/C.joblib` | 实际可用样本训练的冻结 QRF |
| `A_on_B/A_on_C/B_on_C.joblib` | 相同训练对象的波段增量对照 |
| `*_evaluation.npz` | 全部 cal/test 参照候选的 PDF、分位点、模式、类别、支持域和区间 |
| `catalog_XX.fits` | 逐场全部唯一母样本内部预测，扩展名 `BASELINE_V0` |
| `catalog_XX.json` | 实际行数、可用性、覆盖统计、路由与文件哈希 |
| `fit_result.json` | 训练/验证成绩、同星对照、cal 确定的推荐路由 |
| `figures/` | 本轮计算生成的图；审核后将小图复制到Git的`figures/baseline_v0/` |
| `report.json`, `validation.json` | 统一成绩单与实际产物核验 |

北方场13保留相同目录结构、WISE覆盖与身份，但 `route=-1`、距离为NaN。当前没有足够同系统天区建立独立的 BASS/MzLS 验证流程；不套用 DECam 模型冒充已验证的北天结果。

## FITS 主要列

所有长度为5的分位数列依次为 **2.5%、16%、50%、84%、97.5%**。

| 列 | 定义 |
|---|---|
| `identity`, `source_id`, `ls_id`, `field` | Gaia 正身份优先，否则负 LS 身份；原身份和场号均保留 |
| `ra`, `dec`, `release` | 坐标（度）、原始光学发布/相机系统 |
| `r`, `g0/r0/z0/w10/w20`, `sigma_*` | 观测r、各波段改正后的星等、原测光误差 |
| `B`, `C`, `w1_clean/w2_clean`, `snr_w1/snr_w2` | 实际WISE资格及其质量和S/N证据；缺WISE不会删除A预测 |
| `optical_error`, `outside_train_precision` | 最大grz误差；是否≥0.03mag。后者描述超出训练精度入口，不表示该星已被独立验证 |
| `nonstellar_risk` | 已有非恒星风险证据；不等同确定分类，完整应用中保留 |
| `A_mu/B_mu/C_mu` | 各可用模型的原始经验距离模数分位数；缺波段为NaN |
| `A_modes_mu`（B/C同理） | `[mode1_mu, mode2_mu, mass1, mass2]`，按平滑峰高找模式，质量是各模式±0.5mag内的概率；两质量不是全分布的互斥分解 |
| `A_class_weights`（B/C同理） | `[dwarf,evolved,transition_or_other,unknown_label]` 的训练叶权重 |
| `A_supported`, `A_ambiguous`（B/C同理） | 颜色支持域；固定多解诊断：第二峰局部质量≥0.2且两峰间隔≥2mag |
| `route` | -1无同系统模型；0=A、1=B、2=C。升级决定来自cal，同源域外时允许向支持更充分的可用模型回退 |
| `mu`, `distance_kpc` | 推荐模型原始分位数，距离单位kpc |
| `mu_cal` | 推荐模型在cal参照层做CQR扩展后的区间端点，中位数不变；它不是一个新PDF |
| `supported`, `ambiguous`, `internal_uncalibrated` | 推荐模型状态；最后一列恒为True，标识本轮内部研究产品 |

推荐点估计可能落在多解分布中信息较弱的位置；研究多解对象应同时使用模式、分位数和支持状态。`supported=True`只表示参照训练颜色域，不能保证类型已确认或真实距离覆盖率已达到名义值。

当所有可用模型都在支持域外时，仍保存A预测和`route=0`，但`supported=False`。不能仅凭有距离数值或路由编号选出可靠样本。最终计数和分星族表现见[结果报告](baseline_v0_result.md)。

## 复现

在 NERSC checkout 内提交；所有扫描/训练/预测均在 Slurm 分配的计算节点执行：

```bash
sbatch scripts/run_baseline_v0_prepare.sh
sbatch --dependency=afterok:PREP_JOB scripts/run_baseline_v0_fit.sh
sbatch --dependency=afterok:FIT_JOB scripts/run_baseline_v0_apply.sh
sbatch --dependency=afterok:APPLY_ARRAY_JOB scripts/run_baseline_v0_report.sh
```

已有准备数据和目录不会自动覆盖。模型评估可复用已完成的冻结产物，但必须通过训练身份与模型哈希检查。重新设计实验须使用新版本及输出目录，不能覆盖本轮结果。

模型评价区分：同一教师来源下的预测一致性、Gaia几何参照表现、既有整场发展验证，以及独立系统的绝对标定。SPECDIS/SPDist可能共享 Gaia 或恒星模型信息，不能自动称为完全独立真值。弱/负视差图只在观测视差空间展示，S/N选样下的残差中心不作零点推断。
