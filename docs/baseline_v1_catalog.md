# baseline v1 产物与复现

2026-11-03。实验定义：[冻结协议](baseline_v1_protocol.md)。v0所有数据、模型及目录不覆盖；v1单独写入NERSC的`/pscratch/sd/y/yunao/photdis_pipeline/data/baseline_v1/`。

## 计算顺序

以下命令在NERSC checkout提交；脚本只在Slurm计算节点执行。

```bash
sbatch scripts/run_baseline_v1_numerical.sh
sbatch scripts/run_baseline_v1_fit.sh train
sbatch scripts/run_baseline_v1_depth_prepare.sh
sbatch --dependency=afterok:NUM_JOB:TRAIN_JOB scripts/run_baseline_v1_fit.sh evaluate
sbatch --dependency=afterok:EVAL_JOB:DEPTH_INPUT_JOB scripts/run_baseline_v1_finalize.sh
sbatch --dependency=afterok:FINALIZE_JOB scripts/run_baseline_v1_apply.sh
sbatch --dependency=afterok:APPLY_JOB scripts/run_baseline_v1_report.sh
```

母样本、身份和切分读取v0冻结文件。训练与应用脚本拒绝覆盖已有产物；新实验须另起版本。最终全母表预测按不重叠行片段一次计算，按原行顺序合并；并发仅改变运行速度，不改变随机扰动或科学选样。

## 主要文件

| 文件 | 内容 |
|---|---|
| `sample.npz`, `sample.json` | 残差无关的3000源母样本分层抽样及身份哈希 |
| `convergence_A/B/C.*`, `numerical.json` | IID/Sobol、箱内插值、独立512点参照、解析反例及固定数值设置 |
| `A/B/C/A_on_B/A_on_C/B_on_C.joblib` | 宽模型、三个粗类别条件QRF、纯颜色门控和训练比例 |
| `*_train.json` | 对应v0模型及新模型哈希、完全一致的训练身份和类别计数 |
| `*_{wide,natural,balanced,oracle}_evaluation.npz` | cal/test中全部有效距离目录参照的完整经验PDF和摘要 |
| `selection.json` | 只使用cal决定的部署候选、波段升级、区间扩展和研究质量层 |
| `*_selected_evaluation.npz`, `recommended_evaluation.npz` | 固定部署候选与推荐流程；没有输入真实类型 |
| `v0_branch_audit.json` | 所有已知演化/过渡参照的真实参考光度附近质量诊断 |
| `depth_input.*`, `depth_predictions.npz`, `depth_result.json` | 固定原始通量降质、重测质量/可用性/路由、原源聚类不确定度 |
| `shards/`, `catalog_00.fits`–`catalog_13.fits` | 内部全母样本结果；片段不作为额外独立对象计数 |
| `application.json`, `validation.json`, `report.json`, `figures/` | 实际计数、选择及产物哈希、检查结果与主图 |

## FITS新增含义

输入身份、星等、误差、WISE资格、原质量证据延续v0。分位数顺序仍为2.5%、16%、50%、84%、97.5%。

| 列 | 含义 |
|---|---|
| `version` | 恒为1；独立目录，不能覆盖v0 |
| `A_mu/B_mu/C_mu` | 各可用模型最终部署候选的原始分位数，缺波段为NaN |
| `A_modes_mu`等 | `[mode1,mode2,local_mass1,local_mass2]`，模式附近±.5mag局部质量，不是互斥星族概率 |
| `A_weights/B_weights/C_weights` | wide为叶内三类与未知标签比例；混合模型为三个条件分布权重与宽模型回退质量，具体候选由`selection.json`定义 |
| `branch_weights` | 推荐模型上述四个权重；不是已校准银河类别概率 |
| `draws`, `numerical_converged` | 实际扰动次数、所属光学误差层是否通过全部固定收敛容差；后者是层级审计标志，不是每颗星单独的收敛证明 |
| `route` | 0/1/2对应A/B/C；只有域内且cal允许时升级；均域外时保留A且`supported=False` |
| `mu`, `distance_kpc`, `mu_cal` | 推荐原始分位数、对应kpc距离、cal扩展区间；后者不构成新PDF |
| `disagreement` | 本源可用A/B/C距离模数中位数最大差，包含域外模型，用于已登记的质量对照 |
| `research_quality` | 是否通过cal固定研究质量组合；若没有组合满足各类误差/样本量要求则全部False；即使True也不等于真实距离已独立标定 |
| `outside_train_precision`, `supported`, `ambiguous` | 超出训练误差入口、名义宽模型颜色支持、多峰诊断；不单独决定可靠性 |
| `internal_uncalibrated` | 恒为True，说明目录仍是内部研究结果 |

北方场13保留身份/可用性/质量，route=-1且距离NaN。没有套用DECam模型。未知粗类别对象的有效距离标签继续保留于宽模型；oracle只用于解释类内关系，绝不进入目录。

## 结果边界

Gaia、SPECDIS、SPDist仍按来源分别报告，后两者可能共享上游Gaia或恒星模型尺度。原始经验PDF没有去卷积标签误差；cal区间针对带噪参照点，且与模型选择共用发展校准场，不能宣称严格独立的真实距离覆盖保证。旧test已被查看，仍称发展阶段空间验证。

深度实验使用固定实测模板通量与实际母样本噪声/掩膜。模板噪声已计入，无法降噪的配对及少数非变暗配对另列；未模拟新图像、形态变化、真实暗端族群或选择完备率。未知类模板并未被强行视作矮星。
