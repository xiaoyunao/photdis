# 科学收尾复现说明

基线为 `526cc819b320bdac32689081969e54860f2b9043`。此次按照用户粘贴的 Pro WP-A 至 WP-E 工作包执行。科学结论和英文图见[科学报告](report.md)和[图件索引](figures.md)。输出目录为 `manifests/science_closeout_20261120`。既有生产和历史结果按各自的哈希保留。

## 数据入口与约定

- **目录。** `scripts/science_closeout_catalogue.py --core` 在 NERSC 读取先前来源绑定的 32 个专用选择分片，核对实际 PSF 和逐分片哈希，细分 2,281,023 个核心源和 106,196 个 5–10 kpc 研究源。无选项时用紧凑归档生成适用域边际、距离迁移及 GD-1 分层。面积沿用原生 DR11 随机点。共用成像区域不相加，分层内显示地区面积及实际深度。参考表保留拟合与目标人群的各自来源、分母、量和分箱。0–2 与大于 10 kpc 的归档共用剩余 PDF 概率在光谱迁移文件中标明为合并数。
- **训练。** `scripts/science_closeout_sequence.py` 通过既有 `sv18_benchmark.training` 重读原生训练表和序列的相同冻结成员。核验所有树在真实输入训练颜色下都能逐叶重建绝对光度直方图和类别权重。叶节点化学均值按有可靠值的响应质量归一化。北天用唯一 Gaia 键和一致的可靠类别链接历史光谱化学。观测颜色的叶响应和本地训练盒独立于测光误差。在 PDF 的距离轴上，较大的模数对应给定视星等下较亮的训练光度。相同成员的经验直方图比较继承 0.1 mag 双颜色格和四格插值，原有固定扰动定义不变。QRF、简单经验、经验星团、MIST 和混合模型各自有支持条件。
- **已知序列。** NGC 6362/6723/M55 完整精度表来自外部 W1 RR Lyrae 标尺，评价核加入 0.10 mag 的共同尺度量。M5/M13/M30 保留历史共同样本和训练角色，以及对应的原概率模型与原尺度不确定度。M92 是已使用天区，仅重放有相应共同低拥挤与无掩膜资格、且无训练或标定重合的 9 个成员，与原来冻结中位数及分位数核对，评分采用 0.10 mag 的敏感性尺度。M68 的冻结归档支持其分箱群体检验。GD-1 使用实际成员数据及外部模数曲线。CRPS 使用参考中心，log score 对参考误差核卷积。包含率指参考中心，区间宽度取中位数。
- **弱视差。** `scripts/science_closeout_signed.py FIELD` 读取原生 PSF、排除冻结 QRF 训练和过去诊断身份，恢复各视场实际视差质量定义。按合格正负视差人数复核此前原生源关联盘点。若归档与原盘点不合，按原 sweep 唯一身份规则回放。用原哈希冻结固定的 3,000 个上限。预测在原生颜色中运行冻结 QRF。仿真在原信噪选择下归一化，注入 0、±0.3、±0.5 mag 的整份 PDF 平移。各情景分别有 512 个零校准、512 个独立零校验和每个偏移 384 次模拟。场与块共享偏移标准差 0.01 mas，另验证 0.02 mas。每个星等和幅度独立判断功效。阈值写入 `configs/science_closeout_20261120.json`，前向评分只解封通过的视场与幅度。表中评分区间是同对象空间块重抽样。局部尺度诊断使用加权斜率，图误差条为较大零点假设下的一倍模型试验标准差。20–21/21–22 视场的弱视差检验与旧星团的绝对标尺对应两个不同的估计对象。
- **环境。** `scripts/science_closeout_environment.py` 以已有六场加三场固定参考及 A/B/C/D 路径重算共同表。有限范围 B/C 使用同一地图窗口并保持全部原始 SFD A 视图，使用 B 的 PDF 质量 ≥0.5、参考在地图支持、C 收敛及身份条件。D 是已知参考距离查询地图的点诊断。600 次块重采样给出偏差区间和同星 CRPS 差。9 场的参考原始人数和每个迭代退出标志分别列出。旧的 L219 逐星、拥挤和类别匹配诊断仍作为本次解释的依据。
- **迁移。** 径向表的空值代表缺少参考或定义，观测质量、参考距离、选择分母以及硬分箱和概率质量分列。人口情景是固定参考模板视图，对有明确类别的贫金属矮星相对于其他三组标签施加 0.5、1、2 倍，不用来估计总体真实污染。GD-1 信号引用冻结的 DESI、CFHT、SDSS 曲线和原成员选样；新表在 DESI 曲线下对原 PDF 积分，保沿流位置、窄距离窗及 thin/cocoon 身份。
- **图件。** `scripts/science_closeout_figures.py --compact-training` 在 NERSC 将训练颜色与 $M_r$ 聚为可携带的 Hess 数组底表。无选项时从紧凑结果渲染四组新增图，复用 `configs/v3_10_figure_style.json` 校验的 Pal 5/v3.8 字体、轴和输出设置。

## 执行与复现

所有 NERSC 计算脚本在已分配的 CPU 节点上执行。下面各脚本可在固定输入的 canonical checkout 上分别执行，现有完整输出位于 `/pscratch/sd/y/yunao/photdis_pipeline/data/science_closeout_20261120`。历史模型、训练文件、逐源活页及完整预测文件留在计算存储。紧凑表的来源路径和 SHA-256 位于执行记录及九场子表。

```bash
export PYTHONPATH=scripts:src:.deps:.deps/v39_dust
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/science_closeout_catalogue.py --core
python scripts/science_closeout_sequence.py
python scripts/science_closeout_environment.py
python scripts/science_closeout_signed.py L032 --workers 3
python scripts/science_closeout_figures.py --compact-training
```

提取后的紧凑表在任意安装 NumPy 和 Matplotlib 的项目环境中重绘。

```bash
python scripts/science_closeout_catalogue.py
python scripts/science_closeout_figures.py
```

同星独立模型分组复用归档 [cluster_depth](../../manifests/cluster_depth_20261120/results/paired.csv) 和 [sequence mechanisms](../../manifests/sequence_mechanisms_20261119/evaluate/paired.csv)。既有公开对照和实测成像支持见[图件索引](figures.md)。
