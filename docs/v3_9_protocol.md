# Photdis v3.9 扩展协议

以 c935235 和[最新评审](chatgpt-conversation://6af9bbfe-5288-83ea-9e15-f09b45daea62)
为基准。应用入口只用原生测光和 PSF 形态。参考观测用于训练、校准及检验。

- 南天保留高纬低消光和三带误差均小于 0.1 mag，新增观测 20–23 等。
  分 20–21、21–22、22–23 报告深度，另评估较低 S/N 研究层。
- 北区从 11011 原生 sweeps 构建母样本，使用原生 grz。南北常规归属
  采用赤纬 32.375° 和银河北半球划分，并在重叠区保留配对证据。
- 南天低纬低消光范围沿用同样的测光质量。显著消光单独用有限区域
  的视线依赖尘埃试验，依据地图覆盖、有效距离及拥挤程度解释结果。
- 补算冻结目录预测 5–10 kpc 内缺失的名义门槛，并记录已在 Redrock
  回放的值。亮红近端按 v3.8 已验证的四个分区进行完整 PDF 变换。
  逐星保留变换后的距离模数边界、归一化质量及重新积分的距离层。
- 数据目录和验证图按原生领域分区。冻结发布 PDF、主选、严格选样和
  新域的参考覆盖均可独立追溯。本轮交付技术说明和目录产品。

## 目录结构与复现

每个 sweep 保留按原始 FITS 行号排列的 `parent_state.npz`。行号记录
几何边界、原生 RELEASE、TYPE、科学域、质量与精度层。测光分区
以原生 `ls_id` 为键，距离分区通过 `photo_row` 回到同一来源和测光。
预测状态码为 0 测光拒绝、1 低 S/N 层尚未抽到、2 进入预测但颜色
不支持、3 已计算完整 PDF。状态 3 在质量层 6 和 7 分别对应
S/N 约 5–11 研究层与三带误差均小于 0.1 mag 的精确测光层。

南北均从原生源表建立，不以 Gaia 键作为应用门槛。南天和北天原生
分区保留重叠观测。合并显示采用巡天归属字段
`primary_survey_owner`，按 [LS DR11 分区定义](https://www.legacysurvey.org/dr11/description/)
选取；同一 Gaia 身份还须通过位置校验，才进入南北成对校验。
北区模型的 Gaia 几何、SpecDis 矮星、类别训练和留出成员分别留档。
确认输出范围时，以真实预测状态和留出结果为准。

接缝检查使用赤纬 32.375° 两侧各 2°，同时检查冻结南天和新增南北
源。优先采用位置相差小于 1.5″ 的唯一 Gaia DR3 身份。缺少 Gaia 键
时允许小于 0.7″ 的一对一位置匹配，两个有效 Gaia 键冲突时保留
原始归属并记录冲突。确认的同源对用两侧平均赤纬确定唯一归属，
发布 `delivery/seam/paired_sources.npz` 和
`delivery/seam/ownership_overrides.npz`。合并目录应用后者，分域图
记录修正前的巡天归属口径。归档的两次观测、预测状态和 PDF 均可
通过原生 `ls_id` 查询。

显著消光试验统一采用 DECam 原生 `grz` 和冻结 SED，逐条视线在
可靠距离范围内边缘化 Bayestar19 的五条消光曲线。Bayestar19 数值
乘 0.883 转为物理 E(B−V)，DECaPS 输入已是该单位。保留 Legacy
DECam 三带消光系数的形状，按 `E(B−V) = 0.98 E(g−r)` 归一化。
输出 SFD 全距离、SFD 在同一可靠距离区间、以及 3D 尘埃同区间
三份 PDF，并记录 SFD 落入可靠区间的质量。来源包括
[Bayestar19](https://arxiv.org/abs/1905.02734)、[DECaPS](https://arxiv.org/abs/2503.02657)、
[dustmaps 单位说明](https://dustmaps.readthedocs.io/en/latest/examples.html)和
[Legacy DR11 的消光定义](https://www.legacysurvey.org/dr11/catalogs/)。

本轮 3D 尘埃试验准备的 Bayestar19 文件为
`/pscratch/sd/y/yunao/dustmaps/bayestar2019_v3_9.h5`，文件哈希
`f68f14dfba45556b4a073d5ca20b99a100e7ac91c97312f11c24e27b1e32ef99`。
`run_v3_9_dust.sh` 使用独立的 `.deps/v39_dust`，其中包含
`dustmaps==1.0.14`、`progressbar2==4.5.0` 和 `python-utils==3.9.1`。


验收逐行核对原始 FITS 行号、发布键、PSF 与预测状态，并从完整
PDF 重新积分核对五个距离分位数及各距离层的概率。合并前另检
查南、北全量原生键是否重复，以及新增南天区域是否与冻结 v3.5
母样本重复。尘埃试验的 `matched_gaia.csv` 在同星、同一可靠
距离范围内比较 SFD 与 3D 尘埃，用 Gaia 距离误差计算预测评分，
并按拥挤程度给出共同支持样本数。
