# v2.0 审阅落实清单

来源为 docs/pro_review_8b6f744.md。基准模型 expanded_A，应用目录 baseline_v1。
以下交付逐项通过实际数据核对后填写最终路径与数量。

- [x] 冻结模型、配置、训练身份、评价表、refined Gaia 路径与哈希，版本独立。
- [x] 处理规模、14场边界、精确路由和状态；DR11数量尺度比。
- [x] 同口径全DR11分母、掩膜有效面积未统计时明确 null。
- [x] 方法表核对实际代码，含应用PSF与扩展参考无PSF入口的区别。
- [x] 恢复全组装参考TYPE，训练/cal/test形态构成，PSF子集性能。
- [x] 模型独立训练身份检查，chosen和分来源统一逐星视图及PDF定位说明。
- [x] 流量表区分独立失败与顺序损失，范围表区分阈值/实际范围/显示范围。
- [x] 每星重算百分比残差、NMAD、绝对误差、16–84散布、20/50%尾率、1mag尾率及区间覆盖/全宽。
- [x] 表A流量、表B样本范围、表C应用规模、表D分层性能。
- [x] 主图1应用天空/参考足迹/规模，主图2加权母样本与独立证据覆盖。
- [x] 主图3三来源距离+百分比残差（6面板）和四星族附版。
- [x] 主图4四面板完整CMD密度/偏差/绝对误差（N>=20）。
- [x] 主图5四颜色平面及星族附版，明确坐标/模型/参考。
- [x] 主图6Kiel、Teff/logg/FeH和原始/同域原始/校正丰度。
- [x] 主图7亮度偏差/散布/尾率/覆盖/宽度/N，附距离/误差/SNR/EBV。
- [x] 主图8原始参考点区间及refined带符号Gaia PIT，总体与星族。
- [x] 主图9i/WISE可用率、五测量状态、联合状态、精确计数/抽样估计分开。
- [x] 主图10数据/i/r0/弱WISE同星收益和损伤，含既有高SNR匹配波段对照。
- [x] 附图固定PDF（参考误差）、状态风险、固定质量/保留、天空、既有通量降深。
- [x] 外部星团/星流仅整理已存在且合格的证据，并标明开发用途。
- [x] 每图PNG300dpi、PDF、CSV分箱、JSON图注/选样/计数/缺失/图外/稀疏信息。
- [x] scope/methods/status/atlas与可复现统一入口；大型逐星数据留NERSC。


## 最终对应产物

- 范围与分母定义见 [scope](v2_0_scope.md)，方法与实际范围见 [methods](v2_0_methods.md)。
- 表A为 A_selection_flow.csv 和 A_existing_source_flow.csv；表B为 B_sample_ranges.csv；表C为 C_application_fields.csv 和完整 field_configurations.json。
- 表D为 D_performance_key.csv、D_performance_index.csv 及19个分模型性能CSV，raw与cal扩展区间分别命名。
- 10组主图共19张，附图36张，总计55套PNG/PDF/CSV/JSON。目录见 [atlas](v2_0_figure_atlas.md)。
- 19个模型/路由/实验视图的逐星chosen和分来源关系表、训练身份和缓存定位保存在NERSC data/v2_0。
- 391项实际身份、数值、源关联和图像哈希检查通过。完整PNG检查与PDF抽查完成。
- 同口径DR11选样分母和掩膜有效面积为null，外部星团/星流证据只索引既有开发资料。
