# v2.2：Pro审查逐项实现与证据边界

主源：[完整审查](pro_review_fbb8ea0.md)。冻结口径：[协议](v2_2_protocol.md)。

|要求|实现/输出|科学边界|
|---|---|---|
|先修可靠性漏计|selection_v2_2.py；v2_2_audit.py；reliability+denominators；端点测试|只规范1e-7容差内的误差；禁止静默裁掉实质非法概率|
|逐层分箱计数守恒|1280组分母与10bins总和检查|保留旧v2.1；不重训|
|同模型开发/应用完整PDF|v2_2_application.py；26000PSF；expanded_A全PDF；baseline_v1实际ABC路由|冻结catalog测光和路由；不重跑890万对象；不混用模型成绩|
|应用人数与可观测条件|application_summary/distribution/evidence_cells|加权预测人数不等于真近端人数；颜色、亮度、误差、WISE、场分别列出|
|参考支持与PDF结构|固定observable cells；0/1–19/20+reference；宽度/多峰/未知类贡献|占格不保证校准/精度，也不是独立天空证据|
|选择后的完整距离指标|selected_distance.csv，3threshold×4reference×人群/条件/距离分层|标签和校正丰度只作诊断，不用于应用过滤|
|选中后完整Gaia检查|selected_gaia_observation与PIT分箱；原观测空间likelihood|弱/负视差保留；不通过正视差距离反演筛掉失败|
|不确定度与人口迁移|Wilson与2000天区bootstrap；固定cal条件接收率替换远端占比|有限9个dev天区；不能宣称独立最终验证|
|新验证集清点|身份未用42076与缓冲天空5分别保留；逐星身份导出PSF|42076暂无合格类别/距离标签；3596弱视差资格；待成员/外部证据|
|目标优先层|18–22无W1库存；近端贫金属、远巨星、真实PSF非恒星缺口明确|旧光谱控制不是目标目录污染率，未知类别保留|
|可信噪声迁移|v2_2_noise.py；256未拟合dev模板、759颜色/WISE/误差匹配对；4共同波段扰动|同源副本同角色；平衡模板不是目标人口；变暗距离只是条件实验|
|实际完整误差流程|Sobol128全PDF；固定分类器单点与128draw边缘化；原cal参数固定|不是新模型或新校准；不把单次扰动失败等同全PDF性能|
|重复与来源相关性|noise_repeat_stability；source-cluster paired bootstrap|不把4副本当4独立恒星；同时保留所有负结果|
|10–20/20–50远端分支|v2_2_far.py；按Gaia/SpecDis/SPDist分别的窗口质量、条件形状、oracle位置/宽度/覆盖|共同teacher不构成独立距离尺度；真实类条件仅oracle诊断|
|可观测格内分支频率|observable_branch_frequency；color_selection训练/cal/dev/app|已知类条件化显式；应用未知真实频率，不直接估计Galactic prior|
|无标签训练信息|training_luminosity_support，保留43914PSF无类别距离标签|原冻结训练的非PSF/未验证例外不进入新样本|
|极端Gaia失配不删尾|v2_2_extremes.py；保留旧32vs64积分检查；local光度支持对照|不把局部支持差异当因果修复；匹配/参考异常待外部证据|
|只有新证据支持才改分支权重/模型|本轮没有新候选，没有重校准或重新部署|旧Gini候选拒绝保留；×2仅压力，不是真先验|
|可复查交付|完整缓存留NERSC；小CSV/JSON、7EnglishPNG/PDF图、逐项验证|所有原v2.0/v2.1代码/协议/cache冻结；无定时/子代理|

程序与统计交付完成不代表科学缺口关闭：当前不能证明目标暗端近端污染率，不能
确认独立远端分支尺度，也不能完成新身份的系统成员认证。证据没有达到新的权重
校准/结构改动所需条件，本轮以这些事实结束，不通过额外模型搜索填补它们。
