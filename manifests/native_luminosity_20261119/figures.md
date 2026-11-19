# 三组科学图（五张，固定Pal5/v3.8样式）

图面全部英文。PNG300dpi、单页PDF及CSV说明均位于`figures/native_luminosity_20261119/`。颜色、字体、线宽和导出采用固定PAPER函数；原v3.8 F02/F03像素基线通过。全部源／模式保留，图注中的“完整CDF”指覆盖数值上至少1−2e−9的预测概率。

## 1. 距离独立的两色几何

[`two_color_geometry`](../../figures/native_luminosity_20261119/two_color_geometry.png)：上排为完整母样本与原始物理线段；空心为原E支持以外成员。下排为共享r协方差下的正交残差／报告误差，使用symlog保留异常尾部，灰带±3。原始支持保持不变。切向残差在内部投影处按定义为零，不是额外拟合成功证据。

## 2. 留出预测、固定迁移及宽度代价

[`cross_prediction_transfer_scores`](../../figures/native_luminosity_20261119/cross_prediction_transfer_scores.png)：共同E/QRF支持对象。第一排分别显示低主序／转折的尺度及95%区间，灰带±0.10mag。后两排显示同星配对logscore及CRPS变化，方法箭头对应后者减前者；正logscore和负CRPS为改善。M5使用相应训练折参数，M30/M13使用全M5冻结参数。

[`luminosity_mapping_widths`](../../figures/native_luminosity_20261119/luminosity_mapping_widths.png)：上排显示同一组原节点在灰修正前后的原生Mr—颜色关系。M5上排只是全样本拟合的映射示意，下面的M5预测宽度仍来自两折留出。下两排为条件预测W68/W95及共同重采样区间；外部Mix为参数混合PDF，不含目标参考误差。颜色／符号与前图的两序列对应。

## 3. 完整预测分布与固定三颗尾部

[`calibration_complete_PDF`](../../figures/native_luminosity_20261119/calibration_complete_PDF.png)：完全复用旧v7的三个转折个例及M5低主序个例，不按新结果挑选。比较原E、校准E、QRF，以及外部参数混合。上面为模式／主体范围，下面独立小行显示完整CDF；不以高斯替代多峰。灰色参考只调整显示高度，不改变评分核。

[`fixed_three_QRF_tails`](../../figures/native_luminosity_20261119/fixed_three_QRF_tails.png)：旧selected.csv固定的三颗M13尾部身份。黑线原QRF、橙虚线诊断中心化，单独CDF行保留尾部。原统计不删源；身份、测光／自行和评分贡献见`audit_detail/three_source_evidence.csv`。

五张图及五个PDF页面已经放大、联系表与Poppler渲染目视检查；独立公开数组重画在记录的NERSC原生环境中逐像素一致。PDF不属于论文编辑文件。
