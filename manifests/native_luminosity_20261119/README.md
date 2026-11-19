# 原生序列两参数光度校准：本轮已完成

**结论：M5 内部的距离尺度明显改善，但这条两参数修正还不能作为统一的老星族校准。** M30、M13 的低主序与转折区没有同时达到事先约定的尺度目标，因此本轮结束于固定迁移和两个候选盘点，没有打开新系统的距离结果。

本轮完整执行 v8 工作包，起点 `cb2ffcd4`。保留全部933颗原生PSF成员、原E及QRF完整分布、旧五组对照和负结果。程序、数据表和五张图已经交付；论文文件没有改动，生产目录仍固定在 `87069408`。既有天空RGB、physical_selected和各距离图保持原文件字节。

- 主要科学结果与解释：[science_decisions.md](science_decisions.md)。
- 三组新图及图注定义：[figures.md](figures.md)。
- 统计方法、共享参考和参数不确定度：[methods.md](methods.md)。
- 固定源身份／两折设计：`prepare/`；参数、协方差和256组共同参数draw：`fit/`。
- 完整分层结果、同星比较、尺度差和bootstrap数组：`evaluate/`、`decision/`。
- 全933源的支持／外推状态及3147行方法分位数：`predict/`。
- 三颗旧QRF尾部源核查、候选历史和参考组成：`audit/`、`audit_detail/`。
- 候选年龄／丰度原文来源：`candidate_population_metadata.json`。
- `data/`为完整的公开重绘数据；大型逐源PDF、混合成分和评分缓存留在NERSC。精确分布的分解与私有文件哈希见`forecast_factorization.json`。
- 数值、样式、逐像素重画及PDF检查：`validation/`、`delivery/`。

在完整仓库及记录的原生Python环境中独立重画，不读取私有成员数据：

```bash
PYTHONPATH=scripts:src python scripts/nat8_figures.py \
  --source manifests/native_luminosity_20261119/data --out /tmp/photdis_nat8_redraw
```

原生数值环境为NumPy2.5.3、Matplotlib3.11.1、Pillow12.3、FreeType2.14.3。五张图在此环境中逐像素重现；本地PDF另用Poppler与pypdf检查。科学运行使用本次自有NERSC分配61574693，计算和同步记录见`delivery/`。没有新模型搜索、第三个参数、逐系统调零点或全目录重跑属于本轮任务。
