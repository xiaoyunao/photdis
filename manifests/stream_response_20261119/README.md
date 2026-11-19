# 固定星流 CMD 与老星族尺度：2026-11-19

这一轮把覆盖问题与光度尺度问题分开了。放宽消光入口、使用 g/r 并统一形式深度，能够恢复此前缺失的 Pal 5 成像覆盖；给老星族加一个光度零点，则只在低主序上得到迁移支持，转折星仍明显偏近。

详细结论见 [science_decisions.md](science_decisions.md)，主图与配套图见 [figures.md](figures.md)。本目录对应完整 v5 工作包，输入审阅固定在 `884ea87b`，生产模型和距离预测保留 `87069408`。这里新增的全域图是 CMD 筛选响应图，不是新的单星距离目录。论文文件由论文任务维护。

## 结果入口

| 目录 | 内容 |
|---|---|
| `rgb` | 原61节点 cube 的总览 RGB、五个宽距离层内部的三色图；通道、节点权重、stretch 和科学显示网格 |
| `local_source` / `local_area` | 原始成像逐步流失、独立 random 面积；源质量条件与几何面积分别记录 |
| `local` / `core` | Pal 5 与 Sgr 的目标/背景 Hess、完整误差加宽边界、距离响应、角段超额、相位分解；实际 Pal 5 特殊检测圆区单独审计 |
| `sequence` | 原始与当前同星比较、全 QRF-only 成员身份/颜色/参考相位、折返噪声频率、零点及共享 bootstrap |
| `conditioning` | 固定支持集内有/无参考距离的权重、mean/median、balance/ESS、源身份和 signed-Gaia 标尺 |
| `full_replay` | 一条冻结误差宽度曲线在两个系统上的重放及全域累计决定 |
| `full_inventory` / `full_identities` / `full_count` | 全域 raw→gr 筛选、去重和节点计数 |
| `full_area` / `full_atlas` / `full_rgb` / `full_validation` | 同一定义的全域成像面积、cube 来源、RGB 与守恒汇总 |
| `mask_revision` | 逐ROI隔离 z 专属像素限制对 g/r 源数和面积的影响；修正前选择与全域图保留在私有执行历史 |
| `validation` | 样式锁、原 v3.8 F02/F03 像素回归和科学量核对记录 |

`complete.json` 是实际执行的阶段收据，包含输入/输出 SHA、代码版本和作业号。大源表、randoms、完整 cube 和历史修正前产物留在 NERSC：

`/pscratch/sd/y/yunao/photdis_pipeline/data/stream_response_20261119`

g/r入口仅剔除像素位2/3/5/6/10；原strict与z-addback另外保留z专属位4/7。修正前局部图、完整全域累计及选择数组在`execution/before_pure_gr_mask/`，原始测光、生产距离和本轮尺度实验未变。

原61节点 cube仍位于 `data/halo_response_20261118/atlas/cmd_filter_cube.fits`。新增 cube 位于 `data/stream_response_20261119/full_atlas/cmd_filter_cube.fits`，大小3,224,399,040bytes，SHA256为 `c5237cf0023836348478fbf7aca67167cd8c0c064d305d44a7ed22913c3443f6`。

## 不需要大目录的独立重画

在本仓库根目录、具备原有 Python 科学依赖的环境运行。以下命令只读公开的紧凑表和网格，不读取原始成像或大 cube：

```bash
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py rgb --output /tmp/photdis-v5-original-rgb
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py rgb --rgb-stage full_rgb --output /tmp/photdis-v5-full-rgb
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py local --output /tmp/photdis-v5-local
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py phase --output /tmp/photdis-v5-sgr-phase
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py sequence --output /tmp/photdis-v5-sequence
PYTHONPATH=scripts:src:.deps:.deps/v39_dust python scripts/stream_redraw.py conditioning --output /tmp/photdis-v5-conditioning
```

`stream_public_check.py --redraw /tmp/photdis-v5-check` 可核对交付哈希、36组PDF/PNG、文档链接，并从紧凑公开证据独立重画全部图；PDF/PNG检查额外使用已有的Pillow与pypdf。它不需要NERSC原始源表。

重新拟合权重或重扫成像需要收据指定的 NERSC 输入，并使用已分配计算节点。执行顺序为 `stream_full.py replay → scan → identities → count`，独立 randoms 用 `south/north`，随后 `assemble`；`stream_rgb.py --cube-stage .../full_atlas --output-stage full_rgb` 使用相同渲染函数。`configs/stream_full_20261119.json` 固定统一入口，`configs/stream_response_20261119.json` 固定小域与尺度实验。

## 图与计数的含义

- RGB的B/G/R是不同距离模板响应，按真实模数单元的重叠宽度加权，通道内权重之和为1。每个宽层内部采用等模数三分。RGB图例给出实际距离范围，物理密度和 signed residual 的标量图另有色条。
- 密度为 `smooth(weighted counts)/smooth(independent area)`，单位是 `deg^-2`，不是体密度或单星距离概率。全域核为0.3°，背景核为5°。正残差RGB才使用 `max(residual,0)`；科学数组保留负值和实际掩膜。
- RGB逐通道用预先写明的单调asinh拉伸；阈值、饱和比例和节点权重在CSV中。不同图的同一种颜色不代表相同距离或绝对强度。经纬度是Galactic，l=0居中，经度向左增加。
- 局部Hess用独立面积归一化的目标减背景；角向图分别平滑counts与area，再减去同一沿迹位置的两侧背景中位密度。原始段计数和超额区间使用未平滑counts/area。`upper95`是signed超额的单侧95%分位，条件于共同背景模型；它不是独立测得的物理成员数上限，零面积时没有有限约束。沿迹核用实际参考轨迹弧长的局部平均度量，两个注册核宽0.1°/0.3°保持不变。
- 橙线来自计数函数的完整模板边界，包含模数宽度、0.05/0.10颜色宽度和2sigma_g；各列的入口亮限、质量、mask条件另见cutflow。模板曲线延伸到样本亮限之外时不表示这些源可入选。
- source质量、TYPE形态和检测完整率不等价。实际Pal 5检测圆区的Gaia强制测光限制单列；历史键名 `Pal5_core` 是相邻参考轨迹条带，不能当作实际核心。

所有图内文字为英文，沿用固定Pal5/v3.8基础样式；只有本轮授权的RGB表示增加了三通道配色。原生产模型、目录、旧 cube 和负结果均保留。没有自动开启下一轮模型或论文工作。
