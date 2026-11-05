#!/usr/bin/env python3
"""Publish compact, inspectable statistics and the science figure index locally.

Inputs are the small statistics/plot_data JSON exports, never remote catalogs.
Scientific interpretation is maintained in the accompanying result document.
"""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    statpath=args.directory/'statistics.json';plotpath=args.directory/'plot_data.json'
    s=json.loads(statpath.read_text());p=json.loads(plotpath.read_text())
    assert hashlib.sha256(statpath.read_bytes()).hexdigest()==p['statistics_sha256']
    assert all(s['checks'].values())
    for record in p['figures'].values():assert hashlib.sha256((ROOT/record['path']).read_bytes()).hexdigest()==record['sha256']
    models=s.pop('models');s['models']={}
    for name,value in models.items():
        sky=dict(original_sky=value.pop('original_sky'),sky=value.pop('sky'))
        skyfile='manifests/baseline_v1_4_science_sky_'+name+'.json';dump(ROOT/skyfile,sky)
        s['models'][name]=dict(value,sky_file=skyfile)
    checks=s.pop('checks');dump(ROOT/'manifests/baseline_v1_4_science_validation.json',dict(checks=checks,n=len(checks)))
    s.update(statistics_sha256=hashlib.sha256(statpath.read_bytes()).hexdigest(),plot_data_sha256=hashlib.sha256(plotpath.read_bytes()).hexdigest(),actual_checks=len(checks))
    dump(ROOT/'manifests/baseline_v1_4_science.json',s)
    dump(ROOT/'manifests/baseline_v1_4_science_plot_data.json',p)
    lines=['# v1.4 科学诊断：图集与可复查统计','',
           '本图集读取冻结 v1.4 的逐星结果。分类器与 wide QRF 距离模型独立评分；分类分数没有传入这些距离预测。没有重新训练，也没有更新部署或重跑母星表。','',
           '## 关键逐星结果','',
           '距离残差定义为预测中位距离模数减参考距离模数；严重错误为绝对残差 >1 mag。Gaia、SpecDis 和 SPDist 分列；后两者的覆盖率是与发布点估计的一致性，未卷积参考误差。','',
           '| 配置 | 类别 | N | 中位残差 / mag | NMAD / mag | 严重错误率 | 68% / 95% 覆盖 | 中位距离比 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in ['expanded_A','matched_grz','matched_griz']:
        for label in ['dwarf','evolved','other_transition','unknown']:
            v=s['models'][name]['chosen'][label]
            if v['n']:lines.append(f"| {name} | {label} | {v['n']:,} | {v['bias_mag']:+.3f} | {v['nmad_mag']:.3f} | {v['tail_1']:.2%} | {v['coverage68']:.2%} / {v['coverage95']:.2%} | {v['distance_ratio_median']:.3f} |")
    lines+=['','## 改善了谁、损伤了谁','',
            '下表是 chosen 参考下的同星配对。置信区间来自 2000 次天区 cell 整块 bootstrap，重新计算星数加权比例；负差表示改善。它反映当前参考选择与天区样本的波动，不能外推为全母样本精度。','',
            '| 新配置（对原配置） | 类别 | N / cells | 错→对 | 对→错 | 新−旧尾率 / 百分点 | 95% 区间 / 百分点 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for name,group in s['pairs'].items():
        for label in ['all','dwarf','evolved','other_transition','unknown']:
            v=group['chosen'][label]
            if v['n']:
                ci=v['tail_ci95'];interval='未估计' if any(x is None for x in ci) else f'[{100*ci[0]:+.3f}, {100*ci[1]:+.3f}]'
                lines.append(f"| {name} | {label} | {v['n']:,} / {v['cells']} | {v['rescued']} | {v['harmed']} | {100*v['tail_difference']:+.3f} | {interval} |")
    lines+=['','## WISE：质量与深度损失有交集','',
            '原 26,000 颗概率样本保留原权重，只代表原南天 DECam 母样本。以下独立统计 S/N、WISEMASK 和 FRACFLUX；不是顺序互斥筛选比例。','',
            '| 波段 | 所有低 S/N 的加权比例 | 质量失败比例 | 同时低 S/N 且质量失败 | 质量失败组内低 S/N 比例 |',
            '|---|---:|---:|---:|---:|']
    for band,v in s['parent']['joint'].items():
        lines.append(f"| {band} | {v['low_snr_total_fraction']:.2%} | {v['quality_failure_fraction']:.2%} | {v['low_snr_and_quality_fraction']:.2%} | {v['low_snr_given_quality']:.2%} |")
    lines+=['','## 固定 PDF 个例','',
            '每组按预先固定的身份哈希顺序抽取两颗、不重复选星。下表概率为预测 PDF 在参考距离模数 ±1 mag 内的积分。个例仅用于解释，不能据十颗星估计总体失败机制比例。','',
            '| 组 | 身份 | 模型 | 中位残差 / mag | 68% 宽度 / mag | 参考 ±1 mag 概率 | 支持域内 | 歧义标记 |',
            '|---|---|---|---:|---:|---:|---|---|']
    for v in s['pdf_examples']:
        lines.append(f"| {v['category']} | {v['identity']} | {v['model']} | {v['median_delta']:+.2f} | {v['width68']:.2f} | {v['mass_within_1mag']:.2%} | {v['supported']} | {v['ambiguous']} |")
    lines+=['','## 图集','',
            '距离和 CMD 同时展示广覆盖 expanded_A 与含 i 的 matched_griz。其他物理细分主要使用 matched_griz；八种配置的全部分天区统计均已导出。A/B/C 使用不同 WISE 子样本，不能直接跨配置归因波段收益。','',
            '天区性能格要求 N≥20；颜色与趋势格要求 N≥10。稀疏格留白，完整计数保留。残差尾部不裁到 ±1 mag；固定颜色/参数范围的排除数写在图中。','']
    for name,r in p['figures'].items():
        lines += [f'### {name}','',f'![{name}](../{r["path"]})','',r['caption'],'']
    lines+=['## 数据来源、可复现性与边界','',
            '- [诊断协议](baseline_v1_4_diagnostics_protocol.md)；[完整 Pro 建议](pro_review_46cb68a.md)。',
            '- [紧凑统计与输入哈希](../manifests/baseline_v1_4_science.json)；[分箱数值与逐图说明](../manifests/baseline_v1_4_science_plot_data.json)；[实际检查](../manifests/baseline_v1_4_science_validation.json)。',
            f'- 本轮 {s["actual_checks"]} 项实际身份/数值/哈希检查通过。逐星表和 PDF 留在 NERSC `data/baseline_v1_4/diagnostics`；程序为 `scripts/baseline_v1_4_diagnostics.py` 和 `scripts/baseline_v1_4_science_report.py`。',
            '- 逐星关联以精确身份校验，primary 与 refined 评分分别绑定 SHA256；计算开始与结束复核读过的输入未变。',
            '- 完整 Gaia PIT/log density 使用 refined sidecars；弱与负视差从未倒数成距离散点。',
            '- 参考目录之间存在重叠、选择效应和相关信息；参考间一致不等于独立绝对真值。金属贫乏不是银河晕成员分类。',
            '- 本轮按 test 结果作物理诊断。若据此修改模型，这些天区成为开发证据；新的最终结论须另留独立检验。22–25 等仍广泛未验证。',
            '- 原 v1.4 结果、模型及已部署 v1 保持不变；没有自动开启下一轮训练或定时监控。','']
    (ROOT/'docs/baseline_v1_4_science_atlas.md').write_text('\n'.join(lines))
    print('Published compact statistics and',len(p['figures']),'figure references')

if __name__=='__main__':main()
