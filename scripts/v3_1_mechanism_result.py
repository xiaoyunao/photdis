#!/usr/bin/env python3
"""Write a source-backed interpretation of completed mechanism and correction work."""
import csv
import json
import re
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    root = ROOT / 'manifests/v3_1'
    inputs, checks = {}, []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)

    def read(folder, name):
        marker = root / folder / 'complete.json'
        done = json.loads(marker.read_text())
        path = marker.parent / name
        expected = done['summary_sha256'] if name == 'distance.csv' else done['outputs'][name]
        check('source_hash:' + folder + '/' + name, sha(path) == expected)
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            return list(csv.DictReader(stream))

    def one(rows, **where):
        chosen = [r for r in rows if all(r[k] == str(v) for k, v in where.items())]
        check('unique_row:' + repr(where), len(chosen) == 1)
        return chosen[0]

    def interval(row, scale=1, digits=3):
        return (f"{float(row['mean']) * scale:+.{digits}f} "
                f"[{float(row['lo95']) * scale:+.{digits}f}, {float(row['hi95']) * scale:+.{digits}f}]")

    paired = read('diagnostics/diagnostic/paired_report', 'paired.csv')
    diagnostic = json.loads((root / 'diagnostics/diagnostic/complete.json').read_text())
    for relative in ['scripts/v3_1_diagnostics.py', 'src/photdis/v3_1/diagnostics.py']:
        digest = sha(ROOT / relative)
        check('frozen_control_definition:' + relative, digest == diagnostic['inputs'][relative])
        inputs[relative] = digest
    distance = read('diagnostics/diagnostic', 'distance.csv')
    modes = read('diagnostics/diagnostic/mode_support_summary', 'summary.csv')
    latent = read('diagnostics/diagnostic/latent_full', 'summary.csv')
    failure = read('diagnostics/diagnostic/latent_failure_report', 'summary.csv')
    target = {axis: read('target_prior/diagnostic/paired_report', axis + '.csv')
              for axis in ['all', 'feh', 'intrinsic_gr', 'apparent_r', 'optional_bands', 'sky', 'target_field_scope']}
    parameters = read('luminosity_fit_refined', 'parameters.csv')
    guard = read('luminosity_guard', 'comparisons.csv')
    for folder in ['luminosity_calibration', 'luminosity_guard', 'target_ensemble/candidate',
                   'final_validation/report', 'diagnostics/diagnostic/latent_full']:
        p = root / folder / 'complete.json'
        inputs[str(p.relative_to(ROOT))] = sha(p)
    check('11882_complete_stratified_contrasts', sum(map(len, target.values())) == 11882)
    check('1089_original_contrasts', len(paired) == 1089)
    check('six_parameters', len(parameters) == 6)
    bootstrap = [r for r in guard if r['variant'] == 'bootstrap']
    check('15_original_mean_guards', len(bootstrap) == 15 and all(r['passed'] == 'True' for r in bootstrap))
    check('main_numerical_failure_retained', not json.loads((root / 'target_ensemble/candidate/complete.json').read_text())['main_candidate_accepted'])
    check('external_scientific_failure_retained', not json.loads((root / 'final_validation/report/complete.json').read_text())['scientific_acceptance'])
    controls = [('class', '给定可靠类别'), ('feh', '给定 FeH'), ('class_feh', '给定类别及 FeH'),
                ('grz', '仅 grz'), ('griz', '仅 griz'), ('error_i2', 'i 通量误差×2'),
                ('error_optical2', 'grz 通量误差×2'), ('error_w12', 'W1 通量误差×2'),
                ('error_w22', 'W2 通量误差×2'), ('single', '关闭双星代理分支'), ('nominal', '名义响应/校准')]
    table = ['| 对照（相对 response12/参考先验） | 参考距离 n | Δ参考模数绝对偏差 / mag，95%区间 | signed Gaia n | Δ观测空间 log score / nat，95%区间 |',
             '|---|---:|---|---:|---|']
    for control, label in controls:
        p = one(paired, control=control, reference_class=1, reference='primary', metric='absolute_mu50_error_change')
        g = one(paired, control=control, reference_class=1, metric='signed_gaia_logscore_gain')
        table.append(f"| {label} | {p['n']} | {interval(p)} | {g['n']} | {interval(g)} |")
    target_error = one(target['all'], control='response12_target', baseline='response12', reference_class=1,
                       source='primary', metric='absolute_mu_error_change')
    target_coverage = one(target['all'], control='response12_target', baseline='response12', reference_class=1,
                          source='primary', metric='reference_point_agreement68_change')
    class_added = one(target['all'], control='class_feh', baseline='feh', reference_class=1,
                      source='primary', metric='absolute_mu_error_change')
    feh_added = one(target['all'], control='class_feh', baseline='class', reference_class=1,
                    source='primary', metric='absolute_mu_error_change')
    evolved = one(distance, control='class', reference_class=1, reference='primary')
    dwarf = one(distance, control='class', reference_class=0, reference='primary')
    m = one(modes, control='response12', reference_class='all', metric='multimodal_fraction')
    s = one(modes, control='single', reference_class='all', metric='multimodal_fraction')
    different = one(modes, control='response12', reference_class='all', metric='multiple_dominant_classes_fraction')
    rare = one(modes, control='response12', reference_class='all', metric='mass_in_modes_dominated_by_lt10_effective_sources')
    unknown = one(modes, control='response12', reference_class='all', metric='mass_in_modes_dominated_by_majority_missing_class_kernel')
    neg = one(latent, control='response12', reference_class='all', metric='any_negative_upper')
    bad = one(failure, control='response12', reference_class=1, negative_group='below0.001', source='primary', metric='severe_abs_mu_gt1')
    high_negative = one(failure, control='response12', reference_class=-1,
                        negative_group='atleast0.1', source='none', metric='negative_upper')
    high_reference = one(failure, control='response12', reference_class=-1,
                         negative_group='atleast0.1', source='primary', metric='absolute_mu_error')
    check('four_high_negative_unknowns_without_primary',
          int(high_negative['n']) == 4 and int(high_reference['n']) == 0)
    strata = ['| 分层轴 | 原分层标签 | n / 天空块 | Δ参考模数绝对偏差 / mag，95%区间 |', '|---|---|---:|---|']
    for axis in ['feh', 'intrinsic_gr', 'apparent_r']:
        selected = [r for r in target[axis] if r['control'] == 'response12_target' and r['baseline'] == 'response12'
                    and r['reference_class'] == '1' and r['source'] == 'primary' and r['metric'] == 'absolute_mu_error_change']
        check('304_partition:' + axis, sum(int(r['n']) for r in selected) == 304)
        for r in selected:
            strata.append(f"| {axis} | {r['group']} | {r['n']} / {r['cells']} | {interval(r)} |")
    parameter_table = ['| 参考类 | 参数 | 点估计 / mag | 共享天空 bootstrap 95%范围 / mag |', '|---|---|---:|---|']
    labels = {'0': 'dwarf', '1': 'evolved', '2': 'other'}
    for r in parameters:
        parameter_table.append(f"| {labels[r['reference_class']]} | {r['parameter']} | {float(r['point']):.4f} | [{float(r['lower95']):.4f}, {float(r['upper95']):.4f}] |")
    text = f'''# v3.1 巨星失效机制与受约束亮度校正结果

2026-11-14。工作包一的受控诊断、单一预注册校正、校准 guard 与最终验证解释已完成。
结果支持类别分支错误和类内尺度/区间问题同时存在；六参数校正在校准 guard 上通过，
但最终候选数值证据门槛及新验证失败，**不采用为已校准产品**。这不是整个 v3.1 的最终交付。

## 对象与统计范围

原生 TYPE=PSF 的1542个已见开发对象、17386个视图，保留原入选概率、设计权重及天空块。
原始1089个同身份对照，以及包含目标先验、类别与FeH增量对照的11882个分层结果均保留。
“误差”在下文指对合格参考中心的差异；参考点区间一致率不是潜在真实距离覆盖率。
参考源保留各自资格和相关性，不能合并为独立真值。signed Gaia采用观测空间评分，负视差保留。
qnear始终为模型P(d<10 kpc)，高q阈值为0.95；不是P(d<2 kpc)。

## 同身份对照：类别效应大，波段/误差改变不能代替分支修复

以下均为演化参考类，区间来自原设计权重下的天空块bootstrap。
模数绝对偏差变化越负越好，log score变化越正越好。通量误差×2指测量标准差×2；
`error_optical2`只改变grz，i另列。光学子集对照保留同一批对象，原本缺测的可选波段仍缺测。

{chr(10).join(table)}

类别与FeH的合格人数不同，不能直接相减解释增量效应。在完全相同的275个演化星上，
给定FeH后再加入类别使绝对偏差变化为{interval(class_added)}mag；给定类别后再加入FeH为
{interval(feh_added)}mag。真实标签只能做诊断，不能把这些幅度视为光度生产推断已经实现的提升。

给定类别仍未解决区间问题。未加权的同参考子集：演化星{evolved['n']}个的68%/95%参考点
一致率为{100*float(evolved['coverage68']):.2f}%/{100*float(evolved['coverage95']):.2f}%，
严重|Δmu|>1mag比例{100*float(evolved['severe_fraction']):.2f}%；矮星{dwarf['n']}个的对应一致率
为{100*float(dwarf['coverage68']):.2f}%/{100*float(dwarf['coverage95']):.2f}%。这些未加权比例
与上表加权变化的统计对象不同。去掉WISE明显恶化演化星参考表现；简单放大误差的改变小且混合。

## 固定响应换目标先验：迁移退化及分层限制

同304个演化星，response12目标先验相对同响应参考先验的绝对偏差增加
{interval(target_error)}mag，68%参考点一致率变化{interval(target_coverage,100,2)}个百分点。
全部1542个对象都是新格子的超均值迁移，不是已拟合场的原位检验。原1个数值标记及通过的精化侧表保留。

{chr(10).join(strata)}

这里FeH poor/middle/rich的边界是−1.5/−0.5，missing独立保留；不是最终外部库存的另一套贫金属阈值。
颜色为消光修正g−r，分界0.3/0.8/1.3；亮度为观测r。该304对象参考子集在r>=20没有对应证据。
颜色>=1.3仅3对象，不构成该域校准。上述分层为条件诊断，不能把重叠分层当作独立机制或新增显著性检验。
全部来源、可选i/W1/W2掩码和逐天空表见原分层文件；单一天空块不生成伪造的天空bootstrap区间。

同一演化参考类的目标先验log评分变化在primary、SpecDis、SPDist和signed Gaia均为负；
精确Gaia距离参考只有1个，不能据此作该类绝对标定。来源人数不同且教师相关，保留逐来源分母。

## 潜在负通量、稀有支持与模式归属

完整2589视图潜在通量审计无不支持视图，最大任意负通量概率上界0.415346；
response12全1542对象加权平均上界为{float(neg['mean']):.6f}，主要涉及W2。
这是冻结均值线性化尘埃近似下的条件高斯联合事件界，不是独立波段概率乘积，
更不是合法观测负通量的剔除依据。50视图pilot的较小最大值不能替代完整样本结果。

在上界<0.001的{bad['n']}个演化primary参考对象中，严重|Δmu|>1mag比例仍为
{100*float(bad['mean']):.2f}%（天空95%区间[{100*float(bad['lo95']):.2f}, {100*float(bad['hi95']):.2f}]%）。
高负支持的4个unknown对象没有primary距离参考。故负潜在通量不能单独解释已见巨星失效，
也不能据低平均值排除稀有或未验证对象的局部风险。

response12的加权多峰比例为{100*float(m['mean']):.2f}%，关闭双星代理后为{100*float(s['mean']):.2f}%；
{100*float(different['mean']):.2f}%对象的显著模式间主导类别不同。但关闭双星后巨星偏差及Gaia评分
反而略差，不能用模式数减少作为质量改善。弱模式全部保留；显著模式仅指后验质量>0.05。
主导母核有效来源数<10的模式质量均值{100*float(rare['mean']):.3f}%，
主导母核多数训练来源缺类别标签的模式质量均值{100*float(unknown['mean']):.2f}%。
这是母核支持和模型责任分解，不是后验来源ESS、物理星族数或实测双星率；全局均值不能证明稀有尾部已覆盖。
最终3540候选的不同产品身份另见[完整模式结果](v3_1_candidate_modes_result.md)，不可跨样本直接比较多峰比例。

## 单一受约束校正：guard通过，最终不采用

诊断不支持把统一误差膨胀、删除WISE或关闭双星当作主要修复。因此只注册一次类内亮度平移delta和
额外亮度散度sigma：三个参考类共六参数，delta范围[-1,1]、sigma范围[0,1]mag，保留零校正边界。
变换在乘距离先验之前作用于似然，相同变换进入选择归一化；不是对已经归一化的后验再展宽。
不改变response12、类内核比例、双星代理或模型差异通道，不增加颜色/FeH项或模型搜索。

只用原role3，拟合/guard按冻结天空哈希分开；总2044个PSF对象，拟合1113、guard931。
演化星只有89个拟合对象/3天空块、39个guard对象/2块。入选概率及非PSF/缺TYPE排除计数保留，
没有将最终新验证标签用于修正；32次共同天空bootstrap是敏感性抽样，不是精确贝叶斯参数后验。

{chr(10).join(parameter_table)}

演化星sigma的范围包含零并非常宽，不能把点估计的大散度当作已测准的物理散度。
原网格拐点拟合失败保留，数值侧表精化后96项数值检查、15项预注册bootstrap均值guard通过；
这只允许完成冻结候选评价。标签在guard中只分组，不进入预测。

最终主候选因绝对证据种子变化0.248538超过0.05而未数值接受；新天空signed Gaia和新共同尺度
评价也退化，见[冻结最终验证](v3_1_final_validation_result.md)。故校准guard通过不等于外部校准成功，
不能换成通过的等权cut对照来宣布主结果通过。全部旧产品、失败、先验对照与部署v1保留。

## 可复现来源与完成范围

- [原同身份对照](../manifests/v3_1/diagnostics/diagnostic/paired_report/paired.csv)及未加权参考距离表均通过其计算标记哈希认证。
- 目标先验[总体](../manifests/v3_1/target_prior/diagnostic/paired_report/all.csv)、[FeH](../manifests/v3_1/target_prior/diagnostic/paired_report/feh.csv)、[颜色](../manifests/v3_1/target_prior/diagnostic/paired_report/intrinsic_gr.csv)、[亮度](../manifests/v3_1/target_prior/diagnostic/paired_report/apparent_r.csv)、[波段](../manifests/v3_1/target_prior/diagnostic/paired_report/optional_bands.csv)、[天空](../manifests/v3_1/target_prior/diagnostic/paired_report/sky.csv)和[迁移范围](../manifests/v3_1/target_prior/diagnostic/paired_report/target_field_scope.csv)共11882行。
- [校正参数](../manifests/v3_1/luminosity_fit_refined/parameters.csv)、[guard](../manifests/v3_1/luminosity_guard/comparisons.csv)、[完整图集](v3_1_atlas.md)保留已实际目检的配对、校准和潜在通量图。
- [本报告来源与实际检查](../manifests/v3_1/mechanism_result_sources.json)；生成脚本`v3_1_mechanism_result.py`，原计算及数学验证记录见需求台账。

工作包一实施和解释完成，科学接受结论为失败/不采用；总体天空迁移、结构应用和最终整合仍须完成。
'''
    output = ROOT / 'docs/v3_1_mechanism_result.md'
    output.write_text(text)
    for target_path in re.findall(r'\]\(([^)]+)\)', text):
        if target_path.endswith('mechanism_result_sources.json'):
            continue
        check('document_link:' + target_path, (output.parent / target_path).resolve().is_file())
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(root / 'mechanism_result_sources.json', dict(inputs=inputs, report_sha256=sha(output),
        checks=checks, actual_checks=len(checks), passed=True,
        scope='Authenticated completed source tables and unique reported contrasts; exact counts, labels and intervals. Written interpretation is not independent calibration acceptance.'))
    print(json.dumps(dict(actual_checks=len(checks), sources=len(inputs), report=str(output))))


if __name__ == '__main__':
    main()
