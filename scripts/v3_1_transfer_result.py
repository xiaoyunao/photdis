#!/usr/bin/env python3
"""Interpret complete sky-transfer products, retaining unavailable and failed views."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    root = ROOT / 'manifests/v3_1'
    folder = root / 'heldout_transfer/report'
    marker = folder / 'complete.json'
    done = json.loads(marker.read_text())
    inputs = {str(marker.relative_to(ROOT)): sha(marker)}
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def table(name):
        path = folder / name
        check('source_hash:' + name, sha(path) == done['outputs'][name])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            return list(csv.DictReader(stream))

    check('complete102PSF_without_references', done['states'] == 102 and
          done['all_PSF'] and not done['references_read'])
    tables = {name: table(name) for name in ['inventory.csv', 'predictive_summary.csv',
              'counts.csv', 'field_summary.csv', 'checks.csv', 'paired.csv']}
    inventory = tables['inventory.csv']
    check('all_registered_states', len(inventory) == 102 and
          {(r['case'], int(r['correction'])) for r in inventory} ==
          {(case, i) for case in ['broad', 'candidate', 'holdout'] for i in range(34)})
    expected_n = {'broad': 6000, 'candidate': 935}
    for i, r in enumerate(tables['predictive_summary.csv']):
        check(f'fixed_predictive_sample:{i}', int(r['n']) == expected_n[r['envelope']])
    check('frozen_sky_fields', all(int(r['field']) in [4, 6, 12] for r in tables['field_summary.csv']))
    ensembles = {}
    for case in ['broad', 'candidate', 'holdout']:
        path = root / 'target_ensemble' / case / 'complete.json'
        value = json.loads(path.read_text())
        check('ensemble_identity:' + case, done['inputs'][f'data/v3_1/target_ensemble/{case}/complete.json'] == sha(path))
        check('ensemble34:' + case, value['states'] == 34 and value['all_PSF'])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        ensembles[case] = value

    def number(value, digits=3):
        if value in ['', None]:
            return '不可用'
        return f'{float(value):.{digits}f}'

    fit_table = ['| 拟合总体 | 条件迁移状态数值通过 / 34 | 主混合可计算 | 主混合数值接受 | 校正状态 ESS |',
                 '|---|---:|---|---|---:|']
    for case, ensemble in ensembles.items():
        view = next(v for v in ensemble['views'] if v['view'] == 'bootstrap_evidence')
        n = sum(r['case'] == case and r['passed'] == 'True' for r in inventory)
        fit_table.append(f"| {case} | {n} | {view['inputs_passed']} | {view['accepted']} | {number(view.get('state_ESS'))} |")
    support = ['| 拟合总体 | 入选规则 | n | 条件数值接受 | 加权逐对象平均 log 密度 | 单位对象联合 ESS / 半样本差 / 支持通过 | 广义设计联合 ESS / 半样本差 / 支持通过 |',
               '|---|---|---:|---|---:|---|---|']
    statuses = done['views']
    for case in ensembles:
        for envelope in ['broad', 'candidate']:
            status = [v for v in statuses if v['case'] == case and v['view'] == 'bootstrap_evidence'
                      and (v.get('envelope') == envelope or not v['available'])]
            check('unique_main_status:' + case + '/' + envelope, len(status) == 1)
            status = status[0]
            rows = [r for r in tables['predictive_summary.csv'] if r['case'] == case and
                    r['view'] == 'bootstrap_evidence' and r['envelope'] == envelope and r['variant'] == 'base']
            check('main_availability:' + case + '/' + envelope, len(rows) == int(status['available']))
            if not rows:
                support.append(f'| {case} | {envelope} | — | False | 不可用 | 不可用 | 不可用 |')
                continue
            r = rows[0]
            check('heldout_role:' + case + '/' + envelope,
                  (r['population_fit_excludes_heldout'] == 'True') == (case == 'holdout'))
            values = []
            for prefix in ['joint', 'generalized_design_joint']:
                ess = float(r[prefix + '_importance_ess'])
                delta = float(r[prefix + '_half_log_density_delta'])
                supported = ess >= 100 and delta < .2
                check('support_rule:' + case + '/' + envelope + '/' + prefix,
                      supported == (r[prefix + '_supported'] == 'True'))
                values.append(f'{ess:.3f} / {delta:.4f} / {supported}')
            support.append(f"| {case} | {envelope} | {r['n']} | {status['accepted']} | {number(r['design_mean_pointwise_logscore'])} | {' | '.join(values)} |")

    groups = defaultdict(list)
    for r in tables['counts.csv']:
        groups[r['case'], r['view'], r['envelope'], r['variant']].append(r)
    for key, rows in groups.items():
        name = '/'.join(key)
        check('18_joint_bins:' + name, len(rows) == 18 and
              {(int(r['class_index']), int(r['distance_bin'])) for r in rows} ==
              {(c, d) for c in range(3) for d in range(6)})
        n = int(rows[0]['n'])
        check('fixed_count_sample:' + name, n == expected_n[key[2]])
        check('count_conservation:' + name, all(int(r['n']) == n for r in rows) and
              abs(sum(float(r['mean']) for r in rows) - n) < 1e-7)
        check('variance_decomposition:' + name, all(math.isclose(float(r['variance']),
              float(r['conditional_variance']) + float(r['shared_variance']), rel_tol=1e-9, abs_tol=1e-7) for r in rows))
        check('integer_realized_intervals:' + name, all(0 <= int(r['lower95']) <= int(r['median']) <= int(r['upper95']) <= n for r in rows))
    contrasts = ['| 拟合总体−留出拟合 | 入选规则 | n / 天空场数 | 逐对象 log score 变化，天空95%区间 | qnear 变化，天空95%区间 |',
                 '|---|---|---:|---|---|']
    for case in ['broad', 'candidate']:
        for envelope in ['broad', 'candidate']:
            selected = [r for r in tables['paired.csv'] if r['view'] == 'bootstrap_evidence' and
                        r['case'] == case and r['envelope'] == envelope and
                        r['metric'] in ['pointwise_logscore_gain', 'qnear_shift']]
            if not selected:
                contrasts.append(f'| {case} | {envelope} | — | 不可用 | 不可用 |')
                continue
            bymetric = {r['metric']: r for r in selected}
            check('paired_metrics:' + case + '/' + envelope, len(selected) == len(bymetric) == 2)
            values = []
            for metric in ['pointwise_logscore_gain', 'qnear_shift']:
                r = bymetric[metric]
                values.append(f"{number(r['mean'], 5)} [{number(r['lo95'], 5)}, {number(r['hi95'], 5)}]")
            contrasts.append(f"| {case} | {envelope} | {r['n']} / {r['cells']} | {' | '.join(values)} |")
    failures = sum(r['passed'] != 'True' for r in tables['checks.csv'])
    text = f'''# v3.1 总体入选与整块天空迁移结果

此报告由完整102状态、三个总体混合及实际共享计数结果生成。保留所有不可用视图与失败状态，
不使用存活状态重新归一化；表中的数值接受与科学校准分开。

## 冻结比较与产品身份

三组总体为广样本26000、候选3540和排除天空场12/4/6后的拟合。
只有holdout拟合未使用这三个场的光度；broad/candidate是原位对照，不是独立验证。
留出只指该总体拟合步骤，不能据此宣布SED、模板来源和六参数校准训练的全流程独立。
同一组原生TYPE=PSF身份分别按原广样本和候选入选规则评价，选择归一化分母与规则匹配。
主视图固定为32个校正自举状态的证据混合；零校正、点校正及等权cut对照全部留在原表，
不能以更容易通过的对照替换主视图。真实距离和类别参考均未在本实验读取。

{chr(10).join(fit_table)}

## 共享联合预测及支持

{chr(10).join(support)}

逐对象预测先对共享总体后验取平均，再分别评价对象；联合目录预测对所有对象的
条件密度乘积只积分一次共享状态，两者不能互换。广义设计加权联合评分不是归一化的
物理目录概率密度。支持规则固定为ESS>=100且交错半样本log密度差<0.2；
任一失败都保留，不能压平权重或把高逐对象平均分数当作联合支持充分。
条件数值接受包含原总体混合的数值门槛；逐对象概率稳定不解除原证据失败。

## 同对象天空对比与计数

{chr(10).join(contrasts)}

正log score差表示该原位拟合的光度评分较留出拟合高，并非距离更准确；
三场天空的bootstrap只描述这些块的变化，不能推广为全天覆盖保证。
qnear为P(d<10kpc)。原始类别×距离计数采用0–2、2–5、5–10、10–20、20–50、>=50kpc六区间，
共18联合格子。全部{len(groups)}个可用计数组的期望总数守恒、方差分解和整数实现区间已核对。
共享实现计数包含类别指示波动与共同总体/校正状态波动；不能用独立单星误差相加替代。
原报告{len(tables['checks.csv'])}个数值敏感性行中{failures}行失败，全部保留。

本实验仅检验匹配选择的光度预测与模型计数迁移，没有提供真实类别/距离计数。
联合支持不足限制积分解释，即使通过也不能建立距离精度、真实总体恢复或独立污染率。
最终外部科学结论仍见[v3_1_final_validation_result.md](v3_1_final_validation_result.md)。

## 可复现产物

- [102状态清单](../manifests/v3_1/heldout_transfer/report/inventory.csv)、[预测支持](../manifests/v3_1/heldout_transfer/report/predictive_summary.csv)、[共享计数](../manifests/v3_1/heldout_transfer/report/counts.csv)。
- [逐场结果](../manifests/v3_1/heldout_transfer/report/field_summary.csv)、[天空配对](../manifests/v3_1/heldout_transfer/report/paired.csv)、[数值变化](../manifests/v3_1/heldout_transfer/report/checks.csv)。
- [本报告实际检查](../manifests/v3_1/transfer_result_sources.json)；图件需另行渲染、格式检查与目检，此文本生成不代表完成图件验收。
'''
    output = ROOT / 'docs/v3_1_transfer_result.md'
    output.write_text(text)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(root / 'transfer_result_sources.json', dict(inputs=inputs, report_sha256=sha(output),
        checks=checks, actual_checks=len(checks), passed=True, scientific_acceptance=False,
        scope='Complete actual source authentication and count/support checks; no true-distance calibration claim or visual-review acceptance.'))
    print(json.dumps(dict(actual_checks=len(checks), report=str(output))))


if __name__ == '__main__':
    main()
