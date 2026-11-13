#!/usr/bin/env python3
"""Interpret the complete domain join without certifying a failed candidate."""
import csv
import json
from collections import Counter
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    base = ROOT / 'manifests/v3_1'
    folder = base / 'domain_adjudication'
    marker = folder / 'complete.json'
    done = json.loads(marker.read_text())
    inputs = {str(marker.relative_to(ROOT)): sha(marker)}
    checks = []

    def check(name, condition):
        checks.append(dict(check=name, passed=bool(condition)))
        if not condition:
            raise AssertionError(name)

    def read(name):
        path = folder / name
        check('output_hash:' + name, sha(path) == done['outputs'][name])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        with path.open() as stream:
            return json.load(stream) if path.suffix == '.json' else list(csv.DictReader(stream))

    stages = ['domain_evidence', 'domain_assessment', 'candidate_modes',
              'target_ensemble/candidate', 'highq_counts', 'response_probes/report',
              'radial_report', 'radial_summary', 'final_validation/report',
              'heldout_transfer/report', 'structure/posterior', 'structure/report']
    sources = {}
    for stage in stages:
        path = base / stage / 'complete.json'
        check('complete_source_identity:' + stage,
              sha(path) == done['inputs'][f'data/v3_1/{stage}/complete.json'])
        inputs[str(path.relative_to(ROOT))] = sha(path)
        sources[stage] = json.loads(path.read_text())
    check('same_candidate', done['cells'] == 75 and done['candidate_n'] == 3540 and
          done['highq_n'] == 3003 and done['all_PSF'] and not done['new_sample_constructed'])
    check('no_automatic_acceptance', done['calibrated_cells'] == 0 and
          done['scientifically_accepted_ambiguous_cells'] == 0 and
          not done['scientific_domain_adjudication_complete'])
    cells, objects = read('cells.csv'), read('objects.csv')
    limits, support = read('global_limits.json'), read('transfer_support.csv')
    check('75_unique_cells', len(cells) == 75 and {int(r['cell']) for r in cells} == set(range(75)))
    check('3540_unique_PSF', len(objects) == len({r['identity'] for r in objects}) == 3540 and
          all(r['TYPE'] == 'PSF' for r in objects))
    mode_path = base / 'candidate_modes/full_summary.csv'
    check('mode_source_hash', sha(mode_path) == sources['candidate_modes']['outputs']['full_summary.csv'])
    inputs[str(mode_path.relative_to(ROOT))] = sha(mode_path)
    with mode_path.open() as stream:
        mode_rows = list(csv.DictReader(stream))
    modes = {r['identity']: r for r in mode_rows}
    check('same_mode_identities', len(modes) == len(mode_rows) == 3540 and
          set(modes) == {r['identity'] for r in objects})
    for r in objects:
        m = modes[r['identity']]
        check('object:' + r['identity'], r['fixed_main_highq'] == m['fixed_main_highq'] and
              r['significant_modes'] == m['significant_modes'] and
              (r['multimodal'] == 'True') == (int(r['significant_modes']) >= 2) and
              r['mode_numerical_passed'] == m['numerical_passed'] and
              r['scientific_disposition'] == 'failed_main_numerical_gate' and
              r['absolute_calibration_established'] == r['independent_contamination_measured'] == 'False' and
              r['allowed_use'] == 'conditional_model_diagnostics_only' and
              0 <= float(r['distance_mass_without_precise_newsky_reference']) <= 1)
    check('fixed_highq3003', sum(r['fixed_main_highq'] == 'True' for r in objects) == 3003)
    for c in cells:
        key = int(c['cell'])
        group = [r for r in objects if int(r['cell']) == key]
        n = len(group)
        check('cell:' + str(key), int(c['candidate_n']) == n and
              int(c['highq_n']) == sum(r['fixed_main_highq'] == 'True' for r in group) and
              int(c['multimodal_n']) == sum(r['multimodal'] == 'True' for r in group) and
              int(c['highq_multimodal_n']) == sum(r['multimodal'] == r['fixed_main_highq'] == 'True' for r in group) and
              c['scientific_disposition'] == ('failed_main_numerical_gate' if n else 'outside_candidate') and
              c['calibrated_domain'] == c['scientifically_accepted_ambiguous_domain'] == 'False' and
              c['empirical_absolute_distance_status'] == 'unvalidated')
        for source in ['astronn', 'specdis', 'spdist']:
            key = 'precise_newsky_' + source
            check(key + ':' + c['cell'], (c[key + '_below20'] == 'True') == (int(c[key]) < 20))
    occupied = [r for r in cells if int(r['candidate_n'])]
    check('occupied30', len(occupied) == done['occupied_cells'] == 30)
    check('full_transfer', sources['heldout_transfer/report']['states'] == 102 and
          limits['heldout_transfer_views'] == sources['heldout_transfer/report']['views'])
    check('transfer_summary_identity', sha(folder / 'transfer_support.csv') ==
          sources['heldout_transfer/report']['outputs']['predictive_summary.csv'])
    check('global_identity', limits['candidate_main'] == next(v for v in
          sources['target_ensemble/candidate']['views'] if v['view'] == 'bootstrap_evidence') and
          not limits['candidate_main']['accepted'] and
          not limits['final_external_scientific_acceptance'] and
          not limits['independent_fullcandidate_contamination_measured'])
    check('response_and_structure_identity', limits['independent_SED_probe_sets'] ==
          sources['response_probes/report']['sets'] and
          limits['structure_posterior_summary'] == sources['structure/posterior']['summary'] and
          limits['structure_count_report_passed'] == sources['structure/report']['passed'])
    check('complete_global_inventories', sources['response_probes/report']['states'] == 64 and
          sources['radial_report']['catalogs'] == 176 and sources['radial_report']['arms'] == 352 and
          all(sources[s]['states'] == 32 and sources[s]['n'] == 512 and sources[s]['all_PSF']
              for s in ['structure/posterior', 'structure/report']))
    count = Counter(r['wise_label'] for r in occupied)
    table = ['| WISE 格子类型 | 有候选格子数 | 候选 / 固定高q对象 | 显著多峰 / 高q多峰 |',
             '|---|---:|---:|---:|']
    for label in sorted(count):
        group = [r for r in occupied if r['wise_label'] == label]
        sums = {key: sum(int(r[key]) for r in group) for key in
                ['candidate_n', 'highq_n', 'multimodal_n', 'highq_multimodal_n']}
        table.append(f"| {label} | {len(group)} | {sums['candidate_n']} / {sums['highq_n']} | {sums['multimodal_n']} / {sums['highq_multimodal_n']} |")
    sets = ['| 响应探针集合 | 原数值接受 | 响应 ESS |', '|---|---|---:|']
    for r in limits['independent_SED_probe_sets']:
        value = f"{r['ESS']:.6f}" if 'ESS' in r else '不归一化'
        sets.append(f"| {r['set']} | {r['accepted']} | {value} |")
    structure = ['| NGC288 分支 | 后验数值通过 | 模型成员期望 | 响应 ESS |', '|---|---|---:|---:|']
    for r in limits['structure_posterior_summary']:
        structure.append(f"| {r['branch']} | {r['numerically_passed']} | {r['member_mean']:.3f} | {r['response_ESS']:.3f} |")
    missing_n = sum(int(r['candidate_n']) for r in occupied if
                    sum(int(r['precise_newsky_' + s]) for s in ['astronn', 'specdis', 'spdist']) == 0)
    mass = sum(float(r['distance_mass_without_precise_newsky_reference']) for r in objects) / len(objects)
    transfer_main = [r for r in support if r['view'] == 'bootstrap_evidence' and r['variant'] == 'base']
    transfer_table = ['| 总体 / 预测入选 | 单位对象联合 ESS / 支持通过 | 广义设计联合 ESS / 支持通过 |',
                      '|---|---|---|']
    for r in transfer_main:
        transfer_table.append(f"| {r['case']} / {r['envelope']} | {float(r['joint_importance_ess']):.3f} / {r['joint_supported']} | {float(r['generalized_design_joint_importance_ess']):.3f} / {r['generalized_design_joint_supported']} |")
    text = f'''# v3.1 最终候选的科学适用域判定

完整证据合并后，当前候选没有建立可宣称已校准的科学使用域。
75个冻结光度格子中，30个有候选对象，全部继承主混合的数值证据失败；
其余45个格子没有该候选，不能解释为通过、失败或已验证的外推对象。
保留3540个原生PSF对象及固定3003个高q身份，允许带原标记的条件模型诊断。
这不支持精确科学距离、绝对计数、物理密度或独立污染率声明。

## 格子状态与模式信息分别保留

{chr(10).join(table)}

显著多峰按原定义至少两个后验质量大于0.05的距离盆地；弱模式仍在完整产品中。
模式是该失败模型下的条件信息，不证明双星、物理类别或已校准的模糊域。
原逐对象数值结果、native细网格核验状态、模式数和参考缺口都保留，
不能因主状态失败而隐藏，也不能因逐对象稳定而撤销主证据门槛。

## 参考缺口与经验标定

{missing_n}个候选位于没有合格新天空精确距离参考的光度格子。
逐对象模型距离质量中，落在无相应精确新天空参考距离格子的平均比例为
{100 * mass:.4f}%；其补集不是已经验证的概率质量。
各来源少于20参考的标记独立保留；有参考、相对均值guard通过、
或稀疏比较未发现显著差异均不能授予绝对校准。
signed Gaia在观测空间比较，教师相关参考与共同系统成员不作为独立真值副本。
新天空、来源新但已见天空和NGC1261共同尺度的角色保持分开。
完整外部失败与缺口见[最终验证结果](v3_1_final_validation_result.md)。

## 全局支持不能变成局部格子的校准证据

{chr(10).join(sets)}

独立SED探针为固定128目标的有限响应检验；它不是26000总体的响应后验，
也不是主候选的32个亮度校正自举状态。不可把两种ESS互相替代或映射为局部准确率。
失败集合不做幸存状态归一化。详细来源迁移见[响应结果](v3_1_response_result.md)。

{chr(10).join(transfer_table)}

此表只列原报告实际可计算的主视图；缺失视图的原因和数值状态仍在完整报告中。
只有holdout总体拟合排除了预测天空场；其他总体是已见天空比较，且holdout也不
建立全SED/校准训练独立性。重要性支持不是距离真值验证。
见[完整天空迁移](v3_1_transfer_result.md)及原逐视图状态，不用部分状态替代102状态。

全部176目录/352臂的径向恢复及原失败均已纳入合并；窄结构覆盖和来源相关限制
见[径向结果](v3_1_radial_result.md)。模型高q泄漏是条件期望，完整候选独立污染率
仍未测量；共享计数与来源相关误差不因高q选择而消失。

## 结构应用的身份边界

{chr(10).join(structure)}

NGC288计数报告原接受标记为{limits['structure_count_report_passed']}。
两分支均用相同512对象、32共享响应/校准及一次共同尺度先验，撤去旧距离先验；
这是已见系统的方法演示，与主候选产品身份不同。即使数值通过，也不能提供
独立尺度恢复、独立成员污染测量或主候选校准。详见[结构后验结果](v3_1_structure_result.md)。

## 可复核产物及完成边界

- [75格子状态](../manifests/v3_1/domain_adjudication/cells.csv)、[3540对象状态](../manifests/v3_1/domain_adjudication/objects.csv)。
- [完整全局状态](../manifests/v3_1/domain_adjudication/global_limits.json)、[迁移支持](../manifests/v3_1/domain_adjudication/transfer_support.csv)。
- [本文来源与核验](../manifests/v3_1/domain_result_sources.json)。所有原始失败、精化侧表和完整后验保留NERSC。

本文完成的是完整产物上的书面判定。英文图须另行渲染、核验和目检；
最终完成记录必须同时链接本文来源和实际图件验收，不能由生成本文自动授予。
当前判定不采用新模型、不扩展百万目录，也不改变已部署v1。
'''
    # The detailed reports must already be real, source-backed deliverables.
    for stem in ['transfer', 'structure']:
        path = ROOT / f'docs/v3_1_{stem}_result.md'
        proof = base / f'{stem}_result_sources.json'
        record = json.loads(proof.read_text())
        check('actual_report:' + stem, record['passed'] and record['report_sha256'] == sha(path))
        inputs[str(path.relative_to(ROOT))] = sha(path)
        inputs[str(proof.relative_to(ROOT))] = sha(proof)
    output = ROOT / 'docs/v3_1_domain_result.md'
    output.write_text(text)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(base / 'domain_result_sources.json', dict(inputs=inputs,
        report_sha256=sha(output), checks=checks, actual_checks=len(checks), passed=True,
        written_review_complete=True, scientific_acceptance=False,
        scientific_domain_adjudication_complete=False, visual_acceptance_required=True,
        scope='Complete fixed-product domain interpretation and compact provenance checks. Actual figure acceptance and final delivery record are separate.'))
    print(json.dumps(dict(actual_checks=len(checks), report=str(output))))


if __name__ == '__main__':
    main()
