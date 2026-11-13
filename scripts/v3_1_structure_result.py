#!/usr/bin/env python3
"""Write the completed structure comparison with shared-scale and identity limits."""
import csv
import json
import math
from pathlib import Path
from v2_1_near import ROOT, sha, save_json


def main():
    root = ROOT / 'manifests/v3_1/structure'
    inputs, markers, tables, checks = {}, {}, {}, []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            raise AssertionError(name)

    for stage, names in [('posterior', ['summary.csv', 'numerical_checks.csv', 'membership.csv', 'comparison.json']),
                         ('report', ['counts.csv', 'membership.csv', 'membership_comparison.csv'])]:
        marker = root / stage / 'complete.json'
        done = json.loads(marker.read_text())
        markers[stage] = done
        check('complete_shared_sample:' + stage, done['n'] == 512 and done['states'] == 32 and done['all_PSF'])
        inputs[str(marker.relative_to(ROOT))] = sha(marker)
        for name in names:
            path = root / stage / name
            check('source_hash:' + stage + '/' + name, sha(path) == done['outputs'][name])
            inputs[str(path.relative_to(ROOT))] = sha(path)
            with path.open() as stream:
                tables[stage + '/' + name] = json.load(stream) if path.suffix == '.json' else list(csv.DictReader(stream))
    check('count_report_posterior_identity', markers['report']['inputs']['data/v3_1/structure/posterior/complete.json'] == sha(root / 'posterior/complete.json'))
    summary = {r['branch']: r for r in tables['posterior/summary.csv']}
    check('both_branches', len(tables['posterior/summary.csv']) == len(summary) == 2 and set(summary) == {'full', 'gaussian'})
    counts = tables['report/counts.csv']
    check('four_count_rows', len(counts) == 4 and {(r['branch'], r['component']) for r in counts} ==
          {(b, c) for b in summary for c in ['member', 'background']})
    membership = {}
    for branch in summary:
        rows = [r for r in tables['report/membership.csv'] if r['branch'] == branch]
        original = {r['identity']: r for r in tables['posterior/membership.csv'] if r['branch'] == branch}
        membership[branch] = {r['identity']: r for r in rows}
        check('unique512PSF:' + branch, len(rows) == len(membership[branch]) == len(original) == 512 and
              set(original) == set(membership[branch]) and all(r['TYPE'] == 'PSF' for r in rows))
        check('membership_unchanged:' + branch, all(float(r['model_member_probability']) ==
              float(original[r['identity']]['member_probability']) for r in rows))
        check('member_sum:' + branch, abs(sum(float(r['model_member_probability']) for r in rows) -
              float(summary[branch]['member_mean'])) < 1e-7)
        branch_counts = [r for r in counts if r['branch'] == branch]
        check('count_conservation:' + branch, abs(sum(float(r['mean']) for r in branch_counts) - 512) < 1e-8)
        for r in branch_counts:
            name = branch + '/' + r['component']
            check('shared_variance:' + name, math.isclose(float(r['variance']),
                  float(r['conditional_variance']) + float(r['shared_variance']), rel_tol=1e-9, abs_tol=1e-8))
            check('ordered_realized_intervals:' + name, 0 <= int(r['lower95']) <= int(r['lower68']) <=
                  int(r['median']) <= int(r['upper68']) <= int(r['upper95']) <= 512)
            check('original_gate_identity:' + name, r['original_numerical_passed'] == summary[branch]['numerically_passed'])
            check('acceptance_preserved:' + name, (r['accepted'] == 'True') ==
                  (r['original_numerical_passed'] == r['sampling_passed'] == 'True'))
    check('same_branch_identities', set(membership['full']) == set(membership['gaussian']))
    check('same_published_information', all(all(membership['full'][i][k] == membership['gaussian'][i][k]
          for k in ['radius_deg', 'membership_available', 'published_member_probability', 'published_p_ge_0_9'])
          for i in membership['full']))
    numerical = tables['posterior/numerical_checks.csv']
    check('ten_numerical_comparisons', len(numerical) == 10 and len({(r['branch'], r['comparison']) for r in numerical}) == 10)
    for branch, s in summary.items():
        rows = [r for r in numerical if r['branch'] == branch and r['comparison'] != 'quadrature_first']
        check('original_numerical_gate:' + branch, len(rows) == 4 and
              (s['numerically_passed'] == 'True') == all(r['passed'] == 'True' for r in rows))
    comparison = tables['posterior/comparison.json']
    check('full_minus_gaussian_count', math.isclose(comparison['member_mean_full_minus_gaussian'],
          float(summary['full']['member_mean']) - float(summary['gaussian']['member_mean']), abs_tol=1e-10))
    check('maximum_probability_difference', math.isclose(comparison['maximum_member_probability_difference'],
          max(abs(float(membership['full'][i]['model_member_probability']) -
                  float(membership['gaussian'][i]['model_member_probability'])) for i in membership['full']), abs_tol=1e-12))
    result = ['| 分支 | 原后验数值通过 | 成员均值 | 成员实现计数中位数 [95%区间] | 条件 / 共享方差 | 共同距离中位数 [95%区间] / kpc | 响应 ESS / 32 |',
              '|---|---|---:|---|---|---|---:|']
    sampling = ['| 分支 | 成员抽样目录数 | 抽样检查通过 | 后验与抽样联合接受 | 半样本CDF差 |', '|---|---:|---|---|---:|']
    for branch, s in summary.items():
        c = next(r for r in counts if r['branch'] == branch and r['component'] == 'member')
        result.append(f"| {branch} | {s['numerically_passed']} | {float(c['mean']):.3f} | {c['median']} [{c['lower95']}, {c['upper95']}] | {float(c['conditional_variance']):.3f} / {float(c['shared_variance']):.3f} | {float(s['common_distance_median_kpc']):.3f} [{float(s['common_distance_lower95_kpc']):.3f}, {float(s['common_distance_upper95_kpc']):.3f}] | {float(s['response_ESS']):.3f} |")
        sampling.append(f"| {branch} | {c['draws']} | {c['sampling_passed']} | {c['accepted']} | {float(c['CDF_half_delta']):.5f} |")
    member_table = ['| 分支 | 原分组 | 对象数 / 有已发表成员概率数 | 模型期望成员数 | 已发表概率之和（可用对象） |',
                    '|---|---|---:|---:|---:|']
    for r in tables['report/membership_comparison.csv']:
        published = f"{float(r['published_probability_sum']):.3f}" if r['published_probability_sum'] else '缺失'
        member_table.append(f"| {r['branch']} | {r['group']} | {r['n']} / {r['membership_available_n']} | {float(r['expected_model_members']):.3f} | {published} |")
    failed = [r['branch'] + '/' + r['comparison'] for r in numerical if r['passed'] != 'True']
    text = f'''# v3.1 NGC288 结构后验与公平高斯基线结果

本例完成512个光度先选原生PSF对象、32个共享响应/校准状态的结构应用。
这是已见系统的方法演示；共同距离先验已使用外部尺度，不能作为独立距离恢复，
也不能替代修正后主候选的最终验证。数值与抽样失败按原产物逐项保留。

## 相同对象、先验与选择归一化

从原8192对象中按冻结身份哈希选择512个，成员概率与残差不参与入选。
计数只针对这512个对象，不外推到48230个合格源或物理面密度。32个原始响应在
结构光度进入前具有等蒙特卡洛质量；共享响应、共同距离与空间赔率在目录级联合更新。
没有把response12的六参数修正擅自应用到另外31套SED；本例与最终候选有不同产品身份。

full分支撤去原距离先验后使用完整原生似然。gaussian分支对每个类别条件参考后验
匹配均值/方差、保留类别证据，再解析除去原N(15,5)参考先验。两分支采用相同
结构/背景先验与选择积分；不是直接把后验相乘或堆叠。
选入样本的似然为[qR_structure+(1−q)R_background]/[qA_structure+(1−q)A_background]。
空间尺度0.1度、固有距离模数宽度0.02mag预先固定，不是拟合得到的密度剖面或物理深度。

共同尺度先验只使用一次：9.03kpc，下/上尺度0.56/0.48kpc的等半非对称模型。
它不是512个独立标准尺，不按sqrt(512)缩小误差。共同距离后验的接近程度不能证明绝对精度。

## 完整后验与共享实现计数

{chr(10).join(result)}

full−gaussian成员期望之差为{comparison['member_mean_full_minus_gaussian']:+.4f}，
单对象成员概率最大差{comparison['maximum_member_probability_difference']:.6f}。
这描述似然摘要近似对本例的影响，不能解释为两个归一化生成模型的Bayes factor，
也不能由分支差异单独判断哪一个接近真实成员数。

{chr(10).join(sampling)}

每次实现目录先抽一个共享响应/共同距离/赔率索引，再条件抽成员指示；成员+背景恰为512。
表中实现计数区间不同于期望计数误差条或独立Poisson误差。
原65536与必要时262144次目录抽样都保留，检查包括均值、方差、分位数和CDF；
单独CDF通过不能替代全部检查。原后验比较失败项：{', '.join(failed) if failed else '无'}。
第一层quadrature_first变化单独保留；最终接受按原冻结的后四项比较决定，不删除失败响应。

## 拟合后描述性 Gaia 成员模型对比

{chr(10).join(member_table)}

已发表成员概率可用{markers['report']['membership_available']}个，其中p>=0.9为
{markers['report']['published_p_ge_0_9']}个。缺失值保持缺失，不能当作非成员。
全体组的模型期望可能含缺失参考对象，已发表概率和仅限可用对象，分母已逐行标明。
已发表概率也是模型输出，且NGC288已被查看；本对比不是独立硬标签分类准确率，
不产生独立污染估计、独立尺度验证或主候选校准接受。

## 可复现来源与限制

- [后验摘要](../manifests/v3_1/structure/posterior/summary.csv)、[数值变化](../manifests/v3_1/structure/posterior/numerical_checks.csv)、[联合实现计数](../manifests/v3_1/structure/report/counts.csv)。
- [逐对象成员概率](../manifests/v3_1/structure/report/membership.csv)、[分组描述对比](../manifests/v3_1/structure/report/membership_comparison.csv)。
- [本报告来源与实际检查](../manifests/v3_1/structure_result_sources.json)；完整后验、共享索引及协方差保留NERSC。

该演示的主要贡献是正确撤去旧先验、匹配选择归一化和传播目录级共享不确定性。
是否数值可用以上表原门槛为准；既有尺度先验、固定空间/深度模型及有限响应支持
仍限制科学解释。两张英文图须另行实际渲染和目检，此文本生成不代表图件验收。
'''
    output = ROOT / 'docs/v3_1_structure_result.md'
    output.write_text(text)
    inputs[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
    save_json(ROOT / 'manifests/v3_1/structure_result_sources.json', dict(inputs=inputs,
        report_sha256=sha(output), checks=checks, actual_checks=len(checks), passed=True,
        scientific_acceptance=False, scope='Actual shared-count/identity/source consistency. Seen-system supplied-scale demonstration, not independent or final-candidate validation.'))
    print(json.dumps(dict(actual_checks=len(checks), report=str(output))))


if __name__ == '__main__':
    main()
