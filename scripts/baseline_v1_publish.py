#!/usr/bin/env python3
"""Render the checked small NERSC scorecard as the versioned project report."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]

def main():
    r=json.loads((ROOT/'manifests/baseline_v1_report.json').read_text())
    assert r['complete_application'] and r['validation']['all_passed']
    lines=[]
    def add(text=''):lines.append(text)
    def table(headers,rows):
        add('| '+' | '.join(headers)+' |');add('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows:add('| '+' | '.join(map(str,row))+' |')
        add()
    def pct(x):return '—' if x is None else f'{100*x:.1f}%'
    def num(x):return '—' if x is None else f'{x:.3f}'
    def figure(name,caption):add(f'![{caption}](../figures/baseline_v1/{name}.png)\n\n{caption}\n')
    def mr(m):return [f"{m['n']:,}",num(m.get('bias_mag')),num(m.get('nmad_mag')),pct(m.get('tail_1')),
                     num(m.get('width68')),pct(m.get('coverage68')),pct(m.get('coverage95'))]
    mh=['N','Δμ中位数','NMAD','绝对Δμ>1','W68','覆盖68','覆盖95']
    add('# baseline v1：光度分支、数值传播和暗端压力实验\n')
    add('2026-11-03。依据[完整Pro评审](pro_review_494b12c.md)和[预先登记协议](baseline_v1_protocol.md)。'
        '本轮已完成固定实验及完整内部目录；v0模型、预测和历史结果保持冻结。\n')
    add('**采用的v1是原宽QRF加128点Sobol积分和箱内CDF插值。显式类别模型在给定真实类别时明显改善演化星距离，'
        '但两种纯测光门控均未通过校准护栏，因此没有替换部署模型。数值尾部大幅稳定，但仍有分层收敛失败；'
        '也没有找到三类错误率同时达标的质量子目录。上述负结果均保留在最终产物中。**\n')
    add('## 1. 保留整体框架，明确本轮改动\n')
    add('沿用14场、唯一身份、原空间train/cal/test划分、训练精度和应用入口：高银纬、低消光的DR11 PSF光学母样本，'
        'grz S/N≥5及既有clean条件；Gaia、光谱和WISE都不是应用入口条件。北方BASS/MzLS行继续保留且不赋予南方模型距离。'
        '旧test已经用于开发诊断，称为发展阶段空间验证，不称为新的盲测。\n')
    table(['配置','训练对象','矮星','演化星','过渡类','未知类别'],
          [[n,r['training'][n]['n_train'],*[r['training'][n]['classes'].get(str(k),0) for k in (0,1,2,-1)]] for n in 'ABC'])
    add('条件QRF保持原森林参数；仅用训练颜色学习固定二次logistic门控。自然比例和人工等比例是两种实验权重，'
        '不是银河真实类别概率。未知类别对象继续贡献宽QRF；分支支持不足时回退到宽分布。'
        '同一次扰动内先组合分支，再传播共享g到距离。oracle使用已知类别，仅用于诊断，不能进入实际目录。\n')

    add('## 2. 正确光度解可以恢复，纯测光分支权重仍未解决\n')
    rows=[]
    for n in 'ABC':
        for cl,label in [(1,'演化'),(2,'过渡')]:
            a=r['v0_branch_audit'][n][f'chosen:class{cl}:split2']
            rows.append([n,label,a['n'],pct(a['true_Mg_mass_pm0p5_quantiles'][1]),pct(a['fraction_mass_below_0p05']),
                         f"{a['catastrophic_supported_unambiguous']}/{a['catastrophic']}"])
    table(['v0配置','优先参照粗类','N','参照Mg±0.5质量中位数','质量<5%比例','严重错误中域内且未标多解'],rows)
    add('这是全部对应参照对象的原PDF审计，不仅是触发多解者。正确解附近通常有少量质量，但不保证形成独立模式；'
        '参照标签自身也有误差。域内、未触发多解不是可靠性的充分条件。\n')
    rows=[]
    for n in 'ABC':
        for variant in ('wide','natural','balanced','oracle'):
            s=r['model_scores'][n]['variants'][variant]['test']['chosen']
            e=s['class_evolved'];d=s['class_dwarf']
            rows.append([n,variant,e['n'],num(e['bias_mag']),num(e['nmad_mag']),pct(e['tail_1']),pct(d['tail_1'])])
    table(['配置','实验','演化星N','演化偏差','演化NMAD','演化严重错误','矮星严重错误'],rows)
    add('表中每个配置内使用同一批测试对象，A/B/C自身可用对象不同，不能把这张表当作严格同星波段增益。'
        '优先参照采用Gaia合格标签、其次SPECDIS；SPECDIS/SPDist分源成绩完整保存在清单。'
        'oracle的显著改善支持“分支判别/权重是主要障碍之一”，不能证明颜色已达到物理信息上限。'
        '自然门控使演化星更差；均衡权重改善演化星却增加矮星严重错误。\n')
    figure('02_population_branches','按粗星族分别显示严重错误率；oracle不属于可部署成绩。')

    add('## 3. 数值传播改进及未通过的收敛条件\n')
    table(['固定光度解析反例','旧箱中心95%区间所含概率','CDF插值后所含概率'],
          [[k,pct(v['old_center_probability95']),pct(v['mean_probability95'])] for k,v in r['numerical']['analytic'].items()])
    add('解析反例有5000个身份、sigma_g=0.2mag；这不是实际星表覆盖率。CDF插值与更多扰动共同减少尾部数值误差。\n')
    rows=[]
    for layer in ('low','high'):
        for method in ('iid_8_0','iid_32_0','iid_128_0','sobol_128_0'):
            v=r['numerical']['convergence']['A']['comparisons'][method][layer]
            rows.append([layer,method,v['n'],*[num(x) for x in v['p95_abs_mag']],pct(v['ambiguity_flip'])])
    table(['A误差层','积分','N','95下端差P95','中位数差P95','95上端差P95','多解状态变化'],rows)
    table(['独立128重复','误差层','N','95下端差P95','中位数差P95','95上端差P95','多解状态变化'],
          [[name,layer,v['n'],*[num(x) for x in v['p95_abs_mag']],pct(v['ambiguity_flip'])]
           for name,values in r['independent_sobol128_repeat'].items() for layer,v in values.items()])
    add('从母样本按误差、可用波段、支持和多解状态固定抽取3000个源，比较基准是独立数字移位Sobol512。'
        '低/高误差界为最大grz误差0.03mag。预设端点/中位容差为0.10/0.03/0.10mag，且各可用配置都须通过。'
        '**两层均未全部通过，所以按协议采用128次并保留numerical_converged=False。**'
        '它表示误差层级审计未通过，不表示每颗星的全部分位点都不稳定；也不允许把小的平均误差写成普遍收敛。\n')
    figure('01_numerical_convergence','解析概率质量与母样本分层端点稳定性是不同检验。')

    add('## 4. WISE仍是可选增强：同训练、同星比较\n')
    rows=[]
    for tag in ('W1','W1W2'):
        for source in ('Gaia','specdis'):
            for n,v in r['same_star'][tag].items():rows.append([tag,source,n,*mr(v['wide'][source]['all'])])
    table(['共同层','参照','模型',*mh],rows)
    add('Δμ=μphot−μref，星等指标单位mag，W68是中央68%全宽。相同层保持相同训练/测试身份及同源扰动。'
        '真实参照高度集中于亮、高精度对象；同星增益不能外推到所有暗端或演化星。'
        '母样本WISE覆盖不变：W1合格2,821,370/8,899,334=31.70%，W1W2合格1,587,931/8,899,334=17.84%；'
        '22≤r<25分别约14.7%和3.6%，因此纯光学A继续承担主体覆盖。\n')
    figure('03_same_star_information','固定共同训练和测试对象上的颜色增益，保留所有四种实验分布。')

    add('## 5. 固定通量降深：已有源进入暗端后会发生什么\n')
    di=r['depth_input'];de=r['depth']
    add(f"三个留出场共{di['n_sources']:,}个原始源，每源四个目标深度、各两次实现，共{di['n_rows']:,}行；"
        f"重新筛选后{di['accepted']:,}行保留。包含矮星、演化、过渡和未知类别。噪声来自实际母表通量逆方差，"
        '按目标亮度缩放原始通量并给参考距离模数增加相同位移；只补足差额噪声，随后重算S/N、WISE和路由。\n')
    rows=[]
    for key in ('target_depth_0','target_depth_1','target_depth_2','target_depth_3','no_W1'):
        a=de['groups'][key]
        for name in ('v0','v1','v1_cal'):rows.append([key,name,f"{a['n_accepted']}/{a['n_draws']}",*mr(a['models'][name]['metrics'])])
    table(['目标档／状态','流程','保留/模拟',*mh],rows)
    rows=[]
    for cl,label in [(0,'矮星'),(1,'演化'),(2,'过渡')]:
        for lo,hi in [(18,20),(20,22),(22,23),(23,24)]:
            a=de['groups'][f'class{cl}_observed_r_{lo}_{hi}'];m=a['models']['v1']['metrics']
            rows.append([label,f'{lo}–{hi}',a['n_sources'],*mr(m)])
    table(['粗类','最终实测r','原始源数',*mh],rows)
    add(f"非变暗配对{de['non_dimming_pairs']}行；逐波段缩放后的模板噪声已超过目标预算的比例为"
        +', '.join(pct(x) for x in de['template_noise_exceeds_target'])+'（g/r/z/W1/W2）。'
        '这些配对没有在看残差后删除；无法减去已有噪声。无grz基础S/N与clean失败原因可重叠，详见清单。'
        '缺少可用WISE的模板仍进入光学试验；不模拟新增图像形态、混叠或源检测完备率。\n')
    add('置信区间按原始源聚类bootstrap，不能把每源八次模拟当作八颗独立星。输入模板仍来自亮、高精度参照；'
        '按类别限额抽样后的人群比例也不代表母样本，不能把合并严重错误率解释为全目录错误率。'
        '本实验揭示噪声和路由退化，不证明真实暗端族群、参考尺度或污染组成已被验证。\n')
    figure('05_flux_depth_stress','按最终实测r和粗类分层；阴影为原始源聚类95%置信区间。')

    add('## 6. 校准选择、实际保留率和参照成绩\n')
    add('cal按各粗类严重错误的宏平均比较，同时限制矮星及整体严重错误增加≤2百分点、整体绝对残差中位数增加≤0.05mag。'
        '六组配置均保留wide。B/A与C/B在共同支持的cal对象上通过升级规则，仍用C→B→A；全域外时保留A并标域外。'
        'test与降深结果均未用于重训、调整阈值或事后挑选模型。\n')
    q=r['quality_diagnostics'];qc=q['candidates']
    assert r['selection']['quality_rule'] is None
    add(f"预登记{len(qc)}组预测宽度、最大已知类权重、跨配置分歧和光学误差组合；"
        f"其中{sum(min(c['class_counts'])>=30 for c in qc)}组满足每粗类至少30个cal参照，"
        '**没有一组同时达到三类严重错误率≤10%的条件**。所有内部行仍保留，research_quality=False。'
        '这不是说所有对象不可用，而是本次没有得到跨三类统一有效的可观测筛选。\n')
    figure('04_quality_retention','同一组预定筛选在参照和完整光学母样本中的保留率不同；纵轴仍是cal分层错误率。')
    table(['真实参照',*mh],[[s,*mr(r['recommended']['test'][s]['all'])] for s in ('Gaia','specdis','spdist')])
    table(['SPECDIS参照粗类',*mh],[[c,*mr(r['recommended']['test']['specdis'][c])] for c in ('class_dwarf','class_evolved','class_transition')])
    add('Gaia总体仍由矮星主导；SPECDIS/SPDist是一致性参照而非两套独立几何真值。覆盖率针对带噪参照点，'
        '不能解释为真实距离已校准。CQR只在cal扩展区间，不改中心；完整raw/cal成绩均保留。'
        'v1没有重跑弱/负视差后验检验，原v0视差空间诊断作为历史结果保留，不升级为v1校准证据。\n')
    figure('06_distance_references','光度距离与Gaia、SPECDIS、SPDist的密度散点和残差；全部测试对象按相应参照有效性计入。')
    figure('07_same_source_cmd','同源测试CMD：参考Mg与预测中位Mg；预测关系变窄也可能来自模型压缩，不能视为独立精度验证。')

    add('## 7. 完整内部目录、核验与后续边界\n')
    app=r['application'];south=app['fields'][:13]
    table(['实际产物','数量'],[
        ['完整14场唯一身份',f"{app['n']:,}"],['DECam预测',f"{sum(x['n'] for x in south):,}"],
        ['北天无同系统模型',f"{app['fields'][13]['n']:,}"],
        *[[f'推荐{k}',f"{sum(x['routes'][str(i)] for x in south):,}"] for i,k in enumerate('ABC')],
        *[[k,f"{sum(x[k] for x in south):,}"] for k in ('supported','ambiguous','research_quality','numerical_converged')]])
    add('仅在cal规则冻结后完整运行一次，按不重叠分块写入，再按身份原顺序组装。'
        '数据、六套条件模型、逐源分布及FITS留在NERSC的`/pscratch/sd/y/yunao/photdis_pipeline/data/baseline_v1/`；'
        'Git仅保存代码、配置、结果清单和八张图。目录字段、分支权重含义和复现命令见[运行说明](baseline_v1_catalog.md)。\n')
    figure('08_application_states','完整母样本路由和独立状态标记；这些状态不是互斥类别。')
    add(f"实际全产物核验通过{r['validation']['n_checks']}项：v0模型/目录哈希不变，训练与评价身份隔离，"
        '同训练对照，PDF归一化，分位数顺序，共享亮度位移，cal区间包含raw，14场身份/行数/单位/可用波段及全母表身份唯一。'
        '另有8项聚焦数值/噪声测试通过，固定筛选计数的288项浮点边界比较与直接布尔掩码一致。\n')
    add('本轮到此结项：已完成Pro提出的有限实验和版本交付。下一步应从已暴露的整体瓶颈中选择任务：'
        '建立受参照选择函数约束、可校准的分支权重，并单独评价演化与过渡星；为低精度、无WISE人群补充实际验证。'
        '数值剩余误差应针对已识别的不稳定分布处理，避免再次全目录试探重跑。'
        '不自动启动星团扩张、模型族搜索或统一距离零点修正；v1仍为内部研究目录。\n')
    add('机器可读来源：[总成绩与哈希](../manifests/baseline_v1_report.json)、[实际核验](../manifests/baseline_v1_validation.json)、'
        '[目录产物清单](../manifests/baseline_v1_application.json)。')
    (ROOT/'docs/baseline_v1_result.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
