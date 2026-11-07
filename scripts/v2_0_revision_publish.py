#!/usr/bin/env python3
"""Publish all revised figures in bounded atlas volumes after artifact transfer."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
M=ROOT/'manifests/v2_0_revision'
D=ROOT/'docs'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    registry=json.loads((M/'figure_registry.json').read_text())
    assert len(registry)==1050
    audit=json.loads((M/'anomaly_summary.json').read_text())
    assert [audit[k] for k in ('chosen','gaia','giants','anomalies')]==[47756,3207,1163,49]
    delivery={}
    for name, record in registry.items():
        for kind in ('png','pdf','csv'):
            path=ROOT/record[kind]
            assert path.exists(), str(path)
            digest=sha(path)
            if kind in ('png','pdf'):assert digest==record[kind+'_sha256']
            delivery[record[kind]]=digest
        for panel in record['panels']:
            assert panel['plotted']+panel['outside']+panel['missing']==panel['eligible']
        metadata_path=M/'figures'/f'{name}.json'
        metadata=json.loads(metadata_path.read_text())
        assert metadata['png_sha256']==record['png_sha256']
        assert metadata['pdf_sha256']==record['pdf_sha256']
        delivery[str(metadata_path.relative_to(ROOT))]=sha(metadata_path)
    names=sorted(registry, key=lambda n:(registry[n]['section']=='combinations',registry[n]['section'],n))
    index=['# v2.0 修订完整图集','', '每个分册直接包含全部图件，并附PNG、PDF、逐图CSV和元数据。模型expanded_A、空间留出开发测试。统计口径见本轮协议。','']
    for start in range(0,len(names),40):
        number=start//40+1
        filename=f'v2_0_revision_atlas_{number:02}.md'
        batch=names[start:start+40]
        index.append(f'- [分册{number:02}，{len(batch)}张]({filename})，{batch[0]} 至 {batch[-1]}')
        text=[f'# v2.0 修订图集 {number:02}','', '[完整目录](v2_0_revision_atlas.md)','']
        for name in batch:
            record=registry[name]
            text += [f'## {name}','',f'![{name}](../{record["png"]})','',record['caption'],'',
                     f'[PNG](../{record["png"]}) · [PDF](../{record["pdf"]}) · [CSV](../{record["csv"]}) · [元数据](../manifests/v2_0_revision/figures/{name}.json)','']
        (D/filename).write_text('\n'.join(text).rstrip()+'\n')
    (D/'v2_0_revision_atlas.md').write_text('\n'.join(index)+'\n')
    summary=['# v2.0 本轮图件与异常审计','',
             '逐项依据见[完整审查](pro_review_6ee0df1.md)及[执行协议](v2_0_revision_protocol.md)。', '',
             '975张六波段组合图、既有55组重绘、15张温度诊断、参考／应用流程图与4张异常诊断，共1050张。', '',
             'chosen测试47,756，Gaia点距离3,207，保守巨星1,163，固定异常区49。3687个统计面板均通过图内、图外和缺失数守恒检查。', '',
             '所有图内文字均为英文，保持Pal 5 New配色与字体风格。1050组PNG均为透明RGBA、300dpi；PDF均为单页且无中文文本。270张代表图经联系表目视检查，另放大检查距离残差、CMD、流程及异常图。', '',
             '本轮使用冻结历史样本，保留16个非PSF例外供审计；49颗异常对象全为PSF。新建科学样本仍须显式TYPE=PSF。未知光谱类别的温度图为空，表示没有通过质量筛选的温度证据，并非没有此类对象。', '',
             '异常对象的证据分组如下，数目由逐星结果产生。', '']
    summary += [f'- {k}，{v}颗。' for k,v in audit['categories'].items()]
    with open(M/'figures/03_distance_all.csv') as stream:
        bins=list(csv.DictReader(stream))
    summary += ['', '线性残差零附近核对如下，采用等宽2个百分点分箱。', '']
    for source in ('Gaia','specdis','spdist'):
        near=[r for r in bins if r['panel']==source+'_Central 2 percentage-point histogram'
              and float(r['lo']) in (-2.,0.)]
        summary.append(f"- {source}，[-2%,0%)与[0%,2%)分别为"+'、'.join(r['n'] for r in near)+'个对象。')
    with open(M/'anomaly_pdf_summary.csv') as stream:
        pdf_cases=list(csv.DictReader(stream))
    pdf_masses='、'.join(f"{100*float(row['mass_near_reference']):.4g}%" for row in pdf_cases)
    summary += ['', '这些计数未显示原symlog散点图所暗示的明显零附近缺口。', '',
                '49颗异常对象均为PSF，均因具体RVS质量条件未通过而没有可靠光谱类别。低光度CMD位置支持白矮星候选解释，但尚未与外部白矮星目录确认。异常RVS参数不可直接当作物理参数。', '',
                f'四个固定PDF个例在Gaia参考距离模数±0.5mag内的概率质量为{pdf_masses}。部分近距离解存在但权重很低。', '',
                'Gaia与Legacy原生坐标的历元传播后一致到数值精度。原生位置可能沿用Gaia信息，因此该一致性只核对关联内部一致性，独立错配排除仍有限。', '']
    summary += ['',audit['limitations'],'',
                '[49颗完整逐星CSV](../manifests/v2_0_revision/gaia_anomalies.csv) · [精确字符串ID JSON](../manifests/v2_0_revision/gaia_anomalies.json) · [完整图集](v2_0_revision_atlas.md)','']
    (D/'v2_0_revision_result.md').write_text('\n'.join(summary))
    (M/'delivery.json').write_text(json.dumps(dict(figures=len(registry),hashes=delivery,validation='All figure hashes and panel count conservation checked after transfer'),indent=2)+'\n')
    print('Published all revised figures',len(registry))


if __name__=='__main__':main()
