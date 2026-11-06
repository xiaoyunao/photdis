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
        (D/filename).write_text('\n'.join(text)+'\n')
    (D/'v2_0_revision_atlas.md').write_text('\n'.join(index)+'\n')
    summary=['# v2.0 本轮图件与异常审计','',
             '逐项依据见[完整审查](pro_review_6ee0df1.md)及[执行协议](v2_0_revision_protocol.md)。', '',
             '975张六波段组合图、既有55组重绘、15张温度诊断、参考／应用流程图与4张异常诊断，共1050张。', '',
             'chosen测试47,756，Gaia点距离3,207，保守巨星1,163，固定异常区49。每面板核对图内、图外和缺失数。', '',
             '异常对象的证据分组如下，数目由逐星结果产生。', '']
    summary += [f'- {k}，{v}颗。' for k,v in audit['categories'].items()]
    summary += ['',audit['limitations'],'',
                '[49颗完整逐星CSV](../manifests/v2_0_revision/gaia_anomalies.csv) · [精确字符串ID JSON](../manifests/v2_0_revision/gaia_anomalies.json) · [完整图集](v2_0_revision_atlas.md)','']
    (D/'v2_0_revision_result.md').write_text('\n'.join(summary))
    (M/'delivery.json').write_text(json.dumps(dict(figures=len(registry),hashes=delivery,validation='All figure hashes and panel count conservation checked after transfer'),indent=2)+'\n')
    print('Published all revised figures',len(registry))


if __name__=='__main__':main()
