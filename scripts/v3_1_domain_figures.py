#!/usr/bin/env python3
"""Render frozen observable cells; availability is not scientific certification."""
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import FuncNorm, Normalize
from v2_1_near import ROOT, sha, save_json, csv_rows
from photdis_figure_style import STYLE

WISE = ['W1 unavailable', 'W1 signed SNR < 5', 'W1 SNR >= 5']
R_LABELS = ['<16', '16–18', '18–20', '20–22', '>=22']
COLOR_LABELS = ['<0', '0–0.3', '0.3–0.6', '0.6–0.9', '>=0.9']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['photometry','evidence'],required=True); a=p.parse_args()
    source=ROOT/'manifests/v3_1'/('domain_grid' if a.stage=='photometry' else 'domain_evidence')
    marker=source/'complete.json'; done=json.loads(marker.read_text())
    assert done['all_PSF'] and done['cells']==75
    inputs={str(marker.relative_to(ROOT)):sha(marker)}; tables={}
    names=['cells.csv'] if a.stage=='photometry' else ['cells.csv','reference_distance_cells.csv']
    for name in names:
        path=source/name; assert sha(path)==done['outputs'][name]
        inputs[str(path.relative_to(ROOT))]=sha(path)
        with path.open() as f: tables[name]=list(csv.DictReader(f))
    cells=sorted(tables['cells.csv'],key=lambda r:int(r['cell']))
    assert [int(r['cell']) for r in cells]==list(range(75))
    for r in cells:
        assert int(r['cell'])==(int(r['r_bin'])*5+int(r['color_bin']))*3+int(r['wise_bin'])
    out=ROOT/f'figures/v3_1/domain_{a.stage}'; out.mkdir(parents=True,exist_ok=True)
    meta=ROOT/f'manifests/v3_1/domain_{a.stage}_figures'; meta.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update(STYLE); outputs={}; plotted=[]; panel_count=0; labels=[]

    def matrix(values,w):
        return np.array(values,float).reshape(5,5,3)[:,:,w]

    def panels(arrays,titles,norm,cmap,figure,title,subtitle,footer,colorbar_label,ticks=None,percent=False):
        nonlocal panel_count
        nrow=len(arrays); fig,axes=plt.subplots(nrow,3,figsize=(15,3.35*nrow+2),squeeze=False)
        fig.subplots_adjust(left=.08,right=.89,top=.81,bottom=.19,wspace=.23,hspace=.4)
        palette=plt.get_cmap(cmap).copy(); palette.set_bad('#dddddd')
        for i,(values,row_title) in enumerate(zip(arrays,titles)):
            for w in range(3):
                ax=axes[i,w]; data=matrix(values,w); masked=np.ma.masked_invalid(data)
                im=ax.imshow(masked,origin='lower',norm=norm,cmap=palette,aspect='auto')
                ax.set_title(row_title+' | '+WISE[w],fontsize=10)
                ax.set_xticks(range(5),COLOR_LABELS,fontsize=9);ax.set_yticks(range(5),R_LABELS,fontsize=9)
                ax.set_xlabel('Extinction-corrected g - r',fontsize=10)
                if w==0:ax.set_ylabel('Observed r magnitude',fontsize=10)
                ax.grid(False)
                for rb in range(5):
                    for cb in range(5):
                        value=data[rb,cb]; cell=(rb*5+cb)*3+w
                        text='—' if not np.isfinite(value) else f'{100*value:.0f}%' if percent else str(int(value))
                        rgba=palette(norm(value)) if np.isfinite(value) else (.87,.87,.87,1)
                        luminance=.2126*rgba[0]+.7152*rgba[1]+.0722*rgba[2]
                        ax.text(cb,rb,text,ha='center',va='center',fontsize=10,color='black' if luminance>.5 else 'white')
                        plotted.append(dict(figure=figure,panel=row_title,cell=cell,r_bin=rb,color_bin=cb,wise_bin=w,
                            value=None if not np.isfinite(value) else float(value),display=text))
                panel_count+=1
        cax=fig.add_axes([.915,.23,.015,.53]); bar=fig.colorbar(im,cax=cax)
        bar.set_label(colorbar_label,fontsize=10)
        if ticks is not None:bar.set_ticks(ticks)
        fig.suptitle(title,y=.97,fontsize=18);fig.text(.5,.91,subtitle,ha='center',fontsize=10)
        fig.text(.5,.085,footer,ha='center',fontsize=10)
        fig.text(.5,.035,'All native DR11 TYPE=PSF. Fixed lower-inclusive cells; zero is observed zero, not missing data.',ha='center',fontsize=10)
        fig.canvas.draw(); text=[t.get_text() for t in fig.findobj(matplotlib.text.Text)];labels.extend(text)
        assert not any(any('\u4e00'<=c<='\u9fff' for c in t) for t in text)
        for ext in ['png','pdf']:
            path=out/(figure+'.'+ext);fig.savefig(path,dpi=300);outputs[str(path.relative_to(ROOT))]=sha(path)
        plt.close(fig)

    if a.stage=='photometry':
        assert not done['references_read']
        broad=[int(r['application_n']) for r in cells]; candidate=[int(r['application_candidate']) for r in cells]
        assert sum(broad)==26000 and sum(candidate)==3540 and all(0<=c<=b for c,b in zip(candidate,broad))
        maximum=max(broad);norm=FuncNorm((np.log1p,np.expm1),vmin=0,vmax=maximum)
        ticks=[t for t in [0,1,10,100,1000,10000] if t<=maximum]
        panels([broad,candidate],['Application design sample (N=26000)','Candidate admission (N=3540)'],norm,'viridis',
            'photometric_occupancy','Photometric occupancy of the fixed observable cells',
            'Raw sampled-object counts; shared log(1 + count) color scale. No reference outcomes read.',
            'Candidate gate: observed r < 20, grz errors < 0.1 and frozen QRF support. Occupancy is not calibration or physical density.',
            'Objects in cell',ticks)
    else:
        assert done['candidate_n']==3540
        candidate=[int(r['candidate_n']) for r in cells]; assert sum(candidate)==3540
        references=tables['reference_distance_cells.csv'];expected=2*3*2*75*6
        keys=[(r['role'],r['source'],r['quality'],r['cell'],r['distance_bin']) for r in references]
        assert len(references)==expected and len(set(keys))==expected
        arrays=[]
        for name in ['astronn','specdis','spdist']:
            rows=[r for r in references if r['role']=='new_sky_field' and r['source']==name and r['quality']=='precise']
            assert len(rows)==450
            values=np.zeros(75,dtype=int)
            for row in rows:
                count=int(row['n']);assert count>=0;values[int(row['cell'])]+=count
            arrays.append(values)
        maximum=max(1,max(int(v.max()) for v in arrays))
        panels(arrays,['AstroNN (Gaia-trained teacher)','SpecDis source-qualified','SPDist source-qualified'],Normalize(0,maximum),'viridis',
            'precise_newsky_references','Precise new-sky reference availability by source',
            'Candidate-admitted field objects only; counts sum distance bins within each source. Correlated sources are not pooled.',
            'Zero means no qualified precise reference. Reference availability alone does not establish absolute distance accuracy.',
            'Precise reference objects')
        gap=[float(r['mean_distance_mass_without_precise_newsky_reference']) if n else np.nan for r,n in zip(cells,candidate)]
        assert all(not n or (np.isfinite(v) and 0<=v<=1+1e-8) for v,n in zip(gap,candidate))
        flag=any(r['cohort_numerical_passed']!='True' for r in cells)
        panels([gap],['Main candidate'],Normalize(0,1),'magma','missing_reference_mass',
            'Distance probability in bins without a precise new-sky reference',
            'Cell means for the main candidate; — means no candidate object. '+('Original numerical flags retained.' if flag else 'Numerical gates passed; scientific calibration remains separate.'),
            'Same observable cell and distance bin; any precise source makes a bin nonempty. The complement is not validated probability.',
            'Mean posterior mass',ticks=[0,.25,.5,.75,1],percent=True)
    csv_rows(meta/'plotted.csv',plotted);outputs[str((meta/'plotted.csv').relative_to(ROOT))]=sha(meta/'plotted.csv')
    for path in [ROOT/'scripts/v3_1_domain_figures.py',ROOT/'scripts/photdis_figure_style.py']:
        inputs[str(path.relative_to(ROOT))]=sha(path)
    save_json(meta/'complete.json',dict(inputs=inputs,outputs=outputs,stage=a.stage,panels=panel_count,plotted_rows=len(plotted),
        english_text_verified=True,visual_review_pending=True,scientific_domain_adjudication_complete=False,
        scope='Authenticated compact-table rendering. Occupancy and reference availability do not certify scientific use domains.'))
    print('RENDERED',a.stage,panel_count,'panels',len(plotted),'cells',flush=True)


if __name__=='__main__':main()
