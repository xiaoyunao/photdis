#!/usr/bin/env python3
"""Normalize versioned BV21 Table2 and exact published member-system aliases.

Usage: python scripts/kgiant_bv21_prepare.py PAPER_HTML MEMBER_CENTERS
Input HTML is https://arxiv.org/html/2105.09526v2, not the live author table.
"""
import hashlib,json,re,sys
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup

METHODS=['gaia_parallax','gaia_kinematic','hst_kinematic','subdwarf','star_count','literature','combined']

def key(name):
    aliases={'ESO452_SC11':'ESO 452','ESO280_SC06':'ESO 280','Djorg_1':'Djor 1','Djorg_2_ESO456':'Djor 2',
             'Ryu_059_RLGC1':'RLGC 1','Ryu_879_RLGC2':'RLGC 2','Ton2_Pismis26':'Ton 2',
             'VVV_CL001':'VVV-CL001'}
    name=aliases.get(name,name)
    m=re.match(r'(NGC|IC|Terzan|Pal|AM|Arp|BH|HP|Lynga|FSR|Laevens|Liller|Mercer|Rup|UKS|Whiting)[ _]+(\d+)',name)
    if m:name=m[1].replace('Terzan','Ter')+' '+m[2]
    return re.sub('[^a-z0-9]','',name.lower())

def value(cell):
    math=cell.find('math');raw=math.get('alttext') if math else cell.get_text(' ',strip=True)
    if raw=='-':return None
    numbers=[float(v) for v in re.findall(r'\d+\.\d+',raw)]
    if len(numbers)==2:d,e=numbers;lo=hi=e
    elif len(numbers)==3:d,hi,lo=numbers
    else:raise ValueError(raw)
    return dict(distance_kpc=d,minus_kpc=lo,plus_kpc=hi,raw=raw)

def main():
    html,centers=map(Path,sys.argv[1:]);s=BeautifulSoup(html.read_text(),'html.parser');rows={};duplicate=[]
    for t in s.find_all('table'):
        if 'GEDR3 parallax dist.' not in t.get_text():continue
        for tr in t.find_all('tr'):
            cells=tr.find_all('td',recursive=False)
            if len(cells)!=9 or cells[0].get_text(strip=True)=='Name':continue
            name=cells[0].get_text(' ',strip=True)
            r=dict(name=name,methods={m:value(c) for m,c in zip(METHODS,cells[1:8])},n_publication=int(cells[8].get_text()))
            d=r['methods']['combined'];r['mu']=10+5*np.log10(d['distance_kpc'])
            r['mu_minus']=5*np.log10(d['distance_kpc']/(d['distance_kpc']-d['minus_kpc']))
            r['mu_plus']=5*np.log10((d['distance_kpc']+d['plus_kpc'])/d['distance_kpc'])
            r['displayed_methods']=[m for m in METHODS[:-1] if r['methods'][m]]
            r['dependency_flags']=dict(gaia_possible=True,stellar_model_possible=bool(r['methods']['literature'] or r['methods']['subdwarf']),
               dynamical_model=bool(r['methods']['gaia_kinematic'] or r['methods']['hst_kinematic'] or r['methods']['star_count']),
               shared_calibration=True,method_covariance='not supplied; do not independently combine columns')
            if name in rows:assert rows[name]==r;duplicate.append(name)
            rows[name]=r
    assert len(rows)==162
    cat=np.genfromtxt(centers,names=True,dtype=None,encoding='utf8');assert len(cat)==170
    bykey={key(k):k for k in rows};mapping=[]
    for i,c in enumerate(cat):
        name=str(c['Name']);ref=bykey.get(key(name));mapping.append(dict(member_name=name,group=100000+i,reference_name=ref,ra=float(c['ra']),dec=float(c['dec'])))
    matched=[m['reference_name'] for m in mapping if m['reference_name']]
    assert len(matched)==len(set(matched))
    result=dict(source_url='https://arxiv.org/html/2105.09526v2',doi='10.1093/mnras/stab1474',table='Table2 (including continuation pages)',
       source_sha256=hashlib.sha256(html.read_bytes()).hexdigest(),member_centers_sha256=hashlib.sha256(centers.read_bytes()).hexdigest(),
       identical_html_duplicates=duplicate,rows=list(rows.values()),membership_mapping=mapping,
       reference_without_membership=sorted(set(rows)-set(matched)),member_without_reference=[r['member_name'] for r in mapping if r['reference_name'] is None],
       caveat='Blank parallax cells only omit individual low-SNR distance summaries; weak parallaxes may still enter combined distance. Literature methods mix calibrations and can share Gaia or stellar models. Columns are not independent datasets.')
    Path('manifests/kgiant_bv21_reference_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print('unique distances',len(rows),'matched systems',len(matched),'unmatched members',result['member_without_reference'],'no members',result['reference_without_membership'])
if __name__=='__main__':main()
