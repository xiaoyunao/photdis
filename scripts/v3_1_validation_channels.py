"""Same frozen-ID MWS/VAC channels; callable only after final evidence freeze."""
import numpy as np
from photdis.reference_v1_4 import unique_join,pair_keys
from photdis.assembly_v1_4 import representative_rows
from photdis.source_evidence_v1_4 import vac_evidence
from photdis.v3.data import specdis_qualification
from v3_0_prepare import projected,gather


def join_vac(target,source,raw):
    index=unique_join(pair_keys(target,source),pair_keys(raw['targetid'],raw['source_id']))
    index[(target<=0) | (source<=0)]=-1
    fallback=(index<0) & (source>0)
    index[fallback]=unique_join(source[fallback],raw['source_id'])
    index[source<=0]=-1
    return index,fallback & (index>=0)


def multiplicity(query,catalog):
    key=np.sort(catalog)
    return np.searchsorted(key,query,'right')-np.searchsorted(key,query,'left')


def associate_mws(table,base):
    """Reuse frozen source quality definitions without choosing a favored teacher."""
    n=len(table['identity']); ids=table['source_id']
    keys=['identity','source_id','rvs_primary','rvs_sn_r','rvs_targetid','rvs_feh','rvs_feh_err','rvs_teff','rvs_logg']
    source=projected(base/'sources.npz',keys)
    representative=representative_rows(source['identity'],source['rvs_primary'],source['rvs_sn_r'])
    index=unique_join(ids,source['source_id'][representative]); rows=np.full(n,-1,'i8')
    matched=(index>=0) & (ids>0); rows[matched]=representative[index[matched]]
    raw=gather(base/'source_evidence.npz',rows,['class_label','rvs_quality_reason','rvs_parameter_quality',
        'redrock_nonstellar_evidence','class_identity_conflict','target_identity_conflict'])
    conflict=raw['class_identity_conflict'] | raw['target_identity_conflict']
    result=dict(mws_source_match=matched,mws_raw_source_row=rows,mws_identity_conflict=conflict,
        mws_class_label=raw['class_label'],mws_parameter_quality=raw['rvs_parameter_quality'],
        mws_nonstellar=raw['redrock_nonstellar_evidence'],mws_quality_reason=raw['rvs_quality_reason'])
    result['mws_source_matches']=multiplicity(ids,source['source_id'][representative])
    result['mws_class_label'][~matched | conflict]=-1
    result['mws_parameter_quality'][~matched | conflict]=False
    target=np.zeros(n,'i8'); target[matched]=source['rvs_targetid'][rows[matched]]
    result['mws_targetid']=target
    for key in ['rvs_feh','rvs_feh_err','rvs_teff','rvs_logg']:
        values=np.full(n,np.nan); values[matched]=source[key][rows[matched]]; result[key]=values
    audit=dict(raw_MWS_matches=int(matched.sum()),identity_conflicts=int(conflict.sum()))
    audit['ambiguous_MWS_source_ids']=int(((ids>0) & (result['mws_source_matches']>1)).sum())
    for kind,columns in [('specdis',['targetid','source_id','dist','disterr','binary_flag']),
        ('spdist',['targetid','source_id','g0','mg_50','mg_16','mg_84','e_gmag','flag_good','qflag_input'])]:
        catalog=projected(base/(kind+'_raw.npz'),columns)
        j,fallback=join_vac(target,ids,catalog); found=j>=0; q=np.flatnonzero(found)
        result[kind+'_match']=found; result[kind+'_source_only_join']=fallback; result[kind+'_raw_row']=j
        counts=multiplicity(ids,catalog['source_id']); result[kind+'_source_matches']=counts
        audit[kind+'_raw_matches']=int(found.sum()); audit[kind+'_source_only_matches']=int(fallback.sum())
        audit[kind+'_unresolved_duplicate_source']=int(((ids>0) & ~found & (counts>1)).sum())
        if kind=='specdis':
            distance=np.full(n,np.nan); error=distance.copy(); flag=np.full(n,-1,'i8')
            distance[q]=catalog['dist'][j[q]]; error[q]=catalog['disterr'][j[q]]; flag[q]=catalog['binary_flag'][j[q]]
            spectral=matched & ~conflict & ((raw['rvs_quality_reason'].astype('i8') & ~(1<<4))==0)
            valid,sigma=specdis_qualification(distance,error,spectral)
            result.update(specdis_valid=valid,specdis_distance=distance,specdis_sigma_mu=sigma,
                specdis_raw_distance_error=error,specdis_raw_binary_flag=flag)
        else:
            local={key:value[j[q]] for key,value in catalog.items()}; artificial=np.arange(len(q),dtype='i8')+1
            local['source_id']=artificial; local['targetid']=artificial
            values=vac_evidence(artificial,artificial,local,'spdist')
            for key in ['spdist_valid','spdist_distance','spdist_sigma_mu']:
                out=np.full(n,False if key.endswith('valid') else np.nan,dtype=bool if key.endswith('valid') else float)
                out[q]=values[key]; result[key]=out
            result['spdist_valid'][conflict]=False
            # Preserve source flags/measurements even when qualification fails.
            for key in columns[2:]:
                values=catalog[key]; fill='' if values.dtype.kind in 'SU' else False if values.dtype.kind=='b' else np.nan if values.dtype.kind=='f' else -1
                out=np.full(n,fill,dtype=values.dtype); out[q]=values[j[q]]; result['spdist_raw_'+key]=out
        audit[kind+'_qualified']=int(result[kind+'_valid'].sum())
    audit['specdis_qualified_non1_binary_flag']=int((result['specdis_valid'] & (result['specdis_raw_binary_flag']!=1)).sum())
    audit['mws_class_labels']={str(c):int((result['mws_class_label']==c).sum()) for c in [-1,0,1,2]}
    return result,audit
