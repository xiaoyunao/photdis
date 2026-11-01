import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_giant_information_driver import synthetic_table
from giant_candidate_run import inner_thresholds
from photdis.giant_candidate import observable_masks, population_categories, threshold_at_fpr, mechanism_features
from photdis.giant_information import BANDS, ERRORS


def test_application_has_no_temperature_or_w2_gate():
    t=synthetic_table();cfg=json.loads(Path('configs/giant_information_v1.json').read_text())
    t['desi_teff'][:30]=8000;t['label'][:30]=-1
    t['w20'][:90]=np.nan;t['sigma_w2'][:90]=np.nan;t['w2_clean'][:90]=False
    masks=observable_masks(t,cfg)
    assert masks['w1'].sum()==180 and masks['common5'].sum()==90
    t['class_evidence']=np.full(180,'other_star_or_transition',dtype='U32')
    t['redrock_nonstellar_evidence']=np.zeros(180,bool)
    t['redrock_nonstellar_evidence'][30]=True;t['label'][30]=-1
    categories=population_categories(t)
    assert (categories[:30]=='hot_qualified_unclassified').all()
    assert categories[30]=='raw_nonstellar_unassessed'


def test_low_fpr_ties_and_zero_count_uncertainty():
    score=np.r_[np.full(2,.9),np.linspace(0,.8,98)]
    rule=threshold_at_fpr(score,.01)
    assert rule['calibration_false_positives']==0
    assert rule['calibration_fpr']['interval95'][1]>0
    assert not threshold_at_fpr(score,.001)['resolution_ok']
    assert threshold_at_fpr([],.01)['threshold'] is None


def test_inner_threshold_does_not_use_outer_labels(tmp_path):
    t=synthetic_table();t['field_index']=np.repeat(np.arange(6),30)
    t['label']=np.tile(np.r_[np.zeros(15,int),np.ones(15,int)],6)
    cfg=json.loads(Path('configs/giant_information_v1.json').read_text())
    cfg.update(min_train_giants=10,min_train_dwarfs=10,band_draws=2)
    new={'inner_field_groups':3,'seed':17,'target_fpr':[.01,.001]}
    bands=np.column_stack([t[k] for k in BANDS]);errors=np.column_stack([t[k] for k in ERRORS])
    base=np.ones(180,bool)
    first,logs=inner_thresholds(t,bands,errors,base,base,cfg,new,0,tmp_path)
    assert all(0 not in x['held_fields']+x['training_fields'] for x in logs)
    t['label'][:30]=-1
    second,_=inner_thresholds(t,bands,errors,base,base,cfg,new,0,tmp_path)
    assert first==second
    assert all(x['scored'] for x in logs)


def test_mechanism_conditions_on_spectrum_preserves_color_covariance():
    t=synthetic_table();bands=np.column_stack([t[k] for k in BANDS]);errors=np.column_stack([t[k] for k in ERRORS])
    x,v=mechanism_features(t,bands,errors,2)
    assert x.shape==(180,5) and np.all(v[:,:3,:]==0)
    np.testing.assert_allclose(v[:,3,4],-t['sigma_r']**2)
    old=x.copy();t['feh_corrected'][:]=100
    np.testing.assert_array_equal(mechanism_features(t,bands,errors,2)[0],old)


def test_population_aggregation_retains_unknown_and_nonstellar():
    from photdis.giant_candidate import selection_summary
    from giant_candidate_summarize import aggregate
    t=synthetic_table();t['class_evidence']=np.where(t['label']==1,'conservative_giant','conservative_dwarf')
    t['redrock_nonstellar_evidence']=np.zeros(180,bool)
    t['label'][0]=-1;t['class_evidence'][0]='unknown'
    t['label'][1]=-1;t['redrock_nonstellar_evidence'][1]=True
    rows=np.arange(60);score=np.ones(60)*.8;outside=np.zeros(60,bool)
    summ=selection_summary(t,rows,score,outside,np.ones(180,bool),.5)
    assert summ['strata']['all']['unknown']['success']==1
    assert summ['strata']['all']['raw_nonstellar_unassessed']['success']==1
    cfg=json.loads(Path('configs/giant_information_v1.json').read_text())
    result={'parent_config':cfg,'config':{'target_fpr':[.01,.001],'mechanism_panels':['optical']},
            'application':[{'field_index':0,'training_fields':[1,2],'panel':'w1','n_reference':60,
                            'models':{n:{'fixed_half':summ,'inside_half':summ} for n in ['A','B']}}], 'mechanism':[]}
    output=aggregate(result)
    assert all(output['validation'].values())
    assert output['panels']['w1']['models']['B']['fixed_half']['all']['reference_total']['n']==60
