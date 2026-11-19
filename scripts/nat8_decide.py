#!/usr/bin/env python3
"""Apply the registered continuation rule, retaining failed and paired criteria."""
from nat8_common import *
from nat8_evaluate import ci
from photdis.stage1 import require_compute


def main():
    require_compute();d=folder('decision');summary=list(csv.DictReader((OUT/'evaluate/summary.csv').open()));paired=list(csv.DictReader((OUT/'evaluate/paired.csv').open()));metrics=json.loads((OUT/'evaluate/family_metrics.json').read_text());params=json.loads((OUT/'fit/parameters.json').read_text());gaps=[];widths=[];criteria=[]
    def row(s,m,phase,cohort):return next(r for r in summary if r['system']==s and r['method']==m and r['phase']==phase and r['cohort']==cohort and r['domain']=='all')
    def boot(r,key):return np.array(metrics[r['family']]['bootstrap'][key])
    for s in C['systems']:
        for cohort in ('primary','v5','common'):
            for a,b in [('E','CalE'),('CalE','QRF')]+([('CalE','CalE_marg')] if s!='M5' else []):
                records=[row(s,m,p,cohort) for m in (a,b) for p in ('lower_MS','turnoff')];r0,r1,r2,r3=records;ga=float(r1['median_mu_residual'])-float(r0['median_mu_residual']);gb=float(r3['median_mu_residual'])-float(r2['median_mu_residual']);dg=(boot(r3,'center')-boot(r2,'center'))-(boot(r1,'center')-boot(r0,'center'));lo,hi=ci(dg);gaps.append(dict(system=s,cohort=cohort,first=a,second=b,first_gap=ga,second_gap=gb,delta=gb-ga,delta_lo=lo,delta_hi=hi,absolute_gap_reduced=abs(gb)<abs(ga)))
                for phase in ('all','lower_MS','turnoff'):
                    ar=row(s,a,phase,cohort);br=row(s,b,phase,cohort)
                    for metric in ('W68','W95'):
                        ratio=float(br[metric])/float(ar[metric]);lo,hi=ci(boot(br,metric)/boot(ar,metric));widths.append(dict(system=s,cohort=cohort,phase=phase,first=a,second=b,metric=metric,ratio=ratio,lo=lo,hi=hi))
    csv_rows(d/'phase_gap_changes.csv',gaps);csv_rows(d/'width_ratios.csv',widths)
    def condition(s,cohort):
        for phase in ('lower_MS','turnoff'):
            r=row(s,'CalE',phase,cohort);criteria.append(dict(system=s,criterion=phase+'_absolute_scale',value=float(r['median_mu_residual']),pass_condition='absolute value <=0.10mag',passed=abs(float(r['median_mu_residual']))<=.1))
        pair=next(r for r in paired if r['system']==s and r['cohort']==cohort and r['phase']=='all' and r['domain']=='all' and r['first']=='E' and r['second']=='CalE')
        for key,positive in [('raw_logscore_lo',True),('raw_CRPS_hi',False)]:
            value=float(pair[key]);criteria.append(dict(system=s,criterion=key,value=value,pass_condition='>0' if positive else '<0',passed=value>0 if positive else value<0))
        a0=row(s,'E','lower_MS',cohort);a1=row(s,'E','turnoff',cohort);b0=row(s,'CalE','lower_MS',cohort);b1=row(s,'CalE','turnoff',cohort);ga=float(a1['median_mu_residual'])-float(a0['median_mu_residual']);gb=float(b1['median_mu_residual'])-float(b0['median_mu_residual']);criteria.append(dict(system=s,criterion='phase_gap_reduced',value=abs(gb)-abs(ga),pass_condition='<0',passed=abs(gb)<abs(ga)))
        for phase in ('all','lower_MS','turnoff'):
            ratio=float(row(s,'CalE',phase,cohort)['W68'])/float(row(s,'E',phase,cohort)['W68']);criteria.append(dict(system=s,criterion=phase+'_W68_ratio',value=ratio,pass_condition='<=1.5',passed=ratio<=1.5))
        return all(r['passed'] for r in criteria if r['system']==s)
    m5=condition('M5','calibration_region');external={s:condition(s,'primary') for s in ('M30','M13')};gate=m5 and any(external.values());csv_rows(d/'continuation_criteria.csv',criteria)
    decision=dict(M5_gate=m5,external_gates=external,science_gate=gate,prospective_evaluation_performed=False,candidates='both inventories delivered; NGC5053 contains historically exposed members and no isolated kinematic component; NGC1851 prior empty availability inventory, no residual opened',decision='Two-parameter hypothesis insufficient as a universal old-population calibration; M5 cross-prediction benefit and partial transfer improvements retained.',scope='M5 known-population, known-age/FeH, native-photometry conditional calibration only. M30/M13 do not achieve both-phase absolute-scale target. No universal field-star correction or production replacement.',gate_note='M5 uses all525 E-supported predeclared calibration-region members. External combined tests use all primary E-supported objects; both phases tested separately. M5 internal gain is insufficient to authorize independent evaluation when no external system passes.',next_science='Do not add a third parameter or per-system shift in this task. Existing evidence motivates population/phase information and predictive tail/branch representation as future separately scoped questions. All original forecasts and negative results retained.')
    assert not gate,'Continuation passed: freeze one prospective system before any prediction; this script deliberately does not open it.'
    save_json(d/'science_decision.json',decision);a=np.array(params['full']['point']['a']);csv_rows(d/'luminosity_scale.csv',[dict(native_node_gr=c,delta_M=float(a[0]+a[1]*(c-.3)),delta_mu=float(-a[0]-a[1]*(c-.3)),distance_factor=float(10**((-a[0]-a[1]*(c-.3))/5))) for c in (.2,.3,.5,.7)])
    complete(d,{'evaluation':sha(OUT/'evaluate/complete.json'),'audit':sha(OUT/'audit_detail/complete.json')},registered_gate_applied_without_retuning=True,prospective_prediction_opened=False);print('Decision',decision,flush=True)


if __name__=='__main__':main()
