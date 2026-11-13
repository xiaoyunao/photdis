"""Product-matched relative reference screens, never calibration certificates."""
import numpy as np
from .validation_report import paired_field_statistic

BASELINE='candidate/zero'
MARGINS={'log_score':(-.02,True),'severe':(.01,False),'tail':(.01,False),
         'reference_agreement68':(-.01,True),'reference_agreement95':(-.01,True)}


def assess_pair(candidate,baseline,sky,candidate_ok,baseline_ok,metric):
    a,b,sky,ao,bo=map(np.asarray,(candidate,baseline,sky,candidate_ok,baseline_ok))
    if a.ndim!=1 or not all(x.shape==a.shape for x in [b,sky,ao,bo]):raise ValueError('Aligned same-identity reference pairs required')
    if metric not in MARGINS:raise ValueError('Unregistered reference screen')
    # Inputs already passed the independently defined reference qualification.
    # A NaN score here is a failure, not an absent reference to remove.
    limit,lower=MARGINS[metric]; invalid=np.isnan(a) | np.isnan(b)
    summary=paired_field_statistic(np.where(np.isnan(a),np.inf,a),np.where(np.isnan(b),np.inf,b),sky)
    failed=int((~(ao.astype(bool) & bo.astype(bool))).sum())
    sufficient=summary['n']>=20 and summary['cells']>=2
    finite=summary['mean'] is not None and np.isfinite(summary['mean'])
    passes=bool(summary['mean']>=limit if lower else summary['mean']<=limit) if finite else None
    interval='unavailable'
    lo,hi=summary['lo95'],summary['hi95']
    if lo is not None and hi is not None:
        if lower:interval='within_margin' if lo>=limit else 'beyond_margin' if hi<limit else 'crosses_margin'
        else:interval='within_margin' if hi<=limit else 'beyond_margin' if lo>limit else 'crosses_margin'
    if not summary['n']:status='no_qualified_reference'
    elif failed or not finite:status='numerical_or_nonfinite_failure_retained'
    elif not sufficient:status='insufficient_references_or_sky'
    elif not passes:status='mean_guard_failed'
    else:status='mean_guard_passed'
    return dict(**summary,limit=limit,lower_bound=lower,paired_numerical_failed=failed,qualified_nan_pairs=int(invalid.sum()),
        sufficient=sufficient,mean_within_margin=passes,sky_interval_assessment=interval,
        assessment=status,absolute_calibration_established=False)


def summarize_cell(checks):
    statuses=[r['assessment'] for r in checks]
    failed=statuses.count('mean_guard_failed'); passed=statuses.count('mean_guard_passed')
    numerical=statuses.count('numerical_or_nonfinite_failure_retained')
    if numerical:status='numerical_failures_retained'
    elif failed:status='relative_reference_guard_failure'
    elif passed:status='partial_relative_reference_support'
    else:status='unvalidated_sparse_or_missing_reference'
    return dict(relative_assessment=status,metrics_screened=len(checks),sufficient_mean_failures=failed,
        sufficient_mean_passes=passed,numerical_metric_failures=numerical,
        sparse_or_missing_metrics=sum(x in ['no_qualified_reference','insufficient_references_or_sky'] for x in statuses),
        sky_intervals_within_margin=sum(r['sky_interval_assessment']=='within_margin' and r['sufficient'] and not r['paired_numerical_failed'] for r in checks),
        absolute_calibration_established=False)
