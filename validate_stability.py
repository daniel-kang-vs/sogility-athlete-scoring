"""Reproducible synthetic stress probes. No athlete data or predictive claims."""
import json
from pathlib import Path
import numpy as np
from tests.test_scoring import fixture,rps
from final_aps import run_aps,aps_from_benchmarks


def player_scores(df):
    return rps(df)[-1]['overall_player_rps'].set_index('first_name').overall_cohort_rps

base=fixture(repeats=1)
a=player_scores(base)
noisy=base.copy()
noisy['score']*=1+np.random.default_rng(42).normal(0,.01,len(noisy))
delta=(player_scores(noisy)-a).abs()
aps_base=run_aps(base)['players'].set_index('first_name').final_aps_score
aps_noisy=run_aps(noisy)['players'].set_index('first_name').final_aps_score
aps_delta=(aps_noisy-aps_base).abs()
outlier=base.copy()
outlier.loc[outlier.first_name=='synthetic_M_39','score']=1e9
outlier_delta=(player_scores(outlier)-a).abs().drop('synthetic_M_39')
# Crossing the hard 30-observation boundary can change the reference population.
boundary=fixture(n=29,repeats=1)
boundary.loc[boundary.gender_clean=='F','score']+=100
before=player_scores(boundary)
extra=boundary.iloc[[0]].copy()
extra['first_name']='synthetic_extra'
after=player_scores(__import__('pandas').concat([boundary,extra],ignore_index=True))
jump=(after-before).abs().dropna()
constant=base.copy();constant['score']=10
rows,*_,diagnostics=rps(constant)
try:
    run_aps(constant)
    aps_status="unexpected success"
except ValueError as exc:
    aps_status=str(exc)
result={
 'scope':'Synthetic functional and sensitivity checks, not game-performance validation',
 'seed':42,
 'one_percent_raw_score_noise':{'mean_absolute_rps_change':float(delta.mean()),'max_absolute_rps_change':float(delta.max())},
 'one_percent_raw_score_noise_aps':{'mean_absolute_aps_change':float(aps_delta.mean()),'max_absolute_aps_change':float(aps_delta.max())},
 'single_extreme_outlier_other_players':{'mean_absolute_rps_change':float(outlier_delta.mean()),'max_absolute_rps_change':float(outlier_delta.max())},
 '29_to_30_observations_stress_case':{'max_absolute_rps_change':float(jump.max()),'note':'Age distributions intentionally differ; no universal stability guarantee.'},
 'constant_distribution':{'rps_missing_attempts':int(rows.rps_cohort_sigmoid.isna().sum()),'rps_missing_player_scores':int(diagnostics['overall_player_rps'].overall_cohort_rps.isna().sum()),'aps_status':aps_status,'note':'Undefined spans remain missing through RPS aggregation. APS raises an explicit validation error.'},
 'aps_lower_is_better_anchor_probe':{'raw_q50':50,'returned_score':aps_from_benchmarks(50,30,50,70,90,False),'note':'Reversing non-symmetric quantile anchors maps raw median to 70, not 50. Requires a deliberate anchor-policy review.'}
}
Path('validation/results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2))
