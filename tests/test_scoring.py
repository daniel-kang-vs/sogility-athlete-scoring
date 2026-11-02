import unittest
import numpy as np
import pandas as pd
from sigmoid_cohort import compare_global_vs_cohort_sigmoid_rps
from final_aps import run_aps, aps_from_benchmarks, assign_aps_band, get_recent_weights


def fixture(n=40, repeats=3):
    rows=[]
    for gender,count in [('M',n),('F',10)]:
        for i in range(count):
            for attempt in range(repeats):
                rows.append(dict(first_name=f'synthetic_{gender}_{i}',last_name='test',
                    gender_clean=gender,age_group=1,cohort=f'1_{gender}',
                    category_assessment='Circuit Pass find 1',score=10+i+attempt,
                    timestamp=pd.Timestamp('2026-01-01')+pd.Timedelta(days=attempt),
                    higher_is_better=True))
    return pd.DataFrame(rows)


def rps(df, **kw):
    return compare_global_vs_cohort_sigmoid_rps(df, **kw)


class ScoringTests(unittest.TestCase):
    def test_rps_range_and_monotonicity(self):
        rows,*_,d=rps(fixture())
        self.assertTrue(rows.rps_cohort_sigmoid.between(30,99).all())
        self.assertTrue(d['monotonicity_check'].empty)

    def test_rps_recency_weights(self):
        for n,expected in [(1,[1]),(2,[2/3,1/3]),(3,[.5,.3,.2]),(4,[.5,.3,.2,0])]:
            rows,*_=rps(fixture(repeats=n))
            one=rows[rows.first_name=='synthetic_M_0'].sort_values('attempt_recency_rank')
            np.testing.assert_allclose(one.attempt_weight,expected)

    def test_rps_fallback_sources(self):
        rows,*_=rps(fixture(repeats=1))
        self.assertEqual(set(rows[rows.gender_clean=='M'].rps_benchmark_source),{'age_gender_cohort'})
        self.assertEqual(set(rows[rows.gender_clean=='F'].rps_benchmark_source),{'same_age_group_gender_combined'})

    def test_rps_weighted_aggregation(self):
        rows,*_,d=rps(fixture())
        one=rows[rows.first_name=='synthetic_M_0']
        expected=(one.rps_cohort_sigmoid*one.attempt_weight).sum()
        actual=d['player_assessment_rps'].query("first_name == 'synthetic_M_0'").assessment_cohort_rps.iloc[0]
        self.assertAlmostEqual(expected,actual)

    def test_rps_row_order_invariance(self):
        a=rps(fixture())[-1]['overall_player_rps'].sort_values('first_name')
        b=rps(fixture().sample(frac=1,random_state=7))[-1]['overall_player_rps'].sort_values('first_name')
        np.testing.assert_allclose(a.overall_cohort_rps,b.overall_cohort_rps)

    def test_rps_lower_is_better_and_log(self):
        rows,*_=rps(fixture(),skill_type_map={'Circuit Pass find 1':'Log'},good_direction_map={'Circuit Pass find 1':'Lo'})
        one=rows[rows.cohort=='1_M'].sort_values('score')
        self.assertTrue((one.rps_cohort_sigmoid.diff().dropna()<=1e-10).all())

    def test_aps_anchors_and_bands(self):
        for value in [30,50,70,90]:
            self.assertEqual(aps_from_benchmarks(value,30,50,70,90),value)
        self.assertEqual([assign_aps_band(v) for v in [29,30,50,70,90]],
                         ['Foundation','Developing','Approaching','Strong','Elite'])

    def test_aps_direction_and_bounds(self):
        for higher in [True,False]:
            ys=np.array([aps_from_benchmarks(v,30,50,70,90,higher) for v in range(-100,201)])
            self.assertTrue(((ys>=0)&(ys<=100)).all())
            self.assertTrue((np.diff(ys)*(1 if higher else -1)>=0).all())

    def test_aps_weights(self):
        for n in [1,2,3]: self.assertAlmostEqual(sum(get_recent_weights(n)),1)
        np.testing.assert_allclose(get_recent_weights(2),[2/3,1/3])

    def test_aps_pipeline_and_benchmark_grain(self):
        result=run_aps(fixture())
        b=result['benchmarks']
        self.assertEqual(b.query("benchmark_cohort == '1'").n_benchmark_distribution.iloc[0],50)
        self.assertTrue(result['players'].final_aps_score.between(0,100).all())
        self.assertEqual(len(result['players']),50)
        self.assertEqual(result['assessments'].n_recent_scores_used.unique().tolist(),[3])

    def test_aps_row_order_invariance(self):
        a=run_aps(fixture())['players'].sort_values('first_name')
        b=run_aps(fixture().sample(frac=1,random_state=7))['players'].sort_values('first_name')
        np.testing.assert_allclose(a.final_aps_score,b.final_aps_score)

    def test_constant_distribution_has_no_fabricated_rps(self):
        df=fixture();df['score']=10
        result=rps(df)[-1]['overall_player_rps']
        self.assertTrue(result.overall_cohort_rps.isna().all())

    def test_constant_distribution_aps_explicit_error(self):
        df=fixture();df['score']=10
        with self.assertRaisesRegex(ValueError,'No valid APS scores'):
            run_aps(df)

if __name__=='__main__': unittest.main()
