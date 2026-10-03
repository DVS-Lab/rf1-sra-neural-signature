import numpy as np
import pandas as pd
from scipy.stats import binomtest
from reporting import performance, icc31, PERFORMANCE_COLUMNS, reliability
from signature_pilot import PREDICTION_COLUMNS


def test_binomial_ties_bootstrap_and_schema():
    summary = performance([2., -1., 0., 3.], 'synthetic', bootstrap_samples=500)
    assert list(summary) == PERFORMANCE_COLUMNS
    assert summary['n_correct'] == 2 and summary['ties'] == 1
    assert summary['binomial_p'] == binomtest(2,4).pvalue
    assert summary['mean_margin'] == 1
    assert summary == performance([2.,-1.,0.,3.], 'synthetic', bootstrap_samples=500)
    assert performance([], 'empty')['n'] == 0


def test_icc31_known_values_and_additive_run_offset():
    # Consistency ICC is invariant to systematic run offset (absolute agreement is not).
    a = np.array([[1,2], [2,3], [3,4], [4,5]], float)
    assert icc31(a) == 1
    b = np.array([[1,2],[2,1],[3,4],[4,3]], float)
    assert np.isclose(icc31(b), .6)
    b[:,1] += 10
    assert np.isclose(icc31(b), .6)
    assert np.isnan(icc31(np.ones((4,2))))


def test_prediction_schema():
    assert PREDICTION_COLUMNS == ['subject','fold','analysis','train_family','train_tasks','test_family','test_task','model_scope','positive_score','negative_score','paired_margin','correct']


def test_reliability_without_two_run_participants_is_unavailable():
    summary, pairs = reliability(pd.DataFrame(columns=PREDICTION_COLUMNS))
    assert summary['n'] == 0 and np.isnan(summary['pearson_r'])
    assert np.isnan(summary['icc_3_1']) and pairs.empty
    assert list(pairs.columns) == ['run1', 'run2']
