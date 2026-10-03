import json
import nibabel as nib
import pandas as pd
import pytest
from conftest import metadata
from make_split import make_split
from review_qc_attrition import summarize, normalized_qc, review, POLICIES
from utils import PipelineError, sha256, write_tsv


def qc_rows(subjects):
    return pd.DataFrame([{'subject': subject, 'session': '01', 'task': task, 'run': run,
                         **{key: 'FALSE' for key in POLICIES['prior_four_metric_policy']}}
                        for subject in subjects for task, runs in
                        [('sharedreward', [1, 2]), ('trust', [1, 2]),
                         ('socialdoors', [1]), ('doors', [1])] for run in runs])


def test_qc_retained_runs_policies_missingness_and_overlap():
    inv = metadata(4)
    inv['sharedreward_runs'] = ['2', '1,2', '1,2', '1,2']
    split = inv.iloc[:3][['subject']].copy()
    split['split'] = ['development', 'development', 'holdout']
    qc = qc_rows(inv.subject)
    # Unretained SR run must not disqualify the first participant.
    qc.loc[(qc.subject == inv.subject[0]) & (qc.task == 'sharedreward') & (qc.run == 1), 'fd_mean_outlier'] = 'TRUE'
    # The second participant fails only the four-metric policy.
    qc.loc[(qc.subject == inv.subject[1]) & (qc.task == 'trust') & (qc.run == 2), 'tedana_outlier'] = 'TRUE'
    # The holdout participant has overlapping bad tasks, counted once jointly.
    qc.loc[(qc.subject == inv.subject[2]) & qc.task.isin(['trust', 'doors']), 'brain_coverage_outlier'] = 'TRUE'
    # Missing selected metric in an otherwise usable non-locked participant.
    qc.loc[(qc.subject == inv.subject[3]) & (qc.task == 'trust'), 'tsnr_outlier'] = ''
    rows, overlaps = summarize(inv, split, qc)
    def get(policy, partition, task_set='sharedreward_trust'):
        return rows[(rows.policy == policy) & (rows.partition == partition) & (rows.task_set == task_set)].iloc[0]
    assert get('tsnr_coverage_fd', 'development').pass_n == 2
    assert get('prior_four_metric_policy', 'development').pass_n == 1
    assert get('tsnr_coverage_fd', 'holdout', 'all_three_reward_paradigms').qc_outlier_n == 1
    assert get('tsnr_coverage_fd', 'outside_locked_cohort').unknown_qc_n == 1
    assert (rows[['pass_n', 'qc_outlier_n', 'unknown_qc_n', 'unavailable_input_n']].sum(axis=1) == rows.before_n).all()
    overlap = overlaps[(overlaps.policy == 'tsnr_coverage_fd') & (overlaps.partition == 'holdout') &
                       (overlaps.outlier_tasks == 'trust+doors')]
    assert overlap.n.iloc[0] == 1


def test_qc_duplicate_keys_fail():
    qc = qc_rows(['sub-fixture000'])
    with pytest.raises(PipelineError, match='Duplicate'):
        normalized_qc(pd.concat([qc, qc.iloc[:1]]))


def test_qc_review_preserves_split_sources_and_voxel_guard(cfg, monkeypatch):
    inv = metadata(55)
    inv['sharedreward_runs'] = '1,2'
    inv['eligible'] = True
    split = make_split(cfg, inv)
    write_tsv(cfg, 'work/inventory/task_availability.tsv', inv)
    path = cfg.repos['linux2']/cfg.paths['qc_table']
    path.parent.mkdir(parents=True, exist_ok=True)
    qc_rows(inv.subject).to_csv(path, sep='\t', index=False)
    before = {p: sha256(p) for p in [path, cfg.output('work/splits/subject_split_v1.tsv'),
                                    cfg.output('provenance/subject_split_v1.json')]}
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('QC review accessed image'))
    review(cfg)
    assert all(sha256(p) == value for p, value in before.items())
    public = cfg.output('results/aggregate/qc_review_retention.tsv').read_text()
    assert not any(subject in public for subject in inv.subject)
    provenance = json.loads(cfg.output('provenance/qc_review.json').read_text())
    assert provenance['exclusions_applied'] is False and provenance['holdout_scored'] is False
