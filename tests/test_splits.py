import json
import pandas as pd
import pytest
from conftest import metadata
from make_split import make_split, make_folds, guard_from_split
from utils import sha256, PipelineError


def test_exact_reproducible_locked_split(cfg):
    data = metadata(350)
    first = make_split(cfg, data)
    assert (first.split == 'holdout').sum() == 50
    assert (first.split == 'development').sum() == 300
    assert not set(first[first.split == 'holdout'].subject) & set(first[first.split == 'development'].subject)
    before = sha256(cfg.output('work/splits/subject_split_v1.tsv'))
    reused = make_split(cfg, data.sample(frac=1))
    pd.testing.assert_frame_equal(first, reused)
    assert before == sha256(cfg.output('work/splits/subject_split_v1.tsv'))
    with pytest.warns(UserWarning, match='DANGER'):
        regenerated = make_split(cfg, data.sample(frac=1), True)
    pd.testing.assert_frame_equal(first, regenerated)


def test_missing_private_split_and_tampering_fail(cfg):
    make_split(cfg, metadata())
    path = cfg.output('work/splits/subject_split_v1.tsv')
    text = path.read_text()
    path.unlink()
    with pytest.raises(PipelineError, match='private split is missing'): make_split(cfg, metadata())
    path.write_text(text + '\n')
    with pytest.raises(PipelineError, match='hash'): make_split(cfg, metadata())


def test_cohort_drift_fails(cfg):
    make_split(cfg, metadata())
    with pytest.raises(PipelineError, match='Eligibility changed'): make_split(cfg, metadata(101))


def test_missing_age_explicit_fallback(cfg):
    frame = metadata()
    frame['age'] = float('nan')
    make_split(cfg, frame)
    assert json.loads(cfg.output('provenance/subject_split_v1.json').read_text())['method'] == 'flipangle_only'


def test_infeasible_holdout_fails(cfg):
    frame = metadata()
    frame['flip_angle'] = [str(i) for i in range(len(frame))]
    with pytest.raises(PipelineError, match='infeasible'): make_split(cfg, frame)
