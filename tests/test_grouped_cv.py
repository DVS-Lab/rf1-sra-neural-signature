import numpy as np
import pandas as pd
import pytest
from conftest import metadata
from make_split import make_split, make_folds, guard_from_split
from signature_pilot import fit_model, score_pair
from utils import PipelineError, sha256


def test_participant_pairs_share_fold_and_oof_guard(cfg):
    frame = metadata()
    split = make_split(cfg, frame)
    guard = guard_from_split(split)
    dev = frame[frame.subject.isin(guard.development)]
    subjects = sorted(guard.development)
    folds = make_folds(cfg, dev, guard, subjects, sha256(cfg.output('work/splits/subject_split_v1.tsv')))
    pairs = folds.loc[folds.index.repeat(2)]
    assert pairs.groupby('subject').fold.nunique().eq(1).all()
    model = fit_model(np.random.default_rng(8).normal(size=(2, 2, 6)).astype('float32'), subjects[:2], cfg, guard, subjects, 1)
    with pytest.raises(PipelineError, match='OUT-OF-FOLD'): score_pair(np.ones((2, 6)), subjects[0], model, guard, subjects, 'transfer')
    pd.testing.assert_frame_equal(folds, make_folds(cfg, dev, guard, subjects, sha256(cfg.output('work/splits/subject_split_v1.tsv'))))
