import numpy as np
import nibabel as nib
import pytest
from conftest import metadata
from build_mask import load_development_image
from make_split import make_split, guard_from_split
from signature_pilot import fit_model
from utils import PipelineError


def test_holdout_rejected_before_any_image_access(cfg, monkeypatch):
    guard = guard_from_split(make_split(cfg, metadata()))
    holdout = sorted(guard.holdout)[0]
    def forbidden(*args, **kwargs): pytest.fail('Image read happened before holdout check')
    monkeypatch.setattr(nib, 'load', forbidden)
    with pytest.raises(PipelineError, match='HOLDOUT LOCK'):
        load_development_image(cfg, '/not-even-an-image', holdout, guard, sorted(guard.development))
    with pytest.raises(PipelineError, match='HOLDOUT LOCK'):
        load_development_image(cfg, '/not-even-an-image', holdout, guard, [holdout])


def test_holdout_rejected_before_fitting(cfg):
    guard = guard_from_split(make_split(cfg, metadata()))
    with pytest.raises(PipelineError, match='HOLDOUT LOCK'):
        fit_model(np.ones((1,2,4)), [next(iter(guard.holdout))], cfg, guard, sorted(guard.development), 1)
