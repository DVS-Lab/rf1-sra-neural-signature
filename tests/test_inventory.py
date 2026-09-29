import nibabel as nib
import pandas as pd
import pytest
from conftest import make_subject, metadata
from inventory import inventory, flip_angle, feat_dir, inspect_unit, InputUnavailable
from utils import PipelineError


def test_inventory_exclusions_session_missingness_headers_only(cfg, monkeypatch):
    frame = metadata(3)
    for i, subject in enumerate(frame.subject): make_subject(cfg, subject, secondary=False, index=i)
    frame.rename(columns={'subject': 'participant_id'}).to_csv(cfg.bids/'participants.tsv', sep='\t', index=False)
    (cfg.exclusions / ('Smith-SRA-'+frame.subject.iloc[1][4:])).mkdir()
    (feat_dir(cfg, frame.subject.iloc[2], 'sharedreward', 'L1', 2) / 'stats/cope2.nii.gz').unlink()
    monkeypatch.setattr(nib.Nifti1Image, 'get_fdata', lambda *a, **kw: pytest.fail('inventory loaded voxels'))
    table, summary = inventory(cfg)
    assert len(table) == 2 and table.eligible.sum() == 1
    assert frame.subject.iloc[1] not in cfg.output('work/inventory/task_availability.tsv').read_text()
    aggregate = cfg.output('results/aggregate/inventory_summary.tsv').read_text()
    assert 'source_excluded' in aggregate and not any(s in aggregate for s in frame.subject)


def test_flipangle_discrepancy_not_inferred(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    assert flip_angle(cfg, 'sub-fixture000')[0] == '50'
    path = next((cfg.bids/'sub-fixture000/ses-01/func').glob('*run-2*bold.json'))
    path.write_text('{"FlipAngle":20}')
    assert flip_angle(cfg, 'sub-fixture000') == ('discrepant', 'flipangle_run_mismatch')
    path.unlink()
    assert flip_angle(cfg, 'sub-fixture000')[0] == 'missing'


def test_primary_ambiguous_design_fails(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    frame = metadata(1).rename(columns={'subject': 'participant_id'})
    frame.to_csv(cfg.bids/'participants.tsv', sep='\t', index=False)
    path = feat_dir(cfg, 'sub-fixture000', 'sharedreward', 'L1', 1) / 'design.fsf'
    path.write_text(path.read_text().replace('con_real4.4) 1.0', 'con_real4.4) -1.0'))
    with pytest.raises(PipelineError, match='Primary completed design'): inventory(cfg, False)


def test_ambiguous_coordinate_system_is_not_guessed(cfg):
    make_subject(cfg, 'sub-fixture000')
    path = feat_dir(cfg, 'sub-fixture000', 'trust', 'L1', 1) / 'design.fsf'
    path.write_text(path.read_text().replace('MNI152NLin6Asym', 'ambiguous'))
    with pytest.raises(InputUnavailable, match='ambiguous_input_space'):
        inspect_unit(cfg, 'sub-fixture000', 'trust', 'L2')
