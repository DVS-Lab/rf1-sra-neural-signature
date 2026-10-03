import nibabel as nib
import pandas as pd
import pytest
from conftest import make_subject, metadata, register_aging_subject
from inventory import inventory, flip_angle, feat_dir, inspect_unit, InputUnavailable, summarize_feat_paths
from utils import PipelineError


def test_inventory_exclusions_session_missingness_headers_only(cfg, monkeypatch):
    frame = metadata(3)
    for i, subject in enumerate(frame.subject): make_subject(cfg, subject, secondary=True, index=i)
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
    with pytest.raises(PipelineError, match='Core sharedreward completed design'): inventory(cfg, False)


def test_ambiguous_coordinate_system_is_not_guessed(cfg):
    make_subject(cfg, 'sub-fixture000')
    path = feat_dir(cfg, 'sub-fixture000', 'trust', 'L1', 1) / 'design.fsf'
    path.write_text(path.read_text().replace('MNI152NLin6Asym', 'ambiguous'))
    with pytest.raises(InputUnavailable, match='ambiguous_input_space'):
        inspect_unit(cfg, 'sub-fixture000', 'trust', 'L2')


def test_sharedreward_missing_paths_report_alternates_without_selecting(cfg, monkeypatch, capsys):
    # Reproduce Linux2's failure without generating the expected SR derivatives.
    # Literal paths are independent of the path builder under test.
    frame = metadata(1)
    subject = frame.subject.iloc[0]
    (cfg.bids / subject / 'ses-01').mkdir(parents=True)
    register_aging_subject(cfg, subject)
    frame.rename(columns={'subject': 'participant_id'}).to_csv(cfg.bids/'participants.tsv', sep='\t', index=False)
    old = cfg.repos['linux2'] / 'derivatives/fsl' / subject / 'ses-01'
    (old / 'L2_task-sharedreward_ses-01_model-1_type-act_sm-6.gfeat').mkdir(parents=True)
    # Do not publish arbitrary names or scan excluded/noncanonical participants.
    (old / f'L2_task-sharedreward_{subject}_private.gfeat').mkdir()
    excluded = cfg.repos['linux2'] / 'derivatives/fsl/sub-excluded/ses-01'
    (excluded / 'L2_task-sharedreward_ses-01_model-1_type-act_sm-6.gfeat').mkdir(parents=True)
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('path diagnosis loaded an image'))
    table, _ = inventory(cfg)
    assert not table.eligible.any()
    assert 'L2_0:missing_feat_directory' in table.sharedreward_reason.iloc[0]
    report = pd.read_csv(cfg.output('results/aggregate/feat_path_summary.tsv'), sep='\t')
    expected = report[(report.task == 'sharedreward') & report.expected_input]
    assert set(expected.layout) == {
        'sub-<ID>/ses-01/L1_task-sharedreward_model-fulltrial_type-act_run-1_sm-6.feat',
        'sub-<ID>/ses-01/L1_task-sharedreward_model-fulltrial_type-act_run-2_sm-6.feat',
        'sub-<ID>/ses-01/L2_task-sharedreward_model-fulltrial_type-act_sm-6.gfeat'}
    assert expected.directory_n.sum() == 0
    assert not expected.root_exists.any()
    alternate = report[report.directory_n > 0]
    assert len(alternate) == 1 and alternate.directory_n.iloc[0] == 1
    assert not alternate.expected_input.any()
    output = capsys.readouterr().out
    assert 'alternate; NOT selected' in output and 'ZERO complete participants' in output
    assert subject not in output and 'sub-excluded' not in output
    public = cfg.output('results/aggregate/feat_path_summary.tsv').read_text()
    assert subject not in public and '_private.gfeat' not in public and 'sub-excluded' not in public
    assert not cfg.output('work/splits/subject_split_v1.tsv').exists()


def test_sharedreward_expected_path_counts_require_directories(cfg, monkeypatch):
    folder = cfg.repos['aging'] / 'derivatives/fsl/rf1/sub-fixture000/ses-01'
    folder.mkdir(parents=True)
    (folder / 'L1_task-sharedreward_model-fulltrial_type-act_run-1_sm-6.feat').mkdir()
    (folder / 'L1_task-sharedreward_model-fulltrial_type-act_run-2_sm-6.feat').touch()
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('path diagnosis loaded an image'))
    rows = summarize_feat_paths(cfg, ['sub-fixture000'])
    found = [r for r in rows if r['directory_n']]
    assert len(found) == 1 and found[0]['expected_input'] and found[0]['directory_n'] == 1
