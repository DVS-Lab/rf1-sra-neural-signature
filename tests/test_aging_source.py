import json
import shutil
import nibabel as nib
import pandas as pd
import pytest
from conftest import make_subject, metadata, write_fixture_tsv
from aging_source import load_aging_index, read_rows, aging_directory, subject_unit
from inventory import inventory, cope_path, mask_paths, inspect_unit, InputUnavailable
from utils import PipelineError


def test_real_layout_fixed_effects_and_single_run_strategies(cfg, monkeypatch):
    frame = metadata(3)
    for i, subject in enumerate(frame.subject):
        make_subject(cfg, subject, index=i, sr_runs=[(1, 2), (1,), (2,)][i])
    frame.rename(columns={'subject': 'participant_id'}).to_csv(cfg.bids/'participants.tsv', sep='\t', index=False)
    # Unsupported datasets and PPI never become activation inputs, even when present.
    path = cfg.repos['aging']/cfg.paths['aging_audit']/'final/verified-pre-QC-candidates.tsv'
    rows = read_rows(path)
    rows += [{**rows[0], 'dataset': 'ds003745', 'subject': 'external'},
             {**rows[0], 'type': 'ppi_seed-vs', 'cope_path': '/wrong/ppi.nii.gz'}]
    write_fixture_tsv(path, rows)
    original = nib.load
    forbidden = str(aging_directory(cfg, frame.subject.iloc[2], 'L1', 1))
    def header_only(path, *args, **kwargs):
        assert forbidden not in str(path), 'Excluded run was read'
        image = original(path, *args, **kwargs)
        image.get_fdata = lambda *a, **k: pytest.fail('Inventory read voxel intensities')
        return image
    monkeypatch.setattr(nib, 'load', header_only)
    table, _ = inventory(cfg)
    assert table.eligible.all() and table.sharedreward_two_runs.sum() == 1
    assert subject_unit(cfg, frame.subject.iloc[2]) == ('L1', 2)
    assert str(cope_path(cfg, frame.subject.iloc[0], 'sharedreward', 4)).endswith(
        'L2_task-sharedreward_model-fulltrial_type-act_sm-6.gfeat/cope4.feat/stats/cope1.nii.gz')
    assert str(cope_path(cfg, frame.subject.iloc[2], 'sharedreward', 4)).endswith(
        'L1_task-sharedreward_model-fulltrial_type-act_run-2_sm-6.feat/stats/cope4.nii.gz')
    assert mask_paths(cfg, frame.subject.iloc[1], 'sharedreward') == [aging_directory(cfg, frame.subject.iloc[1], 'L1', 1)/'mask.nii.gz']
    assert cfg.aging_provenance['verified_rf1_subjects'] == 3
    assert not cfg.aging_provenance['ratings_filter_applied']


@pytest.mark.parametrize('field,value', [('cope_path', '/wrong/sub-other/cope1.nii.gz'),
                                        ('contrast', 'F_dec'), ('strategy', 'l1_passthrough')])
def test_candidate_identity_and_strategy_fail_closed(cfg, field, value):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    path = cfg.repos['aging']/cfg.paths['aging_audit']/'final/verified-pre-QC-candidates.tsv'
    rows = read_rows(path); rows[0][field] = value; write_fixture_tsv(path, rows)
    with pytest.raises(PipelineError): load_aging_index(cfg, refresh=True)


def test_unverified_or_drifted_frozen_manifest_fails(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    audit = cfg.repos['aging']/cfg.paths['aging_audit']
    summary = audit/'final/summary.json'
    data = json.loads(summary.read_text()); data['computational_gate_passed'] = False
    summary.write_text(json.dumps(data))
    with pytest.raises(PipelineError, match='gate'): load_aging_index(cfg, refresh=True)
    data['computational_gate_passed'] = True; summary.write_text(json.dumps(data))
    manifest = audit/'L1-task-ready.tsv'
    manifest.write_text(manifest.read_text()+'\n')
    with pytest.raises(PipelineError, match='fingerprint'): load_aging_index(cfg, refresh=True)


def test_only_verified_reward_maps_required_and_no_phase_slot_interpretation(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    # Upstream excludes neutral estimates from its verification gate.
    for level, run in [('L1', 1), ('L1', 2), ('L2', None)]:
        base = aging_directory(cfg, 'sub-fixture000', level, run)
        for k in range(7, 29):
            if level == 'L1': (base/f'stats/cope{k}.nii.gz').unlink()
            else: shutil.rmtree(base/f'cope{k}.feat')
        assert inspect_unit(cfg, 'sub-fixture000', 'sharedreward', level, run)


def test_changed_input_image_fingerprint_and_missing_provenance(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    base = aging_directory(cfg, 'sub-fixture000', 'L1', 1)
    stamp = base/'pooled-model-inputs.json'
    data = json.loads(stamp.read_text()); data['images'][0]['mtime_ns'] -= 1
    stamp.write_text(json.dumps(data))
    with pytest.raises(PipelineError, match='image-input fingerprint'):
        inspect_unit(cfg, 'sub-fixture000', 'sharedreward', 'L1', 1)
    stamp.unlink()
    with pytest.raises(InputUnavailable, match='missing_aging_provenance'):
        inspect_unit(cfg, 'sub-fixture000', 'sharedreward', 'L1', 1)


def test_provenance_never_hashes_voxel_images(cfg, monkeypatch):
    from aging_source import verify_stamp
    make_subject(cfg, 'sub-fixture000', secondary=False)
    base = aging_directory(cfg, 'sub-fixture000', 'L1', 1)
    path = base/'pooled-model-inputs.json'
    data = json.loads(path.read_text())
    data['inputs'] = [{'path': data['images'][0]['path'], 'sha256': 'not-to-be-read'}]
    path.write_text(json.dumps(data))
    monkeypatch.setattr('aging_source.sha256', lambda *args: pytest.fail('Image bytes were read'))
    with pytest.raises(PipelineError, match='never voxel images'):
        verify_stamp(cfg, base, 'L1')


def test_completed_fixed_effects_matrix_is_checked(cfg):
    make_subject(cfg, 'sub-fixture000', secondary=False)
    base = aging_directory(cfg, 'sub-fixture000')
    (base/'cope4.feat/design.con').write_text('/Matrix\n-1\n')
    with pytest.raises(PipelineError, match='fixed-effects mean'):
        inspect_unit(cfg, 'sub-fixture000', 'sharedreward', 'L2')
