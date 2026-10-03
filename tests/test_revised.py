import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from conftest import metadata, make_subject, fsf_text, con_text
from test_qc_review import qc_rows
from utils import PipelineError, sha256
from make_split import make_split, make_folds, guard_from_split
from revised_design import revised_config, partner_representations, doors_representations
from revised_samples import reconstruct, inventory
from revised_models import fit, score
from revised_reporting import summarize
from revised_pilot import run


def setup_lock(cfg, n=80):
    frame = metadata(n)
    split = make_split(cfg, frame)
    guard = guard_from_split(split)
    make_folds(cfg, frame[frame.subject.isin(guard.development)], guard, sorted(guard.development),
               sha256(cfg.output('work/splits/subject_split_v1.tsv')))
    return split


def write_qc(cfg, frame):
    path = cfg.repos['linux2']/cfg.paths['qc_table']
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, sep='\t', index=False)
    return path


def test_reconstruction_topup_is_locked_and_all_original_holdouts_stay_protected(cfg, monkeypatch):
    split = setup_lock(cfg)
    frame = metadata(100)
    frame['sharedreward_runs'] = '1,2'
    frame['sharedreward_two_runs'] = True
    qc = qc_rows(frame.subject)
    bad_holdouts = split.loc[split.split == 'holdout', 'subject'].iloc[:4]
    qc.loc[qc.subject.isin(bad_holdouts) & qc.task.eq('trust'), 'fd_mean_outlier'] = 'TRUE'
    write_qc(cfg, qc)
    originals = {p: sha256(p) for p in [cfg.output('work/splits/subject_split_v1.tsv'), cfg.output('work/splits/development_folds.tsv')]}
    monkeypatch.setattr(nib, 'load', lambda *a, **k: pytest.fail('Sample reconstruction read image'))
    policies = ('tsnr_coverage_fd', 'prior_four_metric_policy')
    samples, summary = reconstruct(cfg, frame, policies, True)
    first = summary.iloc[0]
    assert first.qc_qualified_holdout_n == 50 and first.supplemental_holdout_n == 4
    assert first.development_n == 46 and first.new_development_n == 16
    guard, folds = samples[(policies[0], 'partner_pair')]
    old_held = set(split.loc[split.split == 'holdout', 'subject'])
    assert old_held <= guard.holdout and len(guard.holdout) == 54
    assert not guard.development & guard.holdout
    assert set(bad_holdouts) <= guard.holdout
    for subject in guard.holdout:
        with pytest.raises(PipelineError): guard.check([subject], guard.development)
    path = revised_config(cfg, 'samples').output('work/manifest.tsv')
    before = sha256(path)
    reconstruct(cfg, frame.sample(frac=1), policies, True)
    assert sha256(path) == before
    assert all(sha256(p) == value for p, value in originals.items())
    frame.loc[frame.subject == sorted(guard.development)[0], 'trust'] = False
    with pytest.raises(PipelineError, match='eligibility changed'):
        reconstruct(cfg, frame, policies, True)


def test_insufficient_unseen_candidates_stops_without_replacement(cfg):
    split = setup_lock(cfg)
    frame = metadata(82); frame['sharedreward_runs'] = '1,2'
    qc = qc_rows(frame.subject)
    bad = split.loc[split.split == 'holdout', 'subject'].iloc[:3]
    qc.loc[qc.subject.isin(bad), 'tsnr_outlier'] = 'TRUE'
    write_qc(cfg, qc)
    with pytest.raises(PipelineError, match='cannot lock N=50'):
        reconstruct(cfg, frame, ('tsnr_coverage_fd',), True)
    assert not revised_config(cfg, 'samples').output('work/manifest.tsv').exists()


def test_interrupted_voxel_run_cannot_recreate_missing_sample_lock(cfg):
    setup_lock(cfg)
    frame = metadata(82); frame['sharedreward_runs'] = '1,2'
    write_qc(cfg, qc_rows(frame.subject))
    status = revised_config(cfg, 'samples').output('provenance/run_status.json')
    status.parent.mkdir(parents=True, exist_ok=True)
    status.write_text(json.dumps({'status': 'in_progress'}))
    with pytest.raises(PipelineError, match='restore the private manifest'):
        reconstruct(cfg, frame, ('tsnr_coverage_fd',), True)


def test_contrasts_keep_partners_distinct_and_context_separate():
    # Distinct spatial patterns expose label/order/sign mistakes; scalar inputs would center away.
    maps = {k: np.array([k*k, k, -k], dtype=float) for k in range(1, 10)}
    r = partner_representations('sharedreward', maps)
    def centered(a): return a-a.mean()
    np.testing.assert_allclose(r['partner_reward'][0], centered(maps[2]-maps[1]), atol=1e-5)
    np.testing.assert_allclose(r['partner_reward'][1], centered(maps[4]-maps[3]), atol=1e-5)
    np.testing.assert_allclose(r['closeness_positive'][0]-r['closeness_positive'][1], centered(maps[4]-maps[6]), atol=1e-5)
    np.testing.assert_allclose(r['closeness_reward'][0]-r['closeness_reward'][1],
                               centered((maps[4]-maps[3])-(maps[6]-maps[5])), atol=1e-5)
    t = partner_representations('trust', maps)
    np.testing.assert_allclose(t['social_context'][0], centered((maps[7]+maps[6]+maps[9]+maps[8])/4), atol=1e-5)
    d = doors_representations(maps, {k: 2*v for k,v in maps.items()})
    np.testing.assert_allclose(d['social_context'][0], centered((maps[1]+maps[2])/2), atol=1e-5)


def test_multiclass_inference_clusters_three_maps_per_person(cfg):
    rows = []
    for i, accuracy in enumerate([0., 1., 1/3, 2/3]):
        rows.append({'subject': f'sub-fixture{i}', 'family': 'partner_reward', 'train_tasks': 'trust',
                     'test_task': 'sharedreward', 'scope': 'pairwise', 'chance': 1/3,
                     'accuracy': accuracy, 'margin': 0., 'predicted_labels': '0,1,2'})
    result, confusion = summarize(pd.DataFrame(rows), cfg)
    assert result.n.iloc[0] == 4 and result.accuracy.iloc[0] == .5
    assert pd.isna(result.binomial_p.iloc[0]) and confusion.n_maps.sum() == 12


def test_revised_models_block_same_person_across_tasks(cfg):
    split = setup_lock(cfg)
    guard = guard_from_split(split)
    subjects = sorted(guard.development)
    rng = np.random.default_rng(21)
    x = rng.normal(size=(len(subjects), 1, 3, 8)).astype(np.float32)
    model = fit(x, subjects, ('trust',), 'partner_reward', 1, cfg, guard)
    with pytest.raises(PipelineError, match='Participant leakage'):
        score(model, x[0,0], subjects[0], guard, 'sharedreward', 'pairwise', True)


def test_full_revised_pipeline_without_ugr_decision_voxels_and_without_holdout_access(cfg, monkeypatch):
    # Populate actual source templates for win/loss; v3 fixtures previously used unused placeholders there.
    for task in ('doors', 'socialdoors'):
        spec = cfg.contrasts[task]
        spec['copes'][1] = {'name': 'win', 'weights': {1: 1.}}
        spec['copes'][2] = {'name': 'loss', 'weights': {2: 1.}}
        template = cfg.repos['socdoors']/spec['template']
        template.write_text(fsf_text(spec, 'DATA'))
    frame = metadata(74)
    for i, subject in enumerate(frame.subject): make_subject(cfg, subject, index=i, sr_runs=(2,) if i % 4 == 0 else (1, 2))
    frame.rename(columns={'subject': 'participant_id'}).to_csv(cfg.bids/'participants.tsv', sep='\t', index=False)
    source_config = Path(__file__).resolve().parents[1]/'config/revised_analysis.yaml'
    (cfg.root/'config/revised_analysis.yaml').write_text(source_config.read_text())
    split = setup_lock(cfg, 70)
    qc = qc_rows(frame.subject)
    rejected = split.loc[split.split == 'holdout', 'subject'].iloc[:2]
    qc.loc[qc.subject.isin(rejected) & qc.task.eq('trust'), 'fd_mean_outlier'] = 'TRUE'
    write_qc(cfg, qc)
    # Completely remove UGR and decision-only outputs; revised inventory must still succeed.
    import shutil
    shutil.rmtree(cfg.repos['ugr'])
    for subject in frame.subject:
        for task in ('socialdoors', 'doors'):
            from inventory import feat_dir
            (feat_dir(cfg, subject, task, 'L1', 1)/'stats/cope3.nii.gz').unlink()
        shutil.rmtree(feat_dir(cfg, subject, 'trust')/'cope1.feat')
        shutil.rmtree(feat_dir(cfg, subject, 'trust')/'cope2.feat')
        shutil.rmtree(feat_dir(cfg, subject, 'trust')/'cope3.feat')
    original_load = nib.load
    forbidden_subjects = set(split.loc[split.split == 'holdout', 'subject'])
    def header_only(filename, *args, **kwargs):
        image = original_load(filename, *args, **kwargs)
        image.get_fdata = lambda *a, **k: pytest.fail('Dry-run or sample selection read voxel data')
        return image
    monkeypatch.setattr(nib, 'load', header_only)
    summary = run(cfg, dry_run=True)
    assert not revised_config(cfg, 'samples').output('work/manifest.tsv').exists()
    run(cfg, sample_only=True)
    sample_c = revised_config(cfg, 'samples')
    manifest = pd.read_csv(sample_c.output('work/manifest.tsv'), sep='\t')
    forbidden_subjects |= set(manifest.loc[manifest.partition == 'holdout', 'subject'])
    lock_hash = sha256(sample_c.output('work/manifest.tsv'))
    def guarded_load(filename, *args, **kwargs):
        text = str(filename)
        assert 'task-ugr' not in text
        assert not ('task-trust' in text and any(f'/cope{k}.feat/' in text for k in (1,2,3)))
        assert not (('task-socialdoors' in text or 'task-doors' in text) and '/stats/cope3.nii' in text)
        image = original_load(filename, *args, **kwargs)
        if any(s in text for s in forbidden_subjects):
            image.get_fdata = lambda *a, **k: pytest.fail('Protected original/supplemental holdout voxel read')
        return image
    monkeypatch.setattr(nib, 'load', guarded_load)
    run(cfg)
    assert sha256(sample_c.output('work/manifest.tsv')) == lock_hash
    status = json.loads(sample_c.output('provenance/run_status.json').read_text())
    assert status['status'] == 'complete' and len(status['completed_scopes']) == 4
    assert summary.qc_qualified_holdout_n.eq(50).all()
    for policy in ('tsnr_coverage_fd', 'prior_four_metric_policy'):
        for cohort in ('partner_pair', 'three_paradigm'):
            c = revised_config(cfg, f'{policy}/{cohort}')
            result = pd.read_csv(c.output('results/aggregate/performance.tsv'), sep='\t')
            expected_n = int(summary[(summary.policy == policy)&(summary.cohort == cohort)].development_n.iloc[0])
            assert result.n.eq(expected_n).all()
            assert len(result) == (42 if cohort == 'partner_pair' else 30)
            assert json.loads(c.output('provenance/run_status.json').read_text())['status'] == 'complete'
            assert not any(s in c.output('reports/REPORT.md').read_text() for s in frame.subject)
    # A failed rerun must invalidate the previous global completion marker.
    import revised_pilot
    def fail_mask(*args, **kwargs): raise PipelineError('Synthetic mask failure')
    monkeypatch.setattr(revised_pilot, 'build_mask', fail_mask)
    with pytest.raises(PipelineError, match='Synthetic mask failure'): run(cfg)
    status = json.loads(sample_c.output('provenance/run_status.json').read_text())
    assert status['status'] == 'failed' and status['stage'] == 'mask'
    assert status['completed_scopes'] == []
