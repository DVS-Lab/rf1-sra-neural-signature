import hashlib
import os
import sys
import json
import subprocess
from pathlib import Path
import nibabel as nib
import pandas as pd
from pilot import run
from make_split import make_split
from conftest import metadata
from utils import sha256


def source_digest(c):
    h = hashlib.sha256()
    for root in c.repos.values():
        for p in sorted(root.rglob('*')):
            if p.is_file() and '.git' not in p.parts:
                h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()


def test_full_pilot_dry_run_no_writes_and_locked_holdout(cohort, monkeypatch):
    c, frame = cohort
    before = source_digest(c)
    before_files = {str(p) for p in c.root.rglob('*') if p.is_file()}
    run(c, dry_run=True)
    import shutil
    source = Path(__file__).resolve().parents[1] / 'code'
    shutil.copytree(source, c.root/'code', ignore=shutil.ignore_patterns('__pycache__'))
    cli = subprocess.run(['bash', str(c.root/'code/run_pilot.sh'), '--dry-run'], text=True, capture_output=True, env={**os.environ, 'PYTHON': sys.executable})
    assert cli.returncode == 0, cli.stderr
    assert 'DRY RUN PASSED' in cli.stdout
    assert before_files == {str(p) for p in c.root.rglob('*') if p.is_file() and not p.is_relative_to(c.root/'code')}
    assert not c.output('work/splits/subject_split_v1.tsv').exists()
    split = make_split(c, frame)
    holdout = set(split[split.split == 'holdout'].subject)
    original_load = nib.load
    reads = []
    def guarded_load(filename, *args, **kwargs):
        image = original_load(filename, *args, **kwargs)
        if any(subject in str(filename) for subject in holdout):
            def forbidden(*a, **kw): raise AssertionError('Holdout get_fdata() called')
            image.get_fdata = forbidden
        else: reads.append(str(filename))
        return image
    monkeypatch.setattr(nib, 'load', guarded_load)
    run(c)
    assert before == source_digest(c)
    assert reads
    perf = pd.read_csv(c.output('results/aggregate/performance_summary.tsv'), sep='\t')
    assert perf.loc[perf.model_scope != 'reliability', 'n'].eq(20).all()
    from aging_source import aging_entry
    paired_n = sum(len(aging_entry(c, s)['runs']) == 2 for s in set(frame.subject)-holdout)
    assert 0 < paired_n < 20
    assert perf.loc[perf.model_scope == 'reliability', 'n'].eq(paired_n).all()
    assert len(perf[perf.model_scope == 'pairwise']) == 13
    assert len(perf[perf.model_scope == 'lopo']) == 3
    assert len(perf[perf.model_scope == 'cross_family']) == 5
    decision = perf[(perf.model_scope == 'pairwise') & (perf.train_family == 'decision')]
    assert set(decision.test_task) == {'trust', 'socialdoors'}
    assert not perf.test_task.eq('neutral').any()
    diagnostics = pd.read_csv(c.output('work/diagnostics/source_image_metrics.tsv'), sep='\t')
    single = {s for s in set(frame.subject)-holdout if len(aging_entry(c, s)['runs']) == 1}
    assert set(diagnostics.loc[(diagnostics.task == 'sharedreward') & diagnostics.subject.isin(single), 'level']) == {'L1'}
    availability = pd.read_csv(c.output('results/aggregate/analysis_availability.tsv'), sep='\t')
    assert set(availability.loc[availability.status == 'unavailable', 'analysis']) == {'sharedreward_decision', 'sharedreward_neutral'}
    for path in list((c.root/'results').rglob('*.tsv')) + list((c.root/'reports').glob('*.md')) + list((c.root/'provenance').rglob('*.json')):
        assert not any(s in path.read_text() for s in frame.subject), path
    for name in ['cv_predictions.tsv', 'cross_task_predictions.tsv']:
        predictions = pd.read_csv(c.output('work/predictions/'+name), sep='\t')
        assert not set(predictions.subject) & holdout
    assert json.loads(c.output('provenance/model.json').read_text())['holdout_scored'] is False
    assert json.loads(c.output('provenance/run_status.json').read_text())['status'] == 'complete'
    assert len(list((c.root/'results/maps').glob('*.nii.gz'))) == 58
    assert len(list((c.root/'results/figures').glob('*.png'))) == 9
    audit = pd.read_csv(c.output('work/diagnostics/model_membership.tsv'), sep='\t')
    for (_, fold), group in audit.groupby(['model', 'fold']):
        assert not set(group[group.role == 'train'].subject) & set(group[group.role == 'test'].subject)
    models = json.loads(c.output('provenance/model.json').read_text())['models']
    assert all('ugr' not in model['training_tasks'] for model in models)
    for model in models:
        if 'lopo-' in model['model']:
            assert model['model'].split('lopo-')[1] not in model['training_tasks']


def test_shell_launcher_dry_run(cfg):
    from conftest import make_subject
    c = cfg
    make_subject(c, 'sub-fixture000', secondary=False)
    metadata(1).rename(columns={'subject':'participant_id'}).to_csv(c.bids/'participants.tsv', sep='\t', index=False)
    # Actual launcher with temporary fixture checkout, no /ZPOOL dependencies.
    source = Path(__file__).resolve().parents[1] / 'code'
    import shutil
    shutil.copytree(source, c.root/'code', ignore=shutil.ignore_patterns('__pycache__'))
    result = subprocess.run(['bash', str(c.root/'code/run_pilot.sh'), '--dry-run'], text=True, capture_output=True,
                            env={**os.environ, 'PYTHON': sys.executable})
    assert result.returncode == 1
    assert 'STOP before split' in result.stderr
    assert 'Exact task intersections' in result.stdout
    assert not c.output('work/splits/subject_split_v1.tsv').exists()


def test_decision_sharedreward_unavailable_guard(cfg):
    import numpy as np
    import pytest
    from signature_pilot import fit_model
    from make_split import guard_from_split
    from utils import PipelineError
    guard = guard_from_split(make_split(cfg, metadata()))
    subjects = sorted(guard.development)
    with pytest.raises(PipelineError, match='clean paradigm'):
        fit_model(np.ones((2, 1, 2, 9)), subjects[:2], cfg, guard, subjects, 1, ('sharedreward',), 'decision')
