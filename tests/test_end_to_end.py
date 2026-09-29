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
    assert before_files == {str(p) for p in c.root.rglob('*') if p.is_file()}
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
    assert int(perf.loc[perf.analysis == 'primary', 'n'].iloc[0]) == 20
    assert perf.set_index('analysis').loc['socialdoors','n'] > 0
    for path in list((c.root/'results').rglob('*.tsv')) + list((c.root/'reports').glob('*.md')) + list((c.root/'provenance').rglob('*.json')):
        assert not any(s in path.read_text() for s in frame.subject), path
    for name in ['cv_predictions.tsv', 'cross_task_predictions.tsv']:
        predictions = pd.read_csv(c.output('work/predictions/'+name), sep='\t')
        assert not set(predictions.subject) & holdout
    assert json.loads(c.output('provenance/model.json').read_text())['holdout_scored'] is False
    assert json.loads(c.output('provenance/run_status.json').read_text())['status'] == 'complete'
    assert len(list((c.root/'results/maps').glob('*.nii.gz'))) == 7
    assert len(list((c.root/'results/figures').glob('*.png'))) == 5


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
    assert result.returncode == 0, result.stderr
    assert 'DRY RUN PASSED' in result.stdout
    assert not c.output('work/splits/subject_split_v1.tsv').exists()
