"""Read-only adapter for verified RF1 full-trial activation, never pooled participants/PPI."""
import csv
import json
from pathlib import Path
from utils import PipelineError, InputUnavailable, sha256, subject_id

CONTRACT = 'fulltrial-retained-contrasts-v3'


def read_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream, delimiter='\t'))


def aging_directory(c, subject, level='L2', run=None):
    subject_id(subject)
    stem = f'{level}_task-sharedreward_model-fulltrial_type-act'
    if level == 'L1':
        if run not in (1, 2): raise PipelineError('Invalid aging run')
        stem += f'_run-{run}'
    elif level != 'L2':
        raise PipelineError('Invalid aging level')
    return (c.repos['aging'] / 'derivatives/fsl/rf1' / subject / 'ses-01' /
            (stem + '_sm-6' + ('.feat' if level == 'L1' else '.gfeat')))


def verify_contrast_table(c):
    spec = c.contrasts['sharedreward']
    path = c.repos['aging'] / spec['contrast_table']
    rows = read_rows(path)
    if [int(r['candidate_cope']) for r in rows] != list(range(1, 29)):
        raise PipelineError('Aging activation contrast table must contain 28 ordered contrasts')
    for k, required in spec['copes'].items():
        row = rows[k-1]
        weights = [float(x) for x in row['weights_ev1_to_ev10'].split(',')]
        actual = {i+1: value for i, value in enumerate(weights) if value != 0}
        if len(weights) != 10 or row['contrast_name'] != required['name'] or actual != required['weights']:
            raise PipelineError('Aging reward contrast table disagrees with the analysis contract')
    return path


def load_aging_index(c, refresh=False):
    if not refresh and hasattr(c, 'aging_subjects'):
        return c.aging_subjects
    root = c.repos['aging']
    audit = (root / c.paths['aging_audit']).resolve()
    if not audit.is_relative_to(root.resolve()):
        raise PipelineError('Aging audit must be within the configured source repository')
    summary_path = audit / 'final/summary.json'
    try:
        summary = json.loads(summary_path.read_text())
        if summary.get('contract') != CONTRACT or summary.get('computational_gate_passed') is not True:
            raise PipelineError('Aging audit has not passed the full-trial computational gate')
        recorded = {Path(r['path']).name: r['sha256'] for r in summary['inputs']}
        contract = verify_contrast_table(c)
        for path in [audit/'L1-task-ready.tsv', audit/'L2-task-ready.tsv', contract]:
            if recorded.get(path.name) != sha256(path):
                raise PipelineError('Aging frozen manifest/contrast fingerprint changed; obtain a matching audit')
        l1_rows = read_rows(audit/'L1-task-ready.tsv')
        l2_rows = read_rows(audit/'L2-task-ready.tsv')
        candidates_path = audit/'final/verified-pre-QC-candidates.tsv'
        candidates = read_rows(candidates_path)
        selected = {}
        for row in l2_rows:
            if row['dataset'] != 'rf1' or row['session'] != '01': continue
            subject = subject_id('sub-' + row['subject'].removeprefix('sub-'))
            runs = tuple(int(r) for r in row['runs'].split(','))
            strategy = 'fixed_effects' if runs == (1, 2) else 'l1_passthrough'
            if (runs not in [(1,), (2,), (1, 2)] or int(row['n_runs']) != len(runs)
                    or row['subject_level_strategy'] != strategy or subject in selected):
                raise PipelineError('Invalid/duplicate retained RF1 run strategy')
            selected[subject] = {'runs': runs, 'strategy': strategy, 'copes': set(), 'l1': {}}
        for row in l1_rows:
            if row['dataset'] != 'rf1' or row['session'] != '01': continue
            subject = subject_id('sub-' + row['subject'].removeprefix('sub-'))
            run = int(row['run'])
            if subject not in selected or run not in selected[subject]['runs'] or run in selected[subject]['l1']:
                raise PipelineError('Aging L1 and subject manifests disagree')
            selected[subject]['l1'][run] = row
        for row in candidates:
            if row['dataset'] != 'rf1' or row['session'] != '01' or row['type'] != 'act': continue
            k = int(row['cope'])
            if k not in range(1, 7): continue  # No neutral, decision-slot reinterpretation, or PPI.
            subject = subject_id('sub-' + row['subject'].removeprefix('sub-'))
            if subject not in selected: raise PipelineError('Unexpected RF1 activation candidate')
            entry = selected[subject]
            runs = tuple(int(r) for r in row['runs'].split(','))
            if (runs != entry['runs'] or row['strategy'] != entry['strategy'] or k in entry['copes']
                    or row['contrast'] != c.contrasts['sharedreward']['copes'][k]['name']):
                raise PipelineError('Conflicting/duplicate RF1 activation candidate')
            level = 'L2' if len(runs) == 2 else 'L1'
            base = aging_directory(c, subject, level, runs[0] if level == 'L1' else None)
            cope_dir = base/f'cope{k}.feat' if level == 'L2' else base
            index = 1 if level == 'L2' else k
            for key, path in {'cope_path': cope_dir/f'stats/cope{index}.nii.gz',
                              'varcope_path': cope_dir/f'stats/varcope{index}.nii.gz',
                              'mask_path': cope_dir/'mask.nii.gz'}.items():
                actual = Path(row[key])
                if not actual.is_absolute() or actual != path or not actual.resolve().is_relative_to(root.resolve()):
                    raise PipelineError('RF1 candidate path does not match its participant/model/strategy')
            entry['copes'].add(k)
        if not selected or any(e['copes'] != set(range(1, 7)) or set(e['l1']) != set(e['runs']) for e in selected.values()):
            raise PipelineError('Verified aging candidates lack the six reward maps or retained-run evidence')
        c.aging_subjects = selected
        c.aging_provenance = {'contract': CONTRACT, 'epoch': 'full_trial_decision_through_outcome',
                              'audit': c.paths['aging_audit'], 'summary_sha256': sha256(summary_path),
                              'candidates_sha256': sha256(candidates_path),
                              'l1_manifest_sha256': sha256(audit/'L1-task-ready.tsv'),
                              'l2_manifest_sha256': sha256(audit/'L2-task-ready.tsv'),
                              'verified_rf1_subjects': len(selected),
                              'fixed_effects_n': sum(len(e['runs']) == 2 for e in selected.values()),
                              'l1_passthrough_n': sum(len(e['runs']) == 1 for e in selected.values()),
                              'ratings_filter_applied': False}
        return selected
    except (OSError, ValueError, KeyError) as exc:
        raise PipelineError('Cannot validate the configured aging audit; check config/paths.yaml aging_audit') from exc


def aging_entry(c, subject):
    index = load_aging_index(c)
    if subject not in index: raise InputUnavailable('not_in_verified_aging_manifest')
    return index[subject]


def subject_unit(c, subject):
    runs = aging_entry(c, subject)['runs']
    return ('L2', None) if len(runs) == 2 else ('L1', runs[0])


def verify_stamp(c, base, level, parents=()):
    """Check saved input fingerprints without loading any image intensities."""
    path = base/'pooled-model-inputs.json'
    if not path.is_file(): raise InputUnavailable('missing_aging_provenance')
    try:
        data = json.loads(path.read_text())
        if ((data['contract'], data['level'], data['type']) != (CONTRACT, level.lower(), 'act')
                or data['parents'] != [str(p.resolve()) for p in parents]
                or not data['inputs'] or not data['images']):
            raise PipelineError('Aging model provenance contract/parents disagree')
        for item in data['inputs']:
            p = Path(item['path'])
            if p.suffix not in {'.json', '.fsf', '.tsv', '.txt', '.py', '.sh'}:
                raise PipelineError('Aging provenance hash inputs must be small text files, never voxel images')
            if not any(p.resolve().is_relative_to(root.resolve()) for root in c.repos.values()):
                raise PipelineError('Aging provenance input outside configured source repositories')
            if sha256(p) != item['sha256']: raise PipelineError('Aging model input fingerprint changed')
        for item in data['images']:
            p = Path(item['path'])
            if level == 'L2' and p.name not in {f'{kind}{k}.nii.gz' for kind in ('cope', 'varcope') for k in range(1, 7)}:
                continue  # Same primary-only evidence boundary as upstream audit.
            if not any(p.resolve().is_relative_to(root.resolve()) for root in c.repos.values()):
                raise PipelineError('Aging provenance image outside configured source repositories')
            st = p.stat()
            if st.st_size != item['size'] or st.st_mtime_ns != item['mtime_ns']:
                raise PipelineError('Aging model image-input fingerprint changed')
    except (OSError, ValueError, KeyError) as exc:
        raise InputUnavailable('unreadable_aging_provenance') from exc
