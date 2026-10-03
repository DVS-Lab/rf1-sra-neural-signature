"""Persistent pre-model holdout and participant-blocked development folds."""
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from utils import CORE_TASKS, COHORT_DEFINITION, DevelopmentGuard, PipelineError, sha256, write_json, write_tsv


def strata_candidates(frame):
    flip = frame.flip_angle.astype(str).fillna('missing')
    for q in [5, 4, 3]:
        try:
            age = pd.qcut(frame.age, q=q, labels=False, duplicates='raise')
            if age.nunique() != q: continue
            labels = age.map(lambda x: 'missing_age' if pd.isna(x) else str(int(x))) + ':' + flip
            yield f'age_{q}_quantiles_x_flipangle', labels
        except ValueError:
            continue
    yield 'flipangle_only', flip


def describe(frame):
    ages = frame.age.dropna()
    return {'n': len(frame), 'age_n': len(ages), 'age_missing': int(frame.age.isna().sum()),
            'age_mean': float(ages.mean()), 'age_sd': float(ages.std()),
            'age_min': float(ages.min()), 'age_q25': float(ages.quantile(.25)),
            'age_median': float(ages.median()), 'age_q75': float(ages.quantile(.75)),
            'age_max': float(ages.max()), 'flipangle_counts': frame.flip_angle.astype(str).value_counts().to_dict(),
            'sex_counts': frame.sex.fillna('missing').astype(str).value_counts().to_dict()}


def validate_split(split, eligible):
    if list(split.columns) != ['subject', 'split'] or split.subject.duplicated().any():
        raise PipelineError('Invalid persisted split schema or duplicate participants')
    if set(split.split) != {'development', 'holdout'} or (split.split == 'holdout').sum() != 50:
        raise PipelineError('Persisted split must contain exactly N=50 holdout')
    if set(split.subject) != set(eligible.subject):
        raise PipelineError('Eligibility changed since the locked split. Restore inputs or review cohort drift; no automatic regeneration')


def require_multitask_cohort(c, eligible):
    if not set(CORE_TASKS) <= set(eligible.columns) or not eligible[list(CORE_TASKS)].all(axis=None):
        raise PipelineError('Split requires complete core maps in all five task implementations; no single-task fallback')
    minimum = c.analysis['minimum_multitask_n']
    if len(eligible) < minimum:
        raise PipelineError(f'Multitask-complete N={len(eligible)} is below the explicit minimum N={minimum}; STOP before split. See the exact task-overlap Ns above and results/aggregate/task_overlap.tsv on a real launch')


def make_split(c, eligible, regenerate=False):
    require_multitask_cohort(c, eligible)
    eligible = eligible.sort_values('subject').reset_index(drop=True)
    if eligible.subject.duplicated().any() or len(eligible) < 55:
        raise PipelineError('At least 55 unique eligible participants required for N=50 holdout and five folds')
    relative = 'work/splits/subject_split_v1.tsv'
    path = c.output(relative)
    meta_path = c.output('provenance/subject_split_v1.json')
    if path.exists() and not regenerate:
        split = pd.read_csv(path, sep='\t', dtype=str)
        validate_split(split, eligible)
        if meta_path.exists() and json.loads(meta_path.read_text()).get('cohort_definition') != COHORT_DEFINITION:
            raise PipelineError('Legacy split uses a different cohort/model contract; explicit reviewed regeneration is required')
        if not meta_path.exists() or json.loads(meta_path.read_text())['sha256'] != sha256(path):
            raise PipelineError('Locked split hash missing or mismatched; restore matching TSV and provenance')
        print('  Reusing locked N=50 holdout; membership unchanged.', flush=True)
        return split
    if meta_path.exists() and not path.exists() and not regenerate:
        raise PipelineError('Split provenance exists but private split is missing. Restore the private TSV; refusing to redraw')
    if regenerate:
        warnings.warn('DANGER: --regenerate-split replaces independent validation membership. Prior exposure can invalidate the holdout.', stacklevel=2)
        if path.exists():
            old = pd.read_csv(path, sep='\t', dtype=str)
            old_hash = sha256(path)
            write_tsv(c, f'work/splits/archive/split-{old_hash}.tsv', old)
            if meta_path.exists():
                write_json(c, f'work/splits/archive/split-{old_hash}.json', json.loads(meta_path.read_text()))
    chosen = None
    attempts = []
    for method, labels in strata_candidates(eligible):
        try:
            dev, holdout = next(StratifiedShuffleSplit(n_splits=1, test_size=50,
                              random_state=c.analysis['seed']).split(eligible, labels))
            chosen = (method, holdout)
            break
        except ValueError:
            attempts.append(method + ': sparse strata or insufficient split size')
    if chosen is None:
        raise PipelineError('Even FlipAngle-only stratification is infeasible. Resolve metadata; no unstratified holdout fallback')
    method, holdout = chosen
    split = eligible[['subject']].copy()
    split['split'] = 'development'
    split.loc[holdout, 'split'] = 'holdout'
    validate_split(split, eligible)
    write_tsv(c, relative, split)
    summary = {label: describe(eligible[split.split == label]) for label in ['development', 'holdout']}
    write_json(c, 'provenance/subject_split_v1.json',
               {'cohort_definition': COHORT_DEFINITION, 'minimum_multitask_n': c.analysis['minimum_multitask_n'],
                'core_tasks': CORE_TASKS, 'seed': c.analysis['seed'], 'holdout_n': 50, 'development_n': len(eligible)-50,
                'stratification_variables': ['age', 'FlipAngle'], 'method': method,
                'fallback_attempts': attempts, 'distributions': summary, 'sha256': sha256(path)})
    print(f'  Locked exactly 50 holdout participants using {method}. Fallbacks: {attempts}', flush=True)
    return split


def make_folds(c, development, guard, development_subjects, split_hash):
    guard.check(development.subject, development_subjects)
    frame = development.sort_values('subject').reset_index(drop=True)
    path = c.output('work/splits/development_folds.tsv')
    metadata = c.output('provenance/development_folds.json')
    if path.exists() and metadata.exists():
        meta = json.loads(metadata.read_text())
        if meta['split_sha256'] == split_hash:
            folds = pd.read_csv(path, sep='\t', dtype={'subject': str, 'fold': int})
            if (sha256(path) != meta['sha256'] or set(folds.subject) != set(frame.subject)
                    or folds.subject.duplicated().any() or set(folds.fold) != set(range(1, 6))):
                raise PipelineError('Persisted development folds failed integrity validation')
            return folds
    selected, method = None, 'participant_KFold_unstratified'
    attempts = []
    for name, labels in strata_candidates(frame):
        if labels.value_counts().min() >= 5:
            selected = StratifiedKFold(5, shuffle=True, random_state=c.analysis['seed']).split(frame, labels)
            method = name
            break
        attempts.append(name + ': fewer than five participants per stratum')
    if selected is None:
        selected = KFold(5, shuffle=True, random_state=c.analysis['seed']).split(frame)
    folds = frame[['subject']].copy()
    folds['fold'] = 0
    for fold, (_, test) in enumerate(selected, 1): folds.loc[test, 'fold'] = fold
    write_tsv(c, 'work/splits/development_folds.tsv', folds)
    write_json(c, 'provenance/development_folds.json',
               {'seed': c.analysis['seed'], 'method': method, 'fallback_attempts': attempts,
                'sha256': sha256(path), 'split_sha256': split_hash,
                'folds': {str(k): describe(frame[folds.fold == k]) for k in range(1, 6)}})
    print(f'  Five participant folds: {method}; fallbacks: {attempts}', flush=True)
    return folds


def guard_from_split(split):
    return DevelopmentGuard(frozenset(split.loc[split.split == 'development', 'subject']),
                            frozenset(split.loc[split.split == 'holdout', 'subject']))
