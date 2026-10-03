"""Participant-blocked v4 models: social reward, outcome context, and closeness."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from build_mask import vectorize_source, save_image, reconstruct
from revised_design import FAMILIES, partner_representations, doors_representations
from utils import PipelineError, write_tsv, write_json, sha256


@dataclass
class Model:
    estimator: LinearSVC
    subjects: frozenset
    tasks: tuple
    family: str
    fold: int


def fit(x, subjects, tasks, family, fold, c, guard):
    guard.check(subjects, guard.development)
    nclasses = 3 if family == 'partner_reward' else 2
    if x.shape[:3] != (len(subjects), len(tasks), nclasses) or not np.isfinite(x).all():
        raise PipelineError('Invalid revised training array')
    labels = np.arange(3) if nclasses == 3 else np.array([1, -1])
    estimator = LinearSVC(**c.analysis['classifier'], random_state=c.analysis['seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(x.reshape(-1, x.shape[-1]), np.tile(labels, len(subjects)*len(tasks)))
    return Model(estimator, frozenset(subjects), tuple(tasks), family, fold)


def score(model, pair, subject, guard, test_task, scope, unseen=False):
    guard.check([subject], guard.development)
    if subject in model.subjects: raise PipelineError('Participant leakage across revised tasks/conditions')
    if unseen and test_task in model.tasks: raise PipelineError('Test paradigm entered unseen-task training')
    values = model.estimator.decision_function(pair)
    row = {'subject': subject, 'fold': model.fold, 'family': model.family,
           'train_tasks': '+'.join(model.tasks), 'test_task': test_task, 'scope': scope,
           'predicted_labels': '', 'true_labels': ''}
    if model.family == 'partner_reward':
        truth = np.arange(3)
        predicted = model.estimator.predict(pair)
        others = values.copy()
        others[np.arange(3), truth] = -np.inf
        row.update(accuracy=float(np.mean(predicted == truth)), chance=1/3,
                   margin=float(np.mean(values[np.arange(3), truth]-others.max(axis=1))),
                   predicted_labels=','.join(map(str, predicted)), true_labels='0,1,2')
    else:
        margin = float(values[0]-values[1])
        row.update(accuracy=float(margin > 0), chance=.5, margin=margin)
    return row


def load_data(c, subjects, cohort, mask, reference, guard):
    tasks = ('sharedreward', 'trust') if cohort == 'partner_pair' else ('sharedreward', 'trust', 'socialdoors')
    families = FAMILIES[cohort]
    arrays = {family: [] for family in families}
    runs = {}
    diagnostics = []
    norms = []
    def maps(subject, task, level='L2', run=None):
        return {k: vectorize_source(c, subject, task, k, mask, reference, guard,
                                    subjects, diagnostics, level, run) for k in c.contrasts[task]['copes']}
    for index, subject in enumerate(subjects):
        sources = {task: partner_representations(task, maps(subject, task)) for task in ('sharedreward', 'trust')}
        if 'socialdoors' in tasks:
            sources['socialdoors'] = doors_representations(maps(subject, 'socialdoors', 'L1', 1),
                                                          maps(subject, 'doors', 'L1', 1))
        for family in families:
            arrays[family].append(np.stack([sources[task][family] for task in tasks]))
            for task in tasks:
                for label, vector in enumerate(sources[task][family]):
                    norms.append({'subject': subject, 'family': family, 'task': task,
                                  'class_index': label, 'norm': float(np.linalg.norm(vector))})
        # Repeated-run diagnostics use only accepted retained input runs, never a failed run.
        for task in ('sharedreward', 'trust'):
            if task == 'sharedreward' and len(c.aging_subjects[subject]['runs']) != 2: continue
            for run in (1, 2):
                # Cache the six source maps once, rather than all derived families.
                runs[(subject, task, run)] = maps(subject, task, 'L1', run)
        if index and index % 25 == 0: print(f'  Loaded revised development maps: {index}/{len(subjects)}', flush=True)
    write_tsv(c, 'work/source_image_metrics.tsv', diagnostics)
    write_tsv(c, 'work/map_norms.tsv', norms)
    return {key: np.stack(value) for key, value in arrays.items()}, runs, tasks


def run_models(c, cohort, folds, mask, reference, guard, versions, repos, sample_hash):
    subjects = sorted(guard.development)
    arrays, runs, tasks = load_data(c, subjects, cohort, mask, reference, guard)
    fold_map = folds.set_index('subject').fold.to_dict()
    if set(fold_map) != set(subjects): raise PipelineError('Revised folds disagree with development sample')
    predictions, run_predictions, metadata, membership = [], [], [], []
    def save(model, name):
        suffix = f'fold-{model.fold}' if model.fold else 'DEV'
        paths = []
        labels = ('computer', 'friend', 'stranger') if model.family == 'partner_reward' else ('positive',)
        for label, vector in zip(labels, model.estimator.coef_):
            path = f'results/maps/{name}_{suffix}_{label}_weights.nii.gz'
            save_image(c, path, reconstruct(vector.astype(np.float32), mask), reference)
            paths.append({'class': label, 'path': str(c.output(path).relative_to(c.root)),
                          'sha256': sha256(c.output(path))})
        metadata.append({'model': name, 'family': model.family, 'fold': model.fold,
                         'training_tasks': model.tasks, 'training_n': len(model.subjects),
                         'classes': model.estimator.classes_.tolist(), 'intercept': model.estimator.intercept_.tolist(),
                         'n_iter': int(model.estimator.n_iter_), 'maps': paths})
    for fold in range(1, 6):
        test = np.array([fold_map[s] == fold for s in subjects])
        train_subjects = [s for s, keep in zip(subjects, ~test) if keep]
        indices = np.flatnonzero(test)
        if not len(indices) or not train_subjects: raise PipelineError('Empty revised train/test fold')
        for family, x in arrays.items():
            specifications = [(f'task-{task}', (task,), 'pairwise') for task in tasks]
            if cohort == 'three_paradigm':
                specifications += [(f'lopo-{task}', tuple(t for t in tasks if t != task), 'lopo') for task in tasks]
            specifications += [('common', tasks, 'common')]
            for name, training_tasks, scope in specifications:
                task_indices = [tasks.index(t) for t in training_tasks]
                model = fit(x[~test][:, task_indices], train_subjects, training_tasks, family, fold, c, guard)
                save(model, family+'_'+name)
                membership.extend({'family': family, 'model': name, 'fold': fold, 'subject': s,
                                   'role': 'test' if fold_map[s] == fold else 'train'} for s in subjects)
                testing_tasks = (name.removeprefix('lopo-'),) if scope == 'lopo' else tasks
                for task in testing_tasks:
                    for i in indices:
                        predictions.append(score(model, x[i, tasks.index(task)], subjects[i], guard,
                                                 task, scope, unseen=task not in training_tasks or scope == 'lopo'))
                if scope == 'common' and family != 'partner_reward':
                    for i in indices:
                        for task in ('sharedreward', 'trust'):
                            for run in (1, 2):
                                key = (subjects[i], task, run)
                                if key in runs:
                                    pair = partner_representations(task, runs[key])[family]
                                    row = score(model, pair, subjects[i], guard, task, 'reliability')
                                    row['run'] = run
                                    run_predictions.append(row)
        print(f'  Revised fold {fold}/5 completed; test N={len(indices)}', flush=True)
    write_tsv(c, 'work/oof_predictions.tsv', predictions)
    write_tsv(c, 'work/run_predictions.tsv', run_predictions)
    write_tsv(c, 'work/model_membership.tsv', membership)
    # Final DEV fits are distribution artifacts only, never development performance estimates.
    for family, x in arrays.items():
        for name, training_tasks in [(f'task-{task}', (task,)) for task in tasks] + [('common', tasks)]:
            idx = [tasks.index(t) for t in training_tasks]
            model = fit(x[:, idx], subjects, training_tasks, family, 0, c, guard)
            save(model, family+'_'+name)
    write_json(c, 'provenance/model.json', {'architecture': 'v4_social_reward_context_closeness',
               'cohort': cohort, 'development_n': len(subjects), 'holdout_scored': False,
               'sample_manifest_sha256': sample_hash,
               'mask_sha256': sha256(c.output('results/maps/analysis_mask.nii.gz')),
               'classifier': c.analysis['classifier'], 'software_versions': versions,
               'source_repository_shas': {r['repository']: r['head_sha'] for r in repos},
               'preprocessing': 'Per-map spatial mean centering; no scaling or fitted feature selection',
               'multiclass_labels': {'0': 'computer', '1': 'friend', '2': 'stranger'},
               'models': metadata})
    return pd.DataFrame(predictions), pd.DataFrame(run_predictions)
