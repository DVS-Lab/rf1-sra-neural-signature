"""Symmetric trans-task reward and secondary decision/context development models."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from build_mask import vectorize_source, save_image, reconstruct
from utils import REWARD_TASKS, COHORT_DEFINITION, PipelineError, write_tsv, write_json, sha256

FAMILIES = ('reward', 'decision')
PREDICTION_COLUMNS = ['subject', 'fold', 'analysis', 'train_family', 'train_tasks',
                      'test_family', 'test_task', 'model_scope', 'positive_score',
                      'negative_score', 'paired_margin', 'correct']


def representations(task, maps):
    """Existing COPE arithmetic. UGR is a valuation/context probe, never a reward training task."""
    if task == 'sharedreward':
        result = {'reward': (((maps[4]-maps[3])+(maps[6]-maps[5]))/2, maps[2]-maps[1])}
        if 7 in maps:
            result['neutral'] = ((maps[8]+maps[9])/2, maps[7])
        if 27 in maps:
            result['decision'] = ((maps[27]+maps[28])/2, maps[29])
        return result
    if task == 'trust':
        computer, friend, stranger = maps[5]-maps[4], maps[7]-maps[6], maps[9]-maps[8]
        result = {'reward': ((friend+stranger)/2, computer),
                  'friend_computer': (friend, computer), 'stranger_computer': (stranger, computer),
                  'friend_stranger': (friend, stranger)}
        if 1 in maps:
            result['decision'] = ((maps[2]+maps[3])/2, maps[1])
        return result
    if task == 'ugr':
        return {'ugr_pmod': (maps[14], maps[13]),
                'ugr_constant': ((maps[5]+maps[7])/2, (maps[1]+maps[3])/2)}
    raise PipelineError('Unknown representation')


def doors_representations(social, monetary):
    return {'reward': (social[4], monetary[4]), 'decision': (social[3], monetary[3])}


def center_pair(pair):
    return np.stack([np.asarray(v, dtype=np.float32) - np.float32(np.mean(v, dtype=np.float64)) for v in pair])


@dataclass
class FoldModel:
    estimator: LinearSVC
    training_subjects: frozenset
    fold: int
    training_tasks: tuple
    family: str


def fit_model(x, subjects, c, guard, development_subjects, fold,
              training_tasks=('sharedreward',), family='reward'):
    guard.check(subjects, development_subjects)
    if x.ndim == 3:
        x = x[:, None, :, :]
    tasks = tuple(training_tasks)
    if (x.ndim != 4 or x.shape[:3] != (len(subjects), len(tasks), 2)
            or len(set(subjects)) != len(subjects) or not tasks
            or len(set(tasks)) != len(tasks) or not set(tasks) <= set(REWARD_TASKS)
            or family not in FAMILIES):
        raise PipelineError('Expected paired social/nonsocial maps per unique participant and clean paradigm')
    if not np.isfinite(x).all():
        raise PipelineError('Nonfinite model matrix')
    estimator = LinearSVC(**c.analysis['classifier'], random_state=c.analysis['seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(x.reshape(-1, x.shape[-1]), np.tile([1, -1], len(subjects)*len(tasks)))
    if not np.array_equal(estimator.classes_, [-1, 1]):
        raise PipelineError('Unexpected class orientation')
    return FoldModel(estimator, frozenset(subjects), fold, tasks, family)


def score_pair(pair, subject, model, guard, development_subjects, analysis,
               test_family='', test_task='', model_scope='', unseen_task=False):
    guard.check([subject], development_subjects)
    if subject in model.training_subjects:
        raise PipelineError('OUT-OF-FOLD LOCK: participant contributed to this model in any paradigm')
    if unseen_task and test_task in model.training_tasks:
        raise PipelineError('UNSEEN-PARADIGM LOCK: test paradigm entered training')
    values = model.estimator.decision_function(center_pair(pair))
    margin = float(values[0] - values[1])
    return {'subject': subject, 'fold': model.fold, 'analysis': analysis,
            'train_family': model.family, 'train_tasks': '+'.join(model.training_tasks),
            'test_family': test_family, 'test_task': test_task, 'model_scope': model_scope,
            'positive_score': float(values[0]), 'negative_score': float(values[1]),
            'paired_margin': margin, 'correct': bool(margin > 0)}


def load_maps(c, subject, task, mask, reference, guard, development_subjects, diagnostics, level='L2', run=None):
    guard.check([subject], development_subjects)
    keys = range(1, 7) if task == 'sharedreward' and level == 'L1' else c.contrasts[task]['copes']
    return {k: vectorize_source(c, subject, task, k, mask, reference, guard, development_subjects,
                               diagnostics, level, run) for k in keys}


def load_cohort(c, subjects, mask, reference, guard, development_subjects):
    guard.check(subjects, development_subjects)
    shape = (len(subjects), len(REWARD_TASKS), 2, int(mask.sum()))
    arrays = {family: np.empty(shape, dtype=np.float32) for family in FAMILIES}
    probes = {key: np.empty((len(subjects), 2, shape[-1]), dtype=np.float32)
              for key in ['neutral', 'ugr_pmod', 'ugr_constant', 'run1', 'run2',
                          'friend_computer', 'stranger_computer', 'friend_stranger']}
    diagnostics, metrics = [], []
    for i, subject in enumerate(subjects):
        sources = {task: load_maps(c, subject, task, mask, reference, guard, development_subjects,
                                  diagnostics, 'L1' if task in ['socialdoors', 'doors'] else 'L2',
                                  1 if task in ['socialdoors', 'doors'] else None)
                   for task in ['sharedreward', 'trust', 'socialdoors', 'doors', 'ugr']}
        pairs = {'sharedreward': representations('sharedreward', sources['sharedreward']),
                 'trust': representations('trust', sources['trust']),
                 'socialdoors': doors_representations(sources['socialdoors'], sources['doors'])}
        for family in FAMILIES:
            for j, task in enumerate(REWARD_TASKS):
                arrays[family][i, j] = center_pair(pairs[task][family])
                for label, vector in zip(['social', 'nonsocial'], arrays[family][i, j]):
                    metrics.append({'subject': subject, 'family': family, 'task': task,
                                    'condition': label, 'norm': float(np.linalg.norm(vector)),
                                    'spatial_mean': float(vector.mean())})
        probe_pairs = {'neutral': pairs['sharedreward']['neutral'], **representations('ugr', sources['ugr'])}
        probe_pairs.update({key: pairs['trust'][key] for key in ['friend_computer', 'stranger_computer', 'friend_stranger']})
        for run in [1, 2]:
            maps = load_maps(c, subject, 'sharedreward', mask, reference, guard, development_subjects,
                             diagnostics, 'L1', run)
            probe_pairs[f'run{run}'] = representations('sharedreward', maps)['reward']
        for key, pair in probe_pairs.items():
            probes[key][i] = center_pair(pair)
            for label, vector in zip(['positive', 'negative'], probes[key][i]):
                metrics.append({'subject': subject, 'family': 'probe', 'task': key,
                                'condition': label, 'norm': float(np.linalg.norm(vector)),
                                'spatial_mean': float(vector.mean())})
    # A core voxel failure aborts rather than silently changing the already locked cohort.
    write_tsv(c, 'work/diagnostics/image_metrics.tsv', metrics)
    write_tsv(c, 'work/diagnostics/source_image_metrics.tsv', diagnostics)
    return arrays, probes, diagnostics


def run_models(c, inventory, folds, mask, reference, guard, development_subjects, versions, repos):
    subjects = sorted(development_subjects)
    guard.check(subjects, development_subjects)
    if not set(subjects) <= set(inventory.loc[inventory.eligible, 'subject']):
        raise PipelineError('Modeling requires the locked multitask-complete development cohort')
    fold_map = folds.set_index('subject').fold.to_dict()
    if set(fold_map) != set(subjects):
        raise PipelineError('Fold membership differs from development cohort')
    arrays, probes, diagnostics = load_cohort(c, subjects, mask, reference, guard, development_subjects)
    rows, audit, model_metadata = [], [], []
    weights = {}

    def save_model(model, key):
        vector = model.estimator.coef_.ravel().astype(np.float32)
        suffix = f'fold-{model.fold}' if model.fold else 'DEV'
        relative = f'results/maps/{key}_{suffix}_weights.nii.gz'
        save_image(c, relative, reconstruct(vector, mask), reference)
        weights.setdefault(key, {})[model.fold] = vector
        model_metadata.append({'model': key, 'fold': model.fold, 'family': model.family,
                               'training_tasks': model.training_tasks, 'training_n': len(model.training_subjects),
                               'intercept': float(model.estimator.intercept_[0]),
                               'n_iter': int(model.estimator.n_iter_), 'map': relative,
                               'map_sha256': sha256(c.output(relative))})

    for fold in range(1, 6):
        test = np.array([fold_map[s] == fold for s in subjects])
        train_subjects = [s for s, keep in zip(subjects, ~test) if keep]
        test_indices = np.flatnonzero(test)

        def fit(family, tasks, key):
            indices = [REWARD_TASKS.index(task) for task in tasks]
            model = fit_model(arrays[family][~test][:, indices], train_subjects, c, guard,
                              development_subjects, fold, tasks, family)
            save_model(model, key)
            audit.extend({'model': key, 'fold': fold, 'subject': s, 'role': 'train'} for s in train_subjects)
            audit.extend({'model': key, 'fold': fold, 'subject': subjects[i], 'role': 'test'} for i in test_indices)
            return model

        def score(model, pairs, test_family, task, scope, unseen=False):
            name = f'{model.family}:{scope}:{"+".join(model.training_tasks)}->{test_family}:{task}'
            for i in test_indices:
                rows.append(score_pair(pairs[i], subjects[i], model, guard, development_subjects,
                                       name, test_family, task, scope, unseen))

        for family in FAMILIES:
            # Nine cells share the same participant folds, including every off-diagonal test.
            for train_task in REWARD_TASKS:
                model = fit(family, (train_task,), f'{family}_task-{train_task}')
                for j, test_task in enumerate(REWARD_TASKS):
                    score(model, arrays[family][:, j], family, test_task, 'pairwise', train_task != test_task)
            if family == 'reward':
                for held_task in REWARD_TASKS:
                    training = tuple(task for task in REWARD_TASKS if task != held_task)
                    model = fit(family, training, f'reward_lopo-{held_task}')
                    score(model, arrays[family][:, REWARD_TASKS.index(held_task)], family,
                          held_task, 'lopo', unseen=True)
            common = fit(family, REWARD_TASKS, f'common_{family}_signature')
            for j, task in enumerate(REWARD_TASKS):
                score(common, arrays[family][:, j], family, task, 'common')
                other = 'decision' if family == 'reward' else 'reward'
                score(common, arrays[other][:, j], other, task, 'cross_family')
            for name in ['ugr_pmod', 'ugr_constant', 'neutral']:
                score(common, probes[name], 'boundary', name, 'probe')
            if family == 'reward':
                for name in ['run1', 'run2']:
                    score(common, probes[name], 'reliability', name, 'reliability')
                for name in ['friend_computer', 'stranger_computer', 'friend_stranger']:
                    score(common, probes[name], 'trust_decomposition', name, 'probe')
        print(f'  Fold {fold}/5: task models, both matrices, reward LOPO and common-model probes complete; test N={int(test.sum())}', flush=True)

    predictions = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    write_tsv(c, 'work/predictions/all_oof_predictions.tsv', predictions)
    diagonal = (predictions.model_scope == 'pairwise') & (predictions.train_tasks == predictions.test_task)
    write_tsv(c, 'work/predictions/cv_predictions.tsv', predictions[diagonal])
    write_tsv(c, 'work/predictions/cross_task_predictions.tsv', predictions[~diagonal])
    write_tsv(c, 'work/diagnostics/model_membership.tsv', audit)
    print('  All CV predictions saved. Fitting development-only common and task signatures.', flush=True)
    for family in FAMILIES:
        for tasks, key in [(REWARD_TASKS, f'common_{family}_signature'),
                           *[((task,), f'{family}_task-{task}') for task in REWARD_TASKS]]:
            indices = [REWARD_TASKS.index(task) for task in tasks]
            model = fit_model(arrays[family][:, indices], subjects, c, guard, development_subjects,
                              0, tasks, family)
            save_model(model, key)
    write_json(c, 'provenance/model.json',
               {'architecture': 'trans_task_v2', 'cohort_definition': COHORT_DEFINITION,
                'classes': {'social': 1, 'nonsocial': -1}, 'primary_family': 'reward',
                'secondary_family': 'decision/context', 'reward_paradigms': REWARD_TASKS,
                'ugr_role': 'social valuation / broader decision-context boundary probe; never trained',
                'estimator': 'sklearn.svm.LinearSVC', 'parameters': model.estimator.get_params(),
                'software_versions': versions, 'development_n': len(subjects),
                'mask_sha256': sha256(c.output('results/maps/analysis_mask.nii.gz')),
                'split_sha256': sha256(c.output('work/splits/subject_split_v1.tsv')),
                'preprocessing': 'COPE arithmetic; per-map spatial mean centering; no unit normalization; equal pairs per participant and paradigm',
                'source_repository_shas': {row['repository']: row['head_sha'] for row in repos},
                'models': model_metadata, 'holdout_scored': False})
    return predictions, weights, diagnostics
