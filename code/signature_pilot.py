"""One fixed LinearSVC, out-of-participant scoring, and no holdout scoring API."""
from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import LinearSVC
from build_mask import vectorize_source, save_image, reconstruct
from inventory import InputUnavailable
from utils import PipelineError, write_tsv, write_json, sha256

PREDICTION_COLUMNS = ['subject', 'fold', 'analysis', 'positive_score', 'negative_score', 'paired_margin', 'correct']


def representations(task, maps):
    """All arithmetic operates on existing COPEs; no FEAT design is generated."""
    if task == 'sharedreward':
        computer, friend, stranger = maps[2]-maps[1], maps[4]-maps[3], maps[6]-maps[5]
        result = {'primary': ((friend+stranger)/2, computer)}
        if 7 in maps: result['neutral'] = ((maps[8]+maps[9])/2, maps[7])
        if 27 in maps: result['decision'] = ((maps[27]+maps[28])/2, maps[29])
        return result
    if task == 'trust':
        computer, friend, stranger = maps[5]-maps[4], maps[7]-maps[6], maps[9]-maps[8]
        return {'trust': ((friend+stranger)/2, computer), 'trust_friend_computer': (friend, computer),
                'trust_stranger_computer': (stranger, computer), 'trust_friend_stranger': (friend, stranger)}
    if task == 'ugr':
        return {'ugr_pmod': (maps[14], maps[13]), 'ugr_constant': ((maps[5]+maps[7])/2, (maps[1]+maps[3])/2)}
    raise PipelineError('Unknown representation')


def center_pair(pair):
    return np.stack([np.asarray(v, dtype=np.float32) - np.float32(np.mean(v, dtype=np.float64)) for v in pair])


@dataclass
class FoldModel:
    estimator: LinearSVC
    training_subjects: frozenset
    fold: int


def fit_model(x, subjects, c, guard, development_subjects, fold):
    guard.check(subjects, development_subjects)
    if x.ndim != 3 or x.shape[:2] != (len(subjects), 2) or len(set(subjects)) != len(subjects):
        raise PipelineError('Expected one social/computer pair per distinct participant')
    if not np.isfinite(x).all(): raise PipelineError('Nonfinite model matrix')
    estimator = LinearSVC(**c.analysis['classifier'], random_state=c.analysis['seed'])
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator.fit(x.reshape(-1, x.shape[-1]), np.tile([1, -1], len(subjects)))
    if not np.array_equal(estimator.classes_, [-1, 1]): raise PipelineError('Unexpected class orientation')
    return FoldModel(estimator, frozenset(subjects), fold)


def score_pair(pair, subject, model, guard, development_subjects, analysis):
    guard.check([subject], development_subjects)
    if subject in model.training_subjects:
        raise PipelineError('OUT-OF-FOLD LOCK: participant contributed to this model')
    values = model.estimator.decision_function(center_pair(pair))
    margin = float(values[0] - values[1])
    return {'subject': subject, 'fold': model.fold, 'analysis': analysis,
            'positive_score': float(values[0]), 'negative_score': float(values[1]),
            'paired_margin': margin, 'correct': bool(margin > 0)}


def load_maps(c, subject, task, mask, reference, guard, development_subjects, diagnostics, level='L2', run=None):
    guard.check([subject], development_subjects)
    keys = range(1, 7) if task == 'sharedreward' and level == 'L1' else c.contrasts[task]['copes']
    return {k: vectorize_source(c, subject, task, k, mask, reference, guard, development_subjects,
                               diagnostics, level, run) for k in keys}


def run_models(c, inventory, folds, mask, reference, guard, development_subjects, versions, repos):
    subjects = sorted(development_subjects)
    guard.check(subjects, development_subjects)
    fold_map = folds.set_index('subject').fold.to_dict()
    if set(fold_map) != set(subjects): raise PipelineError('Fold membership differs from development cohort')
    diagnostics, image_metrics, primary_rows, cross_rows, missing = [], [], [], [], []
    x = np.empty((len(subjects), 2, int(mask.sum())), dtype=np.float32)
    for index, subject in enumerate(subjects):
        maps = load_maps(c, subject, 'sharedreward', mask, reference, guard, development_subjects, diagnostics)
        x[index] = center_pair(representations('sharedreward', maps)['primary'])
        for label, vector in zip(['social', 'computer'], x[index]):
            image_metrics.append({'subject': subject, 'analysis': 'primary', 'condition': label,
                                  'norm': float(np.linalg.norm(vector)), 'spatial_mean': float(vector.mean())})
    models, weights = {}, []
    for fold in range(1, 6):
        test = np.array([fold_map[s] == fold for s in subjects])
        train_subjects = [s for s, keep in zip(subjects, ~test) if keep]
        model = fit_model(x[~test], train_subjects, c, guard, development_subjects, fold)
        models[fold] = model
        vector = model.estimator.coef_.ravel().astype(np.float32)
        weights.append(vector)
        save_image(c, f'results/maps/fold-{fold}_weights.nii.gz', reconstruct(vector, mask), reference)
        for index in np.flatnonzero(test):
            primary_rows.append(score_pair(x[index], subjects[index], model, guard, development_subjects, 'primary'))
        print(f'  CV fold {fold}/5 complete; held out N={int(test.sum())}', flush=True)
    availability = inventory.set_index('subject')
    for subject in subjects:
        model = models[fold_map[subject]]
        for task in ['sharedreward', 'trust', 'ugr', 'socialdoors']:
            available = bool(availability.loc[subject, task])
            if task == 'socialdoors': available &= bool(availability.loc[subject, 'doors'])
            if not available:
                missing.append({'subject': subject, 'task': task, 'reason': 'unavailable_at_inventory'})
                continue
            try:
                if task == 'socialdoors':
                    social = load_maps(c, subject, task, mask, reference, guard, development_subjects, diagnostics, 'L1', 1)[4]
                    computer = load_maps(c, subject, 'doors', mask, reference, guard, development_subjects, diagnostics, 'L1', 1)[4]
                    pairs = {'socialdoors': (social, computer)}
                else:
                    pairs = representations(task, load_maps(c, subject, task, mask, reference, guard, development_subjects, diagnostics))
                    pairs.pop('primary', None)
                for name, pair in pairs.items():
                    cross_rows.append(score_pair(pair, subject, model, guard, development_subjects, name))
                    for label, vector in zip(['positive', 'negative'], center_pair(pair)):
                        image_metrics.append({'subject': subject, 'analysis': name, 'condition': label,
                                              'norm': float(np.linalg.norm(vector)), 'spatial_mean': float(vector.mean())})
            except (InputUnavailable, OSError) as exc:
                if task == 'sharedreward': raise
                missing.append({'subject': subject, 'task': task, 'reason': 'voxel_or_io_failure'})
        for run in [1, 2]:
            maps = load_maps(c, subject, 'sharedreward', mask, reference, guard, development_subjects, diagnostics, 'L1', run)
            pair = representations('sharedreward', maps)['primary']
            cross_rows.append(score_pair(pair, subject, model, guard, development_subjects, f'run{run}'))
            for label, vector in zip(['positive', 'negative'], center_pair(pair)):
                image_metrics.append({'subject': subject, 'analysis': f'run{run}', 'condition': label,
                                      'norm': float(np.linalg.norm(vector)), 'spatial_mean': float(vector.mean())})
    primary = pd.DataFrame(primary_rows, columns=PREDICTION_COLUMNS)
    cross = pd.DataFrame(cross_rows, columns=PREDICTION_COLUMNS)
    write_tsv(c, 'work/predictions/cv_predictions.tsv', primary)
    write_tsv(c, 'work/predictions/cross_task_predictions.tsv', cross)
    write_tsv(c, 'work/diagnostics/image_metrics.tsv', image_metrics)
    write_tsv(c, 'work/diagnostics/source_image_metrics.tsv', diagnostics)
    missing_frame = pd.DataFrame(missing, columns=['subject', 'task', 'reason'])
    write_tsv(c, 'work/diagnostics/cross_task_missing.tsv', missing_frame)
    write_tsv(c, 'results/aggregate/cross_task_missingness.tsv',
              missing_frame.groupby(['task', 'reason']).size().reset_index(name='n'))
    print('  All out-of-fold predictions complete. Fitting final development signature.', flush=True)
    final = fit_model(x, subjects, c, guard, development_subjects, 0)
    save_image(c, 'results/maps/social_reward_signature_DEV_weights.nii.gz',
               reconstruct(final.estimator.coef_.ravel().astype(np.float32), mask), reference)
    write_json(c, 'provenance/model.json',
               {'intercept': float(final.estimator.intercept_[0]), 'classes': {'social': 1, 'computer': -1},
                'estimator': 'sklearn.svm.LinearSVC', 'parameters': final.estimator.get_params(),
                'software_versions': versions, 'development_n': len(subjects),
                'mask_sha256': sha256(c.output('results/maps/analysis_mask.nii.gz')),
                'split_sha256': sha256(c.output('work/splits/subject_split_v1.tsv')),
                'preprocessing': 'Existing COPE arithmetic; subtract each representation spatial mean inside fixed mask; no unit normalization',
                'source_repository_shas': {row['repository']: row['head_sha'] for row in repos},
                'fold_models': [{'fold': f, 'intercept': float(m.estimator.intercept_[0]),
                                 'n_iter': int(m.estimator.n_iter_)} for f, m in models.items()],
                'n_iter': int(final.estimator.n_iter_), 'holdout_scored': False})
    return primary, cross, np.stack(weights), diagnostics
