"""Read-only v4 audit. No neural outputs until all private membership checks pass."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from revised_design import revised_config, COHORTS, FAMILIES
from revised_samples import RevisedGuard, original_membership
from utils import Config, PipelineError, sha256, write_text

PRIMARY = 'tsnr_coverage_fd'
SENSITIVITY = 'prior_four_metric_policy'
PARAMETERS = dict(C=1.0, dual=True, class_weight=None, max_iter=100000, tol=.0001)
SEED = 20260928
TARGETS = {'three_paradigm': ('social_context', 'social_reward'),
           'partner_pair': ('closeness_positive', 'closeness_negative', 'valence')}
TASKS = {'three_paradigm': ('sharedreward', 'trust', 'socialdoors'),
         'partner_pair': ('sharedreward', 'trust')}


@dataclass
class CharacterizationConfig(Config):
    def output(self, relative):
        parts = Path(relative).parts
        if not parts or parts[0] not in ('work', 'results', 'reports', 'provenance') or '..' in parts:
            raise PipelineError('Characterization output must stay in its own namespace')
        return super().output('/'.join([parts[0], 'revised', 'characterization', *parts[1:]]))


def output_config(base):
    return CharacterizationConfig(**{k: getattr(base, k) for k in Config.__dataclass_fields__})


def require(condition, message):
    if not condition: raise PipelineError('CHARACTERIZATION AUDIT STOP: '+message)


def table(path):
    require(path.is_file(), 'required private/public v4 table is missing; restore original files')
    return pd.read_csv(path, sep='\t', keep_default_na=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def frozen_snapshot(base):
    """Hash existing results, reports, provenance and private inputs, never source voxels."""
    result = {}
    for tree in ('results', 'reports', 'provenance'):
        top = base.root/tree
        if not top.exists(): continue
        for p in sorted(top.rglob('*')):
            rel = p.relative_to(base.root)
            if not p.is_file() or tuple(rel.parts[1:3]) == ('revised', 'characterization'): continue
            result[str(rel)] = sha256(p)
    # Never sweep arbitrary private files: a user may have placed source/holdout
    # images in work/. Fingerprint only the explicitly required metadata tables.
    private=['work/splits/subject_split_v1.tsv','work/splits/development_folds.tsv',
             'work/revised/samples/manifest.tsv']
    for policy in (PRIMARY,SENSITIVITY):
        for cohort in TASKS:
            private.extend(f'work/revised/{policy}/{cohort}/{name}' for name in
                           ('model_membership.tsv','oof_predictions.tsv','run_predictions.tsv',
                            'source_image_metrics.tsv','map_norms.tsv','diagnostics/mask_sources.tsv'))
    for rel in private:
        if (base.root/rel).is_file(): result[rel]=sha256(base.root/rel)
    return result


def audit(base, spec):
    require(base.analysis['classifier'] == PARAMETERS and base.analysis['seed'] == SEED,
            'classifier or seed changed')
    for relative, expected in spec['frozen_code_sha256'].items():
        require(sha256(base.root/relative) == expected, 'frozen v4 implementation/config changed: '+relative)
    samples = revised_config(base, 'samples')
    meta = json.loads(samples.output('provenance/samples.json').read_text())
    status = json.loads(samples.output('provenance/run_status.json').read_text())
    require(status['status'] == 'complete' and status['holdout_scored'] is False, 'v4 run not complete/unscored')
    manifest_path = samples.output('work/manifest.tsv')
    require(manifest_path.is_file(), 'private revised sample manifest unavailable')
    require(sha256(manifest_path) == meta['manifest_sha256'] == status['sample_manifest_sha256'], 'sample fingerprint mismatch')
    split, old_folds, old_hash = original_membership(base)
    require(old_hash == meta['original_split_sha256'], 'original split changed')
    manifest = table(manifest_path)
    require(not manifest.duplicated(['policy', 'cohort', 'subject']).any(), 'duplicate membership')
    require(set(manifest.policy) == {PRIMARY, SENSITIVITY} and set(manifest.cohort) == set(TASKS), 'unexpected sample scopes')
    require(set(manifest.partition) == {'development', 'holdout'}, 'invalid partitions')
    require(manifest.groupby('subject').partition.nunique().eq(1).all(), 'participant changes partition between scopes')
    require(manifest.groupby('subject').fold.nunique().eq(1).all(), 'participant changes fold between scopes')
    held = frozenset(manifest.loc[manifest.partition == 'holdout', 'subject'])
    old_held = set(split.loc[split.split == 'holdout', 'subject'])
    old_dev = set(split.loc[split.split == 'development', 'subject'])
    require(old_held <= held and not held & old_dev, 'original assignments violated')
    old_fold_map = old_folds.set_index('subject').fold.to_dict()
    for r in manifest[manifest.partition == 'development'].itertuples():
        expected = old_fold_map.get(r.subject, 1+int(hashlib.sha256(f'{SEED}:{r.subject}'.encode()).hexdigest()[:16], 16)%5)
        require(r.fold == expected, 'frozen development fold changed')
    require(manifest.loc[manifest.partition == 'holdout', 'fold'].eq(0).all(), 'holdout assigned a training fold')
    scopes = {}
    for policy in (PRIMARY, SENSITIVITY):
        for cohort, tasks in TASKS.items():
            c = revised_config(base, f'{policy}/{cohort}')
            group = manifest[(manifest.policy == policy)&(manifest.cohort == cohort)]
            require(group.eligible.isin([True, False]).all(), 'invalid eligibility flag')
            accepted = group[group.eligible.eq(True)]
            subjects = sorted(accepted.loc[accepted.partition == 'development', 'subject'])
            guard = RevisedGuard(frozenset(subjects), held)
            guard.check(subjects, subjects)
            if policy == PRIMARY:
                require(len(subjects) == spec['expected_development_n'][cohort], 'primary sample N differs from completed v4')
                if cohort == 'partner_pair':
                    require(accepted.partition.eq('holdout').sum() == 50, 'primary qualified validation N must remain 50')
            folds = accepted[accepted.partition == 'development'].set_index('subject').fold.to_dict()
            model = json.loads(c.output('provenance/model.json').read_text())
            run = json.loads(c.output('provenance/run_status.json').read_text())
            mask = json.loads(c.output('provenance/analysis_mask.json').read_text())
            require(run['status'] == 'complete' and run['holdout_scored'] is False and model['holdout_scored'] is False,
                    'scope not complete or holdout protection violated')
            require(model['classifier'] == PARAMETERS and model['development_n'] == len(subjects), 'model contract mismatch')
            require(model['sample_manifest_sha256'] == meta['manifest_sha256'], 'model/sample fingerprint mismatch')
            require(sha256(c.output('results/maps/analysis_mask.nii.gz')) == mask['sha256'] == model['mask_sha256'], 'mask hash mismatch')
            require(mask['coverage_threshold'] == .95 and mask['development_n'] == len(subjects)
                    and mask['coverage_tasks'] == list(COHORTS[cohort]), 'mask cohort/coverage mismatch')
            membership = table(c.output('work/model_membership.tsv'))
            predictions = table(c.output('work/oof_predictions.tsv'))
            runs = table(c.output('work/run_predictions.tsv'))
            for frame in (membership, predictions, runs):
                guard.check(frame.subject, subjects)
            require(set(predictions.subject) == set(subjects), 'OOF coverage incomplete')
            models = {(m['model'], m['fold']): m for m in model['models']}
            require(len(models) == len(model['models']), 'duplicate saved models')
            expected_models = set()
            for family in FAMILIES[cohort]:
                names = ['task-'+t for t in tasks]+['common']
                for fold in range(6):
                    for name in names + (['lopo-'+t for t in tasks] if fold and cohort == 'three_paradigm' else []):
                        expected_models.add((family+'_'+name, fold))
            require(set(models) == expected_models, 'saved model set incomplete')
            require(len(membership) == sum(fold != 0 for _,fold in expected_models)*len(subjects),
                    'extra or missing model membership records')
            require(len(predictions) == (42 if cohort == 'partner_pair' else 30)*len(subjects),
                    'extra or missing OOF records')
            for (name, fold), m in models.items():
                train_tasks = tuple(m['training_tasks'])
                short = name.removeprefix(m['family']+'_')
                expected_tasks = ((short[5:],) if short.startswith('task-') else
                                  tuple(t for t in tasks if t != short[5:]) if short.startswith('lopo-') else tasks)
                require(train_tasks == expected_tasks, 'training task contract / LOPO exclusion violated')
                train = {s for s in subjects if not fold or folds[s] != fold}
                require(m['training_n'] == len(train), 'training count mismatch')
                require(m['classes'] == ([0, 1, 2] if m['family'] == 'partner_reward' else [-1, 1]), 'class order mismatch')
                for weight in m['maps']:
                    path = base.root/weight['path']
                    require(path.resolve().is_relative_to(c.output('results/maps')), 'model map outside its scope')
                    require(sha256(path) == weight['sha256'], 'coefficient fingerprint mismatch')
                if not fold: continue
                rows = membership[(membership.family == m['family'])&(membership.model == short)&(membership.fold == fold)]
                require(len(rows) == len(subjects) and not rows.subject.duplicated().any(), 'membership missing/duplicated')
                require(set(rows.loc[rows.role == 'train', 'subject']) == train, 'training membership disagrees with folds')
                require(set(rows.loc[rows.role == 'test', 'subject']) == set(subjects)-train, 'test membership/leakage problem')
            for keys, rows in predictions.groupby(['family', 'train_tasks', 'test_task', 'scope']):
                family, training, test_task, scope = keys
                name = 'common' if scope == 'common' else 'lopo-'+test_task if scope == 'lopo' else 'task-'+training
                require(len(rows) == len(subjects) and set(rows.subject) == set(subjects), 'OOF cell incomplete/duplicated')
                for r in rows.itertuples():
                    require(r.fold == folds[r.subject], 'OOF participant in wrong fold')
                    m = models[(family+'_'+name, r.fold)]
                    require('+'.join(m['training_tasks']) == training, 'OOF model/task mismatch')
                    if scope == 'lopo': require(test_task not in m['training_tasks'], 'LOPO test task in training')
                if family != 'partner_reward':
                    require(np.array_equal(rows.accuracy.to_numpy(float), (rows.margin.to_numpy(float)>0).astype(float)), 'OOF accuracy/margin mismatch')
            perf = table(c.output('results/aggregate/performance.tsv'))
            require(len(perf) == (42 if cohort == 'partner_pair' else 30), 'performance cell count mismatch')
            for r in perf.itertuples():
                rows = predictions[(predictions.family == r.family)&(predictions.train_tasks == r.train_tasks)&
                                   (predictions.test_task == r.test_task)&(predictions.scope == r.scope)]
                require(len(rows) == r.n and np.isclose(rows.accuracy.mean(), r.accuracy, atol=1e-12), 'public accuracy not reproduced from private OOF')
            for relative in ('work/source_image_metrics.tsv', 'work/diagnostics/mask_sources.tsv'):
                evidence = table(c.output(relative))
                guard.check(evidence.subject, subjects)
                require(set(evidence.subject) == set(subjects), 'image/mask audit lacks development participants')
                for r in evidence.itertuples():
                    path = Path(r.path).resolve()
                    require(r.subject in path.parts and 'ses-01' in path.parts, 'source path/participant mismatch')
                    require(any(path.is_relative_to(p.resolve()) for p in c.repos.values()), 'source image outside configured roots')
                    require(not any(part.startswith('sub-') and part != r.subject for part in path.parts), 'cross-participant source path')
            for r in runs.itertuples():
                require(r.fold == folds[r.subject] and r.scope == 'reliability' and r.run in (1, 2), 'run score fold/role mismatch')
                require(r.family in FAMILIES[cohort] and r.family != 'partner_reward'
                        and r.test_task in ('sharedreward','trust') and r.train_tasks == '+'.join(tasks),
                        'run prediction model/task mismatch')
            require(not runs.duplicated(['subject', 'family', 'test_task', 'run']).any(), 'duplicate run predictions')
            scopes[(policy, cohort)] = dict(c=c, guard=guard, subjects=subjects, folds=folds, model=model,
                                            models=models, predictions=predictions, runs=runs, performance=perf)
    return scopes


AUDIT_TEXT = '''# Implementation and leakage audit\n\n{status}\n\n- Frozen implementation/config hashes checked against the completed v4 code.\n- Estimator: sklearn.svm.LinearSVC(C=1.0, dual=True, class_weight=None, max_iter=100000, tol=0.0001, random_state=20260928).\n- Private manifests, original split/folds, per-model train/test membership, OOF records, source-load logs and mask-source logs are required. Every test participant must be absent from training across all tasks; all maps retain one participant fold. LOPO excludes the entire target paradigm.\n- All original validation participants remain protected. Primary QC-qualified validation N=50 is checked; no validation scoring path is implemented.\n- Per-map spatial centering is independent. No population scaling, feature selection, or hyperparameter search occurs.\n- Frozen masks use 95% development coverage without condition labels. Coverage includes development participants later tested in CV: this is a fixed, label-free development mask, not a strictly training-fold-only mask. It is retained as requested; no validation data contributes.\n- Saved coefficient/mask hashes and OOF aggregate accuracy must match. Reconstruction from float32 saved fold weights is additionally checked before scientific map/plot outputs, using the documented numerical tolerance.\n- Source-load logs and code can establish the pipeline's recorded behavior, not unrecorded manual data access outside this pipeline.\n\nholdout_scored = False\n'''


def write_audit(out, status):
    write_text(out, 'reports/IMPLEMENTATION_AUDIT.md', AUDIT_TEXT.format(status=status))
