"""Fresh reward/context inventory and QC-qualified samples; original holdout immutable."""
import hashlib
from dataclasses import dataclass
import json
from itertools import combinations
import numpy as np
import pandas as pd
from aging_source import load_aging_index
from inventory import flip_angle, inspect_unit
from review_qc_attrition import normalized_qc, task_status, POLICIES
from revised_design import COHORTS, VERSION, revised_config
from utils import DevelopmentGuard, InputUnavailable, PipelineError, sha256, subject_id, write_tsv, write_json


def inventory(c, write=True):
    load_aging_index(c, refresh=True)
    people = pd.read_csv(c.bids/'participants.tsv', sep='\t', dtype={'participant_id': str})
    if people.participant_id.duplicated().any(): raise PipelineError('Duplicate canonical participants')
    people = people.set_index('participant_id')
    rows, geometry = [], []
    for i, folder in enumerate(sorted(c.bids.glob('sub-*'))):
        if not folder.is_dir(): continue
        subject = subject_id(folder.name)
        if (c.exclusions/f'Smith-SRA-{subject[4:]}').is_dir(): continue
        if i and i % 25 == 0: print(f'  Revised header inventory: {i} participants', flush=True)
        meta = people.loc[subject] if subject in people.index else pd.Series(dtype=object)
        age = pd.to_numeric(meta.get(c.paths['age_column'], np.nan), errors='coerce')
        row = {'subject': subject, 'age': age if np.isfinite(age) else np.nan,
               'sex': str(meta.get(c.paths['sex_column'], 'missing'))}
        try: row['flip_angle'], row['metadata_flag'] = flip_angle(c, subject)
        except (ValueError, OSError, TypeError): row['flip_angle'], row['metadata_flag'] = 'missing', 'invalid_bids_metadata'
        entry = c.aging_subjects.get(subject)
        row['sharedreward_runs'] = ','.join(map(str, entry['runs'])) if entry else ''
        row['sharedreward_two_runs'] = bool(entry and len(entry['runs']) == 2)
        for task in c.contrasts:
            reasons = []
            units = [('L1', 1)] if task in ('socialdoors', 'doors') else [('L2', None)]
            if task == 'sharedreward':
                units = [('L1', r) for r in entry['runs']] if entry else []
                if entry and len(entry['runs']) == 2: units.append(('L2', None))
                if not entry: reasons.append('not_in_verified_aging_manifest')
            for level, run in units:
                try: geometry.extend(inspect_unit(c, subject, task, level, run, required_copes_only=True))
                except InputUnavailable as exc: reasons.append(f'{level}_{run or 0}:{exc}')
            row[task] = not reasons
            row[task+'_reason'] = ';'.join(sorted(set(reasons)))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty: raise PipelineError('Empty revised inventory')
    summary = []
    tasks = tuple(c.contrasts)
    for size in range(1, len(tasks)+1):
        for combo in combinations(tasks, size):
            summary.append({'tasks': '+'.join(combo), 'n': int(frame[list(combo)].all(axis=1).sum())})
    if write:
        write_tsv(c, 'work/task_inventory.tsv', frame)
        write_tsv(c, 'work/image_headers.tsv', geometry)
        write_tsv(c, 'results/task_overlap.tsv', summary)
        write_json(c, 'provenance/sharedreward_source.json', c.aging_provenance)
    return frame


def original_membership(base):
    path = base.output('work/splits/subject_split_v1.tsv')
    meta_path = base.output('provenance/subject_split_v1.json')
    folds_path = base.output('work/splits/development_folds.tsv')
    fold_meta = base.output('provenance/development_folds.json')
    if not all(p.is_file() for p in (path, meta_path, folds_path, fold_meta)):
        raise PipelineError('Restore original private split and folds with their provenance; never redraw')
    meta = json.loads(meta_path.read_text())
    fm = json.loads(fold_meta.read_text())
    if meta.get('cohort_definition') != 'multitask_complete_v3_aging_fulltrial':
        raise PipelineError('Expected the completed v3 original split, not a legacy cohort')
    if meta['sha256'] != sha256(path) or fm['split_sha256'] != sha256(path) or fm['sha256'] != sha256(folds_path):
        raise PipelineError('Original split/fold fingerprint mismatch')
    split = pd.read_csv(path, sep='\t', dtype=str)
    folds = pd.read_csv(folds_path, sep='\t', dtype={'subject': str, 'fold': int})
    if (list(split.columns) != ['subject', 'split'] or split.subject.duplicated().any()
            or set(split.split) != {'development', 'holdout'} or split.split.eq('holdout').sum() != 50):
        raise PipelineError('Invalid original N=50 locked split')
    old_dev = set(split.loc[split.split == 'development', 'subject'])
    if folds.subject.duplicated().any() or set(folds.subject) != old_dev or set(folds.fold) != set(range(1, 6)):
        raise PipelineError('Invalid original participant folds')
    return split, folds, sha256(path)


@dataclass(frozen=True)
class RevisedGuard(DevelopmentGuard):
    def __post_init__(self):
        if not self.development or self.development & self.holdout or len(self.holdout) < 50:
            raise PipelineError('Invalid revised protected holdout pool')


def reconstruct(base, frame, policies, include_new, write=True, holdout_mode='top_up_pair'):
    split, folds, old_hash = original_membership(base)
    qc_path = base.repos['linux2']/base.paths['qc_table']
    qc = normalized_qc(pd.read_csv(qc_path, sep='\t', dtype=str, keep_default_na=False))
    old = split.set_index('subject').split.to_dict()
    fold_map = folds.set_index('subject').fold.to_dict()
    original_held = frozenset(split.loc[split.split == 'holdout', 'subject'])
    if frame.subject.duplicated().any(): raise PipelineError('Duplicate revised inventory subjects')
    records = []
    available = frame.set_index('subject').to_dict('index')
    if holdout_mode not in ('top_up_pair', 'original_only'): raise PipelineError('Invalid holdout policy')
    primary = policies[0]
    def pair_passes(subject):
        row = available.get(subject)
        return bool(row and all(task_status(qc, {'subject': subject, **row}, task, POLICIES[primary]) == 'pass'
                                for task in COHORTS['partner_pair']))
    survivors = sorted(s for s in original_held if pair_passes(s))
    candidates = sorted(s for s in available if s not in old and pair_passes(s))
    needed = 50-len(survivors) if holdout_mode == 'top_up_pair' else 0
    if len(candidates) < needed:
        raise PipelineError(f'Only {len(survivors)} original and {len(candidates)} new QC-qualified unseen candidates; cannot lock N=50. No voxels read.')
    supplemental = sorted(np.random.default_rng(base.analysis['seed']).choice(candidates, size=needed, replace=False).tolist())
    held = original_held | frozenset(supplemental)
    # An orphaned prior revision could have exposed new participants. Never rebuild
    # a missing sample manifest after any revised voxel-analysis run has started,
    # including an interrupted run that never reached model provenance output.
    if not revised_config(base, 'samples').output('work/manifest.tsv').exists():
        if (list((base.root/'provenance/revised').glob('*/*/model.json'))
                or revised_config(base, 'samples').output('provenance/run_status.json').exists()
                or list((base.root/'provenance/revised').glob('*/*/run_status.json'))):
            raise PipelineError('Revised analysis outputs exist without their sample lock; restore the private manifest')
    for subject in sorted(set(available) | set(old)):
        row = available.get(subject)
        original = old.get(subject, 'new_candidate')
        assignment = 'holdout' if subject in held else 'development'
        fold = fold_map.get(subject, 1 + int(hashlib.sha256(
            f'{base.analysis["seed"]}:{subject}'.encode()).hexdigest()[:16], 16) % 5)
        for policy in policies:
            statuses = {task: task_status(qc, {'subject': subject, **row}, task, POLICIES[policy])
                        if row else 'unavailable_input' for task in ('sharedreward', 'trust', 'socialdoors', 'doors')}
            for cohort, tasks in COHORTS.items():
                reasons = [task+':'+statuses[task] for task in tasks if statuses[task] != 'pass']
                if original == 'new_candidate' and assignment == 'development' and not include_new: reasons.append('new_participants_not_enabled')
                records.append({'subject': subject, 'policy': policy, 'cohort': cohort,
                                'original_partition': original, 'partition': assignment,
                                'eligible': not reasons, 'fold': 0 if assignment == 'holdout' else fold,
                                'reason': ';'.join(reasons)})
    manifest = pd.DataFrame(records)
    scope = revised_config(base, 'samples')
    path = scope.output('work/manifest.tsv')
    meta_path = scope.output('provenance/samples.json')
    identity = {'version': VERSION, 'original_split_sha256': old_hash,
                'original_folds_sha256': sha256(base.output('work/splits/development_folds.tsv')),
                'qc_sha256': sha256(qc_path), 'policies': list(policies), 'include_new_development': include_new,
                'holdout_mode': holdout_mode, 'supplemental_holdout_n': len(supplemental),
                'primary_qualified_original_holdout_n': len(survivors), 'protected_holdout_total_n': len(held)}
    if path.exists() or meta_path.exists():
        if not (path.exists() and meta_path.exists()): raise PipelineError('Restore revised manifest and provenance together')
        saved_meta = json.loads(meta_path.read_text())
        if any(saved_meta.get(k) != v for k, v in identity.items()) or saved_meta['manifest_sha256'] != sha256(path):
            raise PipelineError('Revised sample configuration/source drift; no automatic overwrite or redraw')
        saved = pd.read_csv(path, sep='\t', keep_default_na=False)
        if saved.to_csv(index=False) != manifest.to_csv(index=False):
            raise PipelineError('Revised participant eligibility changed; review before creating a new revision')
    summary, samples = [], {}
    for policy in policies:
        for cohort in COHORTS:
            group = manifest[(manifest.policy == policy) & (manifest.cohort == cohort)]
            accepted = group[group.eligible]
            dev = frozenset(accepted.loc[accepted.partition == 'development', 'subject'])
            validation = frozenset(accepted.loc[accepted.partition == 'holdout', 'subject'])
            if policy == primary and cohort == 'partner_pair' and holdout_mode == 'top_up_pair' and len(validation) != 50:
                raise PipelineError('Primary revised validation sample must contain exactly 50 QC-qualified participants')
            guard = RevisedGuard(dev, held)  # All original 50 remain protected, even when QC fails.
            dev_folds = accepted[accepted.partition == 'development'][['subject', 'fold']].copy()
            if len(dev) < 10 or set(dev_folds.fold) != set(range(1, 6)):
                raise PipelineError('Revised cohort needs at least ten development participants and all five original folds')
            samples[(policy, cohort)] = (guard, dev_folds)
            summary.append({'policy': policy, 'cohort': cohort, 'development_n': len(dev),
                            'original_development_retained_n': int(((accepted.partition == 'development') &
                                (accepted.original_partition == 'development')).sum()),
                            'new_development_n': int((accepted.original_partition.eq('new_candidate') & accepted.partition.eq('development')).sum()),
                            'original_holdout_total_n': 50, 'protected_holdout_total_n': len(held),
                            'supplemental_holdout_n': len(supplemental), 'qc_qualified_holdout_n': len(validation),
                            'original_holdout_unavailable_n': 50-int((accepted.original_partition.eq('holdout')).sum()),
                            'holdout_scored': False})
    if write:
        write_tsv(scope, 'work/manifest.tsv', manifest)
        write_tsv(scope, 'results/cohort_summary.tsv', summary)
        write_json(scope, 'provenance/samples.json', {**identity, 'manifest_sha256': sha256(path),
                   'original_holdout_redrawn': False, 'holdout_scored': False,
                   'supplement_selection': 'Seeded uniform sample without replacement from sorted new QC-qualified partner-pair candidates; no voxel or performance input',
                   'minimum_is_computational_only': '10 development participants and all five folds; not a power guarantee',
                   'new_fold_assignment': 'SHA256(seed:subject) modulo 5; original folds preserved'})
    print(pd.DataFrame(summary).to_string(index=False), flush=True)
    return samples, pd.DataFrame(summary)
