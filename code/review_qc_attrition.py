"""Review QC attrition without changing eligibility, splits, models, or voxel access."""
import argparse
import json
from itertools import combinations
import pandas as pd
from make_split import validate_split
from utils import PipelineError, load_config, sha256, write_json, write_tsv


POLICIES = {
    'tsnr_coverage_fd': ('tsnr_outlier', 'brain_coverage_outlier', 'fd_mean_outlier'),
    'prior_four_metric_policy': ('tsnr_outlier', 'brain_coverage_outlier',
                                 'fd_mean_outlier', 'tedana_outlier'),
}
TASKS = ('sharedreward', 'trust', 'socialdoors', 'doors')
TASK_SETS = {task: (task,) for task in TASKS}
TASK_SETS.update(sharedreward_trust=('sharedreward', 'trust'),
                 all_three_reward_paradigms=TASKS)


def flag(value):
    text = str(value).strip().lower()
    if text in ('true', '1'): return True
    if text in ('false', '0'): return False
    return None


def normalized_qc(qc):
    qc = qc.copy()
    required = {'subject', 'session', 'task', 'run', *POLICIES['prior_four_metric_policy']}
    if not required <= set(qc.columns): raise PipelineError('QC table lacks required columns')
    qc['subject'] = qc.subject.map(lambda x: str(x) if str(x).startswith('sub-') else 'sub-'+str(x))
    qc['session'] = qc.session.astype(str).str.removeprefix('ses-').str.lstrip('0')
    qc = qc[(qc.session == '1') & qc.task.isin(TASKS)].copy()
    qc['run'] = pd.to_numeric(qc.run, errors='raise').astype(int)
    if qc.duplicated(['subject', 'task', 'run']).any():
        raise PipelineError('Duplicate baseline participant/task/run QC records')
    return qc.set_index(['subject', 'task', 'run'])


def required_runs(row, task):
    if task == 'sharedreward':
        try: runs = tuple(int(r) for r in str(row['sharedreward_runs']).split(','))
        except ValueError as exc: raise PipelineError('Missing retained Shared Reward runs') from exc
        if runs not in ((1,), (2,), (1, 2)): raise PipelineError('Invalid retained Shared Reward runs')
        return runs
    return (1, 2) if task == 'trust' else (1,)


def task_status(qc, row, task, flags):
    if flag(row[task]) is not True: return 'unavailable_input'
    bad, unknown = False, False
    for run in required_runs(row, task):
        key = (row['subject'], task, run)
        if key not in qc.index:
            unknown = True
            continue
        values = [flag(qc.loc[key, name]) for name in flags]
        bad |= any(v is True for v in values)
        unknown |= any(v is None for v in values)
    # Existing L2 maps include every retained input. We cannot remove a bad run
    # from that map by changing a downstream inclusion flag.
    if bad: return 'qc_outlier'
    return 'unknown_qc' if unknown else 'pass'


def summarize(inventory, split, qc):
    qc = normalized_qc(qc)
    membership = split.set_index('subject').split.to_dict()
    rows, overlaps = [], []
    for policy, flags in POLICIES.items():
        assessed = []
        for row in inventory.to_dict('records'):
            statuses = {task: task_status(qc, row, task, flags) for task in TASKS}
            assessed.append({'split': membership.get(row['subject'], 'outside_locked_cohort'), **statuses})
        for group in ('development', 'holdout', 'locked_total', 'outside_locked_cohort'):
            subset = [r for r in assessed if (r['split'] in ('development', 'holdout')
                       if group == 'locked_total' else r['split'] == group)]
            for name, tasks in TASK_SETS.items():
                counts = dict(pass_n=0, qc_outlier_n=0, unknown_qc_n=0, unavailable_input_n=0)
                for row in subset:
                    states = [row[t] for t in tasks]
                    status = next((s for s in ('unavailable_input', 'qc_outlier', 'unknown_qc')
                                   if s in states), 'pass')
                    counts[status+'_n'] += 1
                rows.append({'policy': policy, 'partition': group, 'task_set': name,
                             'before_n': len(subset), **counts})
            # Exact intersections of task-level QC failures, useful because task
            # exclusion counts cannot be added to estimate total attrition.
            for size in range(1, len(TASKS)+1):
                for tasks in combinations(TASKS, size):
                    overlaps.append({'policy': policy, 'partition': group,
                                     'outlier_tasks': '+'.join(tasks),
                                     'n': sum(all(r[t] == 'qc_outlier' for t in tasks) for r in subset)})
    return pd.DataFrame(rows), pd.DataFrame(overlaps)


def review(c):
    inv_path = c.output('work/inventory/task_availability.tsv')
    split_path = c.output('work/splits/subject_split_v1.tsv')
    split_meta = c.output('provenance/subject_split_v1.json')
    qc_path = c.repos['linux2'] / c.paths['qc_table']
    inventory = pd.read_csv(inv_path, sep='\t', dtype=str, keep_default_na=False)
    split = pd.read_csv(split_path, sep='\t', dtype=str)
    if inventory.subject.duplicated().any(): raise PipelineError('Duplicate inventory participants')
    validate_split(split, inventory[inventory.eligible.map(flag).eq(True)])
    if json.loads(split_meta.read_text())['sha256'] != sha256(split_path):
        raise PipelineError('Locked split hash mismatch')
    qc = pd.read_csv(qc_path, sep='\t', dtype=str, keep_default_na=False)
    rows, overlaps = summarize(inventory, split, qc)
    write_tsv(c, 'results/aggregate/qc_review_retention.tsv', rows)
    write_tsv(c, 'results/aggregate/qc_review_overlap.tsv', overlaps)
    write_json(c, 'provenance/qc_review.json', {
        'review_only': True, 'exclusions_applied': False, 'holdout_scored': False,
        'split_sha256': sha256(split_path), 'inventory_sha256': sha256(inv_path),
        'qc_sha256': sha256(qc_path), 'policies': POLICIES,
        'rule': 'Reject task if any run contributing to its existing map has any selected flag; unknown QC does not pass.',
        'thresholds': 'Existing canonical per-paradigm QC flags; no threshold re-estimation.',
        'outside_locked_cohort': 'Descriptive candidates only. No assignment to development/holdout.',
        'limitations': ['Original inventory task flags still include its contrast requirements; this is not a fresh reward-only inventory.',
                        'Shared Reward uses frozen retained runs. Trust current L2 uses both runs. No maps are recombined.',
                        'No UGR or decision-map values accessed. Only metadata tables are read.']})
    joint = rows[(rows.task_set.isin(['sharedreward_trust', 'all_three_reward_paradigms'])) &
                 (rows.partition != 'outside_locked_cohort')]
    print(joint.to_string(index=False))
    print('REVIEW ONLY: no exclusions, membership changes, image reads, or model fits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir')
    args = parser.parse_args()
    review(load_config(args.config_dir))
