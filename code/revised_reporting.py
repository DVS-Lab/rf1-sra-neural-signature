"""Aggregate participant-level v4 summaries; no individual outputs in public reports."""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from reporting import performance, icc31, md_table
from revised_design import DESCRIPTIONS
from utils import write_tsv, write_text, atomic_output


def summarize(predictions, c):
    rows, confusion = [], []
    for keys, group in predictions.groupby(['family', 'train_tasks', 'test_task', 'scope'], sort=False):
        if group.subject.duplicated().any(): raise ValueError('Duplicate test participants within an analysis')
        family, train, test, scope = keys
        label = ':'.join(keys)
        row = {'family': family, 'train_tasks': train, 'test_task': test, 'scope': scope,
               'chance': float(group.chance.iloc[0])}
        if family != 'partner_reward':
            row.update(performance(group.margin.to_numpy(), label, c.analysis['seed'], c.analysis['bootstrap_samples']))
        else:
            rng = np.random.default_rng(c.analysis['seed'])
            a = group.accuracy.to_numpy()
            # Cluster by participant: the three class decisions are not independent trials.
            means = []
            for start in range(0, c.analysis['bootstrap_samples'], 200):
                count = min(200, c.analysis['bootstrap_samples']-start)
                means.extend(a[rng.integers(0, len(a), (count, len(a)))].mean(1))
            low, high = np.quantile(means, [.025, .975])
            row.update(analysis=label, n=len(a), accuracy=float(a.mean()), ci_low=float(low), ci_high=float(high),
                       mean_margin=float(group.margin.mean()), binomial_p=np.nan)
            matrix = np.zeros((3, 3), dtype=int)
            for text in group.predicted_labels:
                for true, pred in enumerate(map(int, text.split(','))): matrix[true, pred] += 1
            for true in range(3):
                for pred in range(3):
                    confusion.append({'train_tasks': train, 'test_task': test, 'scope': scope,
                                      'true_class': ('computer', 'friend', 'stranger')[true],
                                      'predicted_class': ('computer', 'friend', 'stranger')[pred],
                                      'n_maps': int(matrix[true, pred]), 'n_participants': len(group)})
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary['holm_p_binary_family'] = np.nan
    for family, group in summary[summary.family != 'partner_reward'].groupby('family'):
        order = group.binomial_p.sort_values().index
        adjusted = np.maximum.accumulate(summary.loc[order, 'binomial_p'].to_numpy()*np.arange(len(order), 0, -1))
        summary.loc[order, 'holm_p_binary_family'] = np.minimum(1., adjusted)
    return summary, pd.DataFrame(confusion)


def report(c, predictions, run_predictions, cohort_summary):
    summary, confusion = summarize(predictions, c)
    write_tsv(c, 'results/aggregate/performance.tsv', summary)
    if not confusion.empty: write_tsv(c, 'results/aggregate/partner_confusion.tsv', confusion)
    rel = []
    for (family, task), group in run_predictions.groupby(['family', 'test_task']):
        paired = group.pivot(index='subject', columns='run', values='margin').reindex(columns=[1, 2]).dropna()
        values = paired.to_numpy()
        valid = len(values) > 1 and np.all(values.std(axis=0) > 0)
        rel.append({'family': family, 'task': task, 'n': len(values),
                    'pearson_r': float(pearsonr(*values.T).statistic) if valid else np.nan,
                    'spearman_rho': float(spearmanr(*values.T).statistic) if valid else np.nan,
                    'icc_3_1': icc31(values) if len(values) > 1 else np.nan})
    write_tsv(c, 'results/aggregate/reliability.tsv', rel)
    body = '# Revised social reward, context, and closeness development analysis\n\n'
    body += md_table(cohort_summary)+'\n\nAll original holdouts remain barred from training; no validation images were scored.\n\n'
    for family, group in summary.groupby('family', sort=False):
        body += f'## {family}\n\n{DESCRIPTIONS[family]}.\n\n'
        columns = ['train_tasks', 'test_task', 'scope', 'n', 'chance', 'accuracy', 'ci_low', 'ci_high',
                   'mean_margin', 'binomial_p', 'holm_p_binary_family']
        body += md_table(group[columns])+'\n\n'
        pairwise = group[group.scope == 'pairwise']
        tasks = list(pairwise.test_task.unique())
        matrix = pairwise.pivot(index='train_tasks', columns='test_task', values='accuracy').reindex(index=tasks, columns=tasks)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        im = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(tasks)), tasks, rotation=25, ha='right')
        ax.set_yticks(range(len(tasks)), tasks)
        ax.set(xlabel='Test task: unseen participants', ylabel='Training task',
               title=f'{family}; chance={group.chance.iloc[0]:.3f}')
        for i in range(len(tasks)):
            for j in range(len(tasks)): ax.text(j, i, f'{matrix.iloc[i,j]:.3f}', ha='center', va='center', color='white')
        fig.colorbar(im, ax=ax, label='Cross-validated accuracy'); fig.tight_layout()
        with atomic_output(c, f'results/figures/{family}.png') as path: fig.savefig(path, dpi=160)
        plt.close(fig)
    body += '## Repeated-run reliability\n\n'+md_table(pd.DataFrame(rel))+'\n\n'
    body += ('## Interpretation limits\n\n'
             '- Binary accuracy is paired map ranking per participant, with ties incorrect.\n'
             '- Three-class accuracy averages three map classifications within each participant; intervals bootstrap participants, not maps. No independent-map binomial p value is reported.\n'
             '- Binary p values and intervals are descriptive CV summaries; Holm correction is within each family and sample/policy, not across the complete project. Overlapping training folds are not independent replications.\n'
             '- Shared Reward is full-trial; Trust is outcome-phase. Averaging outcome valence does not isolate a pure social mechanism.\n'
             '- The same participant is excluded from training across every task and condition. LOPO additionally excludes the test paradigm.\n'
             '- Fixed masks use development-only coverage. Differences between policies/cohorts can reflect both sample and mask changes.\n'
             '- Newly eligible development participants enter only if explicitly enabled; preserved original fold assignments and deterministic new folds apply across policies and cohorts.\n'
             '- No UGR data or decision-phase maps are used. No settings are tuned to the revised results.\n')
    write_text(c, 'reports/REPORT.md', body)
    return summary
