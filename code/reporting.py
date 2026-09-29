"""Participant-level inference, aggregate-only figures and Markdown reports."""
import json
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir() + '/rf1-neural-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, pearsonr, spearmanr
from make_split import describe
from utils import REWARD_TASKS
from utils import atomic_output, write_json, write_tsv, write_text

PERFORMANCE_COLUMNS = ['analysis', 'n', 'n_correct', 'accuracy', 'ci_low', 'ci_high', 'binomial_p',
                       'mean_margin', 'median_margin', 'margin_ci_low', 'margin_ci_high', 'ties']



def performance(margins, analysis, seed=20260928, bootstrap_samples=10000):
    margins = np.asarray(margins, dtype=float)
    if not np.isfinite(margins).all(): raise ValueError('Nonfinite participant margins')
    n, correct = len(margins), int((margins > 0).sum())
    result = dict.fromkeys(PERFORMANCE_COLUMNS, np.nan)
    result.update(analysis=analysis, n=n, n_correct=correct, ties=int((margins == 0).sum()))
    if not n: return result
    test = binomtest(correct, n, .5, alternative='two-sided')
    ci = test.proportion_ci(confidence_level=.95, method='exact')
    rng = np.random.default_rng(seed)
    # Bootstrap participants (one paired margin each), with bounded memory.
    means = np.empty(bootstrap_samples)
    for start in range(0, bootstrap_samples, 200):
        size = min(200, bootstrap_samples-start)
        means[start:start+size] = margins[rng.integers(0, n, (size, n))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    result.update(accuracy=correct/n, ci_low=ci.low, ci_high=ci.high, binomial_p=test.pvalue,
                  mean_margin=float(margins.mean()), median_margin=float(np.median(margins)),
                  margin_ci_low=low, margin_ci_high=high)
    return result


def icc31(values):
    """Shrout-Fleiss ICC(3,1): two-way mixed, consistency, single measurement."""
    values = np.asarray(values, dtype=float)
    n, k = values.shape
    if n < 2 or k < 2: return np.nan
    rows, cols, grand = values.mean(axis=1), values.mean(axis=0), values.mean()
    ms_subject = k * np.sum((rows-grand)**2) / (n-1)
    residual = values - rows[:, None] - cols[None, :] + grand
    ms_error = np.sum(residual**2) / ((n-1)*(k-1))
    denominator = ms_subject + (k-1)*ms_error
    return (ms_subject-ms_error)/denominator if denominator else np.nan


def reliability(cross):
    runs = cross[cross.model_scope == 'reliability'].copy()
    runs['analysis'] = runs.test_task
    paired = runs.pivot(index='subject', columns='analysis', values='paired_margin').reindex(columns=['run1', 'run2']).dropna()
    values = paired.to_numpy()
    valid = len(paired) > 1 and np.all(np.std(values, axis=0) > 0)
    row = {'n': len(paired), 'pearson_r': float(pearsonr(*values.T).statistic) if valid else np.nan,
           'spearman_rho': float(spearmanr(*values.T).statistic) if valid else np.nan,
           'icc_3_1': icc31(values),
           'run1_accuracy': float((values[:, 0] > 0).mean()) if len(values) else np.nan,
           'run2_accuracy': float((values[:, 1] > 0).mean()) if len(values) else np.nan}
    return row, paired


def save_figure(c, name, fig):
    fig.tight_layout()
    with atomic_output(c, 'results/figures/' + name) as path: fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(frame):
    # Avoid a dependency on tabulate; this contains aggregate fields only.
    def fmt(value):
        if isinstance(value, float): return 'NA' if not np.isfinite(value) else f'{value:.4g}'
        return str(value)
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join(
        '| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None))


SUMMARY_KEYS = ['train_family', 'train_tasks', 'test_family', 'test_task', 'model_scope']
TASK_LABELS = ['Shared Reward', 'Trust', 'Social/monetary Doors']


def summarize_predictions(c, predictions):
    rows = []
    for name, frame in predictions.groupby('analysis', sort=False):
        if frame.subject.duplicated().any():
            raise ValueError('Each inferential cell must have exactly one margin per participant')
        stats = performance(frame.paired_margin, name, c.analysis['seed'], c.analysis['bootstrap_samples'])
        stats.update(frame.iloc[0][SUMMARY_KEYS].to_dict())
        rows.append(stats)
    return pd.DataFrame(rows, columns=SUMMARY_KEYS + PERFORMANCE_COLUMNS)


def spatial_r(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def weight_summaries(c, weights):
    fold_rows, stability, similarities = [], [], []
    for key, maps in weights.items():
        rs = []
        for a in range(1, 6):
            for b in range(1, 6):
                r = spatial_r(maps[a], maps[b])
                fold_rows.append({'model': key, 'fold_a': a, 'fold_b': b, 'r': r})
                if a < b:
                    rs.append(r)
        stability.append({'model': key, 'mean_off_diagonal': float(np.mean(rs)),
                          'min_off_diagonal': float(np.min(rs)), 'max_off_diagonal': float(np.max(rs))})
    keys = [key for key in weights if 0 in weights[key]]
    for fold in range(6):
        for a in keys:
            for b in keys:
                similarities.append({'fold': fold, 'model_a': a, 'model_b': b,
                                     'r': spatial_r(weights[a][fold], weights[b][fold])})
    write_tsv(c, 'results/aggregate/fold_weight_similarity.tsv', fold_rows)
    write_tsv(c, 'results/aggregate/weight_stability_summary.tsv', stability)
    write_tsv(c, 'results/aggregate/signature_spatial_similarity.tsv', similarities)
    write_json(c, 'provenance/weight_stability.json', stability)
    return pd.DataFrame(similarities), pd.DataFrame(stability)


def decoding_heatmap(c, summary, family):
    cells = summary[(summary.model_scope == 'pairwise') & (summary.train_family == family)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, measure, title in zip(axes, ['accuracy', 'mean_margin'], ['Forced-choice accuracy', 'Mean paired margin']):
        values = cells.pivot(index='train_tasks', columns='test_task', values=measure).reindex(index=REWARD_TASKS, columns=REWARD_TASKS).to_numpy()
        limit = max(float(np.abs(values).max()), 1e-6)
        im = ax.imshow(values, vmin=0 if measure == 'accuracy' else -limit,
                       vmax=1 if measure == 'accuracy' else limit, cmap='viridis' if measure == 'accuracy' else 'coolwarm')
        for i, train in enumerate(REWARD_TASKS):
            for j, test in enumerate(REWARD_TASKS):
                n = int(cells[(cells.train_tasks == train) & (cells.test_task == test)].n.iloc[0])
                ax.text(j, i, f'{values[i,j]:.3f}\nN={n}', ha='center', va='center',
                        color='white' if measure == 'accuracy' and values[i,j] < .6 else 'black')
        ax.set_xticks(range(3), TASK_LABELS, rotation=22, ha='right')
        ax.set_yticks(range(3), TASK_LABELS)
        ax.set(xlabel='Test paradigm (held-out participants)', ylabel='Training paradigm', title=title)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle('Reward/outcome cross-decoding' if family == 'reward' else 'SECONDARY decision/context cross-decoding')
    save_figure(c, 'cross_decoding_heatmap.png' if family == 'reward' else 'decision_cross_decoding_heatmap.png', fig)


def accuracy_figure(c, name, frame, labels, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (_, row) in enumerate(frame.iterrows()):
        ax.errorbar(i, row.accuracy, yerr=[[row.accuracy-row.ci_low], [row.ci_high-row.accuracy]],
                    fmt='o', color='#14675f', capsize=4)
    ax.set_xticks(range(len(labels)), labels, rotation=15, ha='right')
    ax.axhline(.5, color='grey', linestyle='--')
    ax.set(ylim=(0, 1.05), ylabel='Forced-choice accuracy\n(exact 95% CI)', title=title)
    save_figure(c, name, fig)


def report(c, inventory, split, predictions, weights, repos, diagnostics):
    summary = summarize_predictions(c, predictions)
    write_tsv(c, 'results/aggregate/performance_summary.tsv', summary)
    pairwise = summary[summary.model_scope == 'pairwise']
    lopo = summary[summary.model_scope == 'lopo']
    cross_family = summary[summary.model_scope == 'cross_family']
    probes = summary[summary.model_scope == 'probe']
    for family in ['reward', 'decision']:
        write_tsv(c, f'results/aggregate/{family}_cross_decoding.tsv', pairwise[pairwise.train_family == family])
        decoding_heatmap(c, summary, family)
    write_tsv(c, 'results/aggregate/reward_leave_one_paradigm_out.tsv', lopo)
    write_tsv(c, 'results/aggregate/cross_family_decoding.tsv', cross_family)
    write_tsv(c, 'results/aggregate/common_signature_probes.tsv', probes)
    samples = []
    for label in ['development', 'holdout']:
        frame = inventory[inventory.subject.isin(split.loc[split.split == label, 'subject'])]
        desc = describe(frame)
        samples.extend({'split': label, 'variable': key, 'category': '', 'value': value}
                       for key, value in desc.items() if not isinstance(value, dict))
        for key in ['flipangle_counts', 'sex_counts']:
            samples.extend({'split': label, 'variable': key, 'category': str(cat), 'value': n}
                           for cat, n in desc[key].items())
    write_tsv(c, 'results/aggregate/sample_summary.tsv', samples)
    rel, paired = reliability(predictions)
    write_tsv(c, 'results/aggregate/run_reliability.tsv', [rel])
    similarity, stability = weight_summaries(c, weights)
    diagonal = pairwise[(pairwise.train_tasks == pairwise.test_task) & (pairwise.train_family == 'reward')]
    accuracy_figure(c, 'development_accuracy.png', diagonal, diagonal.test_task.tolist(), 'Within-task reward decoding; unseen participants')
    accuracy_figure(c, 'lopo_accuracy.png', lopo, lopo.test_task.tolist(), 'Primary: two reward paradigms → an unseen third paradigm')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (_, row) in enumerate(lopo.iterrows()):
        ax.plot([i, i], [row.margin_ci_low, row.margin_ci_high], color='#14675f')
        ax.plot(i, row.mean_margin, 'o', color='#14675f')
    ax.set_xticks(range(3), lopo.test_task, rotation=15)
    ax.axhline(0, color='grey', linestyle='--')
    ax.set(ylabel='Mean paired margin\n(participant bootstrap 95% CI)', title='Leave-one-paradigm-out reward decoding')
    save_figure(c, 'paired_margins.png', fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(paired.run1, paired.run2, alpha=.5, s=18)
    ax.set(xlabel='Shared Reward run 1 OOF margin', ylabel='Shared Reward run 2 OOF margin',
           title=f'Common reward model; N={len(paired)}; r={rel["pearson_r"]:.3f}')
    save_figure(c, 'sharedreward_run_reliability.png', fig)
    task_keys = [f'{family}_task-{task}' for family in ['reward', 'decision'] for task in REWARD_TASKS]
    labels = [f'{family}: {task}' for family in ['reward', 'decision'] for task in REWARD_TASKS]
    final = similarity[similarity.fold == 0].pivot(index='model_a', columns='model_b', values='r').loc[task_keys, task_keys]
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(final, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(6), labels, rotation=30, ha='right')
    ax.set_yticks(range(6), labels)
    ax.set_title('Task-specific development weights: spatial similarity')
    fig.colorbar(im, ax=ax, shrink=.75)
    save_figure(c, 'task_weight_similarity.png', fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, family in zip(axes, ['reward', 'decision']):
        matrix = np.array([[spatial_r(weights[f'common_{family}_signature'][a], weights[f'common_{family}_signature'][b])
                            for b in range(1, 6)] for a in range(1, 6)])
        im = ax.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
        ax.set_xticks(range(5), range(1, 6)); ax.set_yticks(range(5), range(1, 6))
        ax.set(title=f'Common {family}: fold weight stability', xlabel='Fold', ylabel='Fold')
        fig.colorbar(im, ax=ax)
    save_figure(c, 'fold_weight_similarity.png', fig)
    accuracy_figure(c, 'reward_decision_cross_application.png', cross_family,
                    [f'{r.train_family} → {r.test_family}\n{r.test_task}' for r in cross_family.itertuples()],
                    'SECONDARY: common signatures applied across contrast families')
    split_meta = json.loads(c.output('provenance/subject_split_v1.json').read_text())
    mask_meta = json.loads(c.output('provenance/analysis_mask.json').read_text())
    cols = ['train_tasks', 'test_task', 'n', 'accuracy', 'ci_low', 'ci_high', 'mean_margin', 'margin_ci_low', 'margin_ci_high', 'binomial_p']
    warnings = [
        'Exactly N=50 internal multitask holdout participants were never voxel-loaded or scored.',
        'Every test participant is absent from that model’s training data in every paradigm and family.',
        'LOPO omits the test reward paradigm entirely. Pooled within-task CV tests unseen participants, not unseen paradigms.',
        'Decision/context analyses remain secondary pending construct review. UGR is broader social valuation/decision context, not an isolated decision epoch or a fourth reward training task.',
        'No model settings, masks, or contrast definitions are changed in response to decoding performance.',
        'Shared Reward has 6-mm total smoothness; Trust and Doors use 5-mm FEAT smoothing. Cross-task differences can affect generalization.',
        'One fixed development-only mask is used without labels; CV inference is conditional on that mask.',
        'Binomial p values are two-sided and unadjusted. Ties count as incorrect. Intervals do not capture all dependence induced by overlapping CV training sets.',
        'Weight similarity is descriptive and affected by shared participants and correlated voxels. Similarity or failed cross-decoding alone does not establish psychological identity or dissociation.',
        f'Mask: {mask_meta["method"]}; resampled statistical image loads: {sum(bool(x["resampled"]) for x in diagnostics)}.',
    ]
    body = '# RF1-SRA trans-task social-reward pilot\n\n'
    body += md_table(pd.DataFrame(repos)[['repository', 'head_sha', 'status']]) + '\n\n'
    body += f'Multitask-complete development N={int((split.split == "development").sum())}; internal multitask holdout N=50. Minimum cohort N={c.analysis["minimum_multitask_n"]}; split: {split_meta["method"]}.\n\n'
    body += '## A–B. Within-task and pairwise reward decoding\n\n'
    body += md_table(pairwise[pairwise.train_family == 'reward'][cols]) + '\n\n'
    body += '![Reward cross-decoding](../results/figures/cross_decoding_heatmap.png)\n\n'
    body += '## C. Primary leave-one-paradigm-out reward tests\n\n' + md_table(lopo[cols]) + '\n\n'
    body += '## D. Task-specific and common weight similarity\n\n'
    selected_similarity = similarity[(similarity.fold == 0) & (similarity.model_a < similarity.model_b)]
    body += md_table(selected_similarity[['model_a', 'model_b', 'r']]) + '\n\n'
    body += '## E. Common signature probes: UGR and neutral/context boundaries\n\n'
    body += md_table(probes[['train_family'] + cols]) + '\n\n'
    body += 'Strong UGR offer-modulation transfer would be consistent with broader social-value information. Weak UGR transfer alongside successful reward cross-decoding would be consistent with greater reward/outcome specificity, subject to measurement and power limitations. Neither result changes the model.\n\n'
    body += '## F. Secondary decision/context and cross-family decoding\n\n'
    body += md_table(pairwise[pairwise.train_family == 'decision'][cols]) + '\n\n'
    body += md_table(cross_family[['train_family', 'test_family'] + cols]) + '\n\n'
    body += 'Reward/context similarity and bidirectional cross-application describe common versus distinct information; they do not by themselves establish a selective reward mechanism.\n\n'
    body += '## Reliability and stability\n\n' + md_table(pd.DataFrame([rel])) + '\n\n' + md_table(stability) + '\n\n'
    body += '## Warnings\n\n' + '\n'.join('- ' + item for item in warnings) + '\n\n'
    body += 'Contrast formulas: `docs/ANALYSIS_PLAN.md`. Key outputs: `results/aggregate/`, `results/maps/common_reward_signature_DEV_weights.nii.gz`, `results/maps/common_decision_signature_DEV_weights.nii.gz`, `results/figures/`, `provenance/model.json`.\n'
    write_text(c, 'reports/CODEX_REVIEW.md', body)
    body += '\n## Exact cohort intersections\n\n' + md_table(pd.read_csv(c.output('results/aggregate/task_overlap.tsv'), sep='\t'))
    body += '\n\n## Sample description\n\n' + md_table(pd.DataFrame(samples)) + '\n'
    body += '\n\n'.join(f'![{name}](../results/figures/{name}.png)' for name in
                        ['decision_cross_decoding_heatmap', 'lopo_accuracy', 'paired_margins',
                         'sharedreward_run_reliability', 'task_weight_similarity',
                         'fold_weight_similarity', 'reward_decision_cross_application']) + '\n'
    write_text(c, 'reports/REPORT.md', body)
