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
from utils import atomic_output, write_json, write_tsv, write_text

PERFORMANCE_COLUMNS = ['analysis', 'n', 'n_correct', 'accuracy', 'ci_low', 'ci_high', 'binomial_p',
                       'mean_margin', 'median_margin', 'margin_ci_low', 'margin_ci_high', 'ties']
ANALYSES = ['primary', 'socialdoors', 'trust', 'trust_friend_computer', 'trust_stranger_computer',
            'trust_friend_stranger', 'neutral', 'decision', 'ugr_pmod', 'ugr_constant', 'run1', 'run2']


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
    runs = cross[cross.analysis.isin(['run1', 'run2'])]
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


def report(c, inventory, split, primary, cross, weights, repos, diagnostics):
    combined = pd.concat([primary, cross], ignore_index=True)
    summary = pd.DataFrame([performance(combined.loc[combined.analysis == name, 'paired_margin'], name,
                                       c.analysis['seed'], c.analysis['bootstrap_samples']) for name in ANALYSES], columns=PERFORMANCE_COLUMNS)
    write_tsv(c, 'results/aggregate/performance_summary.tsv', summary)
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
    rel, paired = reliability(cross)
    write_tsv(c, 'results/aggregate/run_reliability.tsv', [rel])
    with np.errstate(divide='ignore', invalid='ignore'): similarity = np.corrcoef(weights)
    simframe = pd.DataFrame(similarity, columns=[f'fold_{i}' for i in range(1, 6)])
    simframe.insert(0, 'fold', range(1, 6))
    write_tsv(c, 'results/aggregate/fold_weight_similarity.tsv', simframe)
    offdiag = similarity[np.triu_indices(5, 1)]
    stable = {'mean_off_diagonal': float(np.mean(offdiag)), 'min_off_diagonal': float(np.min(offdiag)),
              'max_off_diagonal': float(np.max(offdiag))}
    write_json(c, 'provenance/weight_stability.json', stable)
    for name, analyses in [('development_accuracy.png', ['primary']), ('cross_task_accuracy.png', ['socialdoors', 'trust', 'neutral', 'decision', 'ugr_pmod', 'ugr_constant'])]:
        subset = summary.set_index('analysis').loc[analyses]
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (label, row) in enumerate(subset.iterrows()):
            if row.n:
                ax.errorbar(i, row.accuracy, yerr=[[row.accuracy-row.ci_low], [row.ci_high-row.accuracy]], fmt='o', color='#14675f', capsize=4)
            else: ax.text(i, .5, 'N=0', ha='center')
        ax.set_xticks(range(len(analyses)), [f'{a}\nN={int(subset.loc[a,"n"])}' for a in analyses], rotation=20)
        ax.axhline(.5, color='grey', linestyle='--'); ax.set_ylim(0, 1.05)
        ax.set_ylabel('Within-person forced-choice accuracy\n(exact 95% CI)')
        save_figure(c, name, fig)
    subset = summary.set_index('analysis').loc[ANALYSES[:3] + ['neutral', 'decision', 'ugr_pmod', 'ugr_constant']]
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (_, row) in enumerate(subset.iterrows()):
        if row.n:
            ax.plot([i, i], [row.margin_ci_low, row.margin_ci_high], color='#14675f')
            ax.plot(i, row.mean_margin, 'o', color='#14675f')
    ax.set_xticks(range(len(subset)), subset.index, rotation=25)
    ax.axhline(0, color='grey', linestyle='--'); ax.set_ylabel('Mean paired margin\n(participant bootstrap 95% CI)')
    save_figure(c, 'paired_margins.png', fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(paired.run1, paired.run2, alpha=.5, s=18)
    ax.set(xlabel='Run 1 OOF margin', ylabel='Run 2 OOF margin', title=f'Development N={len(paired)}; r={rel["pearson_r"]:.3f}')
    save_figure(c, 'sharedreward_run_reliability.png', fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(similarity, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(5), range(1, 6)); ax.set_yticks(range(5), range(1, 6))
    ax.set(xlabel='CV fold', ylabel='CV fold', title='Spatial correlation of fold weights')
    fig.colorbar(im, ax=ax)
    save_figure(c, 'fold_weight_similarity.png', fig)
    mask_meta = json.loads(c.output('provenance/analysis_mask.json').read_text())
    split_meta = json.loads(c.output('provenance/subject_split_v1.json').read_text())
    warnings = [
        'Development analyses only. Exactly N=50 internal holdout participants were not voxel-loaded or scored.',
        'Mask uses all development participants without labels; CV estimates are conditional on this fixed development mask.',
        'Primary Shared Reward uses 6-mm total smoothness; secondary tasks use 5-mm FEAT smoothing.',
        'P values are two-sided, unadjusted exploratory pilot results. Ties count as forced-choice failures.',
        'SVM weights are discriminative coefficients, not psychological localization evidence.',
        'Non-mandatory upstream QC flags are descriptive only. Missing tasks reduce their reported N.',
        f'Mask method: {mask_meta["method"]}.',
        f'Resampling operations (including repeated loads): {sum(bool(x["resampled"]) for x in diagnostics)}.',
    ]
    repo_table = pd.DataFrame(repos)[['repository', 'head_sha', 'status']]
    body = '# RF1-SRA neural signature: development pilot\n\n' + md_table(repo_table) + '\n\n'
    body += f'Development N={int((split.split == "development").sum())}; locked holdout N=50. Split method: {split_meta["method"]}.\n\n'
    body += 'Shared Reward: social = ((cope4−cope3)+(cope6−cope5))/2; computer = cope2−cope1. '
    body += 'Doors: separate socialdoors/doors L1 cope4. Trust: social = ((cope7−cope6)+(cope9−cope8))/2; computer = cope5−cope4. '
    body += 'Neutral = mean(8,9) vs 7; decision = mean(27,28) vs 29; UGR pmod = 14 vs 13; UGR constant = mean(5,7) vs mean(1,3).\n\n'
    body += md_table(summary) + '\n\nReliability (OOF run differences):\n\n' + md_table(pd.DataFrame([rel])) + '\n\n'
    body += 'Weight stability: ' + json.dumps(stable) + '\n\nWarnings:\n\n' + '\n'.join('- '+s for s in warnings)
    body += '\n\nKey outputs: `results/aggregate/`, `results/maps/analysis_mask.nii.gz`, `results/maps/social_reward_signature_DEV_weights.nii.gz`, `results/figures/`, `provenance/model.json`.\n'
    write_text(c, 'reports/CODEX_REVIEW.md', body)
    body += '\nAggregate inventory:\n\n' + md_table(pd.read_csv(c.output('results/aggregate/inventory_summary.tsv'), sep='\t').fillna(''))
    body += '\n\nSample demographics:\n\n' + md_table(pd.DataFrame(samples))
    body += '\n\n' + '\n\n'.join(f'![{name}](../results/figures/{name}.png)' for name in ['development_accuracy', 'cross_task_accuracy', 'paired_margins', 'sharedreward_run_reliability', 'fold_weight_similarity']) + '\n'
    write_text(c, 'reports/REPORT.md', body)
