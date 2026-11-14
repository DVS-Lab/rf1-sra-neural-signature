"""Paired Social Doors outcome contrasts in the frozen N=178 development cohort."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from scipy.ndimage import label
from threadpoolctl import threadpool_limits

from build_mask import reconstruct, save_image, vectorize_source
from characterization_audit import PRIMARY, digest, require, table
from characterization_compute import assert_development, geometry
from inventory import cope_path, feat_dir, header_info, same_grid
from pilot import run_lock
from preflight import parse_fsf, verify_design_con, verify_l1
from revised_design import revised_config
from revised_samples import RevisedGuard, original_membership
from review_qc_attrition import flag, POLICIES
from utils import Config, PipelineError, load_config, sha256, write_json, write_text, write_tsv

N = 178
MASK_HASH = 'bdd44b7b3f2ff442bc9d70213945fbf8270f92f4ea48ac5b7dea7ffeaa3ac10a'
MANIFEST_HASH = '40f9586a877bd9c25b0eaaef7fb03c25b9e0dcde3eeaf9a7301316fed7632b0b'
TASKS = ('socialdoors', 'doors')
CONTRASTS = (
    ('social_win_minus_monetary_win', 'Social win − monetary win', False),
    ('social_loss_minus_monetary_loss', 'Social loss − monetary loss', False),
    ('social_by_valence', 'Win social effect − loss social effect', True),
)
PERMUTATIONS = 10000
SEED = 20261114
ALPHA = .05


@dataclass
class DoorsConfig(Config):
    def output(self, relative):
        p = Path(relative)
        require(not p.is_absolute() and len(p.parts) > 1 and p.parts[0] in
                ('results', 'reports', 'provenance') and '..' not in p.parts, 'Doors output escape')
        namespace = Path(p.parts[0]) / 'revised/doors_voxelwise'
        target = super().output(namespace.joinpath(*p.parts[1:]))
        require(target.is_relative_to(self.root.resolve() / namespace), 'Doors namespace symlink escape')
        return target


def frozen_scope(base):
    """Read protected membership only as an exclusion list; never inventory BIDS subjects."""
    samples = revised_config(base, 'samples')
    manifest_path = samples.output('work/manifest.tsv')
    require(manifest_path.is_file(), 'Linux2 private revised sample manifest unavailable; never reconstruct')
    meta = json.loads(samples.output('provenance/samples.json').read_text())
    status = json.loads(samples.output('provenance/run_status.json').read_text())
    require(sha256(manifest_path) == meta['manifest_sha256'] ==
            status['sample_manifest_sha256'] == MANIFEST_HASH, 'original v4 sample manifest changed')
    require(status['status'] == 'complete' and status['holdout_scored'] is False, 'v4 sample run is not complete/unscored')
    split, folds, split_hash = original_membership(base)
    require(split_hash == meta['original_split_sha256'], 'original split fingerprint changed')
    frame = table(manifest_path)
    require(not frame.duplicated(['policy', 'cohort', 'subject']).any(), 'duplicate v4 membership')
    require(set(frame.partition) == {'development', 'holdout'} and frame.eligible.isin([True, False]).all(),
            'invalid v4 partition or eligibility')
    require(frame.groupby('subject').partition.nunique().eq(1).all(), 'participant changes partition')
    held = frozenset(frame.loc[frame.partition.eq('holdout'), 'subject'])
    old_held = set(split.loc[split.split.eq('holdout'), 'subject'])
    require(old_held <= held and len(held) == meta['protected_holdout_total_n'] == 62 and
            not held.intersection(folds.subject), 'protected pool differs from v4')
    eligible = frame[frame.policy.eq(PRIMARY) & frame.eligible]
    require(len(eligible[eligible.cohort.eq('partner_pair') & eligible.partition.eq('holdout')]) == 50,
            'primary N=50 lock changed')
    rows = eligible[eligible.cohort.eq('three_paradigm') & eligible.partition.eq('development')]
    subjects = sorted(rows.subject)
    require(len(subjects) == N, 'require the exact frozen N=178 development roster')
    guard = RevisedGuard(frozenset(subjects), held)
    guard.check(subjects, subjects)
    c = revised_config(base, f'{PRIMARY}/three_paradigm')
    model = json.loads(c.output('provenance/model.json').read_text())
    status = json.loads(c.output('provenance/run_status.json').read_text())
    mask = json.loads(c.output('provenance/analysis_mask.json').read_text())
    require(status['status'] == 'complete' and status['holdout_scored'] is False and
            model['holdout_scored'] is False and model['development_n'] == N and
            model['sample_manifest_sha256'] == status['sample_manifest_sha256'] == MANIFEST_HASH,
            'original development fit or sample provenance differs')
    require(mask['development_n'] == N and mask['coverage_threshold'] == .95 and
            model['mask_sha256'] == mask['sha256'] == MASK_HASH ==
            sha256(c.output('results/maps/analysis_mask.nii.gz')), 'original development mask changed')
    return dict(c=c, guard=guard, subjects=subjects, model=model)


def source_path(scope, subject, path, root):
    """Validate lexical and resolved participant identity before any image access."""
    assert_development(scope, [subject])
    resolved = path.resolve()
    require(path.is_relative_to(root) and resolved.is_relative_to(root.resolve()) and
            subject in path.parts and subject in resolved.parts and 'ses-01' in resolved.parts and
            all(not part.startswith('sub-') or part == subject or part.startswith(subject+'_')
                for part in resolved.parts), 'unexpected Doors source path or cross-participant alias')
    require(path.is_file(), 'required Doors source file is missing')
    return path


def design_paths(scope, subject, task):
    assert_development(scope, [subject])
    c = scope['c']
    spec = c.contrasts[task]
    root = c.repos['socdoors']
    folder = feat_dir(c, subject, task, 'L1', 1)
    fsf, con, mat = [source_path(scope, subject, folder/name, root)
                     for name in ('design.fsf', 'design.con', 'design.mat')]
    values = parse_fsf(fsf)
    verify_l1(values, spec, rendered=True)
    verify_design_con(con, spec)
    design = np.loadtxt(mat.read_text().split('/Matrix', 1)[1].splitlines(), ndmin=2)
    contrasts = np.loadtxt(con.read_text().split('/Matrix', 1)[1].splitlines(), ndmin=2)
    require(design.shape[1] == contrasts.shape[1] and design.shape[0] > 1 and
            np.isfinite(design).all(), 'invalid completed Doors design matrix')
    projection = np.linalg.pinv(design) @ design
    require(np.allclose(contrasts[:2] @ projection, contrasts[:2], atol=1e-7, rtol=0),
            'win/loss contrasts not estimable in completed design')
    bold = Path(values.get('feat_files(1)', ''))
    require(bold.is_absolute() and bold.name.startswith(subject+'_ses-01_') and
            f'_task-{task}_run-1_' in bold.name and '_space-MNI152NLin6Asym_' in bold.name,
            'unexpected Doors preprocessed BOLD source')
    if not bold.name.endswith('.nii.gz') and not bold.is_file():
        bold = Path(str(bold)+'.nii.gz')
    # Rendered Doors designs have used both fmriprep and fmriprep-24 under Linux2.
    bold_root = c.fmriprep if bold.is_relative_to(c.fmriprep) else c.repos['linux2']
    bold = source_path(scope, subject, bold, bold_root)
    paths = tuple(source_path(scope, subject, cope_path(c, subject, task, k, 'L1', 1), root)
                  for k in (1, 2)) + (source_path(scope, subject, folder/'mask.nii.gz', root),)
    ref = nib.load(bold)
    require(len(ref.shape) == 4 and ref.shape[-1] == design.shape[0] == int(values['fmri(npts)']),
            'Doors BOLD and fitted design volume counts differ')
    for path in paths:
        header_info(path, c.analysis['space'])
        require(same_grid(nib.load(path), ref), 'Doors COPE/FEAT mask/BOLD grid mismatch')
    return paths, (fsf, con, mat)


def audit_sources(base, scope):
    assert_development(scope)
    qc_path = base.repos['linux2']/base.paths['qc_table']
    q = table(qc_path)
    q = q[q.session.astype(str).str.removeprefix('ses-').str.lstrip('0').eq('1') &
          q.task.isin(TASKS) & pd.to_numeric(q.run, errors='coerce').eq(1)].copy()
    q.subject = q.subject.map(lambda s: str(s) if str(s).startswith('sub-') else 'sub-'+str(s))
    q = q[q.subject.isin(scope['subjects'])]
    require(not q.duplicated(['subject', 'task']).any(), 'duplicate Doors QC for development participant')
    q = q.set_index(['subject', 'task'])
    paths, designs = {}, {}
    for i, subject in enumerate(scope['subjects'], 1):
        for task in TASKS:
            require((subject, task) in q.index and
                    all(flag(q.loc[(subject, task), k]) is False for k in POLICIES[PRIMARY]),
                    'frozen development participant lacks passing Doors QC; stop rather than change N')
            paths[subject, task], designs[subject, task] = design_paths(scope, subject, task)
        if i % 25 == 0 or i == len(scope['subjects']):
            print(f'Doors source/QC audit: {i}/{len(scope["subjects"])}', flush=True)
    return paths, designs, sha256(qc_path)


def paired_differences(social, monetary):
    """Order rows as win, loss, then their difference; use uncentered COPE units."""
    win, loss = np.asarray(social, dtype=np.float64) - np.asarray(monetary, dtype=np.float64)
    return np.stack((win, loss, win-loss))


def load_pairs(scope, paths, mask, reference):
    assert_development(scope)
    c = scope['c']
    prior = table(c.output('work/source_image_metrics.tsv'))
    prior = prior[(prior.level.eq('L1')) & prior.task.isin(TASKS) & prior.cope.isin([1, 2])].copy()
    assert_development(scope, prior.subject)
    keys = ['subject', 'task', 'cope']
    require(not prior.duplicated(keys).any(), 'original Doors source log has duplicate keys')
    prior = prior.set_index(keys)
    common = np.ones(int(mask.sum()), dtype=bool)
    differences = np.empty((3, len(scope['subjects']), int(mask.sum())), dtype=np.float64)
    diagnostics = []
    for i, subject in enumerate(scope['subjects']):
        assert_development(scope, [subject])
        task_maps = []
        for task in TASKS:
            mask_path = source_path(scope, subject, paths[subject, task][2], c.repos['socdoors'])
            coverage = nib.load(mask_path)
            binary = coverage.get_fdata(dtype=np.float32)
            require(np.isfinite(binary).all() and np.isin(binary, [0, 1]).all(), 'invalid participant FEAT mask')
            if not same_grid(coverage, reference):
                # The COPE is linearly interpolated. Require the entire interpolating
                # neighborhood to be covered rather than treating missing data as zero.
                # Float input preserves fractional edge coverage during interpolation.
                coverage = resample_from_to(nib.Nifti1Image(binary, coverage.affine),
                                            (reference.shape, reference.affine), order=1)
                binary = coverage.get_fdata(dtype=np.float32)
            common &= binary[mask] >= 1-1e-6
            task_maps.append([vectorize_source(c, subject, task, k, mask, reference, scope['guard'],
                                              scope['subjects'], diagnostics, 'L1', 1) for k in (1, 2)])
        differences[:, i] = paired_differences(*task_maps)
        if (i+1) % 25 == 0 or i+1 == len(scope['subjects']):
            print(f'Paired development maps: {i+1}/{len(scope["subjects"])}', flush=True)
    for row in diagnostics:
        key = tuple(row[k] for k in keys)
        require(key in prior.index, 'Doors source absent from original development log')
        old = prior.loc[key]
        require(np.allclose([row['source_mean'], row['source_norm']],
                            [old.source_mean, old.source_norm], atol=1e-5, rtol=2e-5),
                'raw Doors input differs from original development source')
    require(common.any(), 'no common within-pair coverage across all 178')
    analysis_mask = mask.copy()
    analysis_mask[mask] = common
    return differences[:, :, common], analysis_mask


def max_t_test(x, flips, two_sided=False, block_size=512):
    """Studentized one-sample sign flips; max over voxels, with optional both tails."""
    n, v = x.shape
    require(n > 1 and v > 0 and np.isfinite(x).all() and flips.shape[1] == n, 'invalid paired test inputs')
    effect = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    require(not np.any((sd == 0) & (effect != 0)), 'constant nonzero paired differences; review input')
    t = np.divide(effect * np.sqrt(n), sd, out=np.zeros(v), where=sd > 0)
    # t = r*sqrt(n-1)/sqrt(n-r²), where r is the signed sum divided by the
    # fixed L2 norm. Maximizing r (or abs(r)) is equivalent to maximizing t.
    observed_r = t * np.sqrt(n) / np.hypot(t, np.sqrt(n-1))
    if two_sided:
        observed_r = np.abs(observed_r)
    maxima = np.full(len(flips), -np.inf)
    for start in range(0, v, block_size):
        block = x[:, start:start+block_size]
        norm = np.linalg.norm(block, axis=0)
        unit = np.divide(block, norm, out=np.zeros_like(block), where=norm > 0)
        signed = flips @ unit
        if two_sided:
            np.abs(signed, out=signed)
        maxima = np.maximum(maxima, signed.max(axis=1))
    sorted_max = np.sort(maxima)
    # Include ties despite roundoff, plus the observed assignment.
    exceed = len(flips) - np.searchsorted(sorted_max, observed_r-1e-12, side='left')
    p = (exceed+1)/(len(flips)+1)
    null_t = maxima * np.sqrt(n-1) / np.sqrt(np.maximum(n-maxima**2, np.finfo(float).eps))
    return effect, t, p, null_t


def report(out, results, mask, reference):
    from matplotlib import pyplot as plt
    from scipy.ndimage import generate_binary_structure
    summary, clusters = [], []
    fig, axes = plt.subplots(3, 3, figsize=(11, 9), layout='constrained')
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeeeee')
    coords = np.argwhere(mask)
    for row, ((name, title, two_sided), (effect, t, p, null)) in enumerate(zip(CONTRASTS, results)):
        corrected = np.minimum(1, len(CONTRASTS)*p)
        threshold = corrected <= ALPHA
        peak = int(np.argmax(np.abs(t) if two_sided else t))
        xyz = nib.affines.apply_affine(reference.affine, coords[peak])
        summary.append(dict(contrast=name, tail='two-sided' if two_sided else 'positive',
                            n=N, voxels=len(t), significant_positive=int((threshold & (t>0)).sum()),
                            significant_negative=int((threshold & (t<0)).sum()),
                            peak_t=t[peak], peak_mean_cope=effect[peak], peak_p_fwe=corrected[peak],
                            peak_x=xyz[0], peak_y=xyz[1], peak_z=xyz[2],
                            null_max_t_q95=np.quantile(null, .95)))
        for stem, vector in (('mean_cope', effect), ('t', t), ('significant_t', np.where(threshold, t, 0))):
            save_image(out, f'results/maps/{name}_{stem}.nii.gz', reconstruct(vector, mask), reference)
        for stem, vector in (('p_voxel_fwe', p), ('p_family_fwe', corrected)):
            volume = np.ones(mask.shape, dtype=np.float32)
            volume[mask] = vector
            save_image(out, f'results/maps/{name}_{stem}.nii.gz', volume, reference)
        for sign, direction in ((1, 'positive'), (-1, 'negative')):
            components, count = label(reconstruct(threshold & (sign*t > 0), mask),
                                      structure=generate_binary_structure(3, 1))
            ids = components[mask]
            for k in range(1, count+1):
                indices = np.flatnonzero(ids == k)
                best = indices[np.argmax(sign*t[indices])]
                point = nib.affines.apply_affine(reference.affine, coords[best])
                clusters.append(dict(contrast=name, direction=direction, voxels=len(indices), peak_t=t[best],
                                     peak_mean_cope=effect[best], peak_p_fwe=corrected[best],
                                     peak_x=point[0], peak_y=point[1], peak_z=point[2]))
        volume = reconstruct(t, mask)
        sig = reconstruct(threshold, mask).astype(bool)
        for column, z in enumerate((0, 24, 48)):
            ax = axes[row, column]
            k = int(np.clip(round(nib.affines.apply_affine(np.linalg.inv(reference.affine), [0, 0, z])[2]),
                            0, mask.shape[2]-1))
            view = ax.imshow(np.ma.masked_where(~mask[:, :, k], volume[:, :, k]).T,
                             origin='lower', cmap=cmap, vmin=-8, vmax=8)
            if sig[:, :, k].any() and not sig[:, :, k].all():
                ax.contour(sig[:, :, k].T, levels=[.5], colors='black', linewidths=.65, origin='lower')
            ax.set_title(f'{title}\nz = {nib.affines.apply_affine(reference.affine, [0,0,k])[2]:.0f} mm',
                         fontsize=9)
            ax.set_axis_off()
    fig.colorbar(view, ax=axes.ravel().tolist(), label='Paired t (display clipped at ±8)', shrink=.68)
    fig.suptitle(f'Doors outcome contrasts | development N={N}\nBlack outlines: familywise p ≤ .05')
    from utils import atomic_output
    for ext in ('png', 'pdf'):
        with atomic_output(out, f'results/figures/doors_voxelwise.{ext}') as path:
            fig.savefig(path, dpi=160)
    plt.close(fig)
    write_tsv(out, 'results/aggregate/summary.tsv', summary)
    write_tsv(out, 'results/aggregate/significant_components.tsv', clusters,
              ['contrast','direction','voxels','peak_t','peak_mean_cope','peak_p_fwe','peak_x','peak_y','peak_z'])
    write_tsv(out, 'results/aggregate/max_t_null.tsv',
              pd.DataFrame({name: result[3] for (name, _, _), result in zip(CONTRASTS, results)}))
    text = ('# Paired Doors outcome effects\n\n'
            f'Development N={N}; df={N-1}. The frozen 84,045-voxel search region was restricted to '
            f'{int(mask.sum()):,} voxels covered by both runs for every participant. '
            'The protected N=50 was not accessed or scored.\n\n'
            '| Contrast | Significant positive voxels | Significant negative voxels | Peak t | Peak familywise p |\n'
            '|---|---:|---:|---:|---:|\n')
    for (_, title, _), row in zip(CONTRASTS, summary):
        text += (f'| {title} | {row["significant_positive"]:,} | {row["significant_negative"]:,} | '
                 f'{row["peak_t"]:.2f} | {row["peak_p_fwe"]:.4g} |\n')
    text += ('\nWins and losses are positive-direction tests. The interaction directly tests '
             '(social win − monetary win) − (social loss − monetary loss) in both directions. '
             'It is the test of valence modulation; a difference in thresholded maps is not such a test. '
             'A null interaction does not establish equivalence.\n\n'
             'Inputs are uncentered outcome COPEs. Inference uses a one-sample t on within-person '
             'differences and 10,000 participant sign flips, with voxelwise max-t correction '
             'within contrast and Bonferroni correction across three contrasts. Sign flips assume '
             'independent participants and symmetric difference errors under the null. The simple '
             'effects compare separate task runs and can include task, run or baseline differences. '
             'Component sizes are descriptive; inference is at the voxel level.\n\n'
             '[Figure](../../../results/revised/doors_voxelwise/figures/doors_voxelwise.png) · '
             '[Summary](../../../results/revised/doors_voxelwise/aggregate/summary.tsv) · '
             '[Significant components](../../../results/revised/doors_voxelwise/aggregate/significant_components.tsv)\n')
    write_text(out, 'reports/REPORT.md', text)
    return summary


def run(base, threads, dry_run):
    require(threads > 0, 'threads must be positive')
    out = DoorsConfig(**{key: getattr(base, key) for key in Config.__dataclass_fields__})
    with run_lock(base):
        scope = frozen_scope(base)
        paths, designs, qc_hash = audit_sources(base, scope)
        if dry_run:
            print('DRY RUN PASSED: frozen N=178, all win/loss designs, headers and Doors QC. '
                  'No participant voxels or holdout images loaded.', flush=True)
            return
        write_json(out, 'provenance/run_status.json', dict(status='in_progress', n=N,
            holdout_voxels_accessed=False, holdout_scored=False))
        mask, reference = geometry(scope)
        require(mask.sum() == 84045, 'frozen search mask voxel count changed')
        source_hash = digest([[sha256(p) for p in (*paths[key], *designs[key])] for key in paths])
        x, final_mask = load_pairs(scope, paths, mask, reference)
        print(f'Common coverage: {final_mask.sum():,}/{mask.sum():,} voxels; N={N} everywhere.', flush=True)
        rng = np.random.default_rng(SEED)
        flips = rng.integers(0, 2, size=(PERMUTATIONS, N), dtype=np.int8).astype(float)*2-1
        results = []
        with threadpool_limits(limits=threads, user_api='blas'):
            for (name, _, two_sided), values in zip(CONTRASTS, x):
                results.append(max_t_test(values, flips, two_sided))
                print('Finished permutation inference: '+name, flush=True)
        save_image(out, 'results/maps/common_coverage_mask.nii.gz', final_mask.astype('uint8'), reference)
        summary = report(out, results, final_mask, reference)
        write_json(out, 'provenance/run_status.json', dict(status='complete', n=N, permutations=PERMUTATIONS,
            seed=SEED, manifest_sha256=MANIFEST_HASH, frozen_mask_sha256=MASK_HASH,
            frozen_mask_voxels=int(mask.sum()), common_coverage_voxels=int(final_mask.sum()),
            common_coverage_mask_sha256=sha256(out.output('results/maps/common_coverage_mask.nii.gz')),
            source_fingerprint=source_hash, qc_sha256=qc_hash, code_sha256=sha256(Path(__file__)),
            numpy=np.__version__, nibabel=nib.__version__, threads=threads,
            holdout_voxels_accessed=False, holdout_scored=False, summary=summary))
        print(out.output('reports/REPORT.md'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=16, help='BLAS threads for sign-flip matrix products')
    parser.add_argument('--dry-run', action='store_true', help='audit membership, QC, designs and headers only')
    args = parser.parse_args()
    try:
        run(load_config(), args.threads, args.dry_run)
    except (PipelineError, FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print('ERROR: '+str(exc))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
