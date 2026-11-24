"""Prepare the private COPE handoff and compare group masks under two run coverage rules."""
import argparse
from collections import defaultdict
from hashlib import sha256 as digest
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from build_mask import save_image
from cross_valence_anatomy import asset_sha
from doors_voxelwise import source_path
from export_colleague_copes import COPE_INDICES, DOORS, PARTNERS, TASK_SETS, export, qualifying_runs, select_maps
from inventory import header_info, same_grid
from utils import InputUnavailable, PipelineError, atomic_output, load_config, write_text, write_tsv


RULES = ('historical', 'coverage95')
LABELS = {'partners': 'Shared Reward and Trust', 'both_doors': 'Both Doors games',
          'all_four': 'All four tasks', 'sharedreward': 'Shared Reward', 'trust': 'Trust',
          'socialdoors': 'Social Doors', 'doors': 'Monetary Doors'}
WHOLE_BRAIN = 'tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz'
REGION_SHA = '1335b40c2ad94056cd54c1b41aea100f5063428045c63271f7909432f4e310ed'


def references(base):
    # Only the archived aggregate image's grid is used. Voxels outside its mask can be gained.
    cohort = Path('revised/tsnr_coverage_fd/three_paradigm')
    record = json.loads(base.output(Path('provenance') / cohort / 'analysis_mask.json').read_text())
    reference_path = base.output(Path('results') / cohort / 'maps/analysis_mask.nii.gz')
    directory = base.templateflow / 'tpl-MNI152NLin6Asym'
    template_path = directory / WHOLE_BRAIN
    if asset_sha(reference_path) != record['sha256'] or record['coverage_threshold'] != .95:
        raise PipelineError('Historical analysis grid or group coverage threshold differs from its provenance')
    if not template_path.is_file() or asset_sha(template_path) != record['template_sha256']:
        raise PipelineError(f'The recorded whole-brain TemplateFlow mask is required at {template_path}')
    template = nib.as_closest_canonical(nib.load(template_path))
    for name in ('tpl-MNI152NLin6Asym_res-02_T1w.nii.gz', 'tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz'):
        path = directory / name
        if path.is_file():
            asset_sha(path)  # Reject participant sources before loading a display background.
            anatomy = resample_from_to(nib.load(path), template, order=1).get_fdata(dtype=np.float32)
            break
    else:
        anatomy = template.get_fdata(dtype=np.float32)
    return nib.load(reference_path), template, anatomy


def historical_paths(scope, qc, subject, task, current_paths):
    old, _ = qualifying_runs(scope, qc, subject, task, historical=True)
    if not old:
        return ()
    current, _ = qualifying_runs(scope, qc, subject, task)
    if old == current:
        return current_paths
    # A change can switch an estimate between L1 and L2 even for the same participant.
    try:
        images, _ = select_maps(scope, qc, subject, task, historical=True)
    except InputUnavailable:
        return ()
    return tuple(im.path for _, im in sorted(images.items()))


def task_support(scope, subject, task, paths, reference, cache):
    root = scope['c'].repos['socdoors' if task in DOORS else task]
    support = np.ones(reference.shape, dtype=bool)
    for cope in paths:
        source_path(scope, subject, cope, root)
        if cope.resolve() != cope:
            raise PipelineError('A selected COPE path changed after inventory')
        model = cope.parent.parent
        path = model / 'mask.nii.gz'
        try:
            source_path(scope, subject, path, root)
        except InputUnavailable as exc:
            raise PipelineError(f'Missing FEAT mask for a selected {task} COPE: {path}') from exc
        resolved = path.resolve()
        # A fixed-effects analysis can share one mask across its component COPE fits.
        fit_root = model.parent if model.parent.suffix == '.gfeat' else model
        if not resolved.is_relative_to(fit_root) or resolved.name != 'mask.nii.gz':
            raise PipelineError(f'FEAT mask resolves to a different fitted product: {path}')
        if resolved not in cache:
            header_info(resolved, scope['c'].analysis['space'])
            image = nib.load(resolved)
            data = image.get_fdata(dtype=np.float32)
            if not np.isfinite(data).all() or (data < 0).any() or not (data > 0).any():
                raise PipelineError(f'Invalid FEAT mask: {path}')
            binary = data > 0
            if not same_grid(image, reference):
                native = nib.Nifti1Image(binary.astype('uint8'), image.affine)
                binary = resample_from_to(native, reference, order=0).get_fdata() > 0
            cache[resolved] = image, binary
        image, binary = cache[resolved]
        if not same_grid(image, nib.load(cope)):
            raise PipelineError(f'FEAT mask grid differs from its selected COPE: {cope}')
        support &= binary
    return support


def accumulate(scope, qc, sources, rows, tasks, reference):
    scopes = {t: (t,) for t in tasks}
    for name, required in [('partners', PARTNERS), ('both_doors', DOORS), ('all_four', PARTNERS + DOORS)]:
        if set(required) <= set(tasks):
            scopes[name] = required
    counts = {(name, rule): np.zeros(reference.shape, dtype=np.uint32) for name in scopes for rule in RULES}
    members = {key: set() for key in counts}
    current = defaultdict(list)
    scope['guard'].check((r['participant_id'] for r in rows), scope['subjects'])
    for path, r in zip(sources, rows, strict=True):
        current[r['participant_id'], r['task']].append(path)
    for i, subject in enumerate(scope['subjects'], 1):
        scope['guard'].check([subject], scope['subjects'])
        cache, supports = {}, {}
        for task in tasks:
            paths = tuple(current[subject, task])
            old_paths = historical_paths(scope, qc, subject, task, paths)
            selected = {'coverage95': paths, 'historical': old_paths}
            by_paths = {}
            for rule, products in selected.items():
                if products:
                    if products not in by_paths:
                        by_paths[products] = task_support(scope, subject, task, products, reference, cache)
                    complete = len(products) == len(COPE_INDICES[task])
                    supports[task, rule] = by_paths[products], complete
        for name, required in scopes.items():
            for rule in RULES:
                if not all((t, rule) in supports for t in required):
                    continue
                if len(required) > 1 and not all(supports[t, rule][1] for t in required):
                    continue
                masks = [supports[t, rule][0] for t in required]
                counts[name, rule] += np.logical_and.reduce(masks)
                members[name, rule].add(subject)
        if i % 50 == 0:
            print(f'FEAT mask comparison: {i}/{len(scope["subjects"])} development participants', flush=True)
    return scopes, counts, members


def group_mask(counts, n, brain):
    if n == 0:
        return None
    return brain & (counts >= (95 * n + 99) // 100)


def template_mask(binary, reference, template):
    image = nib.Nifti1Image(binary.astype('uint8'), reference.affine)
    return (resample_from_to(image, template, order=0).get_fdata() > 0) & (np.asanyarray(template.dataobj) > 0)


def render_slices(path, template, anatomy, panels, title, caption):
    """Display aligned binary overlays on a canonical full-template background."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    brain = np.asanyarray(template.dataobj) > 0
    indices = np.where(brain)
    # Choose physical slices throughout the occupied template, including the inferior brain.
    zlo, zhi = indices[2].min(), indices[2].max()
    zs = np.unique(np.rint(zlo + (zhi - zlo) * np.linspace(.10, .92, 6)).astype(int))
    middle_x = int(np.rint(np.median(indices[0])))
    views = [(0, middle_x), *((2, k) for k in zs)]
    background = np.where(brain, anatomy, 0)
    vmax = max(float(np.percentile(background[brain], 99)), 1e-6)
    fig, axes = plt.subplots(len(panels), len(views), figsize=(13.8, 1.92*len(panels)+1.2),
                             squeeze=False, facecolor='white')
    try:
        for i, (label, layers) in enumerate(panels):
            for j, (axis, index) in enumerate(views):
                ax = axes[i, j]
                ax.imshow(np.take(background, index, axis=axis).T, origin='lower', cmap='gray',
                          vmin=0, vmax=vmax, interpolation='nearest')
                for mask, color, alpha in layers:
                    selected = np.take(mask, index, axis=axis).T
                    layer = np.zeros((*selected.shape, 4), dtype=np.float32)
                    layer[selected] = to_rgba(color, alpha)
                    ax.imshow(layer, origin='lower', interpolation='nearest')
                if i == 0:
                    coord = template.affine[axis, axis]*index + template.affine[axis, 3]
                    ax.set_title(f'{"x" if axis == 0 else "z"} = {coord:.0f} mm', fontsize=9)
                if j == 0:
                    ax.set_ylabel(label, rotation=0, ha='right', va='center', labelpad=9, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle(title, x=.015, y=.985, ha='left', fontsize=15, fontweight='bold')
        fig.text(.015, .025, caption, ha='left', va='bottom', fontsize=10)
        fig.subplots_adjust(left=.145, right=.99, bottom=.10, top=.86, hspace=.1, wspace=.035)
        fig.savefig(path, dpi=145, facecolor='white')
    finally:
        plt.close(fig)


def write_review(base, task_set, scopes, counts, members, reference, template, anatomy):
    root = Path('reports/revised/coverage95_masks') / task_set
    brain = resample_from_to(template, reference, order=0).get_fdata() > 0
    full_brain = np.asanyarray(template.dataobj) > 0
    template_n = int(full_brain.sum())
    region_path = base.repos['linux2'] / 'qc/reference/source-cerebellum-brainstem_mask.nii.gz'
    regions = {}
    if region_path.is_file() and asset_sha(region_path) == REGION_SHA:
        excluded = resample_from_to(nib.load(region_path), template, order=0).get_fdata() > 0
        regions = {'Cerebellum and brainstem': full_brain & excluded, 'Remaining brain': full_brain & ~excluded}
    # Publish the report last. An interrupted run cannot mix maps from different comparisons.
    key = digest(b'group-mask-review-v1')
    for name in scopes:
        for rule in RULES:
            key.update(counts[name, rule].tobytes())
            key.update(str(len(members[name, rule])).encode() + b'\0')
        key.update(str(len(members[name, 'historical'] & members[name, 'coverage95'])).encode() + b'\0')
    key.update(reference.affine.tobytes())
    key.update(template.affine.tobytes())
    key.update(np.asanyarray(template.dataobj).tobytes())
    key.update(brain.tobytes())
    key.update(anatomy.tobytes())
    for region in regions.values():
        key.update(region.tobytes())
    assets = root / ('maps-' + key.hexdigest()[:16])
    summary, sections, regional = [], [], []
    for name in scopes:
        original, revised = (members[name, r] for r in RULES)
        row = dict(scope=name, historical_n=len(original), coverage95_n=len(revised),
                   people_in_both=len(original & revised), people_lost=len(original - revised),
                   people_gained=len(revised - original), template_voxels_2mm=template_n)
        row.update(dict.fromkeys(('shared_voxels', 'lost_voxels', 'gained_voxels', 'net_voxels',
                                  'old_mask_retained_pct', 'dice', 'template_change_pp')))
        native, full, panels, links = {}, {}, [], []
        label = LABELS[name]
        for rule, population in zip(RULES, (original, revised)):
            native[rule] = group_mask(counts[name, rule], len(population), brain)
            binary = native[rule]
            if binary is None:
                row[rule + '_voxels'] = row[rule + '_template_pct'] = None
                continue
            full[rule] = template_mask(binary, reference, template)
            row[rule + '_voxels'] = int(binary.sum())
            row[rule + '_template_pct'] = 100 * int(full[rule].sum()) / template_n
            filename = f'{name}_{rule}.nii.gz'
            save_image(base, assets / filename, binary.astype('uint8'), reference)
            title = 'Historical' if rule == 'historical' else '95% run rule'
            links.append(f'[{title} mask]({assets.name}/{filename})')
            panels.append((f'{title}\nN = {len(population)}\n{row[rule + "_template_pct"]:.2f}% of brain',
                           [(full[rule], '#40b6b2', .49)]))
        old, new = (native[r] for r in RULES)
        if old is not None and new is not None:
            both = int((old & new).sum())
            lost = int((old & ~new).sum())
            gained = int((new & ~old).sum())
            total = int(old.sum() + new.sum())
            row.update(shared_voxels=both, lost_voxels=lost, gained_voxels=gained, net_voxels=gained-lost,
                       old_mask_retained_pct=100*both/int(old.sum()) if old.any() else None,
                       dice=2*both/total if total else None,
                       template_change_pp=row['coverage95_template_pct']-row['historical_template_pct'])
            filename = f'{name}_change.nii.gz'
            save_image(base, assets / filename, new.astype('int8')-old.astype('int8'), reference)
            links.append(f'[Signed change map]({assets.name}/{filename})')
            a, b = (full[r] for r in RULES)
            panels.append(('Change', [(a & ~b, '#ff7c43', 1), (b & ~a, '#46c7ff', 1)]))
            for region_name, region in regions.items():
                n = int(region.sum())
                if n:
                    before, after = int((a & region).sum()), int((b & region).sum())
                    regional.append(dict(scope=name, region=region_name, template_voxels=n,
                                         historical_pct=100*before/n, coverage95_pct=100*after/n,
                                         change_pp=100*(after-before)/n,
                                         lost_voxels=int((a & ~b & region).sum()),
                                         gained_voxels=int((b & ~a & region).sum())))
        summary.append(row)
        sections.extend(['## ' + label, ''])
        if panels:
            filename = name + '.png'
            caption = ('Teal shows included brain. Orange shows voxels lost under the 95% run rule. '
                       'Blue shows voxels gained. Axial views have anatomical left on the left.')
            with atomic_output(base, assets / filename) as path:
                render_slices(path, template, anatomy, panels, label + ' • group mask coverage', caption)
            sections.extend([f'![{label} group masks]({assets.name}/{filename})', '', ' · '.join(links), ''])
        if len(full) != 2:
            sections.extend(['At least one rule has no eligible participants for this scope. '
                             'The between-rule comparison is unavailable.', ''])
    frame = pd.DataFrame(summary)
    write_tsv(base, assets / 'mask_summary.tsv', frame)
    if regional:
        write_tsv(base, assets / 'regional_coverage.tsv', regional)
    focus = {'all': 'all_four', 'partners': 'partners', 'doors': 'both_doors'}[task_set]
    write_report(base, root, assets, summary, regional, sections, focus)
    print(frame.to_string(index=False), flush=True)
    print(f'Group mask report: {base.output(root / "REPORT.md")}', flush=True)


def write_report(base, root, assets, summary, regional, sections, focus):
    main = next(r for r in summary if r['scope'] == focus)
    if main['gained_voxels'] is None:
        lead = f'{LABELS[focus]} lacks a complete joint sample under at least one run rule.'
    else:
        lead = (f'The historical group mask for {LABELS[focus]} covered '
                f'{main["historical_template_pct"]:.2f}% of the full template (N={main["historical_n"]}). '
                f'The 95% run rule covered {main["coverage95_template_pct"]:.2f}% (N={main["coverage95_n"]}), '
                f'a change of {main["template_change_pp"]:+.2f} percentage points. On the analysis grid, '
                f'{main["lost_voxels"]:,} voxels were lost and {main["gained_voxels"]:,} were gained.')
        for r in regional:
            if r['scope'] == focus and r['region'] == 'Cerebellum and brainstem':
                lead += (f' Cerebellum and brainstem coverage changed from {r["historical_pct"]:.2f}% '
                         f'to {r["coverage95_pct"]:.2f}%.')
    print(lead, flush=True)
    lines = ['# Group mask comparison', '', lead, '',
             'Both run rules use the same development pool, motion and tSNR checks, phase-resolved outcome '
             'COPEs and L1/L2 selection. The historical rule uses the recorded paradigm-specific coverage '
             'flag. The revised rule accepts a run covering at least 95% of the run-QC target. All 62 '
             'protected participants were excluded before image access.', '',
             'Each participant contributes one vote per voxel. Within a task, support is the intersection '
             'of the FEAT masks for that person’s available outcome estimates. Each single-task row includes '
             'everyone with at least one estimate. Joint rows require every requested condition in every '
             'listed task and intersect their masks before voting. Both group masks require support from '
             'at least 95% of the eligible people in that row and rule.', '',
             'The comparison uses the archived analysis grid and the independent whole-brain MNI152NLin6Asym '
             'template. Masks use nearest-neighbor resampling. Gained and lost voxel counts and Dice use '
             'the analysis grid. Percentages of the full template use the entire 2-mm template brain. '
             'Both arms use the current handoff sources. The archived N=192 and N=178 analyses had '
             'different task and run requirements. The distributed COPEs retain their fitted extent.', '',
             '| Scope | N old → 95% | Full template old → 95% | Change (pp) | Voxels lost | Voxels gained | Old mask retained (%) | Dice |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    def display(value, digits=0):
        return '—' if pd.isna(value) else f'{value:,.{digits}f}'
    for row in summary:
        lines.append('| ' + ' | '.join([
            LABELS[row['scope']], f'{row["historical_n"]} → {row["coverage95_n"]}',
            f'{display(row["historical_template_pct"], 2)} → {display(row["coverage95_template_pct"], 2)}%',
            display(row.get('template_change_pp'), 2), display(row.get('lost_voxels')),
            display(row.get('gained_voxels')), display(row.get('old_mask_retained_pct'), 2),
            display(row.get('dice'), 4)]) + ' |')
    lines.extend(['', f'[Download aggregate counts and overlap metrics]({assets.name}/mask_summary.tsv)', ''])
    if regional:
        lines.extend(['The run-QC target excludes the historical cerebellum and brainstem region. '
                      'Its coverage and that of the remaining brain are measured on the full 2-mm template. '
                      f'[Regional coverage and voxel changes]({assets.name}/regional_coverage.tsv)', ''])
    lines.extend(['The signed NIfTI change maps encode loss as −1, gain as +1 and agreement as 0.', '', *sections])
    write_text(base, root / 'REPORT.md', '\n'.join(lines))


def run(base, task_set):
    scope, qc, sources, rows = export(base, task_set)
    print('Comparing the verified handoff with historical run QC on the same fitted sources.', flush=True)
    reference, template, anatomy = references(base)
    with threadpool_limits(limits=1, user_api='blas'):
        scopes, counts, members = accumulate(scope, qc, sources, rows, TASK_SETS[task_set], reference)
    write_review(base, task_set, scopes, counts, members, reference, template, anatomy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', choices=tuple(TASK_SETS), default='all')
    args = parser.parse_args()
    try:
        run(load_config(), args.tasks)
    except (PipelineError, OSError, ValueError, BadZipFile) as exc:
        sys.exit(f'Package and mask review stopped: {exc}')
