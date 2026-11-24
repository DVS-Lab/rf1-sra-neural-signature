"""Package QC-qualified subject-level outcome COPEs separately for each task on Linux2."""
import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256 as digest_bytes
from io import BytesIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import nibabel as nib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from doors_voxelwise import QC_COMMIT, frozen_development_rows, frozen_qc_tables, source_path
from final_stress_sources import guard_path, phase_dir, phase_index, phase_spec
from inventory import feat_dir, header_info
from preflight import estimable_contrasts, parse_fsf, read_feat_matrix, verify_design_con, verify_l1, verify_l2
from revised_design import revised_config
from revised_samples import RevisedGuard
from review_qc_attrition import flag, normalized_qc
from utils import InputUnavailable, PipelineError, load_config, sha256


PARTNERS = ('sharedreward', 'trust')
DOORS = ('socialdoors', 'doors')
TASK_SETS = {'all': PARTNERS + DOORS, 'partners': PARTNERS, 'doors': DOORS}
COPE_INDICES = {'sharedreward': range(1, 7), 'trust': range(4, 10),
                'socialdoors': (1, 2), 'doors': (1, 2)}
PARTNER_NAMES = {'C': 'computer', 'F': 'friend', 'S': 'stranger'}
OUTCOMES = {'pun': 'punishment', 'rew': 'reward', 'def': 'defection', 'rec': 'reciprocation'}
QC_METRICS = ('tsnr', 'fd_mean', 'brain_coverage_pct')
MIN_COVERAGE_PCT = 95.0


@dataclass(frozen=True)
class SubjectCope:
    path: Path
    name: str
    runs: tuple
    has_zstat: bool


def export_scope(base):
    rows, protected = frozen_development_rows(base, eligible_only=False)
    eligible = rows[rows.eligible]
    pair = frozenset(eligible.loc[eligible.cohort == 'partner_pair', 'subject'])
    doors = frozenset(eligible.loc[eligible.cohort == 'three_paradigm', 'subject'])
    if not doors or not doors <= pair:
        raise PipelineError('The frozen Doors cohort must be a subset of the partner cohort')
    candidates = frozenset(rows.subject)
    guard = RevisedGuard(candidates, protected)
    guard.check(candidates, candidates)
    if not base.exclusions.is_dir() or not os.access(base.exclusions, os.R_OK | os.X_OK):
        raise PipelineError('Current source exclusions are unavailable')
    people = sorted(s for s in candidates if not (base.exclusions / ('Smith-SRA-' + s.removeprefix('sub-'))).is_dir())
    guard.check(people, people)
    print(f'Locked development partition: {len(candidates)}. Current source exclusions: '
          f'{len(candidates) - len(people)}. Protected people excluded: {len(protected)}.', flush=True)
    return dict(c=revised_config(base), guard=guard, subjects=people, paper_pair=pair, paper_three=doors)


def baseline_ages(base, people):
    age_column = base.paths['age_column']
    ages = pd.read_csv(base.bids / 'participants.tsv', sep='\t', dtype=str,
                       usecols=['participant_id', age_column], keep_default_na=False)
    ages = ages[ages.participant_id.isin(people)].set_index('participant_id', verify_integrity=True)
    # BIDS uses n/a for missing ages. Preserve those people and report the missing count.
    ages = pd.to_numeric(ages[age_column].replace({'n/a': np.nan, '': np.nan}), errors='raise')
    if not np.isfinite(ages.dropna()).all():
        raise PipelineError('Invalid age in BIDS participants.tsv')
    return ages.reindex(people)


def frozen_run_qc(base, people):
    frozen, _, _ = frozen_qc_tables(base)
    qc = pd.read_csv(BytesIO(frozen), sep='\t', dtype=str, keep_default_na=False)
    ids = {s.removeprefix('sub-') for s in people}
    qc = normalized_qc(qc[qc.subject.str.removeprefix('sub-').isin(ids)]).sort_index()
    for metric in QC_METRICS:
        if metric not in qc:
            raise PipelineError(f'Frozen run-level QC table lacks {metric}')
        qc[metric] = pd.to_numeric(qc[metric], errors='coerce')
    return qc


def run_covariates(qc, subject, task, runs):
    result = {}
    for metric in QC_METRICS:
        values = {r: qc.loc[(subject, task, r), metric] for r in runs}
        result[metric] = sum(values.values()) / len(values)
        for r in (1, 2):
            result[f'{metric}_run{r}'] = values.get(r, '')
    return result


def run_qc_reason(row):
    """Use the measured coverage percentage and the original motion and tSNR flags."""
    flags = [flag(row['fd_mean_outlier']), flag(row['tsnr_outlier'])]
    if True in flags:
        return 'motion_or_tsnr_outlier'
    if None in flags or not np.isfinite([row[k] for k in QC_METRICS]).all():
        return 'unknown_qc'
    coverage = row['brain_coverage_pct']
    if not 0 <= coverage <= 100:
        return 'unknown_qc'
    return 'coverage_below_95pct' if coverage < MIN_COVERAGE_PCT else ''


def coverage_review(qc, tasks):
    """Count runs and distinct people in the guarded inventory before checking fitted COPEs."""
    frame = qc.reset_index()
    expected_run = frame.run.eq(1) | (frame.task.isin(PARTNERS) & frame.run.eq(2))
    frame = frame[frame.task.isin(tasks) & expected_run].copy()
    measured = frame.brain_coverage_pct.between(0, 100)
    frame['old_fail'] = measured & frame.brain_coverage_outlier.map(flag).eq(True)
    frame['below_95'] = measured & frame.brain_coverage_pct.lt(MIN_COVERAGE_PCT)
    frame['regained'] = frame.old_fail & measured & ~frame.below_95
    frame['pass_after_change'] = [not run_qc_reason(r) for _, r in frame.iterrows()]
    frame['unknown_coverage'] = ~measured
    rows = []
    for task in (*tasks, 'Combined'):
        g = frame if task == 'Combined' else frame[frame.task.eq(task)]
        rows.append(dict(task=task, measured_runs=int((~g.unknown_coverage).sum()),
                         old_runs=int(g.old_fail.sum()),
                         old_people=g.loc[g.old_fail, 'subject'].nunique(),
                         below95_runs=int(g.below_95.sum()),
                         below95_people=g.loc[g.below_95, 'subject'].nunique(),
                         regained_runs=int(g.regained.sum()),
                         regained_people=g.loc[g.regained, 'subject'].nunique(),
                         regained_passing_other_qc=int((g.regained & g.pass_after_change).sum()),
                         unknown_coverage=int(g.unknown_coverage.sum())))
    return pd.DataFrame(rows)


def fitted_maps(scope, subject, task, runs, indices):
    """Verify the fitted run(s), contrast names and weights before selecting COPE files."""
    scope['guard'].check([subject], scope['subjects'])
    c = scope['c']
    if task == 'sharedreward':
        unit = scope['phase_index'][subject]
        if not set(runs) <= set(unit['runs']):
            raise PipelineError('Shared Reward run is outside the verified outcome manifest')
        spec = phase_spec()
        root = c.repos['sharedreward']
        run_dirs = [phase_dir(c, subject, 'L1', r) for r in runs]
        output = unit['output'] if len(runs) == 2 else run_dirs[0]
    else:
        spec = c.contrasts[task]
        root = c.repos[spec['repository']]
        run_dirs = [feat_dir(c, subject, task, 'L1', r) for r in runs]
        output = feat_dir(c, subject, task, 'L2') if len(runs) == 2 else run_dirs[0]

    for r, run_dir in zip(runs, run_dirs):
        values = parse_fsf(source_path(scope, subject, run_dir / 'design.fsf', root))
        verify_l1(values, spec, rendered=True)
        contrasts = verify_design_con(source_path(scope, subject, run_dir / 'design.con', root), spec)
        bold = guard_path(scope, Path(values.get('feat_files(1)', '')), subject)
        if (not bold.name.startswith(subject + '_ses-01_') or f'_task-{task}_run-{r}_' not in bold.name or
                f'_space-{c.analysis["space"]}_' not in bold.name):
            raise PipelineError(f'{subject} {task} L1 uses an unexpected participant, session, run or space')
        design = read_feat_matrix(source_path(scope, subject, run_dir / 'design.mat', root))
        estimable = estimable_contrasts(design, contrasts[np.asarray(indices, dtype=int) - 1])
        indices = [k for k, ok in zip(indices, estimable) if ok]
    if len(runs) == 2:
        values = parse_fsf(source_path(scope, subject, output / 'design.fsf', root))
        verify_l2(values, spec)
        for i, run_dir in enumerate(run_dirs, 1):
            if Path(values.get(f'feat_files({i})', '')).resolve() != run_dir.resolve():
                raise PipelineError(f'{subject} {task} L2 uses unexpected input runs')

    images = {}
    for k in indices:
        stats = output / f'cope{k}.feat/stats' if len(runs) == 2 else output / 'stats'
        number = 1 if len(runs) == 2 else k
        try:
            path = source_path(scope, subject, stats / f'cope{number}.nii.gz', root)
            # A same-person symlink can also point to a different task, contrast, or statistic.
            expected_tail = path.relative_to(root / 'derivatives/fsl').parts
            resolved = path.resolve()
            if resolved.parts[-len(expected_tail):] != expected_tail:
                raise PipelineError(f'{subject} {task} COPE {k} resolves to a different FEAT product')
            header_info(resolved, c.analysis['space'])
        except InputUnavailable:
            continue
        zstat = stats / f'zstat{number}.nii.gz'
        has_zstat = zstat.resolve() == resolved.with_name(zstat.name) and zstat.is_file()
        images[k] = SubjectCope(resolved, spec['copes'][k]['name'], runs, has_zstat)
    return images


def select_maps(scope, qc, subject, task):
    """Select one existing subject estimate per available condition from qualifying runs."""
    scope['guard'].check([subject], scope['subjects'])
    if task == 'sharedreward':
        unit = scope['phase_index'].get(subject)
        if unit is None:
            raise InputUnavailable('no_verified_phase_outcomes')
        candidates = unit['runs']
    else:
        candidates = (1, 2) if task == 'trust' else (1,)

    passing, reasons = [], set()
    for r in candidates:
        key = (subject, task, r)
        reason = run_qc_reason(qc.loc[key]) if key in qc.index else 'unknown_qc'
        if reason:
            reasons.add(reason)
        else:
            passing.append(r)
    runs = tuple(passing)
    if not runs:
        raise InputUnavailable('+'.join(sorted(reasons)))
    indices = tuple(COPE_INDICES[task])
    try:
        images = fitted_maps(scope, subject, task, runs, indices)
    except InputUnavailable:
        if len(runs) == 1:
            raise
        images = {}
    missing = [k for k in indices if k not in images]
    omissions = dict.fromkeys(missing, 'no_existing_estimable_cope')
    if missing and len(runs) == 2:
        singles = []
        for r in runs:
            try:
                singles.append(fitted_maps(scope, subject, task, (r,), missing))
            except InputUnavailable:
                continue
        for k in missing:
            available = [unit[k] for unit in singles if k in unit]
            if len(available) == 1:
                images[k] = available[0]
                del omissions[k]
            elif len(available) == 2:
                omissions[k] = 'two_usable_L1s_without_L2'
    if not images:
        raise InputUnavailable('+'.join(sorted(set(omissions.values()))))
    return images, omissions


def task_counts(rows, task):
    by_person, levels = Counter(), Counter()
    for r in rows:
        if r['task'] == task:
            by_person[r['participant_id']] += 1
            levels[r['source_level']] += 1
    complete = {s for s, n in by_person.items() if n == len(COPE_INDICES[task])}
    return set(by_person), complete, levels


def overlap_text(participants, complete):
    text = (f'{len(set.union(*participants.values()))} people contributed at least one COPE. '
            f'{len(set.intersection(*complete.values()))} contributed every requested condition.')
    if set(DOORS) <= participants.keys():
        text += (f' The Doors games overlap in {len(participants["socialdoors"] & participants["doors"])} '
                 f'participants, including {len(complete["socialdoors"] & complete["doors"])} '
                 'with all four Doors outcome maps.')
    return text


def image_plan(base, tasks, scope, ages, qc):
    if 'sharedreward' in tasks:
        scope['phase_index'] = phase_index(scope)
    sources, rows, participants, complete = [], [], {}, {}
    for task in tasks:
        root = base.repos['socdoors' if task in DOORS else task] / 'derivatives/fsl'
        if not root.is_dir() or not os.access(root, os.R_OK | os.X_OK):
            raise PipelineError(f'{task} FEAT source tree is unavailable: {root}')
        omitted, missing_contrasts = Counter(), Counter()
        for subject in scope['subjects']:
            try:
                images, missing = select_maps(scope, qc, subject, task)
            except InputUnavailable as exc:
                omitted[str(exc)] += 1
                continue
            except PipelineError as exc:
                raise PipelineError(f'{subject}, {task}: {exc}') from exc
            missing_contrasts.update(missing.values())
            quality = {runs: run_covariates(qc, subject, task, runs)
                       for runs in {im.runs for im in images.values()}}
            age = ages.loc[subject]
            for k, im in sorted(images.items()):
                name, runs = im.name, im.runs
                level = 'L2' if len(runs) == 2 else 'L1'
                partner, outcome = '', name
                if task in PARTNERS:
                    prefix, suffix = name.split('_')
                    partner, outcome = PARTNER_NAMES[prefix], OUTCOMES[suffix]
                filename = f'images/{subject}_ses-01_task-{task}_cope-{k:02d}_{name}.nii.gz'
                rows.append(dict(filename=filename, participant_id=subject, session='ses-01', task=task,
                                 phase='outcome', cope_index=k, contrast_name=name, partner=partner,
                                 outcome=outcome, source_level=level,
                                 retained_runs=','.join(map(str, runs)), n_runs=len(runs),
                                 age='' if pd.isna(age) else age, **quality[runs],
                                 complete_task_conditions=not missing,
                                 in_paper_partner_pair=subject in scope['paper_pair'],
                                 in_paper_three_paradigm=subject in scope['paper_three'], space=base.analysis['space'],
                                 source_zstat_available=im.has_zstat, source_sha256=sha256(im.path)))
                sources.append(im.path)
        participants[task], complete[task], levels = task_counts(rows, task)
        print(f'{task}: {len(participants[task])} participants, {len(complete[task])} with every condition. '
              f'COPEs: {levels["L1"]} L1, {levels["L2"]} L2. Omitted participant-tasks: {dict(omitted)}. '
              f'Missing contrasts in partial sets: {dict(missing_contrasts)}.', flush=True)
    print(overlap_text(participants, complete), flush=True)
    if not rows:
        raise PipelineError('No shareable outcome maps in the requested tasks')
    if len(rows) != len(sources) or len({r['filename'] for r in rows}) != len(rows):
        raise PipelineError('Missing source or duplicate subject/contrast entry')
    scope['guard'].check((r['participant_id'] for r in rows), scope['subjects'])
    return sources, rows


def write_cope(source, target):
    """Drop free-text header fields and extensions while preserving stored data and scaling."""
    image = nib.load(source)
    raw = image.dataobj.get_unscaled()
    if not np.isfinite(raw).all():
        raise PipelineError(f'Nonfinite COPE values in {target.name}')
    header = image.header.copy()
    for field in ('data_type', 'db_name', 'descrip', 'aux_file', 'intent_name'):
        header[field] = b''
    header.extensions.clear()
    clean = type(image)(raw, None, header=header)
    clean.header.set_slope_inter(image.dataobj.slope, image.dataobj.inter)
    nib.save(clean, target)
    saved = nib.load(target)
    if not (np.array_equal(saved.affine, image.affine) and saved.get_data_dtype() == image.get_data_dtype()
            and saved.dataobj.slope == image.dataobj.slope and saved.dataobj.inter == image.dataobj.inter
            and np.array_equal(saved.dataobj.get_unscaled(), raw)):
        raise PipelineError(f'Export changed COPE values or geometry in {target.name}')


def release_notes(rows, tasks, coverage_counts):
    people = {r['participant_id'] for r in rows}
    missing_age = len({r['participant_id'] for r in rows if r['age'] == ''})
    descriptions = {'sharedreward': 'Shared Reward, phase-resolved reward and punishment',
                    'trust': 'Trust, reciprocation and defection',
                    'socialdoors': 'Social Doors, win and loss', 'doors': 'monetary Doors, win and loss'}
    participants, complete = {}, {}
    task_lines = ['| Task and conditions | Participants | All conditions | L1 COPEs | L2 COPEs |',
                  '| --- | ---: | ---: | ---: | ---: |']
    for task in tasks:
        participants[task], complete[task], levels = task_counts(rows, task)
        task_lines.append(f'| {descriptions[task]} | {len(participants[task])} | '
                          f'{len(complete[task])} | {levels["L1"]} | {levels["L2"]} |')
    return ('# RF1-SRA subject-level outcome COPEs\n\n'
            f'This collaboration dataset contains {len(rows):,} contrast-of-parameter-estimate images from '
            f'{len(people)} development participants in session 01. Eligibility was evaluated independently '
            'for each task, after excluding the entire 62-person protected validation pool and current '
            'source exclusions.\n\n'
            + '\n'.join(task_lines) + '\n\n' + overlap_text(participants, complete) + '\n\n'
            'Each row of images.csv identifies one signed effect-estimate NIfTI in images/. '
            'Use participant_id to pair images across conditions and tasks. The in_paper_partner_pair '
            'and in_paper_three_paradigm columns identify membership in the archived N=192 '
            'and N=178 development analyses, respectively. Those analyses used the historical '
            'paradigm-specific coverage outlier flag. '
            'complete_task_conditions indicates that this person has every listed condition for this task. '
            'cope_index is the condition contrast number in the run design. '
            'Each condition uses its fitted fixed-effects L2 for two qualifying runs or its L1 for a '
            'single qualifying, available run. A condition with two usable L1s and no fitted L2 is omitted. '
            'The images preserve the original FEAT values and grids in MNI152NLin6Asym space. '
            'Free-text NIfTI header fields and extensions were removed.\n\n'
            'partner identifies the computer, friend or stranger in tasks with individual partners.\n\n'
            f'age is baseline age in years from BIDS participants.tsv ({missing_age} participants missing). '
            'tsnr (temporal signal-to-noise ratio) and fd_mean (mean framewise displacement in mm) '
            'come from MRIQC. brain_coverage_pct is the percentage of the fixed MNI QC target '
            'covered by the fMRIPrep run mask. The target excludes cerebellum and brainstem. '
            'This export requires brain_coverage_pct of at least 95.0 for each contributing run. '
            'High fd_mean or low tsnr excludes a run using the original one-sided 1.5-IQR fences '
            'within paradigm. The Doors games share motion and tSNR fences. Each run needs known, '
            'passing values on all three metrics. The COPEs retain their fitted spatial extent. '
            'Each summary is the unweighted mean over the contributing runs. '
            'The run1 and run2 columns preserve their individual values. A blank run field means the run '
            f'did not contribute. QC uses the original rf1-sra-linux2 snapshot {QC_COMMIT}.\n\n'
            'source_zstat_available records existence of a companion FEAT Z-statistic. '
            'source_sha256 identifies the input COPE and sha256 verifies the distributed file.\n\n'
            '## Development coverage comparison\n\n'
            'These counts cover session 01 QC records after protected and source exclusions, '
            'before checking fitted contrast availability. A person can have several failing runs. '
            'The combined row counts each person once per rule.\n\n'
            + '```text\n' + coverage_counts.to_string(index=False) + '\n```\n')


def verify_package(directory, rows, notes):
    """Verify the folder and ZIP against the current allowlist and source fingerprints."""
    contents = list(directory.iterdir())
    if ({p.name for p in contents} != {'images', 'images.csv', 'README.md', 'subject_copes.zip'} or
            any(p.is_symlink() for p in contents)):
        raise PipelineError('Unrecognized or linked item in the package folder')
    with (directory / 'images.csv').open(newline='') as stream:
        saved = list(csv.DictReader(stream))
    expected_files = {r['filename'] for r in rows}
    actual_files = {p.relative_to(directory).as_posix() for p in (directory / 'images').iterdir()}
    if len(saved) != len(rows) or actual_files != expected_files:
        raise PipelineError('The package folder or CSV differs from the allowed file list')
    if (directory / 'README.md').read_text() != notes:
        raise PipelineError('The README differs from the selected data and QC documentation')
    with ZipFile(directory / 'subject_copes.zip') as archive:
        members = archive.namelist()
        if len(members) != len(rows) + 2 or set(members) != expected_files | {'images.csv', 'README.md'}:
            raise PipelineError('The ZIP differs from the approved file list')
        for name in ('images.csv', 'README.md'):
            if archive.read(name) != (directory / name).read_bytes():
                raise PipelineError(f'{name} differs between the ZIP and the folder')
        for row, entry in zip(rows, saved):
            if set(entry) != set(row) | {'sha256'} or any(entry[k] != str(v) for k, v in row.items()):
                raise PipelineError('CSV differs from the frozen participant, contrast, covariate or source data')
            name = row['filename']
            file = directory / name
            if (file.is_symlink() or sha256(file) != entry['sha256'] or
                    digest_bytes(archive.read(name)).hexdigest() != entry['sha256']):
                raise PipelineError(f'Packaged COPE checksum mismatch: {name}')


def export(base, task_set, dry_run=False, qc_only=False):
    relative = Path('work/colleague_exports/coverage95') / task_set
    destination = base.output(relative)
    if destination != base.root / relative:
        raise PipelineError('The private export location resolves outside its designated work folder')
    tasks = TASK_SETS[task_set]
    scope = export_scope(base)
    qc = frozen_run_qc(base, scope['subjects'])
    coverage_counts = coverage_review(qc, tasks)
    print('Development baseline coverage: historical boxplot flag versus coverage <95%, before map selection.',
          flush=True)
    print(coverage_counts.to_string(index=False), flush=True)
    if qc_only:
        return
    ages = baseline_ages(base, scope['subjects'])
    # The FEAT matrices are small. A single BLAS thread avoids oversubscribing Linux2.
    with threadpool_limits(limits=1, user_api='blas'):
        sources, rows = image_plan(base, tasks, scope, ages, qc)
    print(f'{len(rows)} subject-level COPEs. Companion zstats present for '
          f'{sum(r["source_zstat_available"] for r in rows)}/{len(rows)}.', flush=True)
    if dry_run:
        return
    notes = release_notes(rows, tasks, coverage_counts)
    if destination.exists():
        verify_package(destination, rows, notes)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.preparing-', dir=destination.parent) as temp:
            stage = Path(temp) / 'package'
            (stage / 'images').mkdir(parents=True)
            with (stage / 'images.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=[*rows[0], 'sha256'])
                writer.writeheader()
                for i, (path, row) in enumerate(zip(sources, rows), 1):
                    if path.resolve() != path:
                        raise PipelineError('Source path changed between inventory and export')
                    target = stage / row['filename']
                    write_cope(path, target)
                    if sha256(path) != row['source_sha256']:
                        raise PipelineError('Source changed between inventory and export')
                    writer.writerow({**row, 'sha256': sha256(target)})
                    if i % 500 == 0:
                        print(f'Packaged {i}/{len(rows)} COPEs', flush=True)
            (stage / 'README.md').write_text(notes)
            with ZipFile(stage / 'subject_copes.zip', 'w', compression=ZIP_STORED) as archive:
                for name in ['images.csv', 'README.md', *(r['filename'] for r in rows)]:
                    archive.write(stage / name, name)
            verify_package(stage, rows, notes)
            stage.chmod(0o700)
            stage.rename(destination)
    archive = destination / 'subject_copes.zip'
    print(f'Verified {archive} ({archive.stat().st_size / 2**20:.1f} MiB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', choices=tuple(TASK_SETS), default='all')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Check source maps and metadata without writing a package')
    mode.add_argument('--qc-only', action='store_true', help='Compare run coverage rules without opening any images')
    args = parser.parse_args()
    try:
        export(load_config(), args.tasks, args.dry_run, args.qc_only)
    except (PipelineError, OSError, ValueError, BadZipFile) as exc:
        sys.exit(f'Export stopped: {exc}')
