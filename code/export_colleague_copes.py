"""Package the frozen development participants' subject-level outcome COPEs on Linux2."""
import argparse
import csv
from hashlib import sha256 as digest_bytes
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, BadZipFile, ZipFile

import nibabel as nib
import numpy as np
import pandas as pd

from doors_voxelwise import QC_COMMIT, frozen_development_rows, frozen_qc_tables, source_path
from final_stress_sources import guard_path, phase_dir, phase_index, phase_spec, qc_reason
from inventory import feat_dir, header_info
from preflight import parse_fsf, verify_design_con, verify_l1, verify_l2
from revised_design import revised_config
from revised_samples import RevisedGuard
from review_qc_attrition import normalized_qc
from utils import PipelineError, load_config, sha256


PARTNERS = ('sharedreward', 'trust')
DOORS = ('socialdoors', 'doors')
TASK_SETS = {'all': PARTNERS + DOORS, 'partners': PARTNERS, 'doors': DOORS}
COPE_INDICES = {'sharedreward': range(1, 7), 'trust': range(4, 10),
                'socialdoors': (1, 2), 'doors': (1, 2)}
PARTNER_NAMES = {'C': 'computer', 'F': 'friend', 'S': 'stranger'}
OUTCOMES = {'pun': 'punishment', 'rew': 'reward', 'def': 'defection', 'rec': 'reciprocation'}


def task_rosters(base, tasks):
    eligible, protected = frozen_development_rows(base)
    pair = frozenset(eligible.loc[eligible.cohort == 'partner_pair', 'subject'])
    doors = frozenset(eligible.loc[eligible.cohort == 'three_paradigm', 'subject'])
    if not doors or not doors <= pair:
        raise PipelineError('The frozen Doors cohort must be a subset of the partner cohort')
    allowed = {task: pair if task in PARTNERS else doors for task in tasks}
    people = frozenset.union(*allowed.values())
    guard = RevisedGuard(people, protected)
    guard.check(people, people)
    if not base.exclusions.is_dir():
        raise PipelineError('Current source exclusions are unavailable')
    for s in people:
        if (base.exclusions / ('Smith-SRA-' + s.removeprefix('sub-'))).is_dir():
            raise PipelineError(f'{s} has a current source exclusion. Review before sharing.')
    scope = dict(c=revised_config(base), guard=guard, subjects=sorted(people))
    return scope, allowed


def covariates(base, people):
    age_column = base.paths['age_column']
    ages = pd.read_csv(base.bids / 'participants.tsv', sep='\t', dtype=str,
                       usecols=['participant_id', age_column], keep_default_na=False)
    ages = ages[ages.participant_id.isin(people)].set_index('participant_id', verify_integrity=True)
    # BIDS uses n/a for missing ages. Preserve those people and report the missing count.
    ages = pd.to_numeric(ages[age_column].replace({'n/a': np.nan, '': np.nan}), errors='raise')
    if not np.isfinite(ages.dropna()).all():
        raise PipelineError('Invalid age in BIDS participants.tsv')
    frozen, _, _ = frozen_qc_tables(base)
    qc = pd.read_csv(BytesIO(frozen), sep='\t', dtype=str, keep_default_na=False)
    ids = {s.removeprefix('sub-') for s in people}
    qc = normalized_qc(qc[qc.subject.str.removeprefix('sub-').isin(ids)])
    return ages.reindex(people), qc


def run_covariates(qc, subject, task, runs):
    reason = qc_reason(qc, subject, task, runs)
    if reason:
        raise PipelineError(f'{subject} {task} {runs} fails the frozen run QC ({reason})')
    result = {}
    for metric in ('tsnr', 'fd_mean'):
        values = {r: pd.to_numeric(qc.loc[(subject, task, r), metric], errors='coerce') for r in runs}
        if not np.isfinite(list(values.values())).all():
            raise PipelineError(f'{subject} {task} has missing or nonfinite {metric}')
        result[metric] = sum(values.values()) / len(values)
        for r in (1, 2):
            result[f'{metric}_run{r}'] = values.get(r, '')
    return result


def fitted_maps(scope, subject, task):
    """Verify the fitted run(s), contrast names and weights before selecting COPE files."""
    scope['guard'].check([subject], scope['subjects'])
    c = scope['c']
    if task == 'sharedreward':
        unit = scope['phase_index'].get(subject)
        if unit is None:
            raise PipelineError(f'{subject} missing from verified Shared Reward outcome manifest')
        spec, runs, output = phase_spec(), unit['runs'], unit['output']
        root = c.repos['sharedreward']
        run_dirs = [phase_dir(c, subject, 'L1', r) for r in runs]
    else:
        spec = c.contrasts[task]
        root = c.repos[spec['repository']]
        runs = (1, 2) if task == 'trust' else (1,)
        run_dirs = [feat_dir(c, subject, task, 'L1', r) for r in runs]
        output = feat_dir(c, subject, task, 'L2') if len(runs) == 2 else run_dirs[0]

    for r, run_dir in zip(runs, run_dirs):
        values = parse_fsf(source_path(scope, subject, run_dir / 'design.fsf', root))
        verify_l1(values, spec, rendered=True)
        verify_design_con(source_path(scope, subject, run_dir / 'design.con', root), spec)
        bold = guard_path(scope, Path(values.get('feat_files(1)', '')), subject)
        if (not bold.name.startswith(subject + '_ses-01_') or f'_task-{task}_run-{r}_' not in bold.name or
                f'_space-{c.analysis["space"]}_' not in bold.name):
            raise PipelineError(f'{subject} {task} L1 uses an unexpected participant, session, run or space')
    if len(runs) == 2:
        values = parse_fsf(source_path(scope, subject, output / 'design.fsf', root))
        verify_l2(values, spec)
        for i, run_dir in enumerate(run_dirs, 1):
            if Path(values.get(f'feat_files({i})', '')).resolve() != run_dir.resolve():
                raise PipelineError(f'{subject} {task} L2 uses unexpected input runs')

    images = {}
    for k in COPE_INDICES[task]:
        stats = output / f'cope{k}.feat/stats' if len(runs) == 2 else output / 'stats'
        number = 1 if len(runs) == 2 else k
        path = source_path(scope, subject, stats / f'cope{number}.nii.gz', root)
        # A same-person symlink can also point to a different task, contrast, or statistic.
        expected_tail = path.relative_to(root / 'derivatives/fsl').parts
        resolved = path.resolve()
        if resolved.parts[-len(expected_tail):] != expected_tail:
            raise PipelineError(f'{subject} {task} COPE {k} resolves to a different FEAT product')
        zstat = stats / f'zstat{number}.nii.gz'
        has_zstat = zstat.resolve() == resolved.with_name(zstat.name) and zstat.is_file()
        images[k] = (resolved, spec['copes'][k]['name'], has_zstat)
    return runs, images


def image_plan(base, tasks):
    scope, allowed = task_rosters(base, tasks)
    ages, qc = covariates(base, scope['subjects'])
    if 'sharedreward' in tasks:
        scope['phase_index'] = phase_index(scope)
    sources, rows = [], []
    for task, roster in allowed.items():
        for subject in sorted(roster):
            runs, images = fitted_maps(scope, subject, task)
            quality = run_covariates(qc, subject, task, runs)
            age = ages.loc[subject]
            for k, (path, name, has_zstat) in images.items():
                header_info(path, base.analysis['space'])
                partner, outcome = '', name
                if task in PARTNERS:
                    prefix, suffix = name.split('_')
                    partner, outcome = PARTNER_NAMES[prefix], OUTCOMES[suffix]
                filename = f'images/{subject}_ses-01_task-{task}_cope-{k:02d}_{name}.nii.gz'
                rows.append(dict(filename=filename, participant_id=subject, session='ses-01', task=task,
                                 phase='outcome', cope_index=k, contrast_name=name, partner=partner,
                                 outcome=outcome, source_level='L2' if len(runs) == 2 else 'L1',
                                 retained_runs=','.join(map(str, runs)), n_runs=len(runs),
                                 age='' if pd.isna(age) else age, **quality, space=base.analysis['space'],
                                 source_zstat_available=has_zstat, source_sha256=sha256(path)))
                sources.append(path)
        print(f'{task}: {len(roster)} participants, {len(roster) * len(COPE_INDICES[task])} COPEs', flush=True)
    expected = sum(len(allowed[t]) * len(COPE_INDICES[t]) for t in tasks)
    if len(rows) != expected or len({r['filename'] for r in rows}) != expected:
        raise PipelineError('Incomplete or duplicate subject/contrast set')
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


def release_notes(rows):
    people = {r['participant_id'] for r in rows}
    missing_age = len({r['participant_id'] for r in rows if r['age'] == ''})
    descriptions = {'sharedreward': 'Shared Reward, phase-resolved reward and punishment',
                    'trust': 'Trust, reciprocation and defection',
                    'socialdoors': 'Social Doors, win and loss', 'doors': 'monetary Doors, win and loss'}
    tasks = dict.fromkeys(r['task'] for r in rows)
    task_lines = []
    for task in tasks:
        n = len({r['participant_id'] for r in rows if r['task'] == task})
        task_lines.append(f'- {task} ({descriptions[task]}), N={n}')
    return ('# RF1-SRA subject-level outcome COPEs\n\n'
            f'This collaboration dataset contains {len(rows):,} contrast-of-parameter-estimate images from '
            f'{len(people)} development participants in session 01. All 62 participants ever reserved for '
            'validation were excluded.\n\n'
            + '\n'.join(task_lines) + '\n\n'
            'Each row of images.csv identifies one signed effect-estimate NIfTI in images/. '
            'Use participant_id to pair images across conditions and tasks. '
            'cope_index is the condition contrast number in the run design. '
            'For two retained runs the image is the fitted fixed-effects L2 COPE. '
            'For one retained run the L1 COPE serves as the subject estimate. '
            'The images preserve the original FEAT values and grids in MNI152NLin6Asym space. '
            'Free-text NIfTI header fields and extensions were removed.\n\n'
            'partner identifies the computer, friend or stranger in tasks with individual partners.\n\n'
            f'age is baseline age in years from BIDS participants.tsv ({missing_age} participants missing). '
            'tsnr (temporal signal-to-noise ratio) and fd_mean (mean framewise displacement in mm) '
            'come from MRIQC for that task. Each is the unweighted mean over exactly the retained runs. '
            'The run1 and run2 columns preserve their individual values. A blank run field means the run '
            f'did not contribute. QC uses the original rf1-sra-linux2 snapshot {QC_COMMIT}.\n\n'
            'source_zstat_available records existence of a companion FEAT Z-statistic. '
            'source_sha256 identifies the input COPE and sha256 verifies the distributed file.\n')


def verify_package(directory, rows):
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
    if (directory / 'README.md').read_text() != release_notes(rows):
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


def export(base, task_set, dry_run=False):
    relative = Path('work/colleague_exports') / task_set
    destination = base.output(relative)
    if destination != base.root / relative:
        raise PipelineError('The private export location resolves outside its designated work folder')
    sources, rows = image_plan(base, TASK_SETS[task_set])
    print(f'{len(rows)} subject-level COPEs. Companion zstats present for '
          f'{sum(r["source_zstat_available"] for r in rows)}/{len(rows)}.', flush=True)
    if dry_run:
        return
    if destination.exists():
        verify_package(destination, rows)
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
            (stage / 'README.md').write_text(release_notes(rows))
            with ZipFile(stage / 'subject_copes.zip', 'w', compression=ZIP_STORED) as archive:
                for name in ['images.csv', 'README.md', *(r['filename'] for r in rows)]:
                    archive.write(stage / name, name)
            verify_package(stage, rows)
            stage.chmod(0o700)
            stage.rename(destination)
    archive = destination / 'subject_copes.zip'
    print(f'Verified {archive} ({archive.stat().st_size / 2**20:.1f} MiB)', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', choices=tuple(TASK_SETS), default='all')
    parser.add_argument('--dry-run', action='store_true', help='Check source maps and metadata without writing a package')
    args = parser.parse_args()
    try:
        export(load_config(), args.tasks, args.dry_run)
    except (PipelineError, OSError, ValueError, BadZipFile) as exc:
        sys.exit(f'Export stopped: {exc}')
