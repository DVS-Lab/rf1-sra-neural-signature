"""Header-only inventory: never materialize voxels, even before holdout selection."""
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import re
import hashlib
import nibabel as nib
import numpy as np
import pandas as pd
from preflight import parse_fsf, verify_l1, verify_design_con, verify_l2
from utils import CORE_TASKS, PipelineError, InputUnavailable, subject_id, write_tsv, write_json
from aging_source import aging_directory, aging_entry, subject_unit, load_aging_index, verify_stamp


def feat_dir(c, subject, task, level='L2', run=None):
    subject_id(subject)
    if task == 'sharedreward': return aging_directory(c, subject, level, run)
    spec = c.contrasts[task]
    if task in ['socialdoors', 'doors'] and level != 'L1':
        raise PipelineError('Social/monetary Doors must use separate L1 maps')
    stem = f'{level}_task-{task}_ses-01_model-{spec["model"]}_type-act'
    if level == 'L1':
        if run not in [1, 2] or (task in ['socialdoors', 'doors'] and run != 1):
            raise PipelineError('Invalid run')
        stem += f'_run-{run}'
    stem += '_' + spec['suffix'] + ('.feat' if level == 'L1' else '.gfeat')
    return c.repos[spec['repository']] / 'derivatives/fsl' / subject / 'ses-01' / stem


def cope_path(c, subject, task, cope, level='L2', run=None):
    if task == 'sharedreward' and level == 'L2': level, run = subject_unit(c, subject)
    base = feat_dir(c, subject, task, level, run)
    return base / (f'stats/cope{cope}.nii.gz' if level == 'L1' else f'cope{cope}.feat/stats/cope1.nii.gz')


def mask_paths(c, subject, task):
    level, run = ('L1', 1) if task in ['socialdoors', 'doors'] else ('L2', None)
    if task == 'sharedreward': level, run = subject_unit(c, subject)
    base = feat_dir(c, subject, task, level, run)
    return ([base/'mask.nii.gz'] if level == 'L1' else
            [base/f'cope{k}.feat/mask.nii.gz' for k in c.contrasts[task]['copes']])


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def summarize_feat_paths(c, subjects):
    """Directory-only diagnostics; alternate layouts never become analysis inputs.

    Limit public names to structural FEAT entities, excluding arbitrary directory
    names that could contain participant identifiers or other private text.
    """
    rows = []
    for task in CORE_TASKS:
        spec = c.contrasts[task]
        expected_root = c.repos[spec['repository']] / 'derivatives/fsl'
        if task == 'sharedreward': expected_root /= 'rf1'
        roots = [expected_root]
        if task == 'sharedreward':
            legacy_root = c.repos['linux2'] / 'derivatives/fsl'
            if legacy_root != expected_root:
                roots.append(legacy_root)
        units = [('L1', 1)] if task in ['socialdoors', 'doors'] else [('L1', 1), ('L1', 2), ('L2', None)]
        expected = {feat_dir(c, 'sub-placeholder', task, level, run).name for level, run in units}
        pattern = re.compile(
            rf'L[12]_task-{task}_(?:ses-01_)?model-(?:[0-9]+|fulltrial)_type-act'
            r'(?:_run-[12])?_sm(?:To)?-[0-9]+(?:p[0-9]+)?\.(?:feat|gfeat)')
        for root in roots:
            observed = Counter()
            for subject in subjects:
                folder = root / subject / 'ses-01'
                if folder.is_dir():
                    for path in folder.iterdir():
                        if pattern.fullmatch(path.name) and path.is_dir():
                            observed[path.name] += 1
            for name in sorted(expected | set(observed)):
                rows.append({'task': task, 'source_root': str(root), 'root_exists': root.is_dir(),
                             'layout': f'sub-<ID>/ses-01/{name}',
                             'expected_input': root == expected_root and name in expected,
                             'directory_n': observed[name]})
    return rows


def header_info(path, space='unknown'):
    """Finite geometry is checkable without testing holdout voxel intensities."""
    if not nonempty(path):
        raise InputUnavailable('missing_or_empty_image')
    try:
        img = nib.load(path)
        shape, affine = img.shape, img.affine
        zooms = img.header.get_zooms()
        q, qc = img.get_qform(coded=True)
        s, sc = img.get_sform(coded=True)
        if (len(shape) != 3 or min(shape) <= 0 or not np.isfinite(affine).all()
                or not np.isfinite(zooms).all() or min(zooms) <= 0
                or abs(np.linalg.det(affine[:3, :3])) < 1e-8):
            raise ValueError('invalid_geometry')
        if qc and sc and not np.allclose(q, s, atol=1e-3):
            raise ValueError('conflicting_xforms')
        if not (qc or sc):
            raise ValueError('unknown_xforms')
        if img.header.get_xyzt_units()[0] not in ['mm']:
            raise ValueError('unknown_spatial_units')
        return {'shape': json.dumps(shape), 'zooms': json.dumps(zooms, default=float),
                'affine_fingerprint': hashlib.sha256(np.round(affine, 5).tobytes()).hexdigest(),
                'space': space, 'geometry_finite': True, 'nonempty_file': True,
                'voxel_finite': 'not_read_header_only'}
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as exc:
        raise InputUnavailable('invalid_image_header') from exc


def same_grid(a, b):
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4, rtol=0)


def l1_evidence(c, subject, task, run):
    base = feat_dir(c, subject, task, 'L1', run)
    if not (base / 'design.fsf').is_file() or not (base / 'design.con').is_file():
        raise InputUnavailable('missing_design_evidence')
    values = parse_fsf(base / 'design.fsf')
    verify_l1(values, c.contrasts[task], rendered=True)
    verify_design_con(base / 'design.con', c.contrasts[task])
    data = Path(values.get('feat_files(1)', ''))
    if task == 'sharedreward':
        entry = aging_entry(c, subject)
        if run not in entry['runs']:
            raise PipelineError('Shared Reward run was not retained by the verified audit')
        if data.resolve() != Path(entry['l1'][run]['input']).resolve():
            raise PipelineError('Aging L1 input differs from the frozen retained-run manifest')
        verify_stamp(c, base, 'L1')
    if not data.is_absolute():
        raise InputUnavailable('ambiguous_input_space')
    # Do not trust generic NIfTI MNI codes to identify an MNI template variant.
    tokens = [subject + '_', 'ses-01_', f'task-{task}_', f'run-{run}_', 'space-MNI152NLin6Asym_']
    if not all(token in data.name for token in tokens):
        raise InputUnavailable('ambiguous_input_space')
    if not data.exists() and not str(data).endswith('.nii.gz'):
        data = Path(str(data) + '.nii.gz')
    # This is BOLD *header* access, never raw behavior or voxel access.
    try:
        bold = nib.load(data)
        output = nib.load(cope_path(c, subject, task, next(iter(c.contrasts[task]['copes'])), 'L1', run))
    except (OSError, nib.filebasedimages.ImageFileError) as exc:
        raise InputUnavailable('missing_geometry_evidence') from exc
    if not same_grid(bold, output):
        raise InputUnavailable('input_output_grid_mismatch')
    return values


def inspect_unit(c, subject, task, level, run=None):
    """Validate completion and actual design provenance, then required map headers."""
    spec = c.contrasts[task]
    base = feat_dir(c, subject, task, level, run)
    if not base.is_dir(): raise InputUnavailable('missing_feat_directory')
    if level == 'L1':
        l1_evidence(c, subject, task, run)
        required = [base / x for x in ['design.mat', 'design.con', 'mask.nii.gz', 'cluster_mask_zstat1.nii.gz']]
        required += [base / f'stats/cope{k}.nii.gz' for k in
                     (spec['copes'] if task == 'sharedreward' else range(1, spec['n_copes'] + 1))]
        masks = [base / 'mask.nii.gz']
    else:
        if not (base / 'design.fsf').is_file(): raise InputUnavailable('missing_design_evidence')
        values = parse_fsf(base / 'design.fsf')
        verify_l2(values, spec)
        for r in [1, 2]:
            actual = Path(values.get(f'feat_files({r})', '')).resolve()
            if actual != feat_dir(c, subject, task, 'L1', r).resolve():
                raise PipelineError('Completed L2 input does not match the expected participant/run')
            l1_evidence(c, subject, task, r)
        if task == 'sharedreward':
            verify_stamp(c, base, 'L2', [feat_dir(c, subject, task, 'L1', r) for r in [1, 2]])
        required = [base / 'design.mat', base / 'design.con']
        for k in (spec['copes'] if task == 'sharedreward' else range(1, spec['n_copes'] + 1)):
            required.extend(base / f'cope{k}.feat' / x for x in
                            ['design.mat', 'design.con', 'mask.nii.gz', 'stats/cope1.nii.gz',
                             'stats/zstat1.nii.gz', 'cluster_mask_zstat1.nii.gz'])
        masks = [base / f'cope{k}.feat/mask.nii.gz' for k in spec['copes']]
    if any(not nonempty(p) for p in required): raise InputUnavailable('incomplete_feat')
    if task == 'sharedreward' and level == 'L2':
        for k in spec['copes']:
            for filename, expected in [('design.mat', np.ones((2, 1))), ('design.con', np.ones((1, 1)))]:
                text = (base/f'cope{k}.feat'/filename).read_text()
                if '/Matrix' not in text: raise PipelineError('Aging L2 completed matrix is missing')
                matrix = np.loadtxt(text.split('/Matrix', 1)[1].splitlines(), ndmin=2)
                if matrix.shape != expected.shape or not np.allclose(matrix, expected, atol=1e-8, rtol=0):
                    raise PipelineError('Aging L2 completed matrix must be the two-run fixed-effects mean')
    images = [cope_path(c, subject, task, k, level, run) for k in spec['copes']] + masks
    rows = []
    reference = nib.load(images[0])
    if level == 'L2':
        l1 = nib.load(cope_path(c, subject, task, next(iter(spec['copes'])), 'L1', 1))
        if not same_grid(reference, l1): raise InputUnavailable('l2_grid_mismatch')
    for path in images:
        info = header_info(path, c.analysis['space'])
        if not same_grid(reference, nib.load(path)): raise InputUnavailable('within_unit_grid_mismatch')
        rows.append({'subject': subject, 'task': task, 'level': level, 'run': run,
                     'path': str(path), **info})
    return rows


def flip_angle(c, subject):
    """Resolve applicable BIDS JSON inheritance; reject conflicting echoes/runs."""
    run_angles = []
    for run in [1, 2]:
        folder = c.bids / subject / 'ses-01' / 'func'
        bolds = sorted(folder.glob(f'{subject}_ses-01_task-sharedreward_run-{run}_*bold.nii*'))
        values = []
        for bold in bolds:
            name = bold.name.split('.nii')[0]
            entities = dict(p.split('-', 1) for p in name.split('_') if '-' in p)
            if entities.get('part', 'mag') != 'mag' or 'space' in entities or 'desc' in entities: continue
            angle = None
            for directory in [c.bids, c.bids / subject, c.bids / subject / 'ses-01', folder]:
                matches = []
                for path in directory.glob('*bold.json'):
                    ent = dict(p.split('-', 1) for p in path.stem.split('_') if '-' in p)
                    if all(entities.get(k) == v for k, v in ent.items()): matches.append((len(ent), path))
                by_specificity = {}
                for specificity, path in sorted(matches):
                    payload = json.loads(path.read_text())
                    if 'FlipAngle' in payload:
                        value = float(payload['FlipAngle'])
                        if specificity in by_specificity and by_specificity[specificity] != value:
                            return 'discrepant', 'conflicting_bids_metadata'
                        by_specificity[specificity] = value
                        angle = value
            values.append(angle)
        if not values or any(x is None or not np.isfinite(x) for x in values):
            run_angles.append(None)
        elif not np.allclose(values, values[0], atol=1e-6, rtol=0):
            return 'discrepant', 'flipangle_echo_mismatch'
        else: run_angles.append(values[0])
    if any(x is None for x in run_angles): return 'missing', 'missing_flipangle'
    if not np.isclose(*run_angles, atol=1e-6, rtol=0): return 'discrepant', 'flipangle_run_mismatch'
    return f'{run_angles[0]:g}', ''


def inventory(c, write=True):
    load_aging_index(c, refresh=True)
    people = pd.read_csv(c.bids / 'participants.tsv', sep='\t', dtype={'participant_id': str})
    if 'participant_id' not in people or people.participant_id.duplicated().any():
        raise PipelineError('Invalid canonical participants table')
    people = people.set_index('participant_id')
    subjects = sorted(p.name for p in c.bids.glob('sub-*') if p.is_dir())
    counts = Counter(total_bids_participants=len(subjects), source_excluded=0)
    rows, geometry, missing = [], [], Counter()
    for index, subject in enumerate(subjects):
        if index and index % 25 == 0:
            print(f"  Header inventory: {index}/{len(subjects)} participants", flush=True)
        subject_id(subject)
        if (c.exclusions / f'Smith-SRA-{subject[4:]}').is_dir():
            counts['source_excluded'] += 1
            continue  # No per-person source-exclusion status or reason is retained.
        row = {'subject': subject, 'ses01': (c.bids / subject / 'ses-01').is_dir()}
        meta = people.loc[subject] if subject in people.index else pd.Series(dtype=object)
        age = pd.to_numeric(meta.get(c.paths['age_column'], np.nan), errors='coerce')
        row['age'] = float(age) if pd.notna(age) and np.isfinite(age) else np.nan
        row['sex'] = str(meta.get(c.paths['sex_column'], 'missing'))
        try: row['flip_angle'], row['metadata_flag'] = flip_angle(c, subject)
        except (ValueError, OSError, TypeError): row['flip_angle'], row['metadata_flag'] = 'missing', 'invalid_bids_metadata'
        for task in c.contrasts:
            reasons = []
            units = [('L1', 1)] if task in ['socialdoors', 'doors'] else [('L2', None)]
            if task == 'sharedreward':
                entry = c.aging_subjects.get(subject)
                row['sharedreward_runs'] = ','.join(map(str, entry['runs'])) if entry else ''
                row['sharedreward_strategy'] = entry['strategy'] if entry else 'unavailable'
                row['sharedreward_two_runs'] = bool(entry and len(entry['runs']) == 2)
                units = [('L1', r) for r in entry['runs']] if entry else []
                if entry and len(entry['runs']) == 2: units.append(('L2', None))
                if not entry: reasons.append('not_in_verified_aging_manifest')
            for level, run in units:
                try: geometry.extend(inspect_unit(c, subject, task, level, run))
                except InputUnavailable as exc: reasons.append(f'{level}_{run or 0}:{exc}')
                except PipelineError as exc:
                    raise PipelineError(f'Core {task} completed design contradicts the scientific contract: {exc}') from exc
            if not row['ses01']: reasons.append('missing_ses01')
            row[task] = not reasons
            row[task + '_reason'] = ';'.join(sorted(set(reasons)))
            for reason in set(reasons): missing[(task, reason)] += 1
        row['eligible'] = all(row[task] for task in CORE_TASKS)
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty: raise PipelineError('No non-source-excluded BIDS participants')
    eligible = table[table.eligible]
    for task in CORE_TASKS:
        counts[task + '_complete'] = int(table[task].sum())
    counts['multitask_complete'] = len(eligible)
    overlap = []
    for size in range(1, len(CORE_TASKS) + 1):
        for tasks in combinations(CORE_TASKS, size):
            overlap.append({'tasks': '+'.join(tasks), 'n': int(table[list(tasks)].all(axis=1).sum())})
    print('  Exact task intersections:', flush=True)
    for item in overlap:
        print(f"    {item['tasks']}: N={item['n']}", flush=True)
    summary = [{'category': k, 'reason': '', 'n': v} for k, v in counts.items()]
    summary += [{'category': task, 'reason': reason, 'n': n} for (task, reason), n in sorted(missing.items())]
    summary += [{'category': 'metadata', 'reason': flag, 'n': int(n)}
                for flag, n in eligible.metadata_flag.value_counts().items() if flag]
    summary.append({'category': 'metadata', 'reason': 'age_missing', 'n': int(eligible.age.isna().sum())})
    paths = summarize_feat_paths(c, table.subject)
    if missing:
        print('  Unavailable task inputs (participant counts):', flush=True)
        for (task, reason), n in sorted(missing.items()):
            print(f'    {task}: {reason}: N={n}', flush=True)
    for task in CORE_TASKS:
        if counts[task + '_complete'] == 0:
            print(f'  {task}: ZERO complete participants; directory diagnostics (existence only):', flush=True)
            for item in paths:
                if item['task'] == task and (item['expected_input'] or item['directory_n']):
                    kind = 'expected' if item['expected_input'] else 'alternate; NOT selected'
                    print(f'    {kind}: {item["source_root"]}/{item["layout"]}: '
                          f'N={item["directory_n"]}; root_exists={item["root_exists"]}', flush=True)
    if write:
        write_json(c, 'provenance/sharedreward_source.json', c.aging_provenance)
        write_tsv(c, 'work/inventory/task_availability.tsv', table)
        write_tsv(c, 'work/inventory/image_headers.tsv', geometry)
        write_tsv(c, 'results/aggregate/inventory_summary.tsv', summary)
        write_tsv(c, 'results/aggregate/task_overlap.tsv', overlap)
        write_tsv(c, 'results/aggregate/feat_path_summary.tsv', paths)
        summarize_qc(c, set(table.subject))
    print(f'  BIDS N={len(subjects)}; excluded N={counts["source_excluded"]}; multitask-complete N={len(eligible)}', flush=True)
    return table, summary


def summarize_qc(c, subjects):
    path = c.repos['linux2'] / c.paths['qc_table']
    rows = []
    if path.is_file():
        qc = pd.read_csv(path, sep='\t', dtype=str)
        id_col = next((k for k in ['subject', 'participant_id', 'sub'] if k in qc), None)
        if id_col:
            ids = qc[id_col].map(lambda x: str(x) if str(x).startswith('sub-') else 'sub-' + str(x))
            qc = qc[ids.isin(subjects)]
        if 'session' in qc: qc = qc[qc.session.isin(['01', 'ses-01', '1'])]
        # Only explicit flags/statuses; no free-text reasons or source identities.
        for column in qc:
            if column == 'qc_status' or column.endswith('_flag') or column.startswith('flag_') or column.endswith('_outlier'):
                for value, n in qc[column].fillna('missing').value_counts().items():
                    if str(value).lower() in ['true', 'false', '0', '1', 'yes', 'no', 'pass', 'fail', 'ok', 'incomplete', 'missing', 'outlier']:
                        rows.append({'flag': column, 'value': value, 'n': int(n), 'action': 'descriptive_only'})
    if not rows: rows = [{'flag': 'qc_summary', 'value': 'unavailable_or_no_recognized_flags', 'n': 0, 'action': 'descriptive_only'}]
    write_tsv(c, 'results/aggregate/qc_summary.tsv', rows)
