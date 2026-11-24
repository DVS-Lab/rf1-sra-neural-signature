import csv
import json
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from conftest import con_text, fsf_text, l2_text
import doors_voxelwise
import export_colleague_copes as pack
from final_stress_sources import phase_dir, phase_spec
from inventory import feat_dir
from revised_design import revised_config
from test_final_stress import phase_manifest_fixture
from test_revised import setup_lock
from utils import InputUnavailable, PipelineError, sha256


def write_copes(folder, task, run=None):
    for k in pack.COPE_INDICES[task]:
        path = folder/(f'stats/cope{k}.nii.gz' if run else f'cope{k}.feat/stats/cope1.nii.gz')
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (np.arange(24, dtype='int16') - 12 + k + 100*(run or 0)).reshape(2, 3, 4)
        img = nib.Nifti1Image(raw, np.diag([2., 2., 2., 1.]))
        img.set_qform(img.affine, 4)
        img.set_sform(img.affine, 4)
        img.header.set_xyzt_units('mm')
        img.header.set_slope_inter(2.5, -1.25)
        img.header['descrip'] = b'internal source notes'
        img.header.extensions.append(nib.nifti1.Nifti1Extension(6, b'private text'))
        nib.save(img, path)
        if k % 2 == 0:
            path.with_name(path.name.replace('cope', 'zstat')).write_bytes(b'never read this statistic')


@pytest.fixture
def colleague_data(cfg, monkeypatch):
    split = setup_lock(cfg)
    people = sorted(split.loc[split.split == 'development', 'subject'])[:4]
    original = sorted(split.loc[split.split == 'holdout', 'subject'])
    supplemental = [f'sub-held{i:03d}' for i in range(12)]
    protected = original + supplemental
    records = []
    for s in people + protected:
        for cohort in ('partner_pair', 'three_paradigm'):
            if s in protected:
                eligible = s not in original[38:]
            else:
                eligible = s == people[0] or (cohort == 'partner_pair' and s == people[1])
            records.append(dict(subject=s, cohort=cohort, policy='tsnr_coverage_fd', eligible=eligible,
                                partition='holdout' if s in protected else 'development'))
    samples = revised_config(cfg, 'samples')
    manifest = samples.output('work/manifest.tsv')
    manifest.parent.mkdir(parents=True)
    pd.DataFrame(records).to_csv(manifest, sep='\t', index=False)
    monkeypatch.setattr(doors_voxelwise, 'MANIFEST_HASH', sha256(manifest))

    quality = []
    c = revised_config(cfg)
    phase_manifest_fixture(cfg, dict(subjects=people))
    (cfg.exclusions/('Smith-SRA-' + people[3].removeprefix('sub-'))).mkdir()
    for si, s in enumerate(people):
        for ti, task in enumerate(pack.TASK_SETS['all']):
            runs = (1,) if task in pack.DOORS else (1, 2)
            if si == 1 and task == 'sharedreward':
                runs = (2,)
            spec = phase_spec() if task == 'sharedreward' else c.contrasts[task]
            folders = []
            for r in runs:
                folder = phase_dir(c, s, 'L1', r) if task == 'sharedreward' else feat_dir(c, s, task, 'L1', r)
                folders.append(folder)
                folder.mkdir(parents=True)
                bold = c.fmriprep/s/'ses-01/func'/f'{s}_ses-01_task-{task}_run-{r}_space-MNI152NLin6Asym_bold.nii.gz'
                (folder/'design.fsf').write_text(fsf_text(spec, bold, rendered=True))
                con = con_text(spec)
                (folder/'design.con').write_text(con)
                ncols = len(con.splitlines()[1].split())
                np.savetxt(folder/'design.mat', np.eye(ncols), header='/Matrix', comments='')
                write_copes(folder, task, r)
                coverage = 99.5 + .1*r
                if si == 2 and task == 'trust' and r == 1:
                    coverage = 94.999
                if si == 1 and task in pack.DOORS:
                    coverage = 95.0
                quality.append(dict(subject=s, session='01', task=task, run=r, tsnr=100*ti + 10*r,
                                    fd_mean=ti + .1*r, brain_coverage_pct=coverage,
                                    tsnr_outlier=(si == 2 and task == 'doors'),
                                    fd_mean_outlier=(si == 1 and task == 'socialdoors'),
                                    brain_coverage_outlier=coverage < 99.1,
                                    tedana_outlier=False))
            if len(runs) == 2:
                output = phase_dir(c, s, 'L2') if task == 'sharedreward' else feat_dir(c, s, task, 'L2')
                output.mkdir(parents=True)
                (output/'design.fsf').write_text(l2_text(spec) + ''.join(
                    f'set feat_files({i}) "{folder}"\n' for i, folder in enumerate(folders, 1)))
                write_copes(output, task)
    quality.append({**quality[-1], 'subject': people[1], 'task': 'sharedreward', 'run': 1,
                    'tsnr': 9999, 'fd_mean': 99, 'tsnr_outlier': True})
    qs = cfg.repos['linux2'] / cfg.paths['qc_table']
    qs.parent.mkdir()
    pd.DataFrame(quality).to_csv(qs, sep='\t', index=False)
    monkeypatch.setattr(doors_voxelwise, 'QC_HASH', sha256(qs))
    meta = dict(manifest_sha256=sha256(manifest), original_split_sha256=sha256(cfg.output('work/splits/subject_split_v1.tsv')),
                qc_sha256=sha256(qs), protected_holdout_total_n=62)
    samples.output('provenance/samples.json').parent.mkdir(parents=True)
    samples.output('provenance/samples.json').write_text(json.dumps(meta))
    samples.output('provenance/run_status.json').write_text(json.dumps(
        dict(status='complete', holdout_scored=False, sample_manifest_sha256=sha256(manifest))))
    pd.DataFrame({'participant_id': people, 'age': ['27', 'n/a', '50', '62']}).to_csv(
        cfg.bids/'participants.tsv', sep='\t', index=False)
    return people, protected


def test_private_folder_csv_and_zip_match_frozen_inputs(cfg, colleague_data, monkeypatch):
    people, protected = colleague_data
    original_load = nib.load

    def authorized_load(file, *args, **kwargs):
        path = Path(file)
        assert not set(path.parts).intersection([*protected, people[3]])
        assert 'zstat' not in path.name
        if people[1] in path.parts:
            assert 'task-socialdoors_' not in str(path)
        if people[2] in path.parts:
            assert 'task-doors_' not in str(path) and 'task-sharedreward_' not in str(path)
            if 'task-trust_' in str(path):
                assert 'L1_task-trust' in str(path) and 'run-2_' in str(path)
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(nib, 'load', authorized_load)
    old_zip = cfg.output('work/colleague_exports/taskwise/all/subject_copes.zip')
    old_zip.parent.mkdir(parents=True)
    old_zip.write_bytes(b'previous matched release')
    pack.export(cfg, 'all')
    assert old_zip.read_bytes() == b'previous matched release'
    out = cfg.output('work/colleague_exports/coverage95/all')
    with (out/'images.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 38 and {r['participant_id'] for r in rows} == set(people[:3])
    by_task = {t: {r['participant_id'] for r in rows if r['task'] == t} for t in pack.TASK_SETS['all']}
    assert by_task == dict(sharedreward=set(people[:2]), trust=set(people[:3]),
                           socialdoors={people[0], people[2]}, doors=set(people[:2]))
    assert not any(s in (out/'images.csv').read_text() for s in protected)
    for row in rows:
        path = out / row['filename']
        image = nib.load(path)
        assert sha256(path) == row['sha256']
        assert path.parent == out/'images' and image.get_data_dtype() == np.dtype('int16')
        assert image.header['qform_code'] == image.header['sform_code'] == 4
        assert not bytes(image.header['descrip']).strip(b'\0') and not image.header.extensions
        k = int(row['cope_index'])
        run = int(row['retained_runs']) if row['source_level'] == 'L1' else 0
        expected = (np.arange(24).reshape(2, 3, 4) - 12 + k + 100*run) * 2.5 - 1.25
        np.testing.assert_array_equal(image.get_fdata(), expected)
        assert row['source_zstat_available'] == str(k % 2 == 0)
        assert row['complete_task_conditions'] == 'True'
        assert row['in_paper_partner_pair'] == str(row['participant_id'] in people[:2])
        assert row['in_paper_three_paradigm'] == str(row['participant_id'] == people[0])
        assert row['age'] == {people[0]: '27.0', people[1]: '', people[2]: '50.0'}[row['participant_id']]
        if row['participant_id'] == people[1] and row['task'] == 'sharedreward':
            assert row['age'] == '' and row['source_level'] == 'L1' and row['retained_runs'] == '2'
            assert row['tsnr_run1'] == '' and float(row['tsnr']) == 20 and float(row['fd_mean']) == .2
        if row['task'] == 'trust' and row['participant_id'] in people[:2]:
            assert row['source_level'] == 'L2' and row['retained_runs'] == '1,2'
            assert float(row['tsnr']) == 115 and float(row['fd_mean']) == pytest.approx(1.15)
            assert float(row['brain_coverage_pct']) == pytest.approx(99.65)
        if row['task'] == 'trust' and row['participant_id'] == people[2]:
            assert row['source_level'] == 'L1' and row['retained_runs'] == '2'
            assert row['brain_coverage_pct_run1'] == '' and float(row['brain_coverage_pct']) == 99.7
            assert float(row['tsnr']) == 120 and float(row['fd_mean']) == 1.2
        if row['task'] in pack.DOORS:
            assert row['source_level'] == 'L1' and row['retained_runs'] == '1'
    with ZipFile(out/'subject_copes.zip') as archive:
        assert set(archive.namelist()) == {'images.csv', 'README.md', *(r['filename'] for r in rows)}
        assert archive.read('images.csv') == (out/'images.csv').read_bytes()
        notes = archive.read('README.md').decode()
        assert '1 with all four Doors outcome maps' in notes
        assert '| Trust, reciprocation and defection | 3 | 3 | 6 | 12 |' in notes
    pack.export(cfg, 'all')  # A retry must verify and reuse the existing completed export.
    (out / rows[0]['filename']).write_bytes(b'corrupt')
    with pytest.raises(PipelineError, match='checksum mismatch'):
        pack.export(cfg, 'all')


def test_heldout_source_alias_stops_before_voxels_or_package(cfg, colleague_data, monkeypatch):
    people, protected = colleague_data
    c = revised_config(cfg)
    path = feat_dir(c, people[0], 'socialdoors', 'L1', 1) / 'stats/cope1.nii.gz'
    forbidden = feat_dir(c, protected[-1], 'socialdoors', 'L1', 1) / 'stats/cope1.nii.gz'
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b'sealed')
    path.unlink()
    path.symlink_to(forbidden)
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('Opened a protected input'))
    with pytest.raises(PipelineError, match='cross-participant alias'):
        pack.export(cfg, 'doors')
    assert not cfg.output('work/colleague_exports/coverage95/doors').exists()


def test_missing_person_lock_stops_before_source_access(cfg, monkeypatch):
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('Opened an image without the roster'))
    with pytest.raises(PipelineError, match='private revised sample manifest unavailable'):
        pack.export(cfg, 'all')


def test_single_usable_run_without_l2(cfg, colleague_data):
    scope = pack.export_scope(cfg)
    ages = pack.baseline_ages(cfg, scope['subjects'])
    qc = pack.frozen_run_qc(cfg, scope['subjects'])
    subject = colleague_data[0][0]
    task = 'trust'
    c = scope['c']
    (feat_dir(c, subject, task, 'L2')/'cope4.feat/stats/cope1.nii.gz').unlink()
    images, missing = pack.select_maps(scope, qc, subject, task)
    assert missing == {4: 'two_usable_L1s_without_L2'}
    assert set(images) == set(range(5, 10)) and all(im.runs == (1, 2) for im in images.values())
    _, rows = pack.image_plan(cfg, ('trust',), scope, ages, qc)
    partial = [r for r in rows if r['participant_id'] == subject]
    assert len(partial) == 5 and not any(r['complete_task_conditions'] for r in partial)
    (feat_dir(c, subject, task, 'L1', 1)/'stats/cope4.nii.gz').unlink()
    images, missing = pack.select_maps(scope, qc, subject, task)
    assert not missing and len(images) == 6
    assert images[4].runs == (2,) and images[5].runs == (1, 2)
    assert images[4].path == feat_dir(c, subject, task, 'L1', 2)/'stats/cope4.nii.gz'
    _, rows = pack.image_plan(cfg, ('trust',), scope, ages, qc)
    mixed = {r['cope_index']: r for r in rows if r['participant_id'] == subject}
    assert all(r['complete_task_conditions'] for r in mixed.values())
    assert mixed[4]['tsnr'] == 120 and mixed[5]['tsnr'] == 115
    assert mixed[4]['fd_mean_run1'] == '' and mixed[5]['fd_mean_run1'] == 1.1


def test_unknown_qc_and_unestimable_single_run(cfg, colleague_data, monkeypatch):
    scope = pack.export_scope(cfg)
    qc = pack.frozen_run_qc(cfg, scope['subjects'])
    subject = colleague_data[0][0]
    task = 'socialdoors'
    key = (subject, task, 1)
    original_load = nib.load

    def allowed_load(file, *args, **kwargs):
        assert Path(file).name == 'cope2.nii.gz', 'opened unqualified COPE'
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(nib, 'load', allowed_load)
    without_numeric = qc.copy()
    without_numeric.loc[key, 'brain_coverage_pct'] = np.nan
    with pytest.raises(InputUnavailable, match='unknown_qc'):
        pack.select_maps(scope, without_numeric, subject, task)
    with pytest.raises(InputUnavailable, match='unknown_qc'):
        pack.select_maps(scope, qc.drop(index=key), subject, task)
    mat = feat_dir(scope['c'], subject, task, 'L1', 1)/'design.mat'
    design = np.loadtxt(mat, skiprows=1, ndmin=2)
    design[:, 0] = 0
    np.savetxt(mat, design, header='/Matrix', comments='')
    images, missing = pack.select_maps(scope, qc, subject, task)
    assert set(images) == {2} and missing == {1: 'no_existing_estimable_cope'}


def test_coverage95_replaces_only_coverage_flag_and_counts_distinct_people(cfg, colleague_data, monkeypatch):
    people, _ = colleague_data
    scope = pack.export_scope(cfg)
    qc = pack.frozen_run_qc(cfg, scope['subjects'])
    row = qc.loc[(people[1], 'doors', 1)].copy()
    assert row['brain_coverage_pct'] == 95 and row['brain_coverage_outlier'] == 'True'
    assert not pack.run_qc_reason(row)
    row['brain_coverage_outlier'] = ''
    assert not pack.run_qc_reason(row)
    row['brain_coverage_pct'] = 94.999
    row['brain_coverage_outlier'] = 'False'
    assert pack.run_qc_reason(row) == 'coverage_below_95pct'
    for unknown in (np.nan, np.inf, 101, -1):
        row['brain_coverage_pct'] = unknown
        assert pack.run_qc_reason(row) == 'unknown_qc'
    summary = pack.coverage_review(qc, pack.TASK_SETS['all']).set_index('task')
    total = summary.loc['Combined']
    assert (total.old_runs, total.old_people, total.below95_runs, total.below95_people) == (3, 2, 1, 1)
    assert total.regained_runs == 2 and total.regained_people == 1 and total.regained_passing_other_qc == 1
    assert summary.loc['socialdoors', 'regained_passing_other_qc'] == 0
    assert summary.loc['doors', 'regained_passing_other_qc'] == 1
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('QC comparison opened an image'))
    monkeypatch.setattr(pack, 'baseline_ages', lambda *a: pytest.fail('QC comparison requested demographics'))
    pack.export(cfg, 'all', qc_only=True)
    assert not cfg.output('work/colleague_exports/coverage95/all').exists()
