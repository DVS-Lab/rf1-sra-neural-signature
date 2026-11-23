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
from utils import PipelineError, sha256


@pytest.fixture
def colleague_data(cfg, monkeypatch):
    split = setup_lock(cfg)
    people = sorted(split.loc[split.split == 'development', 'subject'])[:2]
    original = sorted(split.loc[split.split == 'holdout', 'subject'])
    supplemental = [f'sub-held{i:03d}' for i in range(12)]
    protected = original + supplemental
    records = []
    for s in people + protected:
        for cohort in ('partner_pair', 'three_paradigm'):
            eligible = s not in original[38:] and (cohort == 'partner_pair' or s != people[1])
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
    for si, s in enumerate(people):
        for ti, task in enumerate(pack.TASK_SETS['all']):
            if si and task in pack.DOORS:
                continue
            runs = (1,) if task in pack.DOORS else (1, 2)
            if si and task == 'sharedreward':
                runs = (2,)
            spec = phase_spec() if task == 'sharedreward' else c.contrasts[task]
            folders = []
            for r in runs:
                folder = phase_dir(c, s, 'L1', r) if task == 'sharedreward' else feat_dir(c, s, task, 'L1', r)
                folders.append(folder)
                folder.mkdir(parents=True)
                bold = c.fmriprep/s/'ses-01/func'/f'{s}_ses-01_task-{task}_run-{r}_space-MNI152NLin6Asym_bold.nii.gz'
                (folder/'design.fsf').write_text(fsf_text(spec, bold, rendered=True))
                (folder/'design.con').write_text(con_text(spec))
                quality.append(dict(subject=s, session='01', task=task, run=r, tsnr=100*ti + 10*r,
                                    fd_mean=ti + .1*r, tsnr_outlier=False, fd_mean_outlier=False,
                                    brain_coverage_outlier=False, tedana_outlier=False))
            output = folders[0]
            if len(runs) == 2:
                output = phase_dir(c, s, 'L2') if task == 'sharedreward' else feat_dir(c, s, task, 'L2')
                output.mkdir(parents=True)
                (output/'design.fsf').write_text(l2_text(spec) + ''.join(
                    f'set feat_files({i}) "{folder}"\n' for i, folder in enumerate(folders, 1)))
            for k in pack.COPE_INDICES[task]:
                path = output/(f'cope{k}.feat/stats/cope1.nii.gz' if len(runs) == 2 else f'stats/cope{k}.nii.gz')
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = (np.arange(24, dtype='int16') - 12 + k).reshape(2, 3, 4)
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
    pd.DataFrame({'participant_id': people, 'age': ['27', 'n/a']}).to_csv(
        cfg.bids/'participants.tsv', sep='\t', index=False)
    return people, protected


def test_private_folder_csv_and_zip_match_frozen_inputs(cfg, colleague_data, monkeypatch):
    people, protected = colleague_data
    original_load = nib.load

    def authorized_load(file, *args, **kwargs):
        assert not set(Path(file).parts).intersection(protected)
        assert 'zstat' not in Path(file).name
        return original_load(file, *args, **kwargs)

    monkeypatch.setattr(nib, 'load', authorized_load)
    pack.export(cfg, 'all')
    out = cfg.output('work/colleague_exports/all')
    with (out/'images.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 28 and {r['participant_id'] for r in rows} == set(people)
    assert {r['participant_id'] for r in rows if r['task'] in pack.DOORS} == {people[0]}
    assert not any(s in (out/'images.csv').read_text() for s in protected)
    for row in rows:
        path = out / row['filename']
        image = nib.load(path)
        assert sha256(path) == row['sha256']
        assert path.parent == out/'images' and image.get_data_dtype() == np.dtype('int16')
        assert image.header['qform_code'] == image.header['sform_code'] == 4
        assert not bytes(image.header['descrip']).strip(b'\0') and not image.header.extensions
        k = int(row['cope_index'])
        expected = (np.arange(24).reshape(2, 3, 4) - 12 + k) * 2.5 - 1.25
        np.testing.assert_array_equal(image.get_fdata(), expected)
        assert row['source_zstat_available'] == str(k % 2 == 0)
        if row['participant_id'] == people[1] and row['task'] == 'sharedreward':
            assert row['age'] == '' and row['source_level'] == 'L1' and row['retained_runs'] == '2'
            assert row['tsnr_run1'] == '' and float(row['tsnr']) == 20 and float(row['fd_mean']) == .2
        if row['task'] == 'trust':
            assert row['source_level'] == 'L2' and row['retained_runs'] == '1,2'
            assert float(row['tsnr']) == 115 and float(row['fd_mean']) == pytest.approx(1.15)
        if row['task'] in pack.DOORS:
            assert row['source_level'] == 'L1' and row['retained_runs'] == '1' and row['age'] == '27.0'
    with ZipFile(out/'subject_copes.zip') as archive:
        assert set(archive.namelist()) == {'images.csv', 'README.md', *(r['filename'] for r in rows)}
        assert archive.read('images.csv') == (out/'images.csv').read_bytes()
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
    assert not cfg.output('work/colleague_exports/doors').exists()


def test_missing_person_lock_stops_before_source_access(cfg, monkeypatch):
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('Opened an image without the roster'))
    with pytest.raises(PipelineError, match='private revised sample manifest unavailable'):
        pack.export(cfg, 'all')
