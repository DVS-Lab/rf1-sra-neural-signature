import numpy as np
import nibabel as nib
import pytest
from build_mask import spatial_center, reconstruct, load_development_image, vectorize_source
from conftest import save_nii, metadata, make_subject, register_aging_subject
from inventory import header_info, InputUnavailable, cope_path
from make_split import make_split, guard_from_split
from utils import PipelineError, write_text


def test_vector_order_center_reconstruction():
    data = np.arange(60, dtype='float32').reshape((3,4,5))
    mask = data % 3 == 0
    vector = spatial_center(data, mask)
    assert abs(vector.mean()) < 1e-6
    restored = reconstruct(vector, mask)
    np.testing.assert_array_equal(restored[mask], data[mask]-data[mask].mean())
    assert not restored[~mask].any()
    assert np.linalg.norm(vector) != 1


def test_header_inventory_never_get_fdata(tmp_path, monkeypatch):
    path = tmp_path / 'image.nii.gz'
    save_nii(path, np.ones((3,4,5)))
    monkeypatch.setattr(nib.Nifti1Image, 'get_fdata', lambda *args, **kwargs: pytest.fail('voxel read'))
    assert header_info(path)['voxel_finite'] == 'not_read_header_only'


def test_output_escape_symlink_hardlink_and_sources_rejected(cfg, tmp_path):
    with pytest.raises(PipelineError): cfg.output('../escape')
    with pytest.raises(PipelineError): cfg.output(str(cfg.repos['trust'] / 'source'))
    (cfg.root / 'results').symlink_to(cfg.repos['trust'], target_is_directory=True)
    with pytest.raises(PipelineError): write_text(cfg, 'results/source.txt', 'bad')
    (cfg.root / 'results').unlink()
    (cfg.root / 'results').mkdir()
    source = cfg.repos['trust'] / 'source'
    source.write_text('unchanged')
    (cfg.root / 'results/linked').hardlink_to(source)
    with pytest.raises(PipelineError): write_text(cfg, 'results/linked', 'bad')
    assert source.read_text() == 'unchanged'


def test_cross_task_resampling_logged(cfg):
    guard = guard_from_split(make_split(cfg, metadata()))
    subject = sorted(guard.development)[0]
    path = cope_path(cfg, subject, 'trust', 4)
    save_nii(path, np.arange(125).reshape(5,5,5), np.diag([1.,1.,1.,1.]))
    reference = nib.Nifti1Image(np.ones((3,3,3)), np.diag([2.,2.,2.,1.]))
    logs = []
    vec = vectorize_source(cfg, subject, 'trust', 4, np.ones((3,3,3), bool), reference,
                           guard, sorted(guard.development), logs)
    assert len(vec) == 27 and logs[0]['resampled'] and logs[0]['interpolation'] == 'linear'


def test_nonfinite_development_voxels_rejected(cfg):
    guard = guard_from_split(make_split(cfg, metadata()))
    subject = sorted(guard.development)[0]
    path = cope_path(cfg, subject, 'trust', 4)
    data = np.ones((3,3,3)); data[0,0,0] = np.nan
    save_nii(path, data)
    with pytest.raises(InputUnavailable, match='nonfinite'):
        load_development_image(cfg, path, subject, guard, sorted(guard.development))


def test_fixed_95_percent_mask_and_template_fallback(cfg):
    from build_mask import build_mask
    from inventory import feat_dir
    guard = guard_from_split(make_split(cfg, metadata(70)))
    subjects = sorted(guard.development)
    assert len(subjects) == 20
    for i, subject in enumerate(subjects):
        register_aging_subject(cfg, subject)
        data = np.ones((3,3,3))
        if i == 0: data[0,0,0] = 0  # 19/20: retained
        if i < 2: data[0,0,1] = 0  # 18/20: removed
        for task in ['sharedreward', 'trust']:
            for k in cfg.contrasts[task]['copes']:
                save_nii(feat_dir(cfg, subject, task) / f'cope{k}.feat/mask.nii.gz', data)
        for task in ['socialdoors', 'doors']:
            save_nii(feat_dir(cfg, subject, task, 'L1', 1) / 'mask.nii.gz', data)
    save_nii(cope_path(cfg, subjects[0], 'sharedreward', 1), np.ones((3,3,3)))
    mask, reference = build_mask(cfg, guard, subjects)
    assert mask[0,0,0] and not mask[0,0,1]
    import json
    assert 'coverage_only' in json.loads(cfg.output('provenance/analysis_mask.json').read_text())['method']
    template = np.ones((3,3,3)); template[1,1,1] = 0
    save_nii(cfg.templateflow / 'tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz', template)
    mask, _ = build_mask(cfg, guard, subjects)
    assert not mask[1,1,1]
    with pytest.raises(PipelineError, match='HOLDOUT LOCK'):
        build_mask(cfg, guard, subjects + [next(iter(guard.holdout))])


def test_nifti_vector_round_trip(cfg):
    from build_mask import save_image
    mask = (np.arange(60).reshape(3,4,5) % 2 == 1)
    vector = np.arange(mask.sum(), dtype='float32')
    affine = np.diag([2.,2.,2.,1.])
    reference = nib.Nifti1Image(np.ones(mask.shape), affine)
    reference.header['descrip'] = b'private-participant-identity'
    reference.header['aux_file'] = b'private-aux'
    reference.header.extensions.append(nib.nifti1.Nifti1Extension(6, b'private-extension'))
    save_image(cfg, 'results/maps/roundtrip.nii.gz', reconstruct(vector, mask), reference)
    restored = nib.load(cfg.output('results/maps/roundtrip.nii.gz'))
    np.testing.assert_array_equal(restored.get_fdata()[mask], vector)
    np.testing.assert_array_equal(restored.affine, affine)
    assert b'private' not in restored.header['descrip'].tobytes()
    assert not restored.header['aux_file'].tobytes().strip(b'\x00')
    assert not restored.header.extensions
