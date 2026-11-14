"""Actual paired arithmetic, FWE inference, development guard and image/report output."""
from itertools import product

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ttest_1samp

import doors_voxelwise as impl
from revised_samples import RevisedGuard
from utils import PipelineError
from conftest import fsf_text, save_nii
from inventory import feat_dir


def test_paired_contrast_and_sign_flip_maxima_match_brute_force():
    rng = np.random.default_rng(7)
    social = rng.normal(size=(2, 7, 11)) + np.array([2, .5])[:, None, None]
    monetary = rng.normal(size=social.shape) + 12
    differences = impl.paired_differences(social, monetary)
    np.testing.assert_allclose(differences[2], (social[0]-social[1])-(monetary[0]-monetary[1]))
    np.testing.assert_allclose(differences[0], social[0]-monetary[0])
    np.testing.assert_allclose(differences[1], social[1]-monetary[1])
    flips = np.array(list(product([-1., 1.], repeat=7))[:-1])
    for j, two_sided in enumerate((False, False, True)):
        x = differences[j]
        effect, t, p, null = impl.max_t_test(x, flips, two_sided, block_size=3)
        observed = ttest_1samp(x, 0, axis=0).statistic
        perm = ttest_1samp(flips[:, :, None]*x, 0, axis=1).statistic
        maximum = (abs(perm) if two_sided else perm).max(axis=1)
        observed_tail = abs(observed) if two_sided else observed
        expected = (1+(maximum[:, None] >= observed_tail-1e-12).sum(0))/(1+len(flips))
        np.testing.assert_allclose(effect, x.mean(0))
        np.testing.assert_allclose(t, observed)
        np.testing.assert_allclose(null, maximum)
        np.testing.assert_array_equal(p, expected)


def test_holdout_and_alias_rejected_before_image_access(cfg, monkeypatch):
    people = ['sub-dev1', 'sub-dev2']
    held = frozenset(f'sub-held{i}' for i in range(62))
    guard = RevisedGuard(frozenset(people), held)
    scope = dict(c=cfg, guard=guard, subjects=people)
    monkeypatch.setattr(nib, 'load', lambda *_: pytest.fail('must reject before NIfTI access'))
    with pytest.raises(PipelineError, match='HOLDOUT LOCK'):
        impl.design_paths(scope, 'sub-held0', 'doors')
    root = cfg.repos['socdoors']
    fake = root/'sub-held0/ses-01'
    fake.mkdir(parents=True)
    (fake/'cope1.nii.gz').touch()
    alias = root/'sub-dev1/ses-01'
    alias.mkdir(parents=True)
    (alias/'cope1.nii.gz').symlink_to(fake/'cope1.nii.gz')
    with pytest.raises(PipelineError, match='cross-participant alias'):
        impl.source_path(scope, 'sub-dev1', alias/'cope1.nii.gz', root)


def test_headers_contrast_direction_common_coverage_and_report(cfg, monkeypatch):
    from revised_design import revised_config
    c = revised_config(cfg, 'tsnr_coverage_fd/three_paradigm')
    people = ['sub-dev1', 'sub-dev2', 'sub-dev3']
    held = frozenset(f'sub-held{i}' for i in range(62))
    scope = dict(c=c, guard=RevisedGuard(frozenset(people), held), subjects=people)
    reference_file = cfg.root/'reference.nii.gz'
    mask = np.ones((3, 4, 5), bool)
    save_nii(reference_file, mask)
    reference = nib.load(reference_file)
    paths, historical = {}, []
    for i, subject in enumerate(people):
        for task in impl.TASKS:
            folder = feat_dir(c, subject, task, 'L1', 1)
            bold = (c.repos['linux2']/'derivatives/fmriprep-24'/subject/'ses-01/func'/
                    f'{subject}_ses-01_task-{task}_run-1_space-MNI152NLin6Asym_bold.nii.gz')
            save_nii(bold, np.zeros((*mask.shape, 8)))
            folder.mkdir(parents=True)
            spec = c.contrasts[task]
            (folder/'design.fsf').write_text(fsf_text(spec, bold, True)+'set fmri(npts) 8\n')
            contrasts = np.vstack((np.eye(4)[:3], [1, -1, 0, 0]))
            np.savetxt(folder/'design.con', contrasts, header='/Matrix', comments='')
            design = np.tile(np.eye(4), (2, 1))
            np.savetxt(folder/'design.mat', design, header='/Matrix', comments='')
            for k in (1, 2):
                value = (2+i if k == 1 else 1+i*.2) if task == 'socialdoors' else .25
                data = np.full(mask.shape, value, dtype=np.float32)
                save_nii(folder/f'stats/cope{k}.nii.gz', data)
                historical.append(dict(subject=subject, task=task, level='L1', cope=k,
                                       source_mean=float(data.mean()), source_norm=float(np.linalg.norm(data))))
            covered = mask.copy()
            if i == 0 and task == 'doors':
                covered[0, 0, 0] = False
            save_nii(folder/'mask.nii.gz', covered)
            paths[subject, task], _ = impl.design_paths(scope, subject, task)
    prior = c.output('work/source_image_metrics.tsv')
    prior.parent.mkdir(parents=True)
    pd.DataFrame(historical).to_csv(prior, sep='\t', index=False)
    x, common = impl.load_pairs(scope, paths, mask, reference)
    assert not common[0, 0, 0] and common.sum() == mask.size-1
    np.testing.assert_allclose(x[0, :, 0], [1.75, 2.75, 3.75])
    np.testing.assert_allclose(x[1, :, 0], [.75, .95, 1.15], atol=1e-6)
    np.testing.assert_allclose(x[2], x[0]-x[1])
    out = impl.DoorsConfig(**{key: getattr(cfg, key) for key in impl.Config.__dataclass_fields__})
    monkeypatch.setattr(impl, 'N', len(people))
    flips = np.array(list(product([-1., 1.], repeat=len(people))))
    results = [impl.max_t_test(values, flips, tail) for values, (_, _, tail) in zip(x, impl.CONTRASTS)]
    impl.report(out, results, common, reference)
    assert out.output('results/figures/doors_voxelwise.png').stat().st_size > 1000
    name = impl.CONTRASTS[0][0]
    p_map = nib.load(out.output(f'results/maps/{name}_p_family_fwe.nii.gz')).get_fdata()
    np.testing.assert_allclose(p_map[common], np.minimum(1, 3*results[0][2]))
    assert (p_map[~common] == 1).all()
