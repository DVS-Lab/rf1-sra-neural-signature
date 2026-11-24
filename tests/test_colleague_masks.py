import json
from pathlib import Path
from zipfile import ZipFile

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from conftest import save_nii
import export_colleague_copes as pack
from final_stress_sources import phase_dir
from inventory import feat_dir
import review_colleague_masks as review
from revised_design import revised_config
from test_colleague_copes import colleague_data
from utils import PipelineError, sha256


@pytest.fixture
def mask_data(cfg, colleague_data):
    people, protected = colleague_data
    shape = (2, 3, 4)
    brain = np.ones(shape, dtype='uint8')
    reference = cfg.output('results/revised/tsnr_coverage_fd/three_paradigm/maps/analysis_mask.nii.gz')
    # An archived group mask that omits a voxel must still permit new coverage there.
    archived = brain.copy()
    archived[0, 0, 1] = 0
    save_nii(reference, archived)
    folder = cfg.templateflow/'tpl-MNI152NLin6Asym'
    save_nii(folder/review.WHOLE_BRAIN, brain)
    save_nii(folder/'tpl-MNI152NLin6Asym_res-02_T1w.nii.gz', np.arange(24).reshape(shape))
    provenance = cfg.output('provenance/revised/tsnr_coverage_fd/three_paradigm/analysis_mask.json')
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps(dict(sha256=sha256(reference), coverage_threshold=.95,
                                          template_sha256=sha256(folder/review.WHOLE_BRAIN))))
    c = revised_config(cfg)
    for s in people[:3]:
        for task in pack.TASK_SETS['all']:
            mask = brain.copy()
            if task == 'doors':
                mask[0, 0, 0] = 0
                if s == people[1]:
                    mask[-1, -1, -1] = 0
            for r in ((1,) if task in pack.DOORS else (1, 2)):
                run = phase_dir(c, s, 'L1', r) if task == 'sharedreward' else feat_dir(c, s, task, 'L1', r)
                if run.is_dir():
                    save_nii(run/'mask.nii.gz', mask)
            if task in pack.PARTNERS:
                l2 = phase_dir(c, s, 'L2') if task == 'sharedreward' else feat_dir(c, s, task, 'L2')
                for fitted in l2.glob('cope*.feat'):
                    save_nii(fitted/'mask.nii.gz', mask)
    return people, protected


def test_review_creates_both_masks_report_and_private_package(cfg, mask_data, monkeypatch):
    people, protected = mask_data
    region = cfg.repos['linux2']/'qc/reference/source-cerebellum-brainstem_mask.nii.gz'
    roi = np.zeros((2, 3, 4))
    roi[-1] = 1
    save_nii(region, roi)
    monkeypatch.setattr(review, 'REGION_SHA', sha256(region))
    original = nib.load
    def guarded(file, *args, **kwargs):
        assert not set(Path(file).parts).intersection(protected)
        assert 'zstat' not in Path(file).name
        return original(file, *args, **kwargs)
    monkeypatch.setattr(nib, 'load', guarded)
    review.run(cfg, 'all')
    folder = cfg.output('reports/revised/coverage95_masks/all')
    assets, = folder.glob('maps-*')
    summary = pd.read_csv(assets/'mask_summary.tsv', sep='\t').set_index('scope')
    assert len(summary) == 7
    d = summary.loc['doors']
    assert (d.historical_n, d.coverage95_n, d.people_in_both, d.people_gained) == (1, 2, 1, 1)
    assert (d.historical_voxels, d.coverage95_voxels, d.shared_voxels, d.lost_voxels, d.gained_voxels) == (23, 22, 22, 1, 0)
    assert d.dice == pytest.approx(44/45) and d.old_mask_retained_pct == pytest.approx(100*22/23)
    assert d.template_change_pp == pytest.approx(-100/24)
    regional = pd.read_csv(assets/'regional_coverage.tsv', sep='\t').set_index(['scope', 'region'])
    assert regional.loc[('doors', 'Cerebellum and brainstem'), 'change_pp'] == pytest.approx(-100/12)
    assert regional.loc[('doors', 'Remaining brain'), 'change_pp'] == 0
    old, new, change = (nib.load(assets/f'doors_{tag}.nii.gz').get_fdata() for tag in ('historical', 'coverage95', 'change'))
    np.testing.assert_array_equal(change, new-old)
    assert new[0, 0, 1] == 1 and change[-1, -1, -1] == -1
    assert summary.loc['trust', 'coverage95_n'] == 3
    assert summary.loc['all_four', 'coverage95_n'] == 1
    assert summary.loc['socialdoors', 'dice'] == 1
    report = (folder/'REPORT.md').read_text()
    assert f'{assets.name}/all_four.png' in report and (assets/'all_four.png').stat().st_size > 1000
    assert not any(s in report or s in (assets/'mask_summary.tsv').read_text() for s in people + protected)
    with ZipFile(cfg.output('work/colleague_exports/coverage95/all/subject_copes.zip')) as archive:
        assert 'images.csv' in archive.namelist() and not any('mask' in f for f in archive.namelist())


def test_policy_switches_source_and_rejects_protected_mask_alias(cfg, mask_data, monkeypatch):
    people, protected = mask_data
    subject, task = people[0], 'trust'
    scope = pack.export_scope(cfg)
    qc = pack.frozen_run_qc(cfg, scope['subjects'])
    qc.loc[(subject, task, 1), 'brain_coverage_outlier'] = 'True'
    current, _ = pack.select_maps(scope, qc, subject, task)
    paths = tuple(im.path for im in current.values())
    old = review.historical_paths(scope, qc, subject, task, paths)
    single = feat_dir(scope['c'], subject, task, 'L1', 2)
    assert len(old) == 6 and all(p.parent.parent == single for p in old)
    assert all('.gfeat' in str(p) for p in paths)
    mask = single/'mask.nii.gz'
    target = feat_dir(scope['c'], protected[-1], task, 'L1', 2)/'mask.nii.gz'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'sealed')
    mask.unlink()
    mask.symlink_to(target)
    reference, _, _ = review.references(cfg)
    monkeypatch.setattr(nib, 'load', lambda *a, **kw: pytest.fail('opened a protected input'))
    with pytest.raises(PipelineError, match='cross-participant alias'):
        review.task_support(scope, subject, task, old, reference, {})


def test_one_vote_per_person_inclusive_threshold_and_empty_scope():
    brain = np.array([True, True, False])
    assert review.group_mask(np.array([18, 19, 19]), 19, brain).tolist() == [False, True, False]
    assert review.group_mask(np.array([19, 18, 20]), 20, brain).tolist() == [True, False, False]
    assert review.group_mask(np.zeros(3), 0, brain) is None


def test_missing_mask_leaves_verified_cope_package_available(cfg, mask_data):
    subject = mask_data[0][0]
    mask = feat_dir(revised_config(cfg), subject, 'doors', 'L1', 1)/'mask.nii.gz'
    mask.unlink()
    with pytest.raises(PipelineError, match='Missing FEAT mask'):
        review.run(cfg, 'doors')
    assert cfg.output('work/colleague_exports/coverage95/doors/subject_copes.zip').is_file()
    assert not cfg.output('reports/revised/coverage95_masks/doors/REPORT.md').exists()


def test_complete_outcomes_for_joint_masks_and_nearest_neighbor_geometry(cfg, mask_data):
    people, _ = mask_data
    subject = people[0]
    c = revised_config(cfg)
    # This person can share the available win map while the paired Doors mask requires win and loss.
    (feat_dir(c, subject, 'doors', 'L1', 1)/'stats/cope2.nii.gz').unlink()
    scope, qc, sources, rows = pack.export(cfg, 'doors')
    reference, _, _ = review.references(cfg)
    affine = reference.affine.copy()
    affine[0, 3] += 2
    shifted = nib.Nifti1Image(np.zeros(reference.shape), affine)
    scopes, counts, members = review.accumulate(scope, qc, sources, rows, pack.DOORS, shifted)
    assert 'both_doors' in scopes and subject in members['doors', 'coverage95']
    assert not members['both_doors', 'coverage95'] and not counts['both_doors', 'coverage95'].any()
    assert not counts['doors', 'coverage95'][-1].any()  # Outside the source field of view after the shift.
    review.write_review(cfg, 'doors', scopes, counts, members, shifted, *review.references(cfg)[1:])
    folder = cfg.output('reports/revised/coverage95_masks/doors')
    assert 'comparison is unavailable' in (folder/'REPORT.md').read_text()
    assert not list(folder.glob('maps-*/both_doors_change.nii.gz'))
