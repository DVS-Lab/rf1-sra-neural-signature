"""Public-only reporting recovery must neither import Nilearn nor refit models."""
import builtins,json,shutil
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest
from utils import sha256,PipelineError
from task_negative_design import output_config
from render_task_negative_results import orthogonal_slices,validated_tables,run

ROOT=Path(__file__).resolve().parents[1]


def test_ras_slices_preserve_orientation_and_sign():
    shape=(5,5,5); affine=np.diag([2.,2.,2.,1.]); affine[:3,3]=[-4,-56,20]
    values=np.arange(125,dtype=float).reshape(shape)-70
    image=nib.Nifti1Image(values,affine); background=nib.Nifti1Image(np.ones(shape),affine)
    planes,limit,_=orthogonal_slices(image,background)
    np.testing.assert_equal(planes[0][0],values[:,2,:].T)
    np.testing.assert_equal(planes[1][0],values[2,:,:].T)
    np.testing.assert_equal(planes[2][0],values[:,:,2].T)
    assert planes[0][4]==1 and planes[0][2]==[-5.,5.,19.,29.] and limit==70
    # Equivalent stored left-handed x axis must give the same RAS display.
    flipped=affine.copy(); flipped[0,0]=-2; flipped[0,3]=4
    other=orthogonal_slices(nib.Nifti1Image(values[::-1],flipped),background)[0]
    for a,b in zip(planes,other): np.testing.assert_equal(a[0],b[0])


def test_public_recovery_no_nilearn_no_fit_and_rejects_changed_inputs(cfg,monkeypatch):
    import sklearn.svm
    from render_task_negative_results import __file__ as renderer
    contract=json.loads((ROOT/'config/task_negative_render.json').read_text())
    for rel in contract['inputs']:
        dest=cfg.root/rel; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/rel,dest)
    # Fixture uses the actual completed public results, but not other projects' output trees.
    p=cfg.root/'config/task_negative_control.json'; spec=json.loads(p.read_text()); spec['frozen']={}; p.write_text(json.dumps(spec))
    contract['inputs'][str(p.relative_to(cfg.root))]=sha256(p)
    image=nib.load(cfg.root/'results/revised/task_negative_control/maps/DEV_display_signed_mean.nii.gz')
    bg=cfg.root/'standard.nii.gz'; nib.save(nib.Nifti1Image(np.ones(image.shape),image.affine),bg)
    contract['background_sha256']=sha256(bg)
    (cfg.root/'config/task_negative_render.json').write_text(json.dumps(contract))
    original_import=builtins.__import__
    def guard(name,*args,**kwargs):
        assert not name.startswith('nilearn'), 'Recovery depends on incompatible plotting package'
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    monkeypatch.setattr(sklearn.svm.LinearSVC,'fit',lambda *a,**k:pytest.fail('Reporting recovery fitted a classifier'))
    run(cfg,bg); out=output_config(cfg)
    status=json.loads(out.output('provenance/render_status.json').read_text())
    assert status['status']=='complete' and not status['models_refit'] and not status['private_images_accessed']
    assert not (cfg.root/'work').exists()
    assert json.loads(out.output('provenance/run_status.json').read_text())['status']=='failed'
    assert out.output('results/figures/task_negative_comparison.png').stat().st_size>10000
    for rel,h in contract['inputs'].items(): assert sha256(cfg.root/rel)==h
    # Pinning and semantic checks both fail closed rather than silently rendering changed statistics.
    path=out.output('results/aggregate/transfers.tsv'); text=path.read_text(); path.write_text(text.replace('0.824438202247191','0.624438202247191'))
    with pytest.raises(PipelineError,match='input changed'): run(cfg,bg)
    with pytest.raises(PipelineError,match='diagnostic mismatch|difference mismatch'): validated_tables(out)
