"""Specificity guards, real atlas geometry, blocked templates and resumable inference."""
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import pytest
from utils import PipelineError,sha256
from test_characterization import toy_scope
from specificity_design import MODES,DOMAINS,ENDPOINTS,CELLS,output_config,cv,generic_template,endpoint_correct,interval
from specificity_inputs import atlas_mask,verify_previous
from specificity import checkpoint,snapshot


def toy_arrays(n=20):
    rng=np.random.default_rng(391)
    x=rng.normal(0,.3,(n,10,2,12)).astype('float32')
    direction=np.array([1,-1]*6)
    x[:,:,0]+=direction*.12; x[:,:,1]-=direction*.12
    x-=x.mean(-1,keepdims=True)
    generic=rng.normal(.2,.4,(n,3,12)).astype('float32')+direction
    return {'legacy':x.copy(),'outcome':x.copy(),'generic':generic}


def test_blocked_fits_and_generic_template_never_uses_test_subjects(monkeypatch):
    import specificity_design as impl
    scope=toy_scope(20); arrays=toy_arrays(); keep=np.arange(12)<6
    train=np.arange(4,20); target=generic_template(arrays['generic'],train,1)
    changed=arrays['generic'].copy(); changed[:4]=1e6
    np.testing.assert_equal(target,generic_template(changed,train,1))
    original=impl.fit_binary; original_generic=impl.fit_generic; calls=[]
    def recording(x,sc,ids,flips=None):
        excluded=set(sc['subjects'])-set(ids); folds={sc['folds'][s] for s in excluded}
        assert len(folds)==1 and len(excluded)==4
        calls.append(x.shape[-1]); return original(x,sc,ids,flips)
    monkeypatch.setattr(impl,'fit_binary',recording)
    def record_generic(x,sc,ids,flips=None):
        excluded=set(sc['subjects'])-set(ids)
        assert len({sc['folds'][s] for s in excluded})==1 and len(excluded)==4
        calls.append(x.shape[-1]); return original_generic(x,sc,ids,flips)
    monkeypatch.setattr(impl,'fit_generic',record_generic)
    for mode in MODES:
        z=cv(scope,arrays,mode,keep); assert z.shape==(20,10,10)
    assert calls.count(1)==50 and calls.count(6)==100 and calls.count(12)==100
    scope['subjects'][-1]='sub-held000'
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): cv(scope,{},'outcome',keep)


def test_fixed_atlas_real_singleton_volume_and_threshold_free_masks(cfg):
    from types import SimpleNamespace
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'config/specificity.json').read_text())
    # Real downloaded 4D singleton atlas onto a standard 4mm MNI grid.
    shape=(46,55,46); affine=np.diag([4,4,4,1]); affine[:3,3]=[-90,-126,-72]
    ref=nib.Nifti1Image(np.ones(shape,dtype='uint8'),affine); mask=np.ones(shape,bool)
    keep,full=atlas_mask(SimpleNamespace(root=root),mask,ref,spec)
    assert keep.sum()>100 and keep.sum()<mask.sum()/3
    assert np.array_equal(keep,full[mask])
    bad={**spec,'atlas_sha256':'bad'}
    with pytest.raises(PipelineError,match='atlas changed'): atlas_mask(SimpleNamespace(root=root),mask,ref,bad)


def test_phase_swap_preserves_trust_doors_controls_and_comparison_membership(cfg):
    scope=toy_scope(); scope['c']=cfg; arrays=toy_arrays(40); keep=np.arange(12)<6
    old=cv(scope,arrays,'legacy',keep)
    arrays['outcome'][:,:2]*=-1
    phase=cv(scope,arrays,'outcome',keep)
    np.testing.assert_equal(old[:,2:,2:],phase[:,2:,2:])
    assert not np.allclose(old[:,:2],phase[:,:2])
    rows=[dict(cohort='three_paradigm',family='social_context',scope='matrix',train_domain=a,test_domain=b,
               subject=s,fold=scope['folds'][s],margin=old[k,i,j])
          for k,s in enumerate(scope['subjects']) for i,a in enumerate(DOMAINS[:6]) for j,b in enumerate(DOMAINS[:6])]
    path=cfg.root/'work/revised/cross_valence/oof_predictions.tsv'; path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path,sep='\t',index=False)
    assert len(verify_previous(scope,old))==36
    rows[0]['subject']='sub-held000'; pd.DataFrame(rows).to_csv(path,sep='\t',index=False)
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): verify_previous(scope,old)


def test_checkpoint_and_snapshot_exclude_only_new_outputs(cfg):
    out=output_config(cfg); original=cfg.root/'results/frozen.txt'; original.parent.mkdir(parents=True); original.write_text('frozen')
    unrelated=cfg.root/'work/private/sub-held000.nii.gz'; unrelated.parent.mkdir(parents=True); unrelated.write_text('never read')
    before=snapshot(cfg); assert list(before)==['results/frozen.txt']
    x=checkpoint(out,'key','outcome',lambda:np.ones((2,2)))
    np.testing.assert_equal(x,checkpoint(out,'key','outcome',lambda:pytest.fail('unneeded refit')))
    with pytest.raises(PipelineError,match='checkpoint drift'): checkpoint(out,'different','outcome',lambda:x)
    with pytest.raises(PipelineError): out.output('../results/frozen.txt')
    assert snapshot(cfg)==before


def test_synchronized_permutations_serial_parallel_resume_and_report(cfg,monkeypatch):
    import specificity_parallel as par
    from specificity_report import report
    scope=toy_scope(20); out=output_config(cfg); arrays=toy_arrays(); keep=np.arange(12)<6
    for k,a in list(arrays.items()):
        p=cfg.root/(k+'.npy'); np.save(p,a); arrays[k]=np.load(p,mmap_mode='r')
    original=par.run(out,scope,arrays,keep,'fixture',requested=1,count=4)
    assert original.shape==(4,5,9)
    np.testing.assert_equal(original[:,0],original[:,1]) # identical inputs + synchronized flips
    assert .25<original.mean()<.75
    par.initialize(scope,{k:str(a.filename) for k,a in arrays.items()},keep)
    expected=par.one('outcome',0)
    marker=out.output('work/permutations/outcome.json'); data=json.loads(marker.read_text()); del data['results']['0']; marker.write_text(json.dumps(data))
    restored=par.run(out,scope,arrays,keep,'fixture',requested=2,count=4)
    np.testing.assert_equal(original,restored); np.testing.assert_equal(expected,restored[0,1])
    monkeypatch.setattr(par,'one',lambda *a:pytest.fail('completed permutation repeated'))
    np.testing.assert_equal(restored,par.run(out,scope,arrays,keep,'fixture',requested=1,count=4))
    with pytest.raises(PipelineError,match='identity drift'): par.run(out,scope,arrays,keep,'bad',requested=1,count=4)
    margins={m:cv(scope,arrays,m,keep) for m in MODES}
    overlaps=[dict(family=f,pattern=k,dmn_voxel_fraction=.25,absolute_mass_fraction=.3,squared_mass_fraction=.4)
              for f in ('social_context','friend_stranger_context') for k in ('weights','haufe','mean_difference')]
    out.output('results/figures').mkdir(parents=True,exist_ok=True)
    frame=report(out,margins,original,overlaps,dict(mask_voxels=12,dmn_voxels=6))
    assert len(frame)==45 and frame.n.eq(20).all()
    assert (frame.p_maxT_two_sided>=frame.p_two_sided).all()
    assert out.output('results/figures/specificity_comparison.png').stat().st_size>10000
    assert 'No UGR/betrayal models executed' in out.output('reports/SPECIFICITY.md').read_text()
    # A stable path outside the checkout facilitates visual inspection of this synthetic report.
    import shutil
    shutil.copyfile(out.output('results/figures/specificity_comparison.png'),'/private/tmp/rf1-specificity-synthetic.png')


def test_endpoint_bootstrap_keeps_four_maps_within_person():
    x=np.ones((20,10,10)); x[0,:]=-1
    correct=endpoint_correct(x); assert correct.shape==(20,9) and np.all(correct.mean(0)==.95)
    lo,hi=interval(correct[:,0],'test',1000); assert lo<=.95<=hi


def test_raw_feature_adapter_preserves_conditions_and_blocks_non_development(cfg,monkeypatch):
    import specificity_inputs as impl
    from cross_valence_design import representations
    from revised_design import center
    scope=toy_scope(10); scope['c']=cfg
    mask=np.ones((3,2,2),bool); ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4))
    monkeypatch.setattr(impl,'geometry',lambda s:(mask,ref))
    rng=np.random.default_rng(7)
    raw={s:{t:{k:rng.normal(size=12).astype('float32')+k for k in (range(1,7) if t in ('sharedreward','phase') else range(4,10) if t=='trust' else (1,2,4))}
            for t in ('sharedreward','phase','trust','socialdoors','doors')} for s in scope['subjects']}
    old=np.stack([np.concatenate([representations('sharedreward',raw[s]['sharedreward'])['social_context'],
        representations('trust',raw[s]['trust'])['social_context'],representations('socialdoors',raw[s]['socialdoors'],raw[s]['doors'])['social_context']]) for s in scope['subjects']])
    folder=cfg.root/'work/revised/cross_valence/features/three_paradigm'; folder.mkdir(parents=True)
    np.save(folder/'social_context.npy',old)
    (folder/'complete.json').write_text(json.dumps({'hashes':{'social_context':sha256(folder/'social_context.npy')}}))
    rows=[]
    def sources(sc,s,t,m,r,diagnostics):
        assert t in ('sharedreward','trust','socialdoors','doors')
        for k,z in raw[s][t].items():
            diagnostics.append(dict(subject=s,task=t,level='L1' if 'doors' in t else 'L2',run=1 if 'doors' in t else 0,cope=k,source_mean=float(z.mean()),source_norm=float(np.linalg.norm(z))))
        return raw[s][t]
    for s in scope['subjects']:
        for t in ('sharedreward','trust','socialdoors','doors'): sources(scope,s,t,mask,ref,rows)
    historical=cfg.output('work/source_image_metrics.tsv'); historical.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(historical,sep='\t',index=False)
    monkeypatch.setattr(impl,'source_maps',sources)
    monkeypatch.setattr(impl,'vector',lambda sc,s,p,m,r,d:raw[s]['phase'][p])
    paths={s:{k:k for k in range(1,7)} for s in scope['subjects']}
    arrays,_,_=impl.features(output_config(cfg),scope,'fixture',paths)
    assert arrays['outcome'].shape==(10,10,2,12)
    np.testing.assert_allclose(arrays['legacy'][:,:6],old)
    np.testing.assert_allclose(arrays['outcome'][:,2:6],old[:,2:])
    s=scope['subjects'][0]
    np.testing.assert_allclose(arrays['generic'][0,0],center(np.stack(list(raw[s]['phase'].values())).mean(0)))
    np.testing.assert_allclose(arrays['outcome'][0,6:8],representations('sharedreward',raw[s]['phase'])['friend_stranger_context'])
    with pytest.raises(PipelineError,match='identity drift'): impl.features(output_config(cfg),scope,'wrong',paths)
    scope['subjects'][0]='sub-held000'
    # Geometry should be guarded by features itself, even if a read adapter is replaced.
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): impl.features(output_config(cfg),scope,'fixture',paths)
