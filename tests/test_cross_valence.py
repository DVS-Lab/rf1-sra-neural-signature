import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from utils import PipelineError,sha256
from revised_design import center
from characterization_audit import PRIMARY,audit
from test_characterization import toy_scope,completed_characterization_fixture
from cross_valence_design import representations,domains,plans,permutation_plans,output_config
from cross_valence_compute import (source_maps,fit_fold,score_fold,cv_margins,summarize_cell,permute)
from cross_valence import run,snapshot,settings
from cross_valence_anatomy import VISUAL_LABELS,inventory,prepare,standard_image


def test_exact_factorial_arithmetic_and_label_orientation():
    rng=np.random.default_rng(7); maps={k:rng.normal(size=11).astype('float32')+k for k in range(1,10)}
    for task,pk,nk in [('sharedreward',(2,4,6),(1,3,5)),('trust',(5,7,9),(4,6,8))]:
        r=representations(task,maps)
        for j,keys in enumerate((pk,nk)):
            np.testing.assert_allclose(r['social_context'][j],center([(maps[keys[1]]+maps[keys[2]])/2,maps[keys[0]]]),atol=1e-6)
            np.testing.assert_allclose(r['friend_stranger_context'][j],center([maps[keys[1]],maps[keys[2]]]),atol=1e-6)
        np.testing.assert_allclose(r['friend_stranger_context_collapsed'],center([(maps[pk[1]]+maps[nk[1]])/2,(maps[pk[2]]+maps[nk[2]])/2]),atol=1e-6)
        # Main effect and interaction are distinct; centering is independent for each class map.
        np.testing.assert_allclose(r['social_context'].mean(-1),0,atol=2e-6)
        assert not np.allclose(r['social_context_collapsed'],r['social_context'][0]-r['social_context'][1])
    monetary={k:maps[k]*2+3 for k in maps}; doors=representations('socialdoors',maps,monetary)
    np.testing.assert_allclose(doors['social_context'][0],center([maps[1],monetary[1]]))
    np.testing.assert_allclose(doors['social_context'][1],center([maps[2],monetary[2]]))
    np.testing.assert_allclose(doors['social_context_collapsed'],center([(maps[1]+maps[2])/2,(monetary[1]+monetary[2])/2]))


def test_plans_cover_all_cells_and_twelve_prespecified_permutations():
    pair=plans('partner_pair'); secondary=plans('three_paradigm')
    for family in ('social_context','friend_stranger_context'):
        models=[p for p in pair if p['family']==family and p['scope']=='matrix']
        assert {(p['train'][0],j) for p in models for j in p['test']}=={(i,j) for i in range(4) for j in range(4)}
        positive=next(p for p in pair if p['family']==family and p['name']=='pooled_positive')
        negative=next(p for p in pair if p['family']==family and p['name']=='pooled_negative')
        assert positive['train']==[0,2] and positive['test']==[1,3]
        assert negative['train']==[1,3] and negative['test']==[0,2]
    assert sum(len(p['test']) for p in secondary)==36
    selected=permutation_plans()
    assert len(selected)==11
    assert sum(1 if p['scope']=='pooled_cross_valence' else len(p['test']) for p in selected)==12
    for p in selected:
        if p['scope']=='matrix':
            a,b=domains('partner_pair')[p['train'][0]],p['test_names'][0]
            assert a.rsplit('_',1)[0]!=b.rsplit('_',1)[0] and a.rsplit('_',1)[1]!=b.rsplit('_',1)[1]


def test_fold_grouping_uses_correct_domains_and_rejects_overlap(monkeypatch):
    import cross_valence_compute as compute
    scope=toy_scope(20); x=np.arange(20*4*2*6).reshape(20,4,2,6).astype(float)
    p=next(p for p in plans('partner_pair') if p['name']=='pooled_positive')
    calls=[]
    def fit(values,sc,subjects,flips):
        calls.append((values.copy(),list(subjects))); return np.ones(6),0
    monkeypatch.setattr(compute,'fit_binary',fit)
    w,b,train,test=fit_fold(scope,x,p,1)
    np.testing.assert_equal(calls[0][0],x[train][:,[0,2]])
    assert all(scope['folds'][s]!=1 for s in calls[0][1])
    assert all(scope['folds'][scope['subjects'][i]]==1 for i in test)
    result=score_fold(scope,x,p,w,b,train,test)
    np.testing.assert_equal(result,(x[test][:,[1,3],0]-x[test][:,[1,3],1])@w)
    with pytest.raises(PipelineError,match='leakage'): score_fold(scope,x,p,w,b,train,train[:1])
    scope['subjects'][0]='sub-held000'
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): fit_fold(scope,x,p,1)


def test_holdout_rejected_before_source_or_standard_image_load(monkeypatch):
    scope=toy_scope(10)
    monkeypatch.setattr(nib,'load',lambda *a,**k:pytest.fail('Holdout reached image loading'))
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): source_maps(scope,'sub-held000','trust',None,None,[])
    with pytest.raises(PipelineError,match='participant image'): standard_image(scope,'/tmp/sub-held000/T1w.nii.gz','anything')


def test_combined_accuracy_averages_correctness_within_participants():
    # Each person is correct in one task: accuracy .5 despite a strongly positive mean margin.
    m=np.array([[100,-1],[-1,100],[50,-2],[-2,50.]])
    summary=summarize_cell(m,'fixture',100)
    assert summary['accuracy']==.5 and summary['n']==4 and summary['ci_low']==summary['ci_high']==.5
    assert summary['mean_margin']>0


def test_cv_and_permutation_checkpoint_reuse_and_corruption(cfg,monkeypatch):
    import cross_valence_compute as compute
    scope=toy_scope(20); out=output_config(cfg); rng=np.random.default_rng(71)
    x=rng.normal(0,.03,size=(20,4,2,8)); signal=np.array([1,-1]*4)
    x[:,:,0]+=signal; x[:,:,1]-=signal
    p=permutation_plans()[0]; arrays={'social_context':x}
    result,members=cv_margins(out,scope,'partner_pair',arrays,p,'fixture')
    assert np.all(result>0)
    original=compute.fit_binary
    monkeypatch.setattr(compute,'fit_binary',lambda *a,**k:pytest.fail('Repeated completed fold fit'))
    again,_=cv_margins(out,scope,'partner_pair',arrays,p,'fixture'); np.testing.assert_equal(again,result)
    monkeypatch.setattr(compute,'fit_binary',original)
    null=permute(out,scope,arrays,p,{'permutations':30},'fixture')
    assert .30<null.mean()<.70
    monkeypatch.setattr(compute,'fit_binary',lambda *a,**k:pytest.fail('Repeated completed permutation'))
    np.testing.assert_equal(null,permute(out,scope,arrays,p,{'permutations':30},'fixture'))
    marker=next(out.output('work/fold_models').glob('*.json')); state=json.loads(marker.read_text()); state['train_subjects'].append('sub-held000'); marker.write_text(json.dumps(state))
    with pytest.raises(PipelineError,match='membership'): cv_margins(out,scope,'partner_pair',arrays,p,'fixture')


def test_snapshot_never_reads_private_images_and_namespace_isolated(cfg,monkeypatch):
    import cross_valence as module
    p=cfg.root/'work/sub-held000.nii.gz'; p.parent.mkdir(exist_ok=True); p.write_bytes(b'private')
    orig=module.sha256
    def guarded(path):
        assert Path(path)!=p; return orig(path)
    monkeypatch.setattr(module,'sha256',guarded)
    assert str(p.relative_to(cfg.root)) not in snapshot(cfg)
    out=output_config(cfg)
    for rel in ('../results/map.nii.gz','/tmp/result','results/../../maps'):
        with pytest.raises(PipelineError): out.output(rel)
    assert out.output('results/maps/x').is_relative_to(cfg.root/'results/revised/cross_valence')


@pytest.fixture
def cross_fixture(completed_characterization_fixture,monkeypatch):
    base,original=completed_characterization_fixture
    # Production v4 is executed by the fixture; model existing completed-characterization metadata here.
    p=base.root/'provenance/revised/characterization/run_status.json'; p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(dict(status='complete',holdout_scored=False,original_v4_hashes_unchanged=True,fingerprint='synthetic-characterization')))
    spec=settings(base); spec['frozen_code_sha256']={r:sha256(base.root/r) for r in spec['frozen_code_sha256']}
    spec.update(expected_development_n={'partner_pair':12,'three_paradigm':12},permutations=3,bootstrap_samples=100,
                characterization_fingerprint='synthetic-characterization',frozen_outputs={r:h for r,h in snapshot(base).items() if not r.startswith('work/')})
    fsl=base.root.parent/'installed_fsl'; standard=fsl/'data/standard'; standard.mkdir(parents=True)
    atlas=fsl/'data/atlases'; (atlas/'HarvardOxford').mkdir(parents=True)
    data=np.arange(60,dtype=np.float32).reshape(3,4,5)+1; affine=np.diag([2.,2.,2.,1.])
    nib.save(nib.Nifti1Image(data,affine),standard/'MNI152_T1_2mm_brain.nii.gz')
    labels='<atlas><data>'+''.join(f'<label index="{i}">{name}</label>' for i,name in enumerate(VISUAL_LABELS))+'</data></atlas>'
    (atlas/'HarvardOxford-Cortical.xml').write_text(labels)
    nib.save(nib.Nifti1Image((data%9).astype(np.float32),affine),atlas/'HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
    (fsl/'etc').mkdir(); (fsl/'etc/fslversion').write_text('synthetic-test')
    monkeypatch.setenv('FSLDIR',str(fsl))
    return base,spec


def test_end_to_end_preserves_originals_and_plots_restart_without_fits(cross_fixture,monkeypatch):
    base,spec=cross_fixture; out=output_config(base); before=snapshot(base)
    original_load=nib.load
    monkeypatch.setattr(nib,'load',lambda *a,**k:pytest.fail('Dry audit loaded image'))
    run(base,dry_run=True,spec=spec)
    scopes=audit(base,spec); held=scopes[(PRIMARY,'partner_pair')]['guard'].holdout
    def guarded(path,*args,**kwargs):
        assert not set(Path(path).parts)&held,'Holdout image load attempted'
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guarded)
    run(base,spec=spec,workers=1)
    assert snapshot(base)==before
    perf=pd.read_csv(out.output('results/aggregate/performance.tsv'),sep='\t')
    assert len(perf)==88  # 32 primary matrix +12 pooled +6 relational collapsed +2 reused context +36 Doors.
    assert len(pd.read_csv(out.output('results/aggregate/permutation_summary.tsv'),sep='\t'))==12
    assert len(list(out.output('results/maps').glob('*.nii.gz')))==21
    for f in ['REPORT','FREEZE_CANDIDATES','VISUAL_SENSITIVITY_PROPOSAL','IMPLEMENTATION_AUDIT']:
        text=out.output('reports/'+f+'.md').read_text()
        assert not any(s in text for s in held)
    assert json.loads(out.output('provenance/visual_proposal.json').read_text())['executed'] is False
    proposal=json.loads(out.output('provenance/visual_proposal.json').read_text())
    assert proposal['status']=='prepared_for_review' and proposal['counts']['partner_pair']['proposed_excluded_voxels']>0
    models=json.loads(out.output('provenance/models.json').read_text())['models']
    context=next(m for m in models if m['family']=='social_context' and m['mode']=='collapsed')
    assert context['weight_sha256']==context['source_v4_weight_sha256']
    original_model=scopes[(PRIMARY,'partner_pair')]['models'][('social_context_common',0)]
    assert context['weight_sha256']==original_model['maps'][0]['sha256']
    import cross_valence_report as reports
    import cross_valence as main
    monkeypatch.setattr(reports,'render',lambda *a:(_ for _ in ()).throw(RuntimeError('Plot interrupted')))
    with pytest.raises(RuntimeError,match='Plot interrupted'): run(base,plots_only=True,spec=spec,workers=1)
    assert out.output('work/computation_complete.json').exists()
    monkeypatch.setattr(main,'cv_margins',lambda *a:pytest.fail('Plot restart fit models'))
    monkeypatch.setattr(main,'permute',lambda *a:pytest.fail('Plot restart permuted'))
    monkeypatch.setattr(reports,'render',lambda *a:None)
    run(base,plots_only=True,spec=spec,workers=1)
    assert snapshot(base)==before
    status=json.loads(out.output('provenance/run_status.json').read_text())
    assert status['holdout_scored'] is False and status['original_outputs_unchanged']
    # Drift in completed characterization must block before new image/model access.
    p=base.root/'provenance/revised/characterization/run_status.json'; value=json.loads(p.read_text()); value['fingerprint']='changed'; p.write_text(json.dumps(value))
    monkeypatch.setattr(nib,'load',lambda *a,**k:pytest.fail('Drift reached voxel loading'))
    with pytest.raises(PipelineError,match='frozen output changed'): run(base,spec=spec,workers=1)


def test_parallel_permutations_match_serial_and_resume_indexed_results(cfg):
    from cross_valence_parallel import run_permutations
    scope=toy_scope(20); rng=np.random.default_rng(66); arrays={}
    folder=cfg.output('work/parallel_fixture'); folder.mkdir(parents=True)
    for key,nd in [('social_context',4),('friend_stranger_context',4),('friend_stranger_context_collapsed',2)]:
        x=rng.normal(size=(20,nd,2,8)).astype(np.float32)
        p=folder/(key+'.npy'); np.save(p,x); arrays[key]=np.load(p,mmap_mode='r')
    a=deepcopy(cfg); a.root=cfg.root/'serial'; a.root.mkdir()
    b=deepcopy(cfg); b.root=cfg.root/'parallel'; b.root.mkdir()
    serial=run_permutations(output_config(a),scope,arrays,{'permutations':3},'fixture',1)
    parallel=run_permutations(output_config(b),scope,arrays,{'permutations':3},'fixture',2)
    for key in serial: np.testing.assert_equal(serial[key],parallel[key])
    out=output_config(b); key=next(iter(parallel)); p=out.output('work/permutations/'+key+'.json')
    state=json.loads(p.read_text()); state['results'].pop('1'); p.write_text(json.dumps(state))
    resumed=run_permutations(out,scope,arrays,{'permutations':3},'fixture',2)
    for key in serial: np.testing.assert_equal(serial[key],resumed[key])
    progress=json.loads(out.output('provenance/permutation_progress.json').read_text())
    assert progress['completed']==progress['total']==33
    assert json.loads(out.output('provenance/parallel_execution.json').read_text())['blas_threads_per_worker']==1
