import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import nibabel as nib
from sklearn.svm import LinearSVC
from utils import PipelineError, sha256
from characterization_audit import (PRIMARY,SENSITIVITY,PARAMETERS,SEED,audit,output_config,frozen_snapshot,TASKS)
from characterization_compute import haufe,fit_binary,permutation_accuracies,bootstrap,clusters,verify_oof
from characterize_v4 import run,settings,intersection_correlation
from revised_samples import RevisedGuard


def toy_scope(n=40):
    subjects=[f'sub-toy{i:03d}' for i in range(n)]
    held=frozenset(f'sub-held{i:03d}' for i in range(50))
    return {'subjects':subjects,'guard':RevisedGuard(frozenset(subjects),held),
            'folds':{s:1+i%5 for i,s in enumerate(subjects)}}


def test_haufe_matches_covariance_definition():
    rng=np.random.default_rng(13); x=rng.normal(size=(17,9)); x-=x.mean(1,keepdims=True)
    w=rng.normal(size=9); s=x@w+3.
    expected=np.cov(x.T,s,ddof=1)[:-1,-1]/np.var(s,ddof=1)
    np.testing.assert_allclose(haufe(x,w,3),expected,atol=1e-7)
    np.testing.assert_allclose(haufe(x,w,-19),expected,atol=1e-7)
    with pytest.raises(PipelineError,match='variance'): haufe(x,np.zeros(9))


def test_snapshot_never_opens_unrelated_private_images(cfg,monkeypatch):
    path=cfg.root/'work/user_files/sub-held000.nii.gz'
    path.parent.mkdir(parents=True); path.write_bytes(b'forbidden private image')
    import characterization_audit as implementation
    real_hash=implementation.sha256
    def guarded_hash(p):
        assert Path(p)!=path, 'Private image opened during fingerprinting'
        return real_hash(p)
    monkeypatch.setattr(implementation,'sha256',guarded_hash)
    assert str(path.relative_to(cfg.root)) not in frozen_snapshot(cfg)


def test_participant_permutation_scrambles_test_labels_and_blocks_tasks(monkeypatch):
    scope=toy_scope(100); rng=np.random.default_rng(3)
    # Very strong class signal; shuffled training and shuffled TEST labels must approach chance.
    x=rng.normal(0,.03,size=(100,3,2,8)); signal=np.array([1,-1,1,-1,1,-1,1,-1.])
    x[:,:,0]+=signal; x[:,:,1]-=signal
    assert np.all(permutation_accuracies(x,scope,'three_paradigm',np.ones(100))==1)
    null=np.array([permutation_accuracies(x,scope,'three_paradigm',rng.choice([-1,1],100)) for _ in range(30)])
    assert .35<null.mean()<.65
    # A held-out participant passed to a fitting operation must fail before sklearn.fit.
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'):
        fit_binary(x[:1],scope,['sub-held000'])
    calls=[]
    import characterization_compute as compute
    original=compute.fit_binary
    def record(data,sc,subjects,flips=None):
        calls.append((data.shape[1],set(subjects),np.array(flips)))
        return original(data,sc,subjects,flips)
    monkeypatch.setattr(compute,'fit_binary',record)
    permutation_accuracies(x,scope,'three_paradigm',rng.choice([-1,1],100),'lopo')
    assert len(calls)==15 and all(n_tasks==2 and len(ids)==80 for n_tasks,ids,_ in calls)


def test_bootstrap_checkpoint_recovers_without_repeating_completed_fits(cfg,monkeypatch):
    scope=toy_scope(10); out=output_config(cfg)
    x=np.random.default_rng(22).normal(size=(10,2,2,12)).astype(np.float32)
    mask=np.ones((3,2,2),bool); ref=nib.Nifti1Image(mask.astype(np.float32),np.eye(4))
    spec={'bootstrap_samples':4,'sign_stability':.975}
    original=bootstrap(out,scope,'partner_pair','valence',x,mask,ref,spec,'fixture')
    import characterization_compute as compute
    monkeypatch.setattr(compute,'fit_binary',lambda *a,**k:pytest.fail('Completed bootstrap refit'))
    again=bootstrap(out,scope,'partner_pair','valence',x,mask,ref,spec,'fixture')
    np.testing.assert_equal(original['weight']['mean'],again['weight']['mean'])
    path=out.output('work/bootstrap/partner_pair_valence/weight.npy')
    corrupted=np.load(path,mmap_mode='r+'); corrupted[0,0]+=1; corrupted.flush()
    with pytest.raises(PipelineError,match='damaged'):
        bootstrap(out,scope,'partner_pair','valence',x,mask,ref,spec,'fixture')


def test_descriptive_clusters_separate_sign_and_use_world_coordinates():
    mask=np.ones((5,4,3),bool); image=nib.Nifti1Image(np.ones(mask.shape),np.diag([2,3,4,1]))
    values=np.zeros(mask.shape); values[:2]=2; values[3:]=-3
    rows=clusters(values[mask],mask,image,20)
    assert len(rows)==2 and {r['sign'] for r in rows}=={-1,1}
    assert all(r['voxel_count']==24 and r['inferential_p']=='not_applicable' for r in rows)
    negative=next(r for r in rows if r['sign']==-1)
    assert negative['x']==6 and negative['peak_absolute_haufe']==3


def test_oof_mismatch_stops_before_scientific_outputs(cfg,monkeypatch):
    scope=toy_scope(10); out=output_config(cfg)
    x=np.random.default_rng(7).normal(size=(10,2,2,6)).astype(np.float32)
    w=np.arange(6)/7.; rows=[]
    for i,subject in enumerate(scope['subjects']):
        for j,task in enumerate(TASKS['partner_pair']):
            rows.append({'subject':subject,'family':'valence','train_tasks':'sharedreward+trust',
                         'test_task':task,'scope':'common','fold':scope['folds'][subject],
                         'margin':float(x[i,j,0].astype(float)@w-x[i,j,1].astype(float)@w)})
    scope['predictions']=pd.DataFrame(rows)
    scope['models']={('valence_common',fold):{} for fold in range(1,6)}
    import characterization_compute as compute
    monkeypatch.setattr(compute,'coefficients',lambda *a:(w,.2))
    spec={'oof_atol':1e-5,'oof_rtol':2e-5}
    verify_oof(out,scope,'partner_pair',{'valence':x},None,None,spec)
    scope['predictions'].loc[0,'margin']+=.01
    with pytest.raises(PipelineError,match='do not reproduce'):
        verify_oof(out,scope,'partner_pair',{'valence':x},None,None,spec)
    assert not out.output('results').exists()


@pytest.fixture
def completed_characterization_fixture(cfg):
    from conftest import metadata,make_subject,fsf_text
    from test_revised import setup_lock,write_qc
    from test_qc_review import qc_rows
    from revised_pilot import run as revised_run
    for task in ('doors','socialdoors'):
        spec=cfg.contrasts[task]; spec['copes'][1]={'name':'win','weights':{1:1.}}; spec['copes'][2]={'name':'loss','weights':{2:1.}}
        (cfg.repos['socdoors']/spec['template']).write_text(fsf_text(spec,'DATA'))
    frame=metadata(62)
    for i,s in enumerate(frame.subject): make_subject(cfg,s,index=i,sr_runs=(2,) if i%4==0 else (1,2))
    frame.rename(columns={'subject':'participant_id'}).to_csv(cfg.bids/'participants.tsv',sep='\t',index=False)
    setup_lock(cfg,60); write_qc(cfg,qc_rows(frame.subject)); revised_run(cfg)
    root=Path(__file__).resolve().parents[1]
    for p in root.glob('code/*.py'):
        (cfg.root/'code').mkdir(exist_ok=True); (cfg.root/'code'/p.name).write_bytes(p.read_bytes())
    spec=settings(cfg)
    spec['frozen_code_sha256']={r:sha256(cfg.root/r) for r in spec['frozen_code_sha256']}
    spec['expected_development_n']={'partner_pair':12,'three_paradigm':12}
    spec.update(bootstrap_samples=3,permutations=3,glass_brain=False)
    return cfg,spec


def test_characterization_audit_and_end_to_end(completed_characterization_fixture,monkeypatch):
    base,spec=completed_characterization_fixture; out=output_config(base)
    snapshot=frozen_snapshot(base)
    original_load=nib.load
    # Dry audit must not load even an aggregate NIfTI.
    monkeypatch.setattr(nib,'load',lambda *a,**k:pytest.fail('Dry audit loaded an image'))
    run(base,dry_run=True,spec=spec)
    scopes=audit(base,spec)
    held=set(scopes[(PRIMARY,'partner_pair')]['guard'].holdout)
    def guarded_load(path,*args,**kwargs):
        assert not any(s in str(path) for s in held),'Holdout image load attempted'
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(nib,'load',guarded_load)
    # Exercise all main/vector charts and a full representative mosaic/histogram set.
    import characterization_figures as figs
    real_brain=figs.brain_figures; seen=[]
    def representative(*args,**kwargs):
        seen.append(args[1])
        if args[1]=='three_paradigm_social_context_common_weight': return real_brain(*args,**kwargs)
    monkeypatch.setattr(figs,'brain_figures',representative)
    run(base,spec=spec)
    assert frozen_snapshot(base)==snapshot
    assert len(seen)==61  # 17 models x 3 descriptions plus 5 common x 2 stability maps.
    status=json.loads(out.output('provenance/run_status.json').read_text())
    assert status['holdout_scored'] is False and status['original_v4_hashes_unchanged']
    assert len(list(out.output('results/maps').glob('*.nii.gz')))==101  # 51 descriptive + 50 bootstrap summaries
    assert out.output('reports/REPORT.md').is_file()
    assert len(list(out.output('results/figures').glob('Figure[1-6]*.png')))>=6
    for p in out.output('reports').glob('*.md'):
        assert not any(s in p.read_text() for s in scopes[(PRIMARY,'partner_pair')]['subjects'])
    # Plot failure preserves computation checkpoint; then plots-only must not refit.
    import characterize_v4 as main
    real_render=main.render
    monkeypatch.setattr(main,'render',lambda *a:(_ for _ in ()).throw(RuntimeError('Plot failure')))
    with pytest.raises(RuntimeError,match='Plot failure'): run(base,plots_only=True,spec=spec)
    assert out.output('work/computation_complete.json').exists()
    monkeypatch.setattr(main,'render',lambda *a:None)
    monkeypatch.setattr(main,'bootstrap',lambda *a:pytest.fail('Plot restart repeated bootstrap'))
    run(base,plots_only=True,spec=spec)
    assert frozen_snapshot(base)==snapshot
    # Mutated membership is detected before any image/model operation.
    c=scopes[(PRIMARY,'partner_pair')]['c']; p=c.output('work/model_membership.tsv'); rows=pd.read_csv(p,sep='\t')
    rows.loc[0,'role']='test' if rows.loc[0,'role']=='train' else 'train'; rows.to_csv(p,sep='\t',index=False)
    with pytest.raises(PipelineError,match='membership'): audit(base,spec)
