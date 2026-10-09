"""Narrow tests of the new control, including the legacy prediction contract."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from utils import PipelineError,sha256
from test_characterization import toy_scope
from test_specificity import toy_arrays
from conftest import fsf_text,con_text,save_nii
from cross_valence_compute import fold_indices
from task_negative_design import negative_template,remove_direction,train_template,score_fold,old_margins,output_config,MODES
from task_negative_sources import event_baseline,actual_design,contrast_spec,raw_features,prior_inputs


def test_signed_negative_component_before_centering_and_fold_separation():
    raw=np.array([[-4,2,5,1],[-2,4,1,3.]])
    mean,negative,t=negative_template(raw)
    np.testing.assert_equal(mean,[-3,3,3,2]); np.testing.assert_equal(negative,[-3,0,0,0])
    np.testing.assert_allclose(t,(negative-negative.mean())/np.linalg.norm(negative-negative.mean()))
    with pytest.raises(PipelineError,match='no negative'): negative_template([[1,2,3],[3,2,1]])
    with pytest.raises(PipelineError,match='degenerate'): negative_template([[-3,-3,-3]])
    with pytest.raises(PipelineError,match='invalid raw'): negative_template([[np.nan,2]])
    scope=toy_scope(20); x=np.tile(mean,(20,1)); expected=train_template(scope,x,1)
    _,test=fold_indices(scope,1); x[test]=[-1000,5000,-4000,300]
    for a,b in zip(expected,train_template(scope,x,1)): np.testing.assert_equal(a,b)
    scope['subjects'][0]='sub-held000'
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): train_template(scope,x,1)


def test_projection_arithmetic_and_original_margin_reproduction(monkeypatch):
    import task_negative_design as implementation
    from specificity_design import cv
    scope=toy_scope(20); arrays=toy_arrays(); x=arrays['outcome']; t=negative_template([np.linspace(-2,3,12)])[2]
    z=remove_direction(x,t)
    np.testing.assert_allclose(z@t,0,atol=1e-14); np.testing.assert_allclose(remove_direction(z,t),z,atol=1e-14)
    np.testing.assert_allclose(x-z,(x@t)[...,None]*t,atol=1e-14)
    calls=[]; fit=implementation.fit_binary; generic=implementation.fit_generic
    def recording(func):
        def wrapped(values,sc,ids,flips=None):
            missing=set(sc['subjects'])-set(ids)
            assert len(missing)==4 and len({sc['folds'][s] for s in missing})==1
            calls.append(set(ids)); return func(values,sc,ids,flips)
        return wrapped
    monkeypatch.setattr(implementation,'fit_binary',recording(fit)); monkeypatch.setattr(implementation,'fit_generic',recording(generic))
    old=old_margins(cv(scope,arrays,'outcome',np.arange(12)<6)); now=np.empty_like(old)
    for fold in range(1,6):
        _,test=fold_indices(scope,fold); now[test]=score_fold(scope,x,t,fold,'original')
        for mode in MODES[1:]: assert score_fold(scope,x,t,fold,mode).shape==(4,6,4)
    np.testing.assert_allclose(now,old,atol=1e-12); np.testing.assert_equal(now>0,old>0)
    assert len(calls)==120


def event_fixture():
    return pd.DataFrame([dict(onset=1+8*i+j*3.5,duration=3. if j==0 else 1.,trial_type=('decision-missed' if i==0 else 'decision') if j==0 else ('win' if i%2 else 'loss')) for i in range(40) for j in range(2)])


def test_baseline_requires_interpretable_fixation_and_no_invented_rest():
    events=event_fixture(); row=event_baseline(events,330)
    assert row['n_trials']==40 and row['missed_decisions']==1 and row['interior_fixation_seconds']==156.5
    assert event_baseline(events,400)['interior_fixation_seconds']==row['interior_fixation_seconds']
    bad=events.copy(); bad.loc[1,'onset']=4
    with pytest.raises(PipelineError,match='fixation'): event_baseline(bad,330)
    with pytest.raises(PipelineError,match='exceed run'): event_baseline(events,300)
    with pytest.raises(PipelineError,match='forty'): event_baseline(events.iloc[:-2],330)


def fitted_source(cfg):
    from inventory import feat_dir
    scope=toy_scope(10); scope['c']=cfg; s=scope['subjects'][0]; spec=contrast_spec()
    folder=feat_dir(cfg,s,'doors','L1',1); folder.mkdir(parents=True)
    bold=cfg.fmriprep/s/'ses-01/func'/f'{s}_ses-01_task-doors_run-1_space-MNI152NLin6Asym_desc-preproc_bold.nii.gz'
    save_nii(bold,np.zeros((3,2,2,330))); events=event_fixture()
    ep=cfg.bids/s/'ses-01/func'/f'{s}_ses-01_task-doors_run-1_events.tsv'; ep.parent.mkdir(parents=True); events.to_csv(ep,sep='\t',index=False)
    lines=fsf_text(spec,str(bold))+''.join(f'set fmri({k}) {v}\n' for k,v in {'evs_orig':4,'evs_real':4,'ndelete':0,'tr':1,'npts':330}.items())
    for i,label in enumerate(('win','loss','decision','decision-missed'),1):
        ev=folder/f'ev_{label}.txt'; e=events[events.trial_type==label]; np.savetxt(ev,np.column_stack([e.onset,e.duration,np.ones(len(e))]))
        for key,value in {'shape':3,'deriv_yn':0,'convolve':3,'convolve_phase':0,'tempfilt_yn':0,'custom':f'"{ev}"'}.items(): lines+=f'set fmri({key}{i}) {value}\n'
        for j in range(5): lines+=f'set fmri(ortho{i}.{j}) 0\n'
    (folder/'design.fsf').write_text(lines); (folder/'design.con').write_text(con_text(spec))
    x=np.random.default_rng(54).normal(size=(330,4)); (folder/'design.mat').write_text('/Matrix\n'+'\n'.join(' '.join(map(str,z)) for z in x))
    for k in range(1,5): save_nii(folder/f'stats/cope{k}.nii.gz',np.linspace(-1,3,12).reshape(3,2,2))
    for name in ('mask','cluster_mask_zstat1'): save_nii(folder/(name+'.nii.gz'),np.ones((3,2,2)))
    return scope,s,folder


def test_actual_fitted_contrast_event_and_estimability_audit(cfg):
    scope,s,folder=fitted_source(cfg); p,metrics,hashes=actual_design(scope,s)
    assert p.name=='cope3.nii.gz' and metrics['n_trials']==40 and str(folder/'design.mat') in hashes
    con=folder/'design.con'; old=con.read_text(); con.write_text(old.replace('0.0 0.0 1.0','1.0 0.0 0.0'))
    with pytest.raises(PipelineError,match='weights'): actual_design(scope,s)
    con.write_text(old); ev=folder/'ev_decision.txt'; z=np.loadtxt(ev); z[:,0]+=.1; np.savetxt(ev,z)
    with pytest.raises(PipelineError,match='canonical events'): actual_design(scope,s)
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): actual_design(scope,'sub-held000')


def test_raw_sources_preserve_sign_and_mean_and_guard_before_io(cfg,monkeypatch):
    scope,s,folder=fitted_source(cfg); scope['subjects']=[s]; scope['folds']={s:1}
    out=output_config(cfg); p=folder/'stats/cope3.nii.gz'; ref=nib.load(p); mask=np.ones(ref.shape,bool)
    raw=raw_features(out,scope,{s:p},mask,ref,'fixture')
    np.testing.assert_allclose(raw[0],ref.get_fdata().ravel()); assert raw.mean()==pytest.approx(1)
    import task_negative_sources as impl
    monkeypatch.setattr(impl,'load_development_image',lambda *a:pytest.fail('Cached image read'))
    np.testing.assert_equal(raw,raw_features(out,scope,{s:p},mask,ref,'fixture'))
    with pytest.raises(PipelineError,match='checkpoint drift'): raw_features(out,scope,{s:p},mask,ref,'bad')
    scope['subjects']=['sub-held000']
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): raw_features(out,scope,{},mask,ref,'fixture')


def test_output_isolation_identity_and_private_hash_allowlist(cfg,monkeypatch):
    import task_negative_control as impl
    out=output_config(cfg)
    for bad in ('results/../../specificity/x','/tmp/x','config/x'):
        with pytest.raises(PipelineError): out.output(bad)
    prior=cfg.root/'reports/revised/specificity/SPECIFICITY.md'; prior.parent.mkdir(parents=True); prior.write_text('frozen')
    protected=cfg.root/'work/random/sub-held000.nii.gz'; protected.parent.mkdir(parents=True); protected.write_bytes(b'never read')
    expected=impl.snapshot(cfg)
    assert expected=={'reports/revised/specificity/SPECIFICITY.md':sha256(prior)}
    assert impl.identity_check(out,{'test':1},write=True)==impl.identity_check(out,{'test':1})
    with pytest.raises(PipelineError,match='drift'): impl.identity_check(out,{'test':2})
    own=out.output('reports/REPORT.md'); own.parent.mkdir(parents=True); own.write_text('own')
    assert impl.snapshot(cfg)==expected
    own.unlink(); own.symlink_to(prior)
    with pytest.raises(PipelineError,match='namespace'): out.output('reports/REPORT.md')
    # Parent tree cannot point into another output namespace either.
    escape=out.output('results').parent/'task_negative_control'
    escape.parent.mkdir(parents=True,exist_ok=True); escape.symlink_to(cfg.repos['socdoors'],target_is_directory=True)
    with pytest.raises(PipelineError): out.output('results/x')


def test_observed_checkpoints_and_permutations_reproduce_resume(cfg,monkeypatch):
    from specificity_design import cv,endpoint_correct,ENDPOINTS as OLD_ENDPOINTS
    from characterization_compute import replicate_rng
    from task_negative_compute import observed
    import task_negative_parallel as impl
    scope=toy_scope(20); scope['c']=cfg; arrays=toy_arrays(); x=arrays['outcome']
    p=cfg.root/'features.npy'; np.save(p,x); x=np.load(p,mmap_mode='r')
    directions=np.tile(negative_template([np.linspace(-2,3,12)])[2],(5,1)); out=output_config(cfg)
    previous=cv(scope,arrays,'outcome',np.arange(12)<6)
    original,expressions=observed(out,scope,x,directions,previous,'fixture')
    assert len(expressions)==200 and original['original'].shape==(20,6,4)
    import task_negative_compute as comp
    monkeypatch.setattr(comp,'score_fold',lambda *a:pytest.fail('Completed fold refit'))
    again,_=observed(out,scope,x,directions,previous,'fixture')
    for m in MODES: np.testing.assert_equal(again[m],original[m])
    bad=previous.copy(); bad[:,0,2]*=-1
    with pytest.raises(PipelineError,match='benchmark'): observed(out,scope,x,directions,bad,'fixture')
    rows=[]
    for i in range(4):
        flips=replicate_rng('specificity-permutation',i).choice([-1,1],20)
        perf=endpoint_correct(cv(scope,arrays,'outcome',np.arange(12)<6,flips)).mean(0)
        for a,b,_,_,_ in __import__('task_negative_design').ENDPOINTS[:2]:
            idx=OLD_ENDPOINTS.index((0,1) if a=='sharedreward' else (1,0))
            rows.append(dict(mode='outcome',train=a,test=b,iteration=i,accuracy=perf[idx]))
    path=cfg.root/'results/revised/specificity/aggregate/permutation_nulls.tsv'; path.parent.mkdir(parents=True); pd.DataFrame(rows).to_csv(path,sep='\t',index=False)
    serial=impl.run(out,scope,x,directions,'fixture',requested=1,count=4)
    # Remove only fixture checkpoints and verify real spawned workers produce identical indexed nulls.
    for m in MODES[1:]: out.output('work/permutations/'+m+'.json').unlink()
    parallel=impl.run(out,scope,x,directions,'fixture',requested=2,count=4)
    np.testing.assert_equal(serial,parallel)
    monkeypatch.setattr(impl,'one',lambda *a:pytest.fail('Completed permutation refit'))
    np.testing.assert_equal(parallel,impl.run(out,scope,x,directions,'fixture',requested=1,count=4))


def test_template_maps_report_and_figure_with_synthetic_data(cfg,monkeypatch):
    from task_negative_compute import templates,observed
    from task_negative_report import report
    from specificity_design import cv
    import shutil
    scope=toy_scope(20); scope['c']=cfg; out=output_config(cfg)
    shape=(18,22,18); affine=np.diag([10.,10.,10.,1]); affine[:3,3]=[-90,-126,-72]
    mask=np.ones(shape,bool); v=int(mask.sum()); ref=nib.Nifti1Image(mask.astype('uint8'),affine)
    xyz=np.indices(shape).astype(float); pattern=np.sin(xyz[0]/4)*np.cos(xyz[1]/5)+xyz[2]/20-.5
    raw=np.tile(pattern.ravel(),(20,1)); rng=np.random.default_rng(425)
    x=rng.normal(0,.3,(20,10,2,v)).astype('float32'); signal=pattern.ravel()*.03
    x[:,:,0]+=signal; x[:,:,1]-=signal; x-=x.mean(-1,keepdims=True)
    dmn=(xyz[0].ravel()<8); directions=templates(out,scope,raw,x,dmn,mask,ref,'fixture')
    np.testing.assert_allclose(directions.mean(1),0,atol=1e-12)
    # Saved signed map must preserve above-zero as well as below-zero effects.
    mean=nib.load(out.output('results/maps/DEV_display_signed_mean.nii.gz')).get_fdata()
    np.testing.assert_allclose(mean,pattern,atol=1e-7)
    neg=nib.load(out.output('results/maps/DEV_display_negative_component.nii.gz')).get_fdata()
    np.testing.assert_allclose(neg,np.minimum(pattern,0),atol=1e-7)
    import task_negative_compute as impl
    fitter=impl.fit_binary; monkeypatch.setattr(impl,'fit_binary',lambda *a:pytest.fail('Completed template refit'))
    np.testing.assert_equal(directions,templates(out,scope,raw,x,dmn,mask,ref,'fixture'))
    monkeypatch.setattr(impl,'fit_binary',fitter)
    previous=cv(scope,{'outcome':x},'outcome',dmn)
    margins,expressions=observed(out,scope,x,directions,previous,'fixture')
    # Synthetic anatomy and nulls exercise figure/report paths without new source data.
    bg=out.output('work/synthetic_anatomy.nii.gz')
    xx=(xyz[0]-9)/8; yy=(xyz[1]-11)/10; zz=(xyz[2]-9)/8
    anatomy=np.exp(-2*(xx*xx+yy*yy+zz*zz)); save_nii(bg,anatomy,affine)
    out.output('results/figures').mkdir(parents=True,exist_ok=True)
    frame=report(out,margins,expressions,rng.uniform(.4,.6,(500,3,2)),{'path':str(bg)})
    assert len(frame)==18 and frame.n.eq(20).all() and frame[frame.role=='primary'].permutations.eq(500).all()
    assert frame[frame.role!='primary'].permutations.isna().all()
    report_text=out.output('reports/REPORT.md').read_text()
    assert 'refitting' in report_text and 'Protected N=50 remains unscored' in report_text
    image=out.output('results/figures/task_negative_comparison.png'); assert image.stat().st_size>10000
    shutil.copyfile(image,'/private/tmp/rf1-task-negative-SYNTHETIC.png')
    # Public products cannot contain fixture IDs, source paths or private membership.
    for tree in ('results/aggregate','reports','provenance'):
        for p in out.output(tree).rglob('*'):
            if p.is_file() and p.suffix in ('.tsv','.json','.md'):
                assert not any(s in p.read_text() for s in scope['subjects'])


def test_prior_inputs_require_completed_exact_sample_qc_and_cache(cfg):
    from characterization_audit import digest
    scope=toy_scope(20); scope['c']=cfg
    qc=cfg.repos['linux2']/cfg.paths['qc_table']; qc.parent.mkdir(parents=True); qc.write_text('synthetic frozen QC\n')
    root=cfg.root/'work/revised/specificity'; (root/'features').mkdir(parents=True); (root/'observed').mkdir()
    identity=dict(subjects=scope['subjects'],folds=scope['folds'],qc=sha256(qc)); key=digest(identity)
    (root/'identity.json').write_text(json.dumps(dict(identity,fingerprint=key)))
    np.save(root/'features/outcome.npy',toy_arrays()['outcome'])
    (root/'features/complete.json').write_text(json.dumps(dict(fingerprint=key,hashes={'outcome':sha256(root/'features/outcome.npy')})))
    np.save(root/'observed/outcome.npy',np.zeros((20,10,10)))
    (root/'observed/outcome.json').write_text(json.dumps(dict(fingerprint=key,sha256=sha256(root/'observed/outcome.npy'))))
    status=cfg.root/'provenance/revised/specificity/run_status.json'; status.parent.mkdir(parents=True)
    status.write_text(json.dumps(dict(status='complete',holdout_scored=False,original_outputs_unchanged=True,fingerprint=key)))
    paths,hashes=prior_inputs(cfg,scope); assert paths['outcome']==root/'features/outcome.npy'
    qc.write_text('changed QC\n')
    with pytest.raises(PipelineError,match='QC changed'): prior_inputs(cfg,scope)
    qc.write_text('synthetic frozen QC\n'); np.save(paths['outcome'],np.ones((20,10,2,12)))
    with pytest.raises(PipelineError,match='cache changed'): prior_inputs(cfg,scope)


def test_driver_dry_run_has_no_voxel_access_and_rejects_cohort_drift(cfg,monkeypatch,capsys):
    import task_negative_control as impl
    from characterization_audit import PRIMARY
    scope=toy_scope(178); scope['c']=cfg; scope['model']={'mask_sha256':'synthetic-mask'}
    spec_path=cfg.root/'config/task_negative_control.json'; spec=json.loads(spec_path.read_text()); spec['frozen']={}; spec_path.write_text(json.dumps(spec))
    monkeypatch.setattr(impl,'settings',lambda b:{})
    monkeypatch.setattr(impl,'audit',lambda b,s:{(PRIMARY,'three_paradigm'):scope})
    monkeypatch.setattr(impl,'prior_inputs',lambda b,s:({},{}))
    monkeypatch.setattr(impl,'audit_sources',lambda o,s,p:({},{}))
    monkeypatch.setattr(impl,'background',lambda b:{'path':'standard','sha256':'standard'})
    qc=cfg.repos['linux2']/cfg.paths['qc_table']; qc.parent.mkdir(parents=True); qc.write_text('synthetic QC')
    monkeypatch.setattr(impl,'geometry',lambda *a:pytest.fail('Dry run loaded voxel arrays'))
    impl.run(cfg,dry_run=True)
    assert 'DRY RUN PASSED' in capsys.readouterr().out
    assert not output_config(cfg).output('work/identity.json').exists()
    scope['subjects'].pop()
    with pytest.raises(PipelineError,match='exact N=178'): impl.run(cfg,dry_run=True)


def test_template_interruption_resumes_completed_folds(cfg,monkeypatch):
    import task_negative_compute as impl
    scope=toy_scope(20); scope['c']=cfg; out=output_config(cfg); x=toy_arrays()['outcome']
    raw=np.tile(np.linspace(-1,3,12),(20,1)); mask=np.ones((3,2,2),bool)
    ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4)); keep=np.arange(12)<6
    real=impl.fit_binary; calls=[]
    def fail_second(*args):
        calls.append(1)
        if len(calls)==2: raise RuntimeError('synthetic interruption')
        return real(*args)
    monkeypatch.setattr(impl,'fit_binary',fail_second)
    with pytest.raises(RuntimeError,match='interruption'): impl.templates(out,scope,raw,x,keep,mask,ref,'fixture')
    assert out.output('work/templates/fold-1.json').exists()
    resumed=[]
    def record(*args): resumed.append(1); return real(*args)
    monkeypatch.setattr(impl,'fit_binary',record)
    impl.templates(out,scope,raw,x,keep,mask,ref,'fixture')
    assert len(resumed)==5  # four remaining folds and the display-only full-development fit


def test_full_orchestration_uses_frozen_178_and_only_own_outputs(cfg,monkeypatch):
    import task_negative_control as impl
    import task_negative_parallel as parallel
    import task_negative_report as reporting
    from characterization_audit import PRIMARY
    from specificity_design import cv
    scope=toy_scope(178); scope['c']=cfg; scope['model']={'mask_sha256':'synthetic-mask'}
    x=toy_arrays(178)['outcome']; old=cv(scope,{'outcome':x},'outcome',np.arange(12)<6)
    folder=cfg.root/'work/fixture'; folder.mkdir(parents=True)
    np.save(folder/'outcome.npy',x); np.save(folder/'observed.npy',old)
    spec_path=cfg.root/'config/task_negative_control.json'; spec=json.loads(spec_path.read_text()); spec['frozen']={}; spec_path.write_text(json.dumps(spec))
    qc=cfg.repos['linux2']/cfg.paths['qc_table']; qc.parent.mkdir(parents=True); qc.write_text('synthetic QC')
    monkeypatch.setattr(impl,'settings',lambda b:{})
    monkeypatch.setattr(impl,'audit',lambda b,s:{(PRIMARY,'three_paradigm'):scope})
    monkeypatch.setattr(impl,'prior_inputs',lambda b,s:({'outcome':folder/'outcome.npy','observed':folder/'observed.npy'},{}))
    monkeypatch.setattr(impl,'audit_sources',lambda o,s,p:({},{}))
    monkeypatch.setattr(impl,'background',lambda b:{'path':'synthetic background','sha256':'synthetic'})
    mask=np.ones((3,2,2),bool); ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4))
    monkeypatch.setattr(impl,'geometry',lambda s:(mask,ref))
    monkeypatch.setattr(impl,'raw_features',lambda *a:np.tile(np.linspace(-2,3,12),(178,1)))
    import specificity_inputs
    monkeypatch.setattr(specificity_inputs,'atlas_mask',lambda *a:(np.arange(12)<6,None))
    def nulls(out,sc,features,directions,key,requested,count):
        assert requested==3 and count==500 and features.shape==(178,10,2,12)
        return np.random.default_rng(46).uniform(.45,.55,(500,3,2))
    monkeypatch.setattr(parallel,'run',nulls)
    monkeypatch.setattr(reporting,'figure',lambda *a:None) # the real renderer is separately exercised above
    before=impl.snapshot(cfg); impl.run(cfg,workers=3)
    out=output_config(cfg); state=json.loads(out.output('provenance/run_status.json').read_text())
    assert state['status']=='complete' and state['n']==178 and state['holdout_scored'] is False
    assert state['original_outputs_unchanged'] and impl.snapshot(cfg)==before
    assert pd.read_csv(out.output('results/aggregate/transfers.tsv'),sep='\t').n.eq(178).all()
    # Re-running the runner itself must not refit any completed model.
    import task_negative_compute
    monkeypatch.setattr(task_negative_compute,'fit_binary',lambda *a:pytest.fail('Completed template refit'))
    monkeypatch.setattr(task_negative_compute,'score_fold',lambda *a:pytest.fail('Completed prediction refit'))
    impl.run(cfg,workers=3)


def test_ev_comparison_matches_upstream_awk_without_tolerating_other_drift():
    import subprocess
    from task_negative_sources import compare_ev_timing
    # Same serialization expression as pinned BIDSto3col.sh, including integer
    # conversion. Exercise actual awk, not a fixture produced by our comparator.
    canonical=np.array([[123.456789,3.012345678,1.],[312.999789,1.23456789,1.],[1000001.,3.,1.]])
    text='\n'.join(f'{a:.12f}\t{b:.12f}' for a,b,_ in canonical)+'\n'
    formatted=subprocess.check_output(['awk','-F','\t','{printf("%s\\t%s\\t1.0\\n",$1-(0),$2)}'],input=text,text=True)
    actual=np.loadtxt(formatted.splitlines(),ndmin=2)
    assert not np.allclose(actual,canonical,atol=1e-4,rtol=0) # reproduces the old audit failure
    result=compare_ev_timing(actual,canonical)
    assert result['matched'] and result['serialization']=='BIDSto3col_awk_6_significant_digits'
    assert result['max_onset_error_seconds']>1e-4 and result['max_duration_error_seconds']==0
    assert compare_ev_timing(canonical,canonical)['serialization']=='canonical_precision'
    assert compare_ev_timing(actual[::-1],canonical)['matched']
    for col,delta in ((0,1e-5),(0,.1),(1,1e-5),(2,1e-5)):
        wrong=actual.copy(); wrong[0,col]+=delta
        assert not compare_ev_timing(wrong,canonical)['matched']
    assert not compare_ev_timing(actual[:-1],canonical)['matched']
    bad=actual.copy(); bad[0,0]=np.nan
    assert not compare_ev_timing(bad,canonical)['matched']


def test_actual_design_accepts_converter_format_and_writes_sanitized_failure(cfg,monkeypatch):
    import task_negative_sources as impl
    scope,s,folder=fitted_source(cfg); scope['subjects']=[s]
    events_path=cfg.bids/s/'ses-01/func'/f'{s}_ses-01_task-doors_run-1_events.tsv'
    events=pd.read_csv(events_path,sep='\t'); events.onset+=.000789; events.to_csv(events_path,sep='\t',index=False)
    for label in ('win','loss','decision','decision-missed'):
        g=events[events.trial_type==label]
        (folder/f'ev_{label}.txt').write_text(''.join(f'{a:.6g}\t{b}\t1.0\n' for a,b in zip(g.onset,g.duration)))
    checks=[]; impl.actual_design(scope,s,checks)
    assert len(checks)==4 and all(c['matched'] for c in checks)
    assert max(c['max_onset_error_seconds'] for c in checks)>1e-4
    converter=cfg.repos['socdoors']/'code/BIDSto3col.sh'; converter.write_text('synthetic converter identity')
    template=cfg.repos['socdoors']/contrast_spec()['template']; template.write_text(fsf_text(contrast_spec(),'DATA'))
    spec=dict(doors_template_sha256=sha256(template),doors_ev_converter_sha256=sha256(converter),baseline_evidence={})
    monkeypatch.setattr(impl,'qc_table',lambda c:None); monkeypatch.setattr(impl,'qc_reason',lambda *a:'')
    out=output_config(cfg); impl.audit_sources(out,scope,spec)
    public=out.output('provenance/ev_timing_audit.json')
    assert json.loads(public.read_text())['complete_cohort_audit']
    path=folder/'ev_win.txt'; data=np.loadtxt(path); data[0,1]+=.01; np.savetxt(path,data)
    with pytest.raises(PipelineError,match='duration error='):
        impl.audit_sources(out,scope,spec)
    summary=json.loads(public.read_text())
    assert summary['failed_ev_files']==1 and not summary['complete_cohort_audit']
    assert s not in public.read_text() and str(cfg.root) not in public.read_text()
    private=out.output('work/ev_timing_audit.tsv').read_text()
    assert s in private and str(path) in private
