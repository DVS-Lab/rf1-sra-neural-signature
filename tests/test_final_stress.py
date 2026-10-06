from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import nibabel as nib
import pytest
from utils import PipelineError,sha256
from test_characterization import toy_scope
from final_stress_design import output_config,partners,context,trust_norm,ugr_context,make_plans,permutation_targets,reduced
from final_stress_compute import partition,fit_score,run_plan,summarize,specificity,identifier
from final_stress_fairness import nominal_fairness,build_events,contrast_matrix,fsf_text,design_qc,quote,CONDITIONS,ENDOWMENT_CONDITIONS
from final_stress_sources import guard_path,qc_reason,phase_spec
from revised_design import center


def test_equal_partner_valence_weighting_and_norm_orientation():
    m={i:np.random.default_rng(i).normal(size=12).astype('float32')+i for i in range(1,10)}
    for task,indices in [('sharedreward',((1,2),(3,4),(5,6))),('trust',((4,5),(6,7),(8,9)))]:
        p=partners(task,m); actual=context(p,'social_context')
        c,f,s=[(m[a]+m[b])/2 for a,b in indices]
        np.testing.assert_allclose(actual,center([(f+s)/2,c]),atol=2e-6)
        np.testing.assert_allclose(context(p,'friend_stranger_context'),center([f,s]),atol=2e-6)
        keep=np.arange(12)%3!=0
        np.testing.assert_allclose(reduced(actual,keep),center(np.array([(f+s)/2,c])[:,keep]),atol=2e-6)
    np.testing.assert_allclose(trust_norm(m)['trust_norm'],center([(m[6]+m[8])/2,(m[7]+m[9])/2]))
    np.testing.assert_allclose(ugr_context(m)['ugr_context'],center([(m[5]+m[7])/2,(m[1]+m[3])/2]))


def test_plans_seven_primary_permutations_and_matched_samples():
    sc=toy_scope(20); ids=sc['subjects']; names=[]
    for d in ('sr_full','sr_outcome','trust'):
        for f in ('social_context','friend_stranger_context'): names.append(d+'_'+f)
    names+=['ugr_context','ugr_norm','trust_norm']; av={k:ids for k in names}; av['sr_outcome_social_context']=ids[:15]
    plans=make_plans(av); perm=[p for p in plans if p['permutation']]
    assert len(perm)==7
    for p in plans:
        if p['group']=='phase' and p['family']=='social_context': assert p['subjects']==ids[:15]
    assert all(len(permutation_targets(p))==1 for p in perm)
    assert all(not p['visual'] for p in perm)


def test_participant_blocks_and_heldout_before_access(monkeypatch):
    sc=toy_scope(20); p=dict(subjects=sc['subjects'],train=['a','b'],test=['b'],visual=False)
    arrays={k:np.random.default_rng(i).normal(size=(20,2,8)).astype('float32') for i,k in enumerate(('a','b'))}
    for fold in range(1,6):
        train,test=partition(sc,p,fold); assert not set(train)&set(test)
        w,b,tr,te,m=fit_score(sc,arrays,p,fold,np.ones(8,bool)); assert m.shape==(4,1)
    p['subjects']=[*sc['subjects'][:-1],'sub-held000']
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): fit_score(sc,{},p,1,None)


def test_guard_blocks_cross_person_symlink_and_confounds_allowance(cfg):
    sc=toy_scope(10); sc['c']=cfg; s=sc['subjects'][0]
    p=cfg.repos['ugr']/s/f'{s}_ses-01_task-ugr_confounds.tsv'; p.parent.mkdir(parents=True); p.write_text('0')
    assert guard_path(sc,p,s)==p
    other=cfg.repos['ugr']/'sub-held000'/'ses-01'/'map.nii.gz'; other.parent.mkdir(parents=True); other.write_text('never read')
    link=cfg.repos['ugr']/s/'ses-01'/'map.nii.gz'; link.parent.mkdir(); link.symlink_to(other)
    with pytest.raises(PipelineError): guard_path(sc,link,s)
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): guard_path(sc,other,'sub-held000')


def test_qc_unknown_and_any_bad_run_rejected():
    q=pd.DataFrame([dict(subject='s',task='ugr',run=r,tsnr_outlier='False',brain_coverage_outlier='False',fd_mean_outlier='False') for r in (1,2)]).set_index(['subject','task','run'])
    assert qc_reason(q,'s','ugr',(1,2))==''
    q.loc[('s','ugr',2),'fd_mean_outlier']='True'; assert qc_reason(q,'s','ugr',(1,2))=='qc_outlier'
    q.loc[('s','ugr',2),'fd_mean_outlier']='n/a'; assert qc_reason(q,'s','ugr',(1,2))=='unknown_qc'
    assert qc_reason(q,'unknown','ugr',(1,2))=='unknown_qc'


def synthetic_events():
    trials=[]; rows=[]
    for j,name in enumerate(CONDITIONS):
        social,fair=name.split('_')
        for end,endowment in (('high',32),('low',16)):
            for rep in range(2 if (name=='nonsocial_fair' and end=='high') else 4):
                i=len(trials); onset=i*12.; offer=(2 if endowment==32 else 1) if fair=='unfair' else endowment//2
                trials.append(SimpleNamespace(trial_id=str(i),sociality=social,endowment=endowment,offer=offer,missed=False,broad_onset=onset,broad_duration=7.,response_onset=onset+5.,response_time=1.+rep*.1))
                rows.append(dict(trial_id=str(i),trial_type='decision',onset=onset+3.))
    return rows,SimpleNamespace(collapse_trials=lambda _:trials)


def test_fairness_nominal_rounding_timing_and_trial_gate():
    for end,offers in [(16,[1,2,4,8]),(32,[2,3,8,16])]:
        assert [nominal_fairness(v,end)[0] for v in offers]==['unfair','unfair','fair','fair']
    with pytest.raises(PipelineError): nominal_fairness(7,32)
    rows,canonical=synthetic_events(); ev,counts,detail,failed=build_events(rows,canonical)
    assert not failed and counts['social_unfair']==8 and counts['nonsocial_fair']==6
    assert counts['endowment_high']==14 and counts['endowment_low']==16
    assert ev['social_unfair'][0]==(3.,4.,1.)
    assert ev['social_preoffer'][0]==(0.,3.,1.)
    canonical.collapse_trials=lambda _:[SimpleNamespace(trial_id='0',sociality='social',endowment=32,offer=2,missed=False,broad_onset=0.,broad_duration=7.,response_onset=5.,response_time=1.)]
    assert build_events(rows[:1],canonical)[-1]


def test_fsf_preserves_preprocessing_and_exact_contrasts(tmp_path):
    from preflight import parse_fsf
    rows,canonical=synthetic_events(); ev,*_=build_events(rows,canonical)
    original={'fmri(smooth)':'5','fmri(tr)':'1.615','fmri(temphp_yn)':'0','confoundev_files(1)':'/safe/confounds.tsv','feat_files(1)':'/safe/input.nii.gz',
              'fmri(evtitle1)':'old','fmri(con_real10.1)':'1','fmri(ortho3.2)':'1','fmri(conmask9_4)':'1','fmri(evs_orig)':'11'}
    fsf=tmp_path/'test.fsf'; fsf.write_text(fsf_text(original,ev,{k:tmp_path/(k+'.txt') for k in ev},tmp_path/'new.feat')); settings=parse_fsf(fsf)
    assert 'set fmri(con_real1.1) 1.0' in fsf.read_text()
    assert settings['fmri(tr)']=='1.615' and settings['fmri(smooth)']=='5' and settings['fmri(temphp_yn)']=='0'
    assert settings['confoundev_files(1)']=='/safe/confounds.tsv' and 'fmri(con_real10.1)' not in settings
    assert int(settings['fmri(evs_orig)'])==len(ev) and settings['fmri(shape10)']=='10'
    c=contrast_matrix(len(ev)); np.testing.assert_equal(c[6],c[0]-c[1]-c[2]+c[3])
    assert c.shape[0]==8 and c[7,4]==1 and np.count_nonzero(c[7])==1
    assert np.all(c[:,5:]==0) and np.allclose(c[:4].sum(1),1)
    assert '\\$' in quote('$abc') and '\\[' in quote('[exec nope]')


def test_actual_design_metrics_stop_on_collinearity():
    x=np.random.default_rng(5).normal(size=(150,16)); c=contrast_matrix(16)
    metrics,_=design_qc(x,c); assert not metrics['failures'] and len(metrics['relative_contrast_efficiency'])==8
    x[:,1]=x[:,0]; metrics,_=design_qc(x,c)
    assert 'rank-deficient active design' in metrics['failures'] and metrics['max_task_ev_correlation']>.99
    x[:,0]=0
    with pytest.raises(PipelineError,match='zero variance'): design_qc(x,c)


@pytest.mark.parametrize('case',['passing_preview','matrix_failure','trial_failure'])
def test_fairness_qc_report_real_writer(cfg,monkeypatch,case):
    import final_stress_fairness as fairness
    sc=toy_scope(10); sc['c']=cfg; out=output_config(cfg)
    x=np.random.default_rng(5).normal(size=(150,16))
    if case=='matrix_failure': x[:,1]=x[:,0]
    metrics,_=design_qc(x,contrast_matrix(16))
    unit=dict(subject=sc['subjects'][0],run=1,counts={**{k:4 for k in CONDITIONS},**{k:8 for k in ENDOWMENT_CONDITIONS}},metrics=metrics,failures=metrics['failures'])
    if case=='trial_failure':
        unit.pop('metrics'); unit['failures']=['social_unfair: fewer than 3 valid trials']
    monkeypatch.setenv('FSLDIR','synthetic')
    monkeypatch.setattr(fairness.shutil,'which',lambda _: '/synthetic/FSL')
    monkeypatch.setattr(fairness,'upstream_module',lambda _: None)
    monkeypatch.setattr(fairness,'render_unit',lambda *a:unit)
    monkeypatch.setattr(fairness,'pilot_figure',lambda *a:None)
    monkeypatch.setattr(fairness,'execute_unit',lambda *a:pytest.fail('QC/preview must not fit'))
    monkeypatch.setattr(fairness,'execute_l2',lambda *a:pytest.fail('QC/preview must not fit'))
    result=fairness.prepare_and_fit(out,sc,sc['subjects'],'test',preview=case=='passing_preview')
    expected='preview_passed' if case=='passing_preview' else 'stopped_design_qc'
    assert result['status']==expected and result['rendered_runs']==1 and result['holdout_scored'] is False
    table=pd.read_csv(out.output('work/fairness/design_qc.tsv'),sep='\t',keep_default_na=False)
    assert table.failures.tolist()==['; '.join(unit['failures'])]
    saved=json.loads(out.output('provenance/fairness_status.json').read_text())
    assert saved['status']==expected and expected in out.output('reports/FAIRNESS_DESIGN_QC.md').read_text()
    if case!='trial_failure':
        assert saved['pilot_design']['failures']==metrics['failures']
        assert isinstance(unit['metrics']['failures'],list)


def test_preview_restart_preserves_prior_design_and_identity(cfg):
    from final_stress_tests import prepare_identity
    out=output_config(cfg); prepare_identity(out,'old',{}, {})
    design=out.output('work/fairness/sub-dev000/ses-01/run-1/design.mat')
    design.parent.mkdir(parents=True); design.write_text('original design')
    event=design.parent/'social_unfair.txt'; event.write_text('0\t1\t1\n')
    old_event=design.parent/'social_high_unfair.txt'; old_event.write_text('0\t1\t1\n')
    for name in ('endowment_high','endowment_low','endowment_difference'):
        (design.parent/(name+'.txt')).write_text('0\t1\t1\n')
    prepare_identity(out,'old',{}, {},preview=True)
    assert not out.output('work/archived_previews').exists()
    prepare_identity(out,'new',{}, {},preview=True)
    assert not design.exists()
    archived=list(out.output('work/archived_previews').glob('*/work/fairness/sub-dev000/ses-01/run-1/design.mat'))
    assert len(archived)==1 and archived[0].read_text()=='original design'
    archived_events=list(out.output('work/archived_previews').glob('*/work/fairness/sub-dev000/ses-01/run-1/social_unfair.txt'))
    assert len(archived_events)==1
    archived_old_events=list(out.output('work/archived_previews').glob('*/work/fairness/sub-dev000/ses-01/run-1/social_high_unfair.txt'))
    assert len(archived_old_events)==1
    assert len(list(archived[0].parent.glob('endowment_*.txt')))==3
    assert json.loads(out.output('work/identity.json').read_text())['fingerprint']=='new'
    assert json.loads(next(out.output('work/archived_previews').glob('*/work/identity.json')).read_text())['fingerprint']=='old'


@pytest.mark.parametrize('artifact',[
    'work/features/complete.json','work/models/model.npz','work/permutations/checkpoint.json',
    'work/computation_complete.json','work/fairness/sub-dev000/ses-01/run-1/model-signature-fairness.feat',
    'work/fairness/sub-dev000/ses-01/l2_complete.json','results/aggregate/performance.tsv','results/maps/model.nii.gz'])
def test_preview_restart_rejects_any_fitted_or_unknown_checkpoint(cfg,artifact):
    from final_stress_tests import prepare_identity
    out=output_config(cfg); prepare_identity(out,'old',{}, {})
    path=out.output(artifact); path.parent.mkdir(parents=True,exist_ok=True)
    if artifact.endswith('.feat'): path.mkdir()
    else: path.write_text('retain')
    with pytest.raises(PipelineError,match='checkpoint inputs changed'): prepare_identity(out,'new',{}, {},preview=True)
    assert path.exists() and json.loads(out.output('work/identity.json').read_text())['fingerprint']=='old'
    assert not out.output('work/archived_previews').exists()


def test_preview_restart_requires_explicit_preview(cfg):
    from final_stress_tests import prepare_identity
    out=output_config(cfg); prepare_identity(out,'old',{}, {})
    with pytest.raises(PipelineError,match='checkpoint inputs changed'): prepare_identity(out,'new',{}, {})


def test_phase_directory_diagnostics_only_development_and_no_map_reads(cfg,monkeypatch):
    from final_stress_sources import phase_dir,phase_directory_inventory
    sc=toy_scope(10); sc['c']=cfg; out=output_config(cfg)
    phase_dir(cfg,sc['subjects'][0],'L1',1).mkdir(parents=True)
    alternative=phase_dir(cfg,sc['subjects'][1],'L2').with_name('L2_task-sharedreward_ses-01_model-1_type-act_smTo-5.gfeat')
    alternative.mkdir(parents=True)
    phase_dir(cfg,'sub-held000','L2').mkdir(parents=True)
    monkeypatch.setattr(nib,'load',lambda *a,**kw:pytest.fail('directory diagnostics must not read voxels'))
    phase_directory_inventory(out,sc)
    table=pd.read_csv(out.output('results/aggregate/phase_directory_inventory.tsv'),sep='\t')
    assert len(table)==4 and table.n.sum()==2
    assert table.loc[table.path.str.contains('L2_') & table.expected,'n'].iloc[0]==0
    assert table.loc[~table.expected,'n'].tolist()==[1]
    assert all(s not in table.to_csv() for s in [*sc['subjects'],'sub-held000'])


def test_serial_parallel_identical_and_checkpoint_resume(cfg,monkeypatch):
    import final_stress_parallel as parallel
    sc=toy_scope(20); out=output_config(cfg); rng=np.random.default_rng(2); arrays={}
    for name in ('a','b'):
        path=out.output('work/'+name+'.npy'); path.parent.mkdir(parents=True,exist_ok=True)
        a=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(20,2,8)); a[:]=rng.normal(size=a.shape); a.flush(); arrays[name]=np.load(path,mmap_mode='r')
    p=dict(name='toy',family='social_context',subjects=sc['subjects'],train=['a'],test=['b'],visual=False,permutation=True,group='ugr')
    serial=parallel.run(out,sc,arrays,[p],np.ones(8,bool),'test',1,count=4)
    path=out.output('work/permutations/toy_whole_brain.json'); state=json.loads(path.read_text()); state['results'].pop('2'); path.write_text(json.dumps(state))
    resumed=parallel.run(out,sc,arrays,[p],np.ones(8,bool),'test',2,count=4)
    np.testing.assert_array_equal(serial[0][2],resumed[0][2])
    monkeypatch.setattr(parallel,'one',lambda *a:pytest.fail('completed job repeated'))
    parallel.run(out,sc,arrays,[p],np.ones(8,bool),'test',1,count=4)


def test_cv_models_specificity_and_reporting(cfg,tmp_path,monkeypatch):
    import final_stress_report as report
    from final_stress_design import FAMILIES
    sc=toy_scope(10); sc['c']=cfg; out=output_config(cfg); rng=np.random.default_rng(22); arrays={}; v=12
    for task in ('sr_full','trust'):
        p=center(rng.normal(size=(10,3,v))); arrays[task+'_partners']=p
        for f in FAMILIES: arrays[task+'_'+f]=np.stack([context(x,f) for x in p])
    available={k:sc['subjects'] for k in arrays}; plans=[p for p in make_plans(available) if p['group']=='architecture']
    mask=np.ones((3,2,2),bool); ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4)); keep=np.arange(v)%3!=0
    records=[]
    for p in plans:
        rr,members=run_plan(out,sc,arrays,p,keep,'test',mask,ref); records+=rr
        for fold in range(1,6):
            a={r['subject'] for r in members if r['fold']==fold and r['role']=='train'}; b={r['subject'] for r in members if r['fold']==fold and r['role']=='test'}; assert not a&b
    records+=specificity(out,sc,arrays,plans,keep); perf=summarize(records,bootstrap=100)
    assert len(perf)==16
    out.output('results/aggregate').mkdir(parents=True,exist_ok=True)
    pd.DataFrame([dict(comparison='test',kind='weight',spatial_r=.2,n_voxels=v)]).to_csv(out.output('results/aggregate/spatial_comparison.tsv'),sep='\t',index=False)
    # Low-resolution PNG render is enough for synthetic layout QA; no synthetic scientific outputs committed.
    def quick_save(o,fig,name): fig.savefig(tmp_path/(name+'.png'),dpi=90); report.plt.close(fig)
    monkeypatch.setattr(report,'save',quick_save)
    report.render(out,perf,pd.DataFrame(),dict(status='stopped_design_qc',reason='synthetic gate'),keep,[])
    text=out.output('reports/REPORT.md').read_text(); assert 'holdout_scored = False' in text and 'no persistence success criterion' in text
    email=out.output('reports/COLLABORATOR_EMAIL_DRAFT.md').read_text(); assert 500<=len(email.split())<=700
    assert len(list(tmp_path.glob('Figure*.png')))==6


def test_final_models_and_folds_resume_without_refit(cfg,monkeypatch):
    import final_stress_compute as compute
    sc=toy_scope(10); out=output_config(cfg); mask=np.ones((2,2,2),bool); ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4))
    arrays={'a':np.random.default_rng(42).normal(size=(10,2,8))}
    p=dict(name='resume',family='context',group='phase',subjects=sc['subjects'],train=['a'],test=['a'],visual=False)
    original=run_plan(out,sc,arrays,p,np.ones(8,bool),'test',mask,ref)
    monkeypatch.setattr(compute,'fit_binary',lambda *a,**kw:pytest.fail('completed fold/final model refit'))
    assert run_plan(out,sc,arrays,p,np.ones(8,bool),'test',mask,ref)==original
    path=out.output('work/models/resume_whole_brain_fold-1.json'); data=json.loads(path.read_text()); data['train'].append('sub-held000');path.write_text(json.dumps(data))
    with pytest.raises(PipelineError,match='drift'): run_plan(out,sc,arrays,p,np.ones(8,bool),'test',mask,ref)


def test_new_namespace_and_snapshot_preserve_old_artifacts(cfg):
    from final_stress_tests import snapshot
    out=output_config(cfg); path=cfg.output('results/original.txt');path.parent.mkdir(parents=True);path.write_text('frozen')
    before=snapshot(cfg); new=out.output('results/aggregate/toy.tsv');new.parent.mkdir(parents=True);new.write_text('new')
    assert snapshot(cfg)==before
    for rel in ('../old','/tmp/escape','results/../../old','code/old.py'):
        with pytest.raises(PipelineError): out.output(rel)


def test_orchestration_completed_stopped_fairness_and_plots_resume(cfg,monkeypatch):
    import final_stress_tests as main
    import final_stress_parallel as parallel
    import final_stress_fairness as fairness
    import final_stress_report as report
    from characterization_audit import PRIMARY
    from final_stress_design import FAMILIES
    sc=toy_scope(10); sc['c']=cfg; mask=np.ones((2,2,2),bool); ref=nib.Nifti1Image(mask.astype('float32'),np.eye(4)); keep=np.array([True]*6+[False]*2)
    out=output_config(cfg); arrays={}; rng=np.random.default_rng(100)
    for task in ('sr_full','sr_outcome','trust'):
        x=center(rng.normal(size=(10,3,8))); arrays[task+'_partners']=x
        for family in FAMILIES: arrays[task+'_'+family]=np.stack([context(v,family) for v in x])
    for name in ('ugr_context','ugr_high','ugr_low','trust_norm','trust_computer_norm'): arrays[name]=center(rng.normal(size=(10,2,8)))
    available={k:sc['subjects'] for k in arrays}
    for k,x in list(arrays.items()):
        path=out.output('work/features/'+k+'.npy');path.parent.mkdir(parents=True,exist_ok=True);np.save(path,x);arrays[k]=np.load(path,mmap_mode='r')
    qc=cfg.repos['linux2']/cfg.paths['qc_table'];qc.parent.mkdir(parents=True,exist_ok=True);qc.write_text('fixed')
    monkeypatch.setattr(main,'verify_contract',lambda c:({},{}));monkeypatch.setattr(main,'prior_settings',lambda c:{})
    monkeypatch.setattr(main,'audit',lambda *a:{(PRIMARY,'partner_pair'):sc})
    monkeypatch.setattr(main,'inventory',lambda *a:({'baseline':sc['subjects'],'sr_outcome':sc['subjects'],'ugr':sc['subjects']},{}))
    monkeypatch.setattr(main,'build_features',lambda *a:(arrays,available,mask,ref));monkeypatch.setattr(main,'visual_mask',lambda *a:keep)
    monkeypatch.setattr(main,'fixed_probes',lambda *a:[])
    monkeypatch.setattr(main,'verify_baseline',lambda *a:None)
    def spatial(*args):
        p=out.output('results/aggregate/spatial_comparison.tsv');pd.DataFrame([dict(comparison='toy',kind='weight',spatial_r=.5,n_voxels=8)]).to_csv(p,sep='\t',index=False)
    monkeypatch.setattr(main,'spatial_comparison',spatial)
    monkeypatch.setattr(fairness,'prepare_and_fit',lambda *a,**kw:dict(status='stopped_design_qc',reason='synthetic collinearity'))
    actual=parallel.run
    monkeypatch.setattr(parallel,'run',lambda *a,**kw:actual(*a,**kw,count=2))
    # Rendering is tested separately; keep this orchestration check focused on stage/checkpoint contracts.
    monkeypatch.setattr(report,'render',lambda *a:None)
    main.run(cfg,workers=1)
    state=json.loads(out.output('provenance/run_status.json').read_text()); assert state['holdout_scored'] is False and state['original_outputs_unchanged']
    monkeypatch.setattr(main,'run_plan',lambda *a,**kw:pytest.fail('plots-only refit'))
    monkeypatch.setattr(fairness,'prepare_and_fit',lambda *a,**kw:pytest.fail('plots-only FEAT'))
    main.run(cfg,workers=1,plots_only=True)


@pytest.mark.parametrize('runs',[(1,2),(1,),(2,)])
@pytest.mark.parametrize('empty_neutral',[False,True])
def test_phase_sources_require_authoritative_design_and_canonical_outcome_timing(cfg,runs,empty_neutral):
    from final_stress_sources import phase_dir,phase_unit
    from conftest import fsf_text as fixture_fsf,con_text,l2_text
    sc=toy_scope(10); sc['c']=cfg; s=sc['subjects'][0]; spec=phase_spec()
    labels=['event_computer_punish','event_computer_reward','event_friend_punish','event_friend_reward',
            'event_stranger_punish','event_stranger_reward','event_computer_neutral','event_friend_neutral','event_stranger_neutral',
            'missed_decision','missed_outcome','friend_face','stranger_face','computer_non-face']
    for run in (1,2):
        folder=phase_dir(cfg,s,'L1',run);folder.mkdir(parents=True)
        data=cfg.repos['sharedreward']/s/'ses-01'/f'{s}_ses-01_task-sharedreward_run-{run}_space-MNI152NLin6Asym_bold.nii.gz'
        content=fixture_fsf(spec,data,rendered=True)+'set fmri(evs_orig) 14\n'
        evdir=cfg.repos['sharedreward']/s/'ses-01'/'evs';evdir.mkdir(parents=True,exist_ok=True)
        events=[]
        for i,label in enumerate(labels,1):
            ev=evdir/f'run-{run}_{label}.txt';ev.write_text(f'{10*i} 1 1\n')
            content+=f'set fmri(custom{i}) "{ev}"\nset fmri(shape{i}) 3\n'
            events.append(dict(onset=10*i,duration=1,trial_type=label))
        (folder/'design.fsf').write_text(content);(folder/'design.con').write_text(con_text(spec));(folder/'cluster_mask_zstat1.nii.gz').write_text('fixture')
        (folder/'stats').mkdir(); (folder/'mask.nii.gz').write_text('fixture')
        for k in range(1,7):
            (folder/f'stats/cope{k}.nii.gz').write_text('fixture')
            (folder/f'stats/varcope{k}.nii.gz').write_text('fixture')
        if empty_neutral:
            events=[r for r in events if r['trial_type']!='event_computer_neutral']
            (folder/'design.fsf').write_text(content.replace('set fmri(shape7) 3','set fmri(shape7) 10'))
            (evdir/f'run-{run}_event_computer_neutral.txt').write_text('')
        path=cfg.bids/s/'ses-01/func'/f'{s}_ses-01_task-sharedreward_run-{run}_events.tsv';path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(events).to_csv(path,sep='\t',index=False)
    l2=phase_dir(cfg,s,'L2');l2.mkdir(parents=True)
    text=l2_text(spec)+''.join(f'set feat_files({r}) "{phase_dir(cfg,s,"L1",r)}"\n' for r in (1,2));(l2/'design.fsf').write_text(text)
    for k in range(1,7):
        d=l2/f'cope{k}.feat';(d/'stats').mkdir(parents=True)
        for rel in ('stats/cope1.nii.gz','stats/zstat1.nii.gz','mask.nii.gz','cluster_mask_zstat1.nii.gz'): (d/rel).write_text('fixture')
        (d/'design.mat').write_text('/Matrix\n1\n1\n');(d/'design.con').write_text('/Matrix\n1\n')
    output=l2 if len(runs)==2 else phase_dir(cfg,s,'L1',runs[0])
    sc['phase_index']={s:dict(runs=runs,output=output,strategy='fixed_effects' if len(runs)==2 else 'l1_passthrough')}
    paths=phase_unit(sc,s); assert len(paths)==6
    assert paths[6]==(l2/'cope6.feat/stats/cope1.nii.gz' if len(runs)==2 else output/'stats/cope6.nii.gz')
    ev=cfg.repos['sharedreward']/s/f'ses-01/evs/run-{runs[0]}_event_friend_reward.txt';ev.write_text('3 12 1\n')
    with pytest.raises(PipelineError,match='timings differ'): phase_unit(sc,s)
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'): phase_unit(sc,'sub-held000')


def phase_manifest_fixture(cfg,sc):
    from final_stress_sources import phase_dir
    contract=json.loads((cfg.root/'config/final_stress_sources.json').read_text())
    audit=contract['phase_resolved_audit']; directory=cfg.repos['sharedreward']/audit['directory']; directory.mkdir(parents=True)
    rows=[]
    for s,runs in [(sc['subjects'][0],'1,2'),(sc['subjects'][1],'2'),('sub-held000','1,2')]:
        paired=runs=='1,2'
        output=phase_dir(cfg,s,'L2') if paired else phase_dir(cfg,s,'L1',2)
        rows.append(dict(subject=s[4:],session='01',runs=runs,strategy='fixed_effects' if paired else 'l1_passthrough',output=str(output),source_repo_commit=audit['model_commit']))
    pd.DataFrame(rows).to_csv(directory/'verified-subject-outputs.tsv',sep='\t',index=False)
    (directory/'summary.json').write_text(json.dumps(dict(verified=True,type='activation',source_repo_commit=audit['model_commit'],subjects=3,runs=5,fixed_effects=2,passthrough=1)))
    (directory/'provenance.json').write_text('{}')
    for name in ('summary.json','verified-subject-outputs.tsv','provenance.json'):
        contract['runtime_sources']['sharedreward'][audit['directory']+'/'+name]=sha256(directory/name)
    (cfg.root/'config/final_stress_sources.json').write_text(json.dumps(contract))
    return directory


def test_phase_manifest_pins_strategies_and_filters_before_image_access(cfg,monkeypatch):
    from final_stress_sources import phase_index,phase_unit
    sc=toy_scope(10);sc['c']=cfg; directory=phase_manifest_fixture(cfg,sc)
    monkeypatch.setattr(nib,'load',lambda *a,**kw:pytest.fail('manifest inventory cannot load images'))
    result=phase_index(sc)
    assert set(result)==set(sc['subjects'][:2]) and result[sc['subjects'][1]]['runs']==(2,)
    sc['phase_index']=result
    from utils import InputUnavailable
    with pytest.raises(InputUnavailable,match='absent_from_verified'): phase_unit(sc,sc['subjects'][2])
    path=directory/'verified-subject-outputs.tsv';path.write_text(path.read_text()+'\n')
    with pytest.raises(PipelineError,match='fingerprint changed'): phase_index(sc)


@pytest.mark.parametrize('change',['strategy','output','duplicate'])
def test_phase_manifest_rejects_inconsistent_metadata(cfg,change):
    from final_stress_sources import phase_index
    sc=toy_scope(10);sc['c']=cfg; directory=phase_manifest_fixture(cfg,sc)
    path=directory/'verified-subject-outputs.tsv';frame=pd.read_csv(path,sep='\t',dtype=str)
    if change=='strategy': frame.loc[1,'strategy']='fixed_effects'
    elif change=='output': frame.loc[1,'output']=frame.loc[0,'output']
    else: frame=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    frame.to_csv(path,sep='\t',index=False)
    contract=json.loads((cfg.root/'config/final_stress_sources.json').read_text())
    contract['runtime_sources']['sharedreward'][contract['phase_resolved_audit']['directory']+'/verified-subject-outputs.tsv']=sha256(path)
    (cfg.root/'config/final_stress_sources.json').write_text(json.dumps(contract))
    with pytest.raises(PipelineError,match='phase'): phase_index(sc)


def test_primary_phase_absence_stops_full_run_before_features(cfg,monkeypatch):
    import final_stress_tests as main
    from characterization_audit import PRIMARY
    sc=toy_scope(10);sc['c']=cfg
    monkeypatch.setattr(main,'verify_contract',lambda c:({},{}))
    monkeypatch.setattr(main,'prior_settings',lambda c:{})
    monkeypatch.setattr(main,'audit',lambda *a:{(PRIMARY,'partner_pair'):sc})
    monkeypatch.setattr(main,'inventory',lambda *a:({'baseline':sc['subjects'],'sr_outcome':[],'ugr':sc['subjects']},{}))
    monkeypatch.setattr(main,'build_features',lambda *a:pytest.fail('must stop before images/features'))
    with pytest.raises(PipelineError,match='primary SR outcome'): main.run(cfg,workers=96)


def test_phase_inventory_applies_qc_to_exact_retained_runs(cfg,monkeypatch):
    import final_stress_sources as sources
    sc=toy_scope(10); sc['c']=cfg; directory=phase_manifest_fixture(cfg,sc)
    cfg.aging_subjects={s:{'runs':(2,) if s==sc['subjects'][1] else (1,2)} for s in sc['subjects']}
    rows=[]
    for s in sc['subjects']:
        for task in ('sharedreward','trust','ugr'):
            for run in (1,2):
                if s==sc['subjects'][1] and task=='sharedreward' and run==1: continue
                rows.append(dict(subject=s,task=task,run=run,tsnr_outlier='False',brain_coverage_outlier='False',fd_mean_outlier='False'))
    q=pd.DataFrame(rows).set_index(['subject','task','run'])
    monkeypatch.setattr(sources,'load_aging_index',lambda *a,**kw:None)
    monkeypatch.setattr(sources,'qc_table',lambda _:q)
    monkeypatch.setattr(sources,'phase_unit',lambda scope,s:{1:scope['phase_index'][s]['output']})
    monkeypatch.setattr(sources,'inspect_unit',lambda *a,**kw:None)
    out=output_config(cfg); eligible,_=sources.inventory(out,sc)
    assert eligible['sr_outcome']==sc['subjects'][:2]
    q.loc[(sc['subjects'][1],'sharedreward',2),'fd_mean_outlier']='True'
    eligible,_=sources.inventory(out,sc)
    assert eligible['sr_outcome']==sc['subjects'][:1]
