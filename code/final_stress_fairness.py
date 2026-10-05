"""New categorical UGR FEAT model, canonical events only, with mandatory design QC."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util,sys,json,re,os,time,subprocess,shutil
import numpy as np
import pandas as pd
import nibabel as nib
from utils import PipelineError,sha256,write_json,write_tsv,write_text
from characterization_audit import require,digest
from characterization_compute import assert_development
from preflight import parse_fsf
from inventory import feat_dir
from final_stress_sources import guard_path,vector
from revised_design import center
from cross_valence_parallel import available_memory

CONDITIONS=tuple(f'{social}_{end}_{fair}' for social in ('social','nonsocial') for end in ('high','low') for fair in ('unfair','fair'))
CONTRASTS=('social_unfair','social_fair','nonsocial_unfair','nonsocial_fair','social_unfair_minus_fair','nonsocial_unfair_minus_fair','social_minus_nonsocial_fairness')
MIN_TRIALS=3
MAX_IMBALANCE=4.
MAX_CORRELATION=.95
MAX_VIF=100.
MAX_CONDITION=1e8

def nominal_fairness(offer,endowment):
    require(endowment in (16,32),'unexpected endowment')
    # User approved nominal 5/10/25/50% with rounded dollars; never infer from response.
    candidates=[p for p in (.05,.10,.25,.50) if np.isclose(offer,np.floor(endowment*p+.5),atol=1e-8,rtol=0)]
    require(len(candidates)==1,'unexpected or ambiguous dollar offer; stop categorical model')
    return ('unfair' if candidates[0]<.25 else 'fair'),candidates[0]

def upstream_module(base):
    p=base.repos['ugr']/'code/gen_model3_evs.py'
    spec=importlib.util.spec_from_file_location('_stress_ugr_canonical',p); module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module

def build_events(rows,canonical):
    trials=canonical.collapse_trials(rows)  # Enforces consistent per-trial metadata and phase bounds.
    grouped={t.trial_id:[r for r in rows if str(r['trial_id'])==t.trial_id] for t in trials}
    names=(*CONDITIONS,*(f'{s}_{e}_preoffer' for s in ('social','nonsocial') for e in ('high','low')),'rt_constant','rt_pmod','missed_trial','missed_feedback')
    ev={k:[] for k in names}; counts={k:0 for k in CONDITIONS}; detail=[]
    valid=[t for t in trials if not t.missed]; require(valid,'no valid UGR trials')
    mean_rt=np.mean([t.response_time for t in valid])
    for t in trials:
        rr=grouped[t.trial_id]
        if t.missed:
            ev['missed_trial'].append((t.broad_onset,t.broad_duration,1.))
            for r in rr:
                if r['trial_type']=='missed_feedback': ev['missed_feedback'].append((float(r['onset']),float(r['duration']),1.))
            continue
        fair,nominal=nominal_fairness(t.offer,t.endowment); end='high' if t.endowment==32 else 'low'
        decision=next(r for r in rr if r['trial_type']=='decision'); onset=float(decision['onset'])
        endpoint=t.broad_onset+t.broad_duration
        require(endpoint>onset>=t.broad_onset,'invalid offer-to-decision endpoint')
        name=f'{t.sociality}_{end}_{fair}'; ev[name].append((onset,endpoint-onset,1.)); counts[name]+=1
        ev[f'{t.sociality}_{end}_preoffer'].append((t.broad_onset,onset-t.broad_onset,1.))
        ev['rt_constant'].append((t.response_onset,0.,1.)); ev['rt_pmod'].append((t.response_onset,0.,t.response_time-mean_rt))
        detail.append(dict(trial_id=t.trial_id,condition=name,nominal_percentage=100*nominal,offer=t.offer,endowment=t.endowment,onset=onset,endpoint=endpoint))
    failures=[]
    for name,n in counts.items():
        if n<MIN_TRIALS: failures.append(f'{name}: fewer than {MIN_TRIALS} valid trials')
    for s in ('social','nonsocial'):
        for e in ('high','low'):
            a,b=(counts[f'{s}_{e}_{f}'] for f in ('unfair','fair'))
            if min(a,b)==0 or max(a,b)/min(a,b)>MAX_IMBALANCE: failures.append(f'{s}_{e}: fair/unfair imbalance exceeds {MAX_IMBALANCE}:1')
    return ev,counts,detail,failures

def contrast_matrix(n):
    c=np.zeros((7,n))
    c[0,[0,2]]=.5; c[1,[1,3]]=.5; c[2,[4,6]]=.5; c[3,[5,7]]=.5
    c[4]=c[0]-c[1]; c[5]=c[2]-c[3]; c[6]=c[4]-c[5]
    return c

def quote(value):
    value=str(value)
    require('\n' not in value and '\r' not in value,'newline in FSF value')
    return '"'+value.replace('\\','\\\\').replace('"','\\"').replace('$','\\$').replace('[','\\[').replace(']','\\]')+'"'

def fsf_text(original,ev,paths,output):
    # Keep authoritative preprocessing, confounds, TR and nuisance conventions.
    settings={k:v for k,v in original.items() if not re.match(r'fmri\((evtitle|shape\d|convolve\d|convolve_phase|tempfilt_yn\d|deriv_yn|custom\d|ortho\d|conpic_|conname_|con_real\d|con_orig\d|conmask|ftest_|con_mode)',k)}
    n=len(ev); settings.update({'fmri(outputdir)':str(output),'fmri(evs_orig)':n,'fmri(evs_real)':n,'fmri(ncon_orig)':7,'fmri(ncon_real)':7,'fmri(nftests_orig)':0,'fmri(nftests_real)':0,'fmri(overwrite_yn)':0})
    for i,(name,rows) in enumerate(ev.items(),1):
        active=bool(rows) and any(abs(v[2])>1e-12 for v in rows)
        settings.update({f'fmri(evtitle{i})':name,f'fmri(shape{i})':3 if active else 10,f'fmri(convolve{i})':3,
            f'fmri(convolve_phase{i})':0,f'fmri(tempfilt_yn{i})':0,f'fmri(deriv_yn{i})':0,f'fmri(custom{i})':str(paths[name])})
        for j in range(n+1): settings[f'fmri(ortho{i}.{j})']=0
    c=contrast_matrix(n)
    for mode in ('orig','real'):
        settings[f'fmri(con_mode_{mode})']='orig'
        for j,name in enumerate(CONTRASTS,1):
            settings[f'fmri(conname_{mode}.{j})']=name; settings[f'fmri(conpic_{mode}.{j})']=1
            for i,value in enumerate(c[j-1],1): settings[f'fmri(con_{mode}{j}.{i})']=float(value)
    for i in range(1,8):
        for j in range(1,8): settings[f'fmri(conmask{i}_{j})']=0
    return '\n'.join('set '+k+' '+quote(v) for k,v in settings.items())+'\n'

def fsl_matrix(path):
    text=Path(path).read_text(); require('/Matrix' in text,'missing FSL matrix')
    return np.loadtxt(text.split('/Matrix',1)[1].splitlines(),ndmin=2)

def design_qc(x,contrasts):
    require(x.ndim==2 and np.isfinite(x).all(),'invalid rendered design')
    centered=x-x.mean(0); norms=np.linalg.norm(centered,axis=0); active=norms>1e-10
    require(active[:8].all(),'categorical task EV has zero variance')
    z=centered[:,active]/norms[active]; singular=np.linalg.svd(z,compute_uv=False)
    rank=int(np.linalg.matrix_rank(z)); cond=float(singular[0]/singular[-1]) if singular[-1]>0 else float('inf')
    corr=np.corrcoef(centered[:,:8],rowvar=False); maxcorr=float(np.max(np.abs(corr-np.eye(8))))
    inv=np.linalg.pinv(z.T@z); vifs=np.diag(inv)[:8]
    # Relative efficiency (to mutually orthogonal unit-norm EVs), scale-free.
    cs=np.pad(contrasts,((0,0),(0,x.shape[1]-contrasts.shape[1])))[:,active]
    efficiency=[]
    for c in cs:
        denominator=float(c@inv@c); efficiency.append(float(c@c/denominator) if denominator>0 else 0.)
    failures=[]
    if rank<z.shape[1]: failures.append('rank-deficient active design')
    if cond>MAX_CONDITION: failures.append('condition number exceeds 1e8')
    if maxcorr>MAX_CORRELATION: failures.append('categorical EV correlation exceeds .95')
    if max(vifs)>MAX_VIF: failures.append('categorical EV VIF exceeds 100')
    return dict(n_volumes=len(x),active_columns=int(active.sum()),rank=rank,condition_number=cond,max_task_ev_correlation=maxcorr,
                task_ev_vif=vifs.tolist(),relative_contrast_efficiency=efficiency,failures=failures),corr

def run_command(args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'FSLSUB_PARALLEL':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with log.open('a') as stream: subprocess.run([str(x) for x in args],stdout=stream,stderr=subprocess.STDOUT,check=True,env=env)
    log.chmod(0o600)

def render_unit(out,scope,s,run,canonical,key):
    assert_development(scope,[s]); base=scope['c']; folder=out.output(f'work/fairness/{s}/ses-01/run-{run}')
    folder.mkdir(parents=True,exist_ok=True)
    events=guard_path(scope,base.bids/s/'ses-01/func'/f'{s}_ses-01_task-ugr_run-{run}_events.tsv',s)
    rows=canonical.read_events(events); ev,counts,detail,failures=build_events(rows,canonical)
    original_path=guard_path(scope,feat_dir(base,s,'ugr','L1',run)/'design.fsf',s); original=parse_fsf(original_path)
    for k in ('feat_files(1)','confoundev_files(1)'): guard_path(scope,Path(original[k]),s)
    require(original['fmri(smooth)']=='5' and original['fmri(temphp_yn)']=='0' and original['fmri(regstandard_yn)']=='0','UGR preprocessing conventions changed')
    bold=nib.load(original['feat_files(1)']); require(len(bold.shape)==4 and bold.shape[3]==int(original['fmri(npts)']),'BOLD/design length mismatch')
    confounds=np.loadtxt(original['confoundev_files(1)'],ndmin=2)
    require(confounds.shape[0]==bold.shape[3] and np.isfinite(confounds).all(),'invalid canonical confounds')
    paths={name:folder/(name+'.txt') for name in ev}
    for name,values in ev.items():
        paths[name].write_text(''.join(f'{a:.8f}\t{b:.8f}\t{c:.8f}\n' for a,b,c in values)); paths[name].chmod(0o600)
    fsf=folder/'design.fsf'; output=folder/'model-signature-fairness.feat'
    content=fsf_text(original,ev,paths,output)
    identity=dict(fingerprint=key,events_sha256=sha256(events),original_design_sha256=sha256(original_path),confounds_sha256=sha256(original['confoundev_files(1)']),rendered_sha256=digest(content))
    marker=folder/'identity.json'
    if marker.exists(): require(json.loads(marker.read_text())==identity,'fairness inputs changed since rendering')
    fsf.write_text(content); fsf.chmod(0o600); marker.write_text(json.dumps(identity,indent=2)); marker.chmod(0o600)
    if failures: return dict(subject=s,run=run,folder=str(folder),counts=counts,failures=failures,identity=identity)
    # feat_model includes the authoritative confound columns in the inspected matrix.
    run_command(['feat_model',fsf.with_suffix(''),original['confoundev_files(1)']],folder/'render.log')
    x=fsl_matrix(folder/'design.mat'); c=fsl_matrix(folder/'design.con')
    require(np.allclose(c[:,:len(ev)],contrast_matrix(len(ev))) and np.all(c[:,len(ev):]==0),'rendered fairness contrast mismatch')
    metrics,corr=design_qc(x,contrast_matrix(len(ev)))
    pd.DataFrame(detail).to_csv(folder/'trial_counts_detail.tsv',sep='\t',index=False)
    (folder/'trial_counts_detail.tsv').chmod(0o600)
    return dict(subject=s,run=run,folder=str(folder),counts=counts,failures=metrics['failures'],metrics=metrics,identity=identity)

def pilot_figure(out,unit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path=Path(unit['folder'])/'design.mat'
    if not path.exists(): return
    x=fsl_matrix(path); fig,ax=plt.subplots(1,2,figsize=(13,6))
    z=x-x.mean(0); scale=np.max(np.abs(z),axis=0); scale[scale==0]=1
    ax[0].imshow(z/scale,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1); ax[0].set(xlabel='EV / confound column',ylabel='Volume',title='First eligible development run: rendered FEAT design')
    corr=np.corrcoef(x[:,:8],rowvar=False); im=ax[1].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax[1].set(xticks=range(8),yticks=range(8),title='Categorical EV correlation'); fig.colorbar(im,ax=ax[1]); fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(out.output('results/figures/fairness_design_preview.'+ext),dpi=200)
    plt.close(fig)

def wait_complete(folder,level):
    required=([folder/f'stats/cope{k}.nii.gz' for k in range(1,8)]+[folder/f'stats/varcope{k}.nii.gz' for k in range(1,8)]+[folder/'mean_func.nii.gz',folder/'mask.nii.gz'] if level=='L1' else
              [folder/f'cope{k}.feat/stats/cope1.nii.gz' for k in range(1,8)]+[folder/f'cope{k}.feat/stats/varcope1.nii.gz' for k in range(1,8)])
    for _ in range(720):
        if all(p.is_file() and p.stat().st_size for p in required): return required
        time.sleep(10)
    raise PipelineError('Timed out waiting for categorical FEAT outputs; inspect private logs')

def execute_unit(out,scope,unit):
    s=unit['subject']; assert_development(scope,[s]); folder=Path(unit['folder'])
    require(folder.resolve().is_relative_to(out.output('work/fairness')),'fairness output escape')
    output=folder/'model-signature-fairness.feat'; marker=folder/'fit_complete.json'
    if marker.exists():
        for p,h in json.loads(marker.read_text())['products'].items(): require(sha256(p)==h,'categorical L1 changed')
        return
    require(not output.exists(),'incomplete categorical output exists; inspect it, no automatic deletion')
    run_command(['feat',folder/'design.fsf'],folder/'fit.log'); products=wait_complete(output,'L1')
    reg=output/'reg'; reg.mkdir(exist_ok=True)
    ident=Path(os.environ['FSLDIR'])/'etc/flirtsch/ident.mat'
    for name,target in [('example_func2standard.mat',ident),('standard2example_func.mat',ident),('standard.nii.gz',output/'mean_func.nii.gz')]:
        dest=reg/name
        require(not dest.exists() and not dest.is_symlink(),'unexpected categorical registration output')
        dest.symlink_to(target)
    marker.write_text(json.dumps(dict(products={str(p):sha256(p) for p in products}),indent=2)); marker.chmod(0o600)

def execute_l2(out,scope,s):
    assert_development(scope,[s]); folder=out.output(f'work/fairness/{s}/ses-01'); dest=folder/'L2_model-signature-fairness.gfeat'
    marker=folder/'l2_complete.json'
    if marker.exists():
        for p,h in json.loads(marker.read_text())['products'].items(): require(sha256(p)==h,'categorical L2 changed')
        return
    require(not dest.exists(),'incomplete categorical L2 exists; no automatic overwrite')
    template=parse_fsf(scope['c'].repos['ugr']/'templates/L2_task-ugr_model-3_type-act.fsf')
    template['fmri(outputdir)']=str(dest); template['fmri(ncopeinputs)']='7'
    for run in (1,2): template[f'feat_files({run})']=str(folder/f'run-{run}/model-signature-fairness.feat')
    template={k:v for k,v in template.items() if not re.match(r'fmri\(copeinput\.',k)}
    for k in range(1,8): template[f'fmri(copeinput.{k})']='1'
    fsf=folder/'L2.fsf'; fsf.write_text('\n'.join('set '+k+' '+quote(v) for k,v in template.items())+'\n'); fsf.chmod(0o600)
    run_command(['feat',fsf],folder/'l2.log'); products=wait_complete(dest,'L2')
    marker.write_text(json.dumps(dict(products={str(p):sha256(p) for p in products}),indent=2)); marker.chmod(0o600)

def prepare_and_fit(out,scope,ids,key,workers=96,preview=False):
    assert_development(scope,ids)
    require(shutil.which('feat_model') and shutil.which('feat') and os.environ.get('FSLDIR'),'FSL feat/feat_model/FSLDIR required for categorical model')
    canonical=upstream_module(scope['c']); units=[]; stopped=None
    for s in ids:
        for run in (1,2):
            try: unit=render_unit(out,scope,s,run,canonical,key)
            except (ValueError,PipelineError) as exc:
                # Scientific design failure stops the family; never silently drops difficult runs.
                stopped=str(exc); break
            units.append(unit)
            if len(units)==1: pilot_figure(out,unit)
            if unit['failures']: stopped='; '.join(unit['failures']); break
            if preview: break
        if stopped or preview: break
    private=[dict(subject=u['subject'],run=u['run'],**u['counts'],**u.get('metrics',{}),failures='; '.join(u['failures'])) for u in units]
    write_tsv(out,'work/fairness/design_qc.tsv',private)
    status=dict(status='stopped_design_qc' if stopped else 'preview_passed' if preview else 'designs_passed',reason=stopped,
                rendered_runs=len(units),development_n=len(ids),min_trials_per_cell=MIN_TRIALS,max_imbalance=MAX_IMBALANCE,
                max_task_ev_correlation=MAX_CORRELATION,max_vif=MAX_VIF,max_condition_number=MAX_CONDITION,holdout_scored=False)
    if units:
        status['pilot_counts']=units[0]['counts']; status['pilot_design']=units[0].get('metrics')
    write_json(out,'provenance/fairness_status.json',status)
    write_text(out,'reports/FAIRNESS_DESIGN_QC.md','# Categorical UGR design gate\n\n'+json.dumps(status,indent=2)+'\n\nFirst eligible run rendered and its actual convolved FEAT matrix checked before fitting. All runs must pass the same fixed thresholds; no performance-guided redefinition. The primary categorical regressors begin at offer onset and end at canonical choice-feedback end. Separate pre-offer, response/RT and missed-trial regressors retained. Relative efficiencies are design diagnostics, not power estimates.\n')
    print('Categorical UGR design gate: '+status['status'],flush=True)
    if stopped or preview: return status
    memory=available_memory(); cap=max(1,int(.7*memory//(8*1024**3))) if memory else 1
    count=min(workers,cap,os.cpu_count() or 1); status['feat_workers']=count
    print(f'Categorical UGR fits: {len(units)} L1 units; {count} concurrent FEAT jobs, FSLSUB_PARALLEL=1',flush=True)
    with ThreadPoolExecutor(max_workers=count) as pool:
        for future in as_completed([pool.submit(execute_unit,out,scope,u) for u in units]): future.result()
        for future in as_completed([pool.submit(execute_l2,out,scope,s) for s in ids]): future.result()
    status['status']='complete'; write_json(out,'provenance/fairness_status.json',status)
    return status

def categorical_features(out,scope,ids,mask,ref):
    # Only paths in the guarded downstream work tree, never a validation option.
    n=len(scope['subjects']); paths={name:out.output('work/features/'+name+'.npy') for name in ('ugr_norm','ugr_computer_norm')}
    arrays={k:np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(n,2,int(mask.sum()))) for k,p in paths.items()}
    for k,a in arrays.items(): a[:]=np.nan; paths[k].chmod(0o600)
    from nibabel.processing import resample_from_to
    for s in ids:
        assert_development(scope,[s]); folder=out.output(f'work/fairness/{s}/ses-01/L2_model-signature-fairness.gfeat'); m={}
        for k in range(1,5):
            p=folder/f'cope{k}.feat/stats/cope1.nii.gz'; resolved=p.resolve()
            require(resolved.is_relative_to(out.output('work/fairness')) and s in resolved.parts and not any(x.startswith('sub-') and x!=s for x in resolved.parts),'categorical participant/path mismatch')
            img=nib.load(p)
            if img.shape!=ref.shape or not np.allclose(img.affine,ref.affine): img=resample_from_to(img,(ref.shape,ref.affine),order=1)
            m[k]=img.get_fdata(dtype=np.float32)[mask]; require(np.isfinite(m[k]).all(),'categorical nonfinite voxels')
        i=scope['subjects'].index(s); arrays['ugr_norm'][i]=center([m[1],m[2]]); arrays['ugr_computer_norm'][i]=center([m[3],m[4]])
    for a in arrays.values(): a.flush()
    write_json(out,'work/features/fairness_complete.json',dict(subjects=ids,hashes={k:sha256(p) for k,p in paths.items()}))
    return {k:np.load(p,mmap_mode='r') for k,p in paths.items()}
