"""Audit actual decision-versus-fixation designs, then read signed raw COPE3."""
import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from utils import sha256,write_json,write_tsv
from preflight import parse_fsf,verify_l1,verify_design_con
from inventory import feat_dir,inspect_unit,same_grid,header_info
from build_mask import load_development_image
from characterization_audit import require,digest
from characterization_compute import assert_development
from final_stress_sources import guard_path,qc_table,qc_reason


def contrast_spec():
    return dict(model=1,n_copes=4,smoothing='5',suffix='sm-5',repository='socdoors',
        template='templates/L1_task-doors_model-1_type-act.fsf',ev_titles={1:'win',2:'loss',3:'decision',4:'miss'},
        copes={1:dict(name='win',weights={1:1.}),2:dict(name='loss',weights={2:1.}),
               3:dict(name='decision',weights={3:1.}),4:dict(name='win-loss',weights={1:1.,2:-1.})})

def event_baseline(events,seconds):
    """Task source identifies every interior gap as fixation, not inferred rest."""
    e=events.sort_values('onset').reset_index(drop=True)
    require(set(e.trial_type)<= {'win','loss','decision','decision-missed'},'unexpected events: baseline interpretation requires review')
    times=e[['onset','duration']].to_numpy(float)
    require(np.isfinite(times).all() and (times[:,0]>=0).all() and (times[:,1]>0).all(),'invalid event times')
    require(len(e)==80 and e.iloc[::2].trial_type.isin(['decision','decision-missed']).all() and e.iloc[1::2].trial_type.isin(['win','loss']).all(),'expected forty decision/feedback pairs')
    end=times[:,0]+times[:,1]; gaps=times[1:,0]-end[:-1]
    require(np.all(gaps>0) and end[-1]<=seconds+1e-3,'no interpretable interior fixation gaps or events exceed run')
    # Both pre-feedback gaps and intertrial gaps must remain present. Do not
    # impose a new response-rate exclusion or count post-scan instructions as rest.
    interior=float(gaps.sum()); modeled=float(times[:,1].sum())
    return dict(n_trials=40,missed_decisions=int(e.trial_type.eq('decision-missed').sum()),
        decision_seconds=float(e.iloc[::2].duration.sum()),modeled_seconds=modeled,
        interior_fixation_seconds=interior,interior_fixation_fraction=interior/seconds,
        minimum_decision_feedback_gap=float(gaps[::2].min()),minimum_intertrial_gap=float(gaps[1::2].min()),
        maximum_intertrial_gap=float(gaps[1::2].max()),run_seconds=float(seconds))

def actual_design(scope,subject):
    assert_development(scope,[subject]); c=deepcopy(scope['c']); c.contrasts['doors']=contrast_spec()
    folder=guard_path(scope,feat_dir(c,subject,'doors','L1',1),subject)
    fsf=folder/'design.fsf'; values=parse_fsf(fsf)
    verify_l1(values,contrast_spec(),rendered=True); verify_design_con(folder/'design.con',contrast_spec())
    require(int(values['fmri(evs_orig)'])==4 and int(values['fmri(evs_real)'])==4,'unexpected Doors EV basis')
    require(int(values['fmri(ndelete)'])==0,'unreviewed event/BOLD time shift')
    for i in range(1,5):
        require(values[f'fmri(tempfilt_yn{i})']=='0' and values[f'fmri(convolve_phase{i})']=='0','unexpected EV filtering/phase')
        require(values[f'fmri(deriv_yn{i})']=='0' and values[f'fmri(convolve{i})']=='3','unexpected HRF basis')
        require(all(float(v)==0 for k,v in values.items() if k.startswith(f'fmri(ortho{i}.')),'orthogonalized task EV changes signed interpretation')
    events_path=guard_path(scope,c.bids/subject/'ses-01/func'/f'{subject}_ses-01_task-doors_run-1_events.tsv',subject)
    events=pd.read_csv(events_path,sep='\t'); inputs=[fsf,folder/'design.con',folder/'design.mat',events_path]
    for ev,label in enumerate(('win','loss','decision','decision-missed'),1):
        expected=events[events.trial_type==label]
        if expected.empty:
            require(ev==4 and values[f'fmri(shape{ev})']=='10','missing required decision/outcome EV'); continue
        require(values[f'fmri(shape{ev})']=='3','expected three-column timing EV')
        path=guard_path(scope,Path(values[f'fmri(custom{ev})']),subject); inputs.append(path)
        a=np.loadtxt(path,ndmin=2); b=np.column_stack([expected.onset,expected.duration,np.ones(len(expected))])
        a=a[np.argsort(a[:,0])]; b=b[np.argsort(b[:,0])]
        require(a.shape==b.shape and np.allclose(a,b,atol=1e-4,rtol=0),'fitted EV differs from canonical events')
    bold_path=Path(values['feat_files(1)'])
    if not bold_path.exists() and not str(bold_path).endswith('.nii.gz'): bold_path=Path(str(bold_path)+'.nii.gz')
    bold_path=guard_path(scope,bold_path,subject)
    require(all(t in bold_path.name for t in (subject+'_','ses-01_','task-doors_','run-1_','space-MNI152NLin6Asym_')),'decision input space unresolved')
    bold=nib.load(bold_path); tr=float(values['fmri(tr)']); n=int(values['fmri(npts)'])
    require(len(bold.shape)==4 and bold.shape[-1]==n and np.isclose(bold.header.get_zooms()[3],tr,atol=.001),'run duration/TR mismatch')
    design_text=(folder/'design.mat').read_text(); require('/Matrix' in design_text,'missing fitted design matrix')
    x=np.loadtxt(design_text.split('/Matrix',1)[1].splitlines(),ndmin=2)
    require(x.shape[0]==n and x.shape[1]>=4 and np.isfinite(x).all(),'invalid fitted design')
    contrast=np.zeros(x.shape[1]); contrast[2]=1
    projected=x.T@np.linalg.lstsq(x.T,contrast,rcond=None)[0]
    require(np.allclose(projected,contrast,atol=1e-7,rtol=0),'decision contrast not estimable in actual design')
    inspect_unit(c,subject,'doors','L1',1,required_copes_only=True)
    cope=guard_path(scope,folder/'stats/cope3.nii.gz',subject); header_info(cope)
    require(same_grid(nib.load(cope),bold),'signed decision COPE and BOLD geometry differ')
    inputs.extend([cope,folder/'mask.nii.gz'])
    return cope,event_baseline(events,n*tr),{str(p):sha256(p) for p in inputs}

def audit_sources(out,scope,spec):
    assert_development(scope); template=scope['c'].repos['socdoors']/contrast_spec()['template']
    require(sha256(template)==spec['doors_template_sha256'],'authoritative Doors template changed')
    verify_l1(parse_fsf(template),contrast_spec())
    q=qc_table(scope['c']); records=[]; paths={}; fingerprints={}
    for i,s in enumerate(scope['subjects']):
        require(not qc_reason(q,s,'doors',(1,)),'frozen cohort no longer passes Doors QC; do not alter N')
        path,row,hashes=actual_design(scope,s); paths[s]=path; fingerprints.update(hashes)
        records.append(dict(subject=s,**row))
        if (i+1)%20==0 or i+1==len(scope['subjects']): print(f'Decision contrast/baseline audit {i+1}/{len(scope["subjects"])}',flush=True)
    write_tsv(out,'work/baseline_audit.tsv',records)
    frame=pd.DataFrame(records).drop(columns='subject')
    summary=[dict(metric=k,minimum=float(frame[k].min()),median=float(frame[k].median()),maximum=float(frame[k].max())) for k in frame]
    write_tsv(out,'results/aggregate/baseline_audit.tsv',summary)
    write_json(out,'provenance/baseline_audit.json',dict(status='passed',n=len(records),contrast='decision versus implicit fixation baseline',
        code_review=spec['baseline_evidence'],condition='actual fitted EVs match canonical events; fixation gaps present in every retained run',holdout_scored=False))
    return paths,fingerprints

def prior_inputs(base,scope):
    assert_development(scope); root=base.root/'work/revised/specificity'
    original=json.loads((root/'identity.json').read_text()); execution_path=root/'execution_identity.json'
    executed=json.loads(execution_path.read_text()) if execution_path.exists() else original
    for identity in (original,executed):
        require(digest({k:v for k,v in identity.items() if k!='fingerprint'})==identity['fingerprint'],'specificity private identity damaged')
    require({k:v for k,v in original.items() if k not in ('code','fingerprint')}=={k:v for k,v in executed.items() if k not in ('code','fingerprint')},'prior execution changed beyond solver repair')
    status=json.loads((base.root/'provenance/revised/specificity/run_status.json').read_text())
    require(status['status']=='complete' and not status['holdout_scored'] and status['original_outputs_unchanged'],'completed specificity run required')
    require(status['fingerprint']==executed['fingerprint'],'specificity execution identity mismatch')
    require(original['subjects']==scope['subjects'] and original['folds']==scope['folds'],'specificity membership/folds changed')
    require(original['qc']==sha256(base.repos['linux2']/base.paths['qc_table']),'QC changed since completed specificity analysis')
    marker=root/'features/complete.json'; features=json.loads(marker.read_text())
    require(features['fingerprint']==original['fingerprint'],'specificity feature identity mismatch')
    path=root/'features/outcome.npy'; require(sha256(path)==features['hashes']['outcome'],'prior outcome cache changed')
    m=root/'observed/outcome.json'; p=root/'observed/outcome.npy'; meta=json.loads(m.read_text())
    require(meta['fingerprint']==original['fingerprint'] and sha256(p)==meta['sha256'],'prior outcome predictions changed')
    return dict(outcome=path,observed=p),{str(z):sha256(z) for z in (root/'identity.json',marker,path,m,p)}

def raw_features(out,scope,paths,mask,ref,key):
    assert_development(scope); marker=out.output('work/decision_complete.json'); dest=out.output('work/decision_raw.npy')
    if marker.exists():
        state=json.loads(marker.read_text()); require(state['fingerprint']==key and sha256(dest)==state['sha256'],'raw decision checkpoint drift')
    else:
        dest.parent.mkdir(parents=True,exist_ok=True)
        raw=np.lib.format.open_memmap(dest,mode='w+',dtype='float32',shape=(len(scope['subjects']),int(mask.sum()))); dest.chmod(0o600)
        records=[]
        for i,s in enumerate(scope['subjects']):
            assert_development(scope,[s]); path=guard_path(scope,paths[s],s)
            image=load_development_image(scope['c'],path,s,scope['guard'],scope['subjects'])
            resampled=not same_grid(image,ref)
            if resampled: image=resample_from_to(image,(ref.shape,ref.affine),order=1)
            values=image.get_fdata(dtype=np.float32)[mask]
            require(np.isfinite(values).all(),'nonfinite raw signed decision effect')
            raw[i]=values # Crucially NO spatial centering, sign inversion or thresholding.
            records.append(dict(subject=s,path=str(path),sha256=sha256(path),resampled=resampled,
                mean=float(values.mean()),negative_fraction=float((values<0).mean()),minimum=float(values.min()),maximum=float(values.max())))
        raw.flush(); write_tsv(out,'work/decision_source_metrics.tsv',records)
        write_json(out,'work/decision_complete.json',dict(fingerprint=key,sha256=sha256(dest),centered=False,contrast='cope3 decision'))
    result=np.load(dest,mmap_mode='r'); require(result.shape==(len(scope['subjects']),int(mask.sum())),'raw decision shape changed')
    return result
