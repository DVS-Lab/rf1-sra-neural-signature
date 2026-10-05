"""Development-only source inventory. New phase maps never alias aging full-trial maps."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from utils import PipelineError, InputUnavailable, sha256, write_json, write_tsv
from characterization_audit import require
from characterization_compute import assert_development,geometry
from cross_valence_compute import source_maps
from cross_valence_anatomy import standard_image,label_names
from final_stress_design import FAMILIES,VISUAL_IDS,partners,context,trust_norm,ugr_context
from build_mask import load_development_image,save_image
from inventory import inspect_unit,feat_dir,cope_path,same_grid
from preflight import parse_fsf,verify_l1,verify_l2,verify_design_con
from review_qc_attrition import flag,POLICIES
from aging_source import load_aging_index


def phase_spec():
    names=['C_pun','C_rew','F_pun','F_rew','S_pun','S_rew','C_neu','F_neu','S_neu','missed_decision','missed_outcome','F_dec','S_dec','C_dec']
    return dict(model=1,n_copes=34,smoothing='0',ev_titles=dict(enumerate(names,1)),
                copes={i:dict(name=n,weights={i:1.}) for i,n in enumerate(names[:6],1)})

def phase_dir(c,s,level,run=None):
    stem=f'{level}_task-sharedreward_ses-01_model-1_type-act'
    if level=='L1': stem+=f'_run-{run}'
    return c.repos['sharedreward']/'derivatives/fsl'/s/'ses-01'/(stem+'_smTo-6'+('.feat' if level=='L1' else '.gfeat'))

def guard_path(scope,path,s):
    assert_development(scope,[s]); path=Path(path); resolved=path.resolve()
    require(s in path.parts and s in resolved.parts and ('ses-01' in resolved.parts or '_ses-01_' in resolved.name),'source participant/session mismatch')
    require(not any(x.startswith('sub-') and not (x==s or x.startswith(s+'_')) for x in resolved.parts),'cross-person source symlink')
    require(any(resolved.is_relative_to(p.resolve()) for p in scope['c'].repos.values()),'source outside configured repositories')
    return path

def phase_unit(scope,s):
    """Only authoritative two-run fixed effects. No undocumented single-run substitution."""
    assert_development(scope,[s]); c=scope['c']; spec=phase_spec(); l2=phase_dir(c,s,'L2')
    if not l2.is_dir(): raise InputUnavailable('missing_phase_resolved_L2')
    for run in (1,2):
        l1=phase_dir(c,s,'L1',run); fsf=guard_path(scope,l1/'design.fsf',s)
        if not fsf.is_file(): raise InputUnavailable('missing_phase_L1_design')
        values=parse_fsf(fsf); verify_l1(values,spec,rendered=True); verify_design_con(l1/'design.con',spec)
        require(int(values['fmri(evs_orig)'])==14,'phase model must have 14 EVs including decisions')
        events=guard_path(scope,c.bids/s/'ses-01/func'/f'{s}_ses-01_task-sharedreward_run-{run}_events.tsv',s)
        canonical=pd.read_csv(events,sep='\t')
        labels=['event_computer_punish','event_computer_reward','event_friend_punish','event_friend_reward',
                'event_stranger_punish','event_stranger_reward','event_computer_neutral','event_friend_neutral','event_stranger_neutral',
                'missed_decision','missed_outcome','friend_face','stranger_face','computer_non-face']
        for ev,label in enumerate(labels,1):
            target=canonical[canonical.trial_type==label]
            if target.empty:
                require(ev in (10,11) and values.get(f'fmri(shape{ev})')=='10','required phase EV missing from canonical events')
                continue
            path=guard_path(scope,Path(values[f'fmri(custom{ev})']),s)
            require(path.is_file(),'completed phase EV file missing')
            actual=np.loadtxt(path,ndmin=2)
            expected=np.column_stack([target.onset,target.duration,np.ones(len(target))])
            actual=actual[np.argsort(actual[:,0])]; expected=expected[np.argsort(expected[:,0])]
            require(actual.shape==expected.shape and np.allclose(actual,expected,atol=1e-4,rtol=0),'phase EV timings differ from canonical events')
        data=guard_path(scope,Path(values['feat_files(1)']),s)
        require(all(x in data.name for x in (s+'_','ses-01_','task-sharedreward_',f'run-{run}_','space-MNI152NLin6Asym_')),'phase input space unresolved')
        for k in range(1,7):
            if not (l1/f'stats/cope{k}.nii.gz').is_file(): raise InputUnavailable('incomplete_phase_L1')
        if not (l1/'cluster_mask_zstat1.nii.gz').is_file(): raise InputUnavailable('incomplete_phase_L1')
    values=parse_fsf(guard_path(scope,l2/'design.fsf',s)); verify_l2(values,spec)
    for run in (1,2): require(Path(values[f'feat_files({run})']).resolve()==phase_dir(c,s,'L1',run).resolve(),'phase L2 retained-run mismatch')
    paths={k:l2/f'cope{k}.feat/stats/cope1.nii.gz' for k in range(1,7)}
    for k,path in paths.items():
        for p in (path,path.parent/'zstat1.nii.gz',path.parent.parent/'mask.nii.gz',path.parent.parent/'cluster_mask_zstat1.nii.gz'):
            if not guard_path(scope,p,s).is_file(): raise InputUnavailable('incomplete_phase_L2')
        for name,expected in [('design.mat',np.ones((2,1))),('design.con',np.ones((1,1)))]:
            text=(path.parent.parent/name).read_text()
            require('/Matrix' in text,'phase L2 lacks matrix')
            a=np.loadtxt(text.split('/Matrix',1)[1].splitlines(),ndmin=2)
            require(a.shape==expected.shape and np.allclose(a,expected),'phase L2 is not two-run fixed effects mean')
    return paths

def qc_table(base):
    q=pd.read_csv(base.repos['linux2']/base.paths['qc_table'],sep='\t',dtype=str,keep_default_na=False)
    q.subject=q.subject.map(lambda s:s if s.startswith('sub-') else 'sub-'+s)
    q=q[q.session.str.removeprefix('ses-').str.lstrip('0')=='1'].copy(); q['run']=q.run.astype(int)
    require(not q.duplicated(['subject','task','run']).any(),'duplicate source QC')
    return q.set_index(['subject','task','run'])

def qc_reason(q,s,task,runs):
    for r in runs:
        if (s,task,r) not in q.index: return 'unknown_qc'
        flags=[flag(q.loc[(s,task,r),k]) for k in POLICIES['tsnr_coverage_fd']]
        if any(x is True for x in flags): return 'qc_outlier'
        if any(x is None for x in flags): return 'unknown_qc'
    return ''

def inventory(out,scope):
    assert_development(scope); c=scope['c']; require(c.exclusions.is_dir(),'source exclusions unavailable')
    load_aging_index(c,refresh=True); q=qc_table(c); rows=[]; paths={}
    for s in scope['subjects']:
        assert_development(scope,[s])
        excluded=(c.exclusions/f'Smith-SRA-{s[4:]}').is_dir()
        base_reason='current_source_exclusion' if excluded else (qc_reason(q,s,'sharedreward',c.aging_subjects[s]['runs']) or qc_reason(q,s,'trust',(1,2)))
        for task in ('baseline','sr_outcome','ugr'):
            reason=base_reason
            if not reason and task!='baseline': reason=qc_reason(q,s,'sharedreward' if task=='sr_outcome' else 'ugr',(1,2))
            if not reason and task=='sr_outcome':
                try: paths[s]=phase_unit(scope,s)
                except InputUnavailable as exc: reason=str(exc)
            if not reason and task=='ugr':
                try: inspect_unit(c,s,'ugr','L2',required_copes_only=True)
                except InputUnavailable as exc: reason=str(exc)
            rows.append(dict(subject=s,domain=task,eligible=not reason,reason=reason))
    frame=pd.DataFrame(rows); write_tsv(out,'work/inventory.tsv',frame)
    summary=frame.groupby(['domain','eligible','reason'],dropna=False).size().reset_index(name='n')
    write_tsv(out,'results/aggregate/inventory.tsv',summary)
    print('Development inventory (before fitting):\n'+summary.to_string(index=False),flush=True)
    return {k:sorted(g.loc[g.eligible,'subject']) for k,g in frame.groupby('domain')},paths

def visual_mask(out,scope,mask,ref):
    p=json.loads((scope['c'].root/'provenance/revised/cross_valence/visual_proposal.json').read_text())
    require(p['status']=='prepared_for_review' and tuple(p['label_values'])==VISUAL_IDS,'visual proposal changed')
    asset=p['atlas']; labels=label_names(asset)
    require(set(labels[i] for i in VISUAL_IDS)==set(p['labels']),'visual labels changed')
    img=standard_image(scope,asset['image'],asset['image_sha256'])
    binary=np.isin(img.get_fdata(),VISUAL_IDS)
    union=resample_from_to(nib.Nifti1Image(binary.astype('uint8'),img.affine),(mask.shape,ref.affine),order=0).get_fdata()>0
    retained=mask&~union; expected=p['counts']['partner_pair']
    require(int(retained.sum())==expected['proposed_remaining_voxels'] and int(mask.sum())==expected['existing_mask_voxels'],'visual exclusion counts changed')
    save_image(out,'results/maps/visual_excluded_analysis_mask.nii.gz',retained.astype('uint8'),ref)
    write_json(out,'provenance/visual_exclusion.json',{**p,'executed':True,'original_mask_sha256':scope['model']['mask_sha256']})
    return retained[mask]

def vector(scope,s,path,mask,ref,diagnostics):
    guard_path(scope,path,s)
    img=load_development_image(scope['c'],path,s,scope['guard'],scope['subjects'])
    changed=not same_grid(img,ref)
    if changed: img=resample_from_to(img,(ref.shape,ref.affine),order=1)
    x=img.get_fdata(dtype=np.float32)[mask]; require(np.isfinite(x).all(),'nonfinite new map')
    diagnostics.append(dict(subject=s,path=str(path),sha256=sha256(path),resampled=changed,mean=float(x.mean()),norm=float(np.linalg.norm(x))))
    return x

def build_features(out,scope,eligible,phase_paths,key):
    assert_development(scope); mask,ref=geometry(scope); subjects=scope['subjects']; n=len(subjects); v=int(mask.sum())
    available={}; diagnostics=[]; arrays={}; marker=out.output('work/features/complete.json')
    if marker.exists():
        saved=json.loads(marker.read_text()); require(saved['fingerprint']==key,'feature checkpoint changed')
        for item in saved.get('source_hashes',[]):
            guard_path(scope,item['path'],item['subject'])
            require(sha256(item['path'])==item['sha256'],'new source image changed since feature checkpoint')
        for name,h in saved['hashes'].items():
            p=out.output('work/features/'+name+'.npy'); require(sha256(p)==h,'feature cache damaged'); arrays[name]=np.load(p,mmap_mode='r')
        return arrays,saved['available'],mask,ref
    def add(name,i,value,s):
        if name not in arrays:
            p=out.output('work/features/'+name+'.npy'); p.parent.mkdir(parents=True,exist_ok=True)
            arrays[name]=np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(n,*value.shape)); p.chmod(0o600)
            arrays[name][:]=np.nan; available[name]=[]
        arrays[name][i]=value; available[name].append(s)
    for i,s in enumerate(subjects):
        if s not in eligible['baseline']: continue
        for task,tag in [('sharedreward','sr_full'),('trust','trust')]:
            m=source_maps(scope,s,task,mask,ref,diagnostics)
            p=partners(task,m); add(tag+'_partners',i,p,s)
            for f in FAMILIES: add(tag+'_'+f,i,context(p,f),s)
            if task=='trust':
                for name,value in trust_norm(m).items(): add(name,i,value,s)
        if s in eligible['sr_outcome']:
            m={k:vector(scope,s,p,mask,ref,diagnostics) for k,p in phase_paths[s].items()}
            p=partners('sharedreward',m); add('sr_outcome_partners',i,p,s)
            for f in FAMILIES: add('sr_outcome_'+f,i,context(p,f),s)
        if s in eligible['ugr']:
            m={k:vector(scope,s,cope_path(scope['c'],s,'ugr',k),mask,ref,diagnostics) for k in (1,3,5,7)}
            for name,value in ugr_context(m).items(): add(name,i,value,s)
        if (i+1)%20==0: print(f'  Source maps {i+1}/{n}',flush=True)
    # Reconstructed historical source vectors must agree with the frozen v4 logs.
    original=pd.read_csv(scope['c'].output('work/source_image_metrics.tsv'),sep='\t').fillna({'run':0})
    original=original.drop_duplicates(['subject','task','level','run','cope']).set_index(['subject','task','level','run','cope'])
    for row in diagnostics:
        if 'task' not in row: continue
        lookup=(row['subject'],row['task'],row['level'],row.get('run') or 0,row['cope'])
        require(lookup in original.index,'historical source absent from frozen v4 log')
        old=original.loc[lookup]
        require(np.allclose([row['source_mean'],row['source_norm']],[old.source_mean,old.source_norm],atol=1e-5,rtol=2e-5),'historical source changed')
    for a in arrays.values(): a.flush()
    write_tsv(out,'work/source_metrics.tsv',diagnostics)
    write_json(out,'work/features/complete.json',dict(fingerprint=key,available=available,source_hashes=[{k:r[k] for k in ('subject','path','sha256')} for r in diagnostics if 'sha256' in r],hashes={k:sha256(a.filename) for k,a in arrays.items()}))
    return {k:np.load(a.filename,mmap_mode='r') for k,a in arrays.items()},available,mask,ref
