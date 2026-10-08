"""Read-only fixed development adapters and independent atlas reference."""
import json
import numpy as np
import pandas as pd
import nibabel as nib
from nibabel.processing import resample_from_to
from utils import sha256, write_json, write_tsv
from characterization_audit import require
from characterization_compute import assert_development, geometry, correlation
from cross_valence_compute import source_maps
from cross_valence_design import representations
from aging_source import load_aging_index
from final_stress_sources import phase_index, phase_unit, qc_table, qc_reason, vector
from revised_design import center
from specificity_design import MODES, DOMAINS


def atlas_mask(base,mask,ref,spec):
    path=base.root/spec['atlas_path']
    require(sha256(path)==spec['atlas_sha256'],'independent atlas changed')
    image=nib.squeeze_image(nib.load(path))
    # Atlas is distributed in FSL nonlinear MNI152 coordinates; nearest labels only.
    data=resample_from_to(image,(ref.shape,ref.affine),order=0).get_fdata()
    require(np.isfinite(data).all() and np.isin(data,range(8)).all(),'atlas labels/grid invalid')
    keep=(data==7)[mask]
    require(keep.any() and (~keep).any(),'empty DMN or complement')
    return keep, data==7


def inventory(scope):
    assert_development(scope); scope['phase_index']=phase_index(scope); q=qc_table(scope['c'])
    load_aging_index(scope['c'],refresh=True); paths={}
    for i,s in enumerate(scope['subjects']):
        assert_development(scope,[s]); unit=phase_unit(scope,s)
        for task,runs in [('sharedreward',scope['phase_index'][s]['runs']),('sharedreward',scope['c'].aging_subjects[s]['runs']),('trust',(1,2)),('socialdoors',(1,)),('doors',(1,))]:
            reason=qc_reason(q,s,task,runs)
            require(not reason,f'Frozen N=178 cohort no longer passes {task} QC: {reason}; stop rather than alter sample')
        paths[s]=unit
        if (i+1)%20==0 or i+1==len(scope['subjects']): print(f'Specificity phase/QC inventory {i+1}/{len(scope["subjects"])}',flush=True)
    return paths


def features(out,scope,key,phase_paths):
    assert_development(scope)
    mask,ref=geometry(scope); n=len(scope['subjects']); v=int(mask.sum())
    root=scope['c'].root/'work/revised/cross_valence'
    old_meta=json.loads((root/'features/three_paradigm/complete.json').read_text())
    legacy=root/'features/three_paradigm/social_context.npy'
    require(sha256(legacy)==old_meta['hashes']['social_context'],'old three-paradigm features changed')
    old=np.load(legacy,mmap_mode='r'); require(old.shape==(n,6,2,v),'old feature geometry mismatch')
    marker=out.output('work/features/complete.json')
    paths={k:out.output('work/features/'+k+'.npy') for k in ('legacy','outcome','generic')}
    if marker.exists():
        state=json.loads(marker.read_text()); require(state['fingerprint']==key,'specificity feature identity drift')
        for k,p in paths.items(): require(sha256(p)==state['hashes'][k],'specificity cache changed')
    else:
        marker.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=(n,10,2,v) if k!='generic' else (n,3,v)) for k,p in paths.items()}
        for p in paths.values(): p.chmod(0o600)
        diagnostics=[]; phase_log=[]
        for i,s in enumerate(scope['subjects']):
            assert_development(scope,[s])
            maps={k:vector(scope,s,p,mask,ref,phase_log) for k,p in phase_paths[s].items()}
            arrays['legacy'][i,:6]=old[i]
            phase=representations('sharedreward',maps)
            arrays['outcome'][i,:2]=phase['social_context']
            arrays['outcome'][i,6:8]=phase['friend_stranger_context']
            full=source_maps(scope,s,'sharedreward',mask,ref,diagnostics)
            previous=representations('sharedreward',full)
            require(np.allclose(previous['social_context'],old[i,:2],atol=1e-5,rtol=2e-5),'old SR source drift')
            arrays['legacy'][i,6:8]=previous['friend_stranger_context']
            arrays['generic'][i,0]=center(np.stack(list(maps.values())).mean(0))
            trust=source_maps(scope,s,'trust',mask,ref,diagnostics)
            doors=source_maps(scope,s,'socialdoors',mask,ref,diagnostics)
            money=source_maps(scope,s,'doors',mask,ref,diagnostics)
            arrays['outcome'][i,2:4]=representations('trust',trust)['social_context']
            arrays['outcome'][i,4:6]=representations('socialdoors',doors,money)['social_context']
            require(np.allclose(arrays['outcome'][i,2:6],old[i,2:],atol=1e-5,rtol=2e-5),'Trust/Doors source changed from frozen arrays')
            arrays['outcome'][i,8:10]=representations('trust',trust)['friend_stranger_context']
            arrays['legacy'][i,8:10]=arrays['outcome'][i,8:10]
            arrays['generic'][i,1]=center(np.stack([trust[k] for k in range(4,10)]).mean(0))
            arrays['generic'][i,2]=center(np.stack([d[k] for d in (doors,money) for k in (1,2)]).mean(0))
            if (i+1)%20==0 or i+1==n: print(f'Specificity inputs {i+1}/{n}',flush=True)
        # Original uncentered means/norms also protect the generic baseline inputs.
        historical=pd.read_csv(scope['c'].output('work/source_image_metrics.tsv'),sep='\t').fillna({'run':0})
        keys=['subject','task','level','run','cope']; historical=historical.drop_duplicates(keys).set_index(keys)
        for r in pd.DataFrame(diagnostics).fillna({'run':0}).to_dict('records'):
            k=tuple(r[t] for t in keys); require(k in historical.index,'source absent from original log')
            z=historical.loc[k]
            require(np.allclose([r['source_mean'],r['source_norm']],[z.source_mean,z.source_norm],atol=1e-5,rtol=2e-5),'raw source drift')
        for a in arrays.values(): a.flush()
        write_tsv(out,'work/source_metrics.tsv',diagnostics); write_tsv(out,'work/phase_metrics.tsv',phase_log)
        write_json(out,'work/features/complete.json',dict(fingerprint=key,hashes={k:sha256(p) for k,p in paths.items()}))
    return {k:np.load(p,mmap_mode='r') for k,p in paths.items()},mask,ref


def verify_previous(scope,margins):
    root=scope['c'].root; old=pd.read_csv(root/'work/revised/cross_valence/oof_predictions.tsv',sep='\t')
    old=old[(old.cohort=='three_paradigm')&(old.family=='social_context')&(old.scope=='matrix')]
    assert_development(scope,old.subject)
    require(len(old)==len(scope['subjects'])*36,'old Doors matrix incomplete')
    errors=[]
    for i,a in enumerate(DOMAINS[:6]):
        for j,b in enumerate(DOMAINS[:6]):
            group=old[(old.train_domain==a)&(old.test_domain==b)]
            require(not group.subject.duplicated().any() and set(group.subject)==set(scope['subjects']),'old prediction membership changed')
            group=group.set_index('subject').loc[scope['subjects']]
            require(all(group.fold==[scope['folds'][s] for s in scope['subjects']]),'old OOF fold mismatch')
            z=margins[:,i,j]; expected=group.margin.to_numpy()
            require(np.allclose(z,expected,atol=1e-5,rtol=2e-5) and np.array_equal(z>0,expected>0),'old full-trial OOF reproduction failed')
            errors.append(dict(train=a,test=b,n=len(z),max_absolute_error=float(abs(z-expected).max())))
    return errors


def overlap(base,scope,spec):
    """Threshold-free descriptive overlap on frozen candidates' ORIGINAL mask."""
    root=base.root; rows=[]
    for family in ('social_context','friend_stranger_context'):
        meta=json.loads((root/f'provenance/revised/cross_valence/candidate_{family}.json').read_text())
        mask_path=root/meta['mask_path']; require(sha256(mask_path)==meta['mask_sha256'],'candidate mask drift')
        ref=nib.load(mask_path); mask=ref.get_fdata().astype(bool); keep,_=atlas_mask(base,mask,ref,spec)
        for kind in ('weights','haufe','mean_difference'):
            path=root/meta['weight_path'].replace('_weights.nii.gz','_'+kind+'.nii.gz')
            img=nib.load(path); require(img.shape==mask.shape and np.allclose(img.affine,ref.affine),'candidate overlap grid')
            z=img.get_fdata()[mask]; energy=float((z*z).sum()); mass=float(abs(z).sum())
            require(energy>0 and mass>0,'empty frozen candidate')
            rows.append(dict(family=family,pattern=kind,training_n=meta['training_n'],n_voxels=int(mask.sum()),dmn_voxels=int(keep.sum()),dmn_voxel_fraction=float(keep.mean()),
                absolute_mass_fraction=float(abs(z[keep]).sum()/mass),squared_mass_fraction=float((z[keep]**2).sum()/energy),
                mass_enrichment=float(abs(z[keep]).sum()/mass/keep.mean()),signed_spatial_r=correlation(z,keep.astype(float))))
    return rows
