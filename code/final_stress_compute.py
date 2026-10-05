"""Participant-blocked models; descriptive fixed-model probes explicitly labeled."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from utils import atomic_output,sha256,write_json,write_tsv
from characterization_audit import require
from characterization_compute import assert_development,fit_binary,save_vector,haufe,correlation,coefficients
from cross_valence_compute import summarize_cell
from final_stress_design import FAMILIES,reduced,usable


def partition(scope,p,fold):
    assert_development(scope,p['subjects']); require(usable(scope,p['subjects']),'insufficient matched sample/folds')
    index={s:i for i,s in enumerate(scope['subjects'])}
    train=[index[s] for s in p['subjects'] if scope['folds'][s]!=fold]
    test=[index[s] for s in p['subjects'] if scope['folds'][s]==fold]
    require(train and test and not set(train)&set(test),'participant overlap or empty fold')
    return np.array(train),np.array(test)

def selected(arrays,keys,idx,keep=None):
    x=np.stack([arrays[k][idx] for k in keys],axis=1)
    require(np.isfinite(x).all(),'missing/nonfinite matched input')
    return reduced(x,keep) if keep is not None else x

def fit_score(scope,arrays,p,fold,keep,flips=None,weights=None):
    train,test=partition(scope,p,fold); feature_keep=keep if p['visual'] else None
    if weights is None:
        x=selected(arrays,p['train'],train,feature_keep)
        weights=fit_binary(x,scope,[scope['subjects'][i] for i in train],None if flips is None else flips[train])
    w,b=weights; x=selected(arrays,p['test'],test,feature_keep)
    scores=x.astype(float)@w+b; margins=scores[:,:,0]-scores[:,:,1]
    if flips is not None: margins*=flips[test,None]
    return w,b,train,test,margins

def identifier(p): return p['name']+('_visual_excluded' if p['visual'] else '_whole_brain')

def run_plan(out,scope,arrays,p,keep,key,mask,ref):
    assert_development(scope,p['subjects']); stem=identifier(p); records=[]; members=[]
    for fold in range(1,6):
        train,test=partition(scope,p,fold); rel=f'work/models/{stem}_fold-{fold}'
        identity=dict(fingerprint=key,plan=p,fold=fold,train=[scope['subjects'][i] for i in train],test=[scope['subjects'][i] for i in test])
        marker=out.output(rel+'.json'); weights=None
        if marker.exists():
            old=json.loads(marker.read_text()); require(all(old.get(k)==v for k,v in identity.items()),'fold checkpoint drift')
            require(sha256(out.output(rel+'.npz'))==old['sha256'],'fold checkpoint damaged')
            with np.load(out.output(rel+'.npz')) as z: weights=(z['w'],float(z['b']))
        w,b,train,test,margins=fit_score(scope,arrays,p,fold,keep,weights=weights)
        if weights is None:
            with atomic_output(out,rel+'.npz') as dest: np.savez(dest,w=w,b=b)
            write_json(out,rel+'.json',{**identity,'sha256':sha256(out.output(rel+'.npz'))})
        for row,i in enumerate(test):
            for j,target in enumerate(p['test']):
                records.append(dict(model=stem,family=p['family'],group=p['group'],visual=p['visual'],test=target,
                    subject=scope['subjects'][i],fold=fold,margin=float(margins[row,j]),evaluation='participant_blocked_cv'))
        members.extend(dict(model=stem,fold=fold,subject=scope['subjects'][i],role=role) for role,ids in [('train',train),('test',test)] for i in ids)
    # Separate characterization model; never overwrites either frozen candidate.
    idx=[scope['subjects'].index(s) for s in p['subjects']]
    x=selected(arrays,p['train'],idx,keep if p['visual'] else None)
    w,b=fit_binary(x,scope,p['subjects']); active=mask.copy()
    if p['visual']: active[mask]=keep
    for kind,values in [('weights',w),('haufe',haufe(x,w,b)),('mean_difference',(x[:,:,0]-x[:,:,1]).mean((0,1)))]:
        save_vector(out,stem+'_DEV_'+kind,values,active,ref)
    write_json(out,'provenance/models/'+stem+'.json',dict(plan={k:v for k,v in p.items() if k!='subjects'},training_n=len(idx),
        weight_sha256=sha256(out.output('results/maps/'+stem+'_DEV_weights.nii.gz')),intercept=b,holdout_scored=False))
    return records,members

def summarize(records,bootstrap=10000):
    rows=[]; frame=pd.DataFrame(records)
    if frame.empty: return frame
    keys=['model','family','group','visual','test','evaluation']
    for values,g in frame.groupby(keys):
        require(not g.subject.duplicated().any(),'duplicate participant endpoint')
        rows.append({**dict(zip(keys,values)),**summarize_cell(g.margin.to_numpy(),':'.join(map(str,values)),bootstrap)})
    return pd.DataFrame(rows)

def candidate(scope,family,mask,ref):
    assert_development(scope); root=scope['c'].root
    meta=json.loads((root/f'provenance/revised/cross_valence/candidate_{family}.json').read_text())
    p=root/meta['weight_path']; require(p.resolve().is_relative_to(root/'results/revised/cross_valence/maps'),'candidate outside frozen tree')
    require(sha256(p)==meta['weight_sha256'] and sha256(root/meta['mask_path'])==meta['mask_sha256'],'frozen candidate changed')
    img=nib.load(p); require(img.shape==mask.shape and np.allclose(img.affine,ref.affine),'candidate geometry')
    return img.get_fdata()[mask],meta['intercept']

def fixed_probes(scope,arrays,available,mask,ref):
    records=[]
    for family in FAMILIES:
        w,b=candidate(scope,family,mask,ref)
        targets=['sr_outcome_'+family]+(['ugr_context','ugr_high','ugr_low'] if family=='social_context' else [])
        for target in targets:
            for s in available.get(target,[]):
                assert_development(scope,[s]); i=scope['subjects'].index(s); z=arrays[target][i].astype(float)@w+b
                records.append(dict(model='frozen_DEV_'+family,family=family,group='fixed_probe',visual=False,test=target,subject=s,
                    fold=scope['folds'][s],margin=float(z[0]-z[1]),evaluation='development_training_participant_overlap'))
    return records

def specificity(out,scope,arrays,plans,keep):
    records=[]; expressions=[]
    for p in plans:
        if p['group']!='architecture' or p['visual']: continue
        for fold in range(1,6):
            _,test=partition(scope,p,fold); path=out.output(f'work/models/{identifier(p)}_fold-{fold}.npz')
            with np.load(path) as z: w,b=z['w'],float(z['b'])
            for task in ('sr_full','trust'):
                x=arrays[task+'_partners'][test].astype(float); scores=x@w+b
                for row,i in enumerate(test):
                    s=scope['subjects'][i]
                    for partner,value in zip(('computer','friend','stranger'),scores[row]):
                        expressions.append(dict(model=p['family'],task=task,subject=s,fold=fold,partner=partner,expression=float(value)))
                    for name,value in [('human_vs_computer',scores[row,1:].mean()-scores[row,0]),('friend_vs_stranger',scores[row,1]-scores[row,2])]:
                        records.append(dict(model=p['family'],family=p['family'],group='specificity',visual=False,test=task+'_'+name,
                            subject=s,fold=fold,margin=float(value),evaluation='participant_blocked_cv'))
    write_tsv(out,'work/partner_expression.tsv',expressions)
    return records

def spatial_comparison(out,scope,mask,ref,plans,keep):
    rows=[]; root=scope['c'].root/'results/revised/cross_valence/maps'
    for kind,suffix in [('weight','weights'),('haufe','haufe'),('mean_difference','mean_difference')]:
        vectors=[]
        for f in FAMILIES:
            path=root/f'partner_pair_{f}_collapsed_common_DEV_{suffix}.nii.gz'
            img=nib.load(path); require(img.shape==ref.shape and np.allclose(img.affine,ref.affine),'frozen pattern grid')
            vectors.append(img.get_fdata()[mask])
        rows.append(dict(comparison='frozen_context_vs_friend_stranger',kind=kind,spatial_r=correlation(*vectors),n_voxels=int(mask.sum())))
    for p in plans:
        if p['visual']: continue
        a=out.output('results/maps/'+identifier(p)+'_DEV_weights.nii.gz')
        b=out.output('results/maps/'+identifier({**p,'visual':True})+'_DEV_weights.nii.gz')
        if a.exists() and b.exists():
            rows.append(dict(comparison=p['name']+'_whole_vs_visual_excluded',kind='weight_on_retained_voxels',
                spatial_r=correlation(nib.load(a).get_fdata()[mask][keep],nib.load(b).get_fdata()[mask][keep]),n_voxels=int(keep.sum())))
    write_tsv(out,'results/aggregate/spatial_comparison.tsv',rows)

def valence_controls(scope,arrays,available,mask,ref):
    records=[]
    for target in ('trust_norm','ugr_norm'):
        for s in sorted(set(available.get(target,[])) & set(available.get('ugr_norm',[]))):
            assert_development(scope,[s]); fold=scope['folds'][s]
            w,b=coefficients(scope,scope['models'][('valence_common',fold)],mask,ref)
            z=arrays[target][scope['subjects'].index(s)].astype(float)@w+b
            # Frozen valence positive=reward. Orient to violation by reversing score, not retraining.
            records.append(dict(model='existing_valence_OOF_reversed',family='generic_valence_control',group='norm',visual=False,
                test=target,subject=s,fold=fold,margin=float(z[1]-z[0]),evaluation='participant_blocked_existing_OOF'))
    return records
