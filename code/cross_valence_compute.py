"""Guarded, checkpointed fitting and participant-level summaries."""
import json
import shutil
import numpy as np
import pandas as pd
from aging_source import load_aging_index
from inventory import inspect_unit
from build_mask import vectorize_source
from characterization_audit import require, digest, PRIMARY
from characterization_compute import (assert_development, geometry, coefficients, fit_binary,
    haufe, save_vector, correlation, replicate_rng, verify_oof)
from cross_valence_design import representations, domains, plans, training_names, TASKS, FAMILIES
from reporting import performance
from utils import sha256, write_json, write_tsv, atomic_output


def source_maps(scope,subject,task,mask,reference,diagnostics):
    # Guard executes before path inspection or a source image can be opened.
    assert_development(scope,[subject]); c=scope['c']
    level,run=('L1',1) if task in ('socialdoors','doors') else ('L2',None)
    units=[(level,run)]
    if task=='sharedreward':
        runs=c.aging_subjects[subject]['runs']
        units=[('L1',r) for r in runs]+([('L2',None)] if len(runs)==2 else [])
    for lev,r in units: inspect_unit(c,subject,task,lev,r,required_copes_only=True)
    return {k:vectorize_source(c,subject,task,k,mask,reference,scope['guard'],scope['subjects'],diagnostics,level,run)
            for k in c.contrasts[task]['copes']}


def build_arrays(out,scope,cohort,fingerprint,spec):
    assert_development(scope); mask,ref=geometry(scope)
    count=2 if cohort=='partner_pair' else 3; n=len(scope['subjects']); v=int(mask.sum())
    families=FAMILIES if count==2 else FAMILIES[:1]
    shapes={f:(n,2*count,2,v) for f in families}
    shapes.update({f+'_collapsed':(n,count,2,v) for f in families})
    marker=out.output(f'work/features/{cohort}/complete.json')
    paths={k:out.output(f'work/features/{cohort}/{k}.npy') for k in shapes}
    if marker.exists():
        state=json.loads(marker.read_text()); require(state['fingerprint']==fingerprint,'cross-valence feature inputs changed')
        for k,p in paths.items(): require(sha256(p)==state['hashes'][k],'cross-valence cache damaged')
    else:
        marker.parent.mkdir(parents=True,exist_ok=True)
        arrays={k:np.lib.format.open_memmap(p,mode='w+',dtype='float32',shape=shapes[k]) for k,p in paths.items()}
        for p in paths.values(): p.chmod(0o600)
        load_aging_index(scope['c'],refresh=True); diagnostics=[]
        for i,s in enumerate(scope['subjects']):
            assert_development(scope,[s])
            for j,task in enumerate(TASKS[:count]):
                maps=source_maps(scope,s,task,mask,ref,diagnostics)
                monetary=source_maps(scope,s,'doors',mask,ref,diagnostics) if task=='socialdoors' else None
                values=representations(task,maps,monetary)
                for f in families:
                    arrays[f][i,2*j:2*j+2]=values[f]
                    arrays[f+'_collapsed'][i,j]=values[f+'_collapsed']
            if (i+1)%25==0 or i+1==n: print(f'  Cross-valence inputs {cohort}: {i+1}/{n}',flush=True)
        old=pd.read_csv(scope['c'].output('work/source_image_metrics.tsv'),sep='\t').fillna({'run':0})
        current=pd.DataFrame(diagnostics).fillna({'run':0}); keys=['subject','task','level','run','cope']
        old=old.drop_duplicates(keys).set_index(keys)
        for r in current.to_dict('records'):
            key=tuple(r[k] for k in keys); require(key in old.index,'source absent from original v4 log')
            row=old.loc[key]
            require(np.allclose([r['source_mean'],r['source_norm']],[row.source_mean,row.source_norm],atol=1e-5,rtol=2e-5),
                    'source input differs from frozen v4')
        for a in arrays.values(): a.flush()
        write_tsv(out,f'work/features/{cohort}/source_metrics.tsv',diagnostics)
        write_json(out,f'work/features/{cohort}/complete.json',dict(fingerprint=fingerprint,hashes={k:sha256(p) for k,p in paths.items()}))
    arrays={k:np.load(p,mmap_mode='r') for k,p in paths.items()}
    require(all(a.shape==shapes[k] for k,a in arrays.items()),'feature dimensions changed')
    old_arrays={'social_context':arrays['social_context_collapsed']}
    if count==2:
        old_arrays.update(closeness_positive=arrays['friend_stranger_context'][:,[0,2]],
                          closeness_negative=arrays['friend_stranger_context'][:,[1,3]])
    checks=verify_oof(out,scope,cohort,old_arrays,mask,ref,spec)
    return arrays,mask,ref,checks


def fold_indices(scope,fold):
    assert_development(scope)
    values=np.array([scope['folds'][s] for s in scope['subjects']])
    require(set(values)==set(range(1,6)),'five nonempty fixed folds required')
    train=np.where(values!=fold)[0]; test=np.where(values==fold)[0]
    require(len(train)>0 and len(test)>0,'empty CV partition')
    require(not {scope['subjects'][i] for i in train}&{scope['subjects'][i] for i in test},'participant overlap')
    return train,test


def fit_fold(scope,x,p,fold,flips=None):
    train,test=fold_indices(scope,fold)
    ids=[scope['subjects'][i] for i in train]
    selected=x[train][:,p['train']]
    w,b=fit_binary(selected,scope,ids,None if flips is None else flips[train])
    return w,b,train,test


def score_fold(scope,x,p,w,b,train,test,flips=None):
    assert_development(scope,[scope['subjects'][i] for i in test])
    require(not set(train)&set(test),'Participant leakage before scoring')
    scores=x[test][:,p['test']].astype(np.float64)@w+b
    margins=scores[:,:,0]-scores[:,:,1]
    if flips is not None: margins=margins*flips[test,None]
    require(np.isfinite(margins).all(),'nonfinite cross-valence predictions')
    return margins


def cv_margins(out,scope,cohort,arrays,p,fingerprint):
    x=arrays[p['key']]; result=np.empty((len(x),len(p['test'])))
    stem=cohort+'_'+p['family']+'_'+p['name']; members=[]
    for fold in range(1,6):
        assert_development(scope); train,test=fold_indices(scope,fold)
        rel=f'work/fold_models/{stem}_fold-{fold}'
        path=out.output(rel+'.npz'); marker=out.output(rel+'.json')
        expected=dict(fingerprint=fingerprint,plan=p,fold=fold,
                      train_subjects=[scope['subjects'][i] for i in train],test_subjects=[scope['subjects'][i] for i in test])
        if marker.exists():
            meta=json.loads(marker.read_text()); require(all(meta.get(k)==v for k,v in expected.items()),'fold checkpoint membership changed')
            require(sha256(path)==meta['sha256'],'fold checkpoint damaged')
            with np.load(path,allow_pickle=False) as saved: w,b=saved['w'],float(saved['b'])
        else:
            w,b,_,_=fit_fold(scope,x,p,fold)
            with atomic_output(out,rel+'.npz') as dest: np.savez(dest,w=w,b=b)
            write_json(out,rel+'.json',{**expected,'sha256':sha256(path)})
        result[test]=score_fold(scope,x,p,w,b,train,test)
        members.extend(dict(cohort=cohort,family=p['family'],model=p['name'],fold=fold,subject=s,role='test' if i in test else 'train',
                            training_domains='+'.join(training_names(p,cohort))) for i,s in enumerate(scope['subjects']))
    return result,members


def summarize_cell(margins,key,bootstrap_samples):
    """For combined tasks, average correctness WITHIN participant, never binarize average margin."""
    margins=np.asarray(margins); margins=margins[:,None] if margins.ndim==1 else margins
    if margins.shape[1]==1:
        row=performance(margins[:,0],key,bootstrap_samples=bootstrap_samples)
        row['ci_method']='exact binomial descriptive OOF interval'
        return row
    correct=(margins>0).mean(1); means=margins.mean(1)
    rng=replicate_rng('summary:'+key,0); acc=[]; avg=[]
    for start in range(0,bootstrap_samples,200):
        idx=rng.integers(0,len(means),(min(200,bootstrap_samples-start),len(means)))
        acc.extend(correct[idx].mean(1)); avg.extend(means[idx].mean(1))
    lo,hi=np.quantile(acc,[.025,.975]); ml,mh=np.quantile(avg,[.025,.975])
    return dict(n=len(means),accuracy=float(correct.mean()),ci_low=lo,ci_high=hi,
                mean_margin=float(means.mean()),median_margin=float(np.median(means)),margin_ci_low=ml,margin_ci_high=mh,
                ci_method='participant bootstrap; task correctness averaged within participant',binomial_p=np.nan)


def summaries(cohort,p,margins,spec):
    rows=[]
    targets=list(enumerate(p['test_names']))
    if p['scope']=='pooled_cross_valence': targets.append((None,'combined'))
    for j,name in targets:
        key=cohort+':'+p['family']+':'+p['name']+':'+name
        rows.append(dict(cohort=cohort,family=p['family'],scope=p['scope'],train_domain=p['name'],test_domain=name,
                         maps_per_participant=margins.shape[1] if j is None else 1,
                         **summarize_cell(margins if j is None else margins[:,j],key,spec['bootstrap_samples'])))
    return rows


def final_patterns(out,scope,arrays,mask,ref,spec,fingerprint):
    """Six common forward patterns plus task-specific relational candidate weights."""
    entries=[]; patterns={}; subjects=scope['subjects']
    mask_dest='results/maps/partner_pair_analysis_mask.nii.gz'
    with atomic_output(out,mask_dest) as dest: shutil.copyfile(scope['c'].output('results/maps/analysis_mask.nii.gz'),dest)
    for family in FAMILIES:
        for mode,indices in [('positive',[0,2]),('negative',[1,3]),('collapsed',[0,1])]:
            x=arrays[family+('_collapsed' if mode=='collapsed' else '')][:,indices]
            stem='partner_pair_'+family+'_'+mode+'_common_DEV'
            assert_development(scope)
            original=scope['models'][('social_context_common',0)] if family=='social_context' and mode=='collapsed' else None
            if original:
                w,b=coefficients(scope,original,mask,ref)
                rel='results/maps/'+stem+'_weights.nii.gz'
                with atomic_output(out,rel) as dest: shutil.copyfile(scope['c'].root/original['maps'][0]['path'],dest)
            else:
                w,b=fit_binary(x,scope,subjects)
                rel='results/maps/'+stem+'_weights.nii.gz'; save_vector(out,stem+'_weights',w,mask,ref)
            a=haufe(x,w,b); mean=(x[:,:,0]-x[:,:,1]).mean((0,1),dtype=np.float64)
            save_vector(out,stem+'_haufe',a,mask,ref); save_vector(out,stem+'_mean_difference',mean,mask,ref)
            patterns[(family,mode)]={'weight':w,'haufe':a,'mean_difference':mean}
            entries.append(dict(family=family,mode=mode,model='common',training_n=len(subjects),
                training_domains=[t+'_'+mode for t in TASKS[:2]],weight_path=str(out.output(rel).relative_to(out.root)),
                weight_sha256=sha256(out.output(rel)),intercept=b,classes=[-1,1],mask_path=str(out.output(mask_dest).relative_to(out.root)),
                mask_sha256=sha256(out.output(mask_dest)),reused_v4=bool(original),
                source_v4_weight_sha256=original['maps'][0]['sha256'] if original else None))
    family='friend_stranger_context'
    for j,task in enumerate(TASKS[:2]):
        assert_development(scope); w,b=fit_binary(arrays[family+'_collapsed'][:,[j]],scope,subjects)
        stem='partner_pair_'+family+'_collapsed_'+task+'_DEV'; save_vector(out,stem+'_weights',w,mask,ref)
        entries.append(dict(family=family,mode='collapsed',model=task,training_n=len(subjects),training_domains=[task+'_collapsed'],
            weight_path=str(out.output('results/maps/'+stem+'_weights.nii.gz').relative_to(out.root)),
            weight_sha256=sha256(out.output('results/maps/'+stem+'_weights.nii.gz')),intercept=b,classes=[-1,1],
            mask_path=str(out.output(mask_dest).relative_to(out.root)),mask_sha256=sha256(out.output(mask_dest)),reused_v4=False))
    rows=[]
    for family in FAMILIES:
        for a,b in [('positive','negative'),('positive','collapsed'),('negative','collapsed')]:
            for kind in ('weight','haufe','mean_difference'):
                rows.append(dict(family=family,pattern_a=a,pattern_b=b,kind=kind,spatial_r=correlation(patterns[(family,a)][kind],patterns[(family,b)][kind]),n_voxels=int(mask.sum())))
    write_tsv(out,'results/aggregate/spatial_similarity.tsv',rows)
    write_json(out,'provenance/models.json',dict(fingerprint=fingerprint,models=entries,holdout_scored=False,
        classifier=spec['classifier'],seed=spec['seed'],normalization='independent per-map spatial mean centering; no feature scaling'))
    return entries


def permute(out,scope,arrays,p,spec,fingerprint):
    key=p['family']+'_'+p['name']; path=out.output('work/permutations/'+key+'.json')
    state=json.loads(path.read_text()) if path.exists() else dict(fingerprint=fingerprint,plan=p,null=[])
    require(state['fingerprint']==fingerprint and state['plan']==p and len(state['null'])<=spec['permutations'],'permutation checkpoint mismatch')
    x=arrays[p['key']]; n=len(x)
    for iteration in range(len(state['null']),spec['permutations']):
        # Same deterministic flip for each participant across all tasks, valences, and constructs.
        flips=replicate_rng('cross_valence:partner_pair:labels',iteration).choice([-1,1],n)
        margins=np.empty((n,len(p['test'])))
        for fold in range(1,6):
            w,b,train,test=fit_fold(scope,x,p,fold,flips)
            margins[test]=score_fold(scope,x,p,w,b,train,test,flips)
        accuracy=(margins>0).mean(0)
        # The two tasks are repeated observations from n participants, not 2n people.
        state['null'].append([float((margins>0).mean(axis=1).mean())] if p['scope']=='pooled_cross_valence' else accuracy.tolist())
        write_json(out,'work/permutations/'+key+'.json',state)
        if (iteration+1)%10==0 or iteration+1==spec['permutations']:
            print(f'  Permutation {key}: {iteration+1}/{spec["permutations"]}',flush=True)
    return np.asarray(state['null'])
