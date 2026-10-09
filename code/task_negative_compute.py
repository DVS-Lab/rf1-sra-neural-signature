"""Fold templates, signed expression and checkpointed matched predictions."""
import json
import numpy as np
from utils import sha256,write_json,write_tsv,atomic_output
from characterization_audit import require
from characterization_compute import assert_development,fit_binary,haufe,save_vector,correlation
from cross_valence_compute import fold_indices
from specificity_design import DOMAINS
from task_negative_design import MODES,ENDPOINTS,negative_template,score_fold,old_margins,sanity


def templates(out,scope,raw,x,dmn,mask,ref,key):
    assert_development(scope); folder='work/templates'; marker=out.output(folder+'/complete.json')
    if marker.exists():
        meta=json.loads(marker.read_text()); require(meta['fingerprint']==key,'template identity drift')
        for rel,h in meta['products'].items(): require(sha256(out.output(rel))==h,'template/map checkpoint damaged')
        return np.load(out.output(folder+'/directions.npy'),mmap_mode='r')
    rows=[]; similarity=[]; directions=[]; products=[]
    # Fold-level markers also survive interruption while deriving descriptive maps.
    for fold in (1,2,3,4,5,0):
        train=fold_indices(scope,fold)[0] if fold else np.arange(len(raw))
        ids=[scope['subjects'][i] for i in train]
        stem=f'fold-{fold}' if fold else 'DEV_display'
        checkpoint=out.output(folder+'/'+stem+'.json')
        direction_rel=folder+'/'+stem+'_direction.npy'
        if checkpoint.exists():
            saved=json.loads(checkpoint.read_text())
            require(saved['fingerprint']==key and saved['train']==ids,'template fold identity drift')
            for rel,h in saved['products'].items(): require(sha256(out.output(rel))==h,'template fold product damaged')
            t=np.load(out.output(direction_rel)); row=saved['row']; association=saved['association']
        else:
            mean,negative,t=negative_template(raw[train])
            values=np.asarray(x[train,:4]); w,b=fit_binary(values,scope,ids); h=haufe(values,w,b)
            require(np.isfinite(h).all(),'nonfinite outcome Haufe pattern')
            row=dict(fold=fold,training_n=len(train),**sanity(mean,negative,t,dmn))
            association=dict(comparison='task_negative_vs_outcome_social_context_haufe' if fold else 'DEV_display_task_negative_vs_haufe',
                             fold_a=fold,fold_b=fold,spatial_r=correlation(t,h))
            entries=[]
            for name,z in [('signed_mean',mean),('negative_component',negative),('direction',t),('outcome_social_context_haufe',h)]:
                save_vector(out,stem+'_'+name,z,mask,ref); entries.append('results/maps/'+stem+'_'+name+'.nii.gz')
            with atomic_output(out,direction_rel) as dest: np.save(dest,t)
            entries.append(direction_rel)
            saved=dict(fingerprint=key,train=ids,row=row,association=association,products={p:sha256(out.output(p)) for p in entries})
            write_json(out,folder+'/'+stem+'.json',saved)
        require(t.shape==(int(mask.sum()),) and np.isfinite(t).all() and abs(t.mean())<1e-10 and np.isclose(t@t,1,atol=1e-10),'invalid cached template geometry')
        products.extend(saved['products']); rows.append(row); similarity.append(association)
        if fold: directions.append(t)
        print(f'Task-negative template {stem}: negative={100*row["negative_fraction"]:.1f}%; DMN={100*row["dmn_negative_fraction"]:.1f}%; non-DMN={100*row["non_dmn_negative_fraction"]:.1f}%',flush=True)
    for a in range(5):
        for b in range(a+1,5): similarity.append(dict(comparison='cross_fold_task_negative',fold_a=a+1,fold_b=b+1,spatial_r=correlation(directions[a],directions[b])))
    rel=folder+'/directions.npy'
    with atomic_output(out,rel) as dest: np.save(dest,np.stack(directions))
    products.append(rel)
    for rel,values in [('results/aggregate/template_sanity.tsv',rows),('results/aggregate/spatial_similarity.tsv',similarity)]:
        write_tsv(out,rel,values); products.append(rel)
    write_json(out,folder+'/complete.json',dict(fingerprint=key,products={p:sha256(out.output(p)) for p in products}))
    return np.load(out.output(folder+'/directions.npy'),mmap_mode='r')


def observed(out,scope,x,directions,previous,key):
    assert_development(scope); benchmark=old_margins(previous); result={}; reconstruction=[]; expressions=[]; membership=[]
    for fold in range(1,6):
        train,test=fold_indices(scope,fold); t=directions[fold-1]
        scores=np.asarray(x[test],float)@t
        for j,i in enumerate(test):
            for d,name in enumerate(DOMAINS):
                expressions.append(dict(subject=scope['subjects'][i],fold=fold,domain=name,first_score=float(scores[j,d,0]),
                    second_score=float(scores[j,d,1]),signed_difference=float(scores[j,d,0]-scores[j,d,1])))
        membership.extend(dict(subject=s,fold=fold,role='test' if i in test else 'train') for i,s in enumerate(scope['subjects']))
    write_tsv(out,'work/raw_expression.tsv',expressions); write_tsv(out,'work/membership.tsv',membership)
    for mode in MODES:
        margins=np.empty((len(x),len(ENDPOINTS),4))
        for fold in range(1,6):
            train,test=fold_indices(scope,fold); rel=f'work/predictions/{mode}_fold-{fold}'
            expected=dict(fingerprint=key,mode=mode,fold=fold,train=[scope['subjects'][i] for i in train],test=[scope['subjects'][i] for i in test])
            marker=out.output(rel+'.json'); path=out.output(rel+'.npy')
            if marker.exists():
                meta=json.loads(marker.read_text()); require(all(meta.get(k)==v for k,v in expected.items()) and sha256(path)==meta['sha256'],'fold checkpoint drift')
                z=np.load(path)
            else:
                z=score_fold(scope,x,directions[fold-1],fold,mode)
                if mode=='original':
                    require(np.allclose(z,benchmark[test],atol=1e-5,rtol=2e-5) and np.array_equal(z>0,benchmark[test]>0),'outcome-only benchmark not reproduced')
                with atomic_output(out,rel+'.npy') as dest: np.save(dest,z)
                write_json(out,rel+'.json',dict(**expected,sha256=sha256(path)))
            require(z.shape==(len(test),len(ENDPOINTS),4) and np.isfinite(z).all(),'invalid fold predictions')
            margins[test]=z
            print(f'Task-negative control: {mode}, fold {fold}/5 complete',flush=True)
        if mode=='original':
            require(np.allclose(margins,benchmark,atol=1e-5,rtol=2e-5) and np.array_equal(margins>0,benchmark>0),'cached benchmark not reproduced')
            for e,(a,b,_,_,_) in enumerate(ENDPOINTS): reconstruction.append(dict(train=a,test=b,n=len(x),max_absolute_error=float(abs(margins[:,e]-benchmark[:,e]).max())))
            write_tsv(out,'results/aggregate/benchmark_reconstruction.tsv',reconstruction)
        result[mode]=margins
    return result,expressions
