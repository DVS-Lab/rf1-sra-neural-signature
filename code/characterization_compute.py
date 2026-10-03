"""Guarded development-only map reconstruction, forward patterns and resampling."""
from pathlib import Path
import json
import warnings
import hashlib
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from aging_source import load_aging_index
from inventory import inspect_unit
from build_mask import vectorize_source, reconstruct, save_image
from revised_design import partner_representations, doors_representations
from characterization_audit import require, digest, TARGETS, TASKS, SEED, PARAMETERS, PRIMARY
from utils import PipelineError, atomic_output, sha256, write_json, write_tsv


def assert_development(scope, subjects=None):
    scope['guard'].check(scope['subjects'] if subjects is None else subjects, scope['subjects'])


def frozen_image(scope, path, expected_hash=None):
    assert_development(scope)
    path = Path(path)
    require(path.resolve().is_relative_to(scope['c'].output('results/maps')), 'image outside frozen development map tree')
    if expected_hash: require(sha256(path) == expected_hash, 'saved map changed before image load')
    return nib.load(path)


def geometry(scope):
    path = scope['c'].output('results/maps/analysis_mask.nii.gz')
    image = frozen_image(scope, path, scope['model']['mask_sha256'])
    data = image.get_fdata()
    require(np.isfinite(data).all() and np.isin(data, [0, 1]).all(), 'invalid frozen mask')
    return data.astype(bool), image


def coefficients(scope, entry, mask, reference):
    require(entry['classes'] == [-1, 1] and len(entry['maps']) == 1, 'binary coefficient class contract')
    item = entry['maps'][0]
    image = frozen_image(scope, scope['c'].root/item['path'], item['sha256'])
    require(image.shape == mask.shape and np.allclose(image.affine, reference.affine), 'coefficient/mask geometry mismatch')
    data = image.get_fdata(dtype=np.float32)
    require(np.isfinite(data).all(), 'nonfinite coefficients')
    return data[mask], float(entry['intercept'][0])


def save_vector(out, stem, values, mask, reference):
    save_image(out, 'results/maps/'+stem+'.nii.gz', reconstruct(np.asarray(values, np.float32), mask), reference)


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a)&np.isfinite(b)
    if good.sum()<3 or np.std(a[good]) == 0 or np.std(b[good]) == 0: return np.nan
    return float(np.corrcoef(a[good], b[good])[0, 1])


def haufe(x, w, b=0.):
    """Cov(X,s)/Var(s), without predictor standardization or V-by-V covariance."""
    x = np.asarray(x).reshape(-1, x.shape[-1])
    s = x @ np.asarray(w, float) + b
    centered = s-s.mean()
    denominator = centered@centered
    require(np.isfinite(denominator) and denominator>0, 'zero/nonfinite decoder-output variance')
    result = np.empty(x.shape[1], np.float32)
    for start in range(0, len(result), 4096):
        block = np.asarray(x[:, start:start+4096], float)
        result[start:start+4096] = (block-block.mean(axis=0)).T@centered/denominator
    return result


def build_arrays(out, scope, cohort, fingerprint):
    """Cache exact representations in private memmaps; never call v4 output writers."""
    assert_development(scope)
    mask, reference = geometry(scope)
    subjects, tasks = scope['subjects'], TASKS[cohort]
    families = TARGETS[cohort] + (('closeness_reward',) if cohort == 'partner_pair' else ())
    shape = (len(subjects), len(tasks), 2, int(mask.sum()))
    marker = out.output(f'work/features/{cohort}/complete.json')
    paths = {f: out.output(f'work/features/{cohort}/{f}.npy') for f in families}
    if marker.exists():
        info = json.loads(marker.read_text())
        require(info['fingerprint'] == fingerprint, 'feature cache fingerprint changed')
        for family, path in paths.items(): require(sha256(path) == info['hashes'][family], 'feature cache damaged')
        arrays = {f: np.load(p, mmap_mode='r') for f,p in paths.items()}
        require(all(a.shape == shape for a in arrays.values()), 'feature cache dimensions changed')
        return arrays, mask, reference
    marker.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f: np.lib.format.open_memmap(p, mode='w+', dtype='float32', shape=shape) for f,p in paths.items()}
    for p in paths.values(): p.chmod(0o600)
    c = scope['c']; load_aging_index(c, refresh=True)
    diagnostics = []
    def maps(subject, task, level='L2', run=None):
        assert_development(scope, [subject])
        return {k: vectorize_source(c, subject, task, k, mask, reference, scope['guard'], subjects,
                                    diagnostics, level, run) for k in c.contrasts[task]['copes']}
    for i, subject in enumerate(subjects):
        assert_development(scope, [subject])
        # Verify completed designs and aging input-stat fingerprints before loading
        # development intensities, retaining the original one/two-run strategy.
        for task in c.contrasts:
            units = [('L1',1)] if task in ('socialdoors','doors') else [('L2',None)]
            if cohort == 'partner_pair' and task in ('socialdoors','doors'): continue
            if task == 'sharedreward':
                retained=c.aging_subjects[subject]['runs']
                units=[('L1',r) for r in retained]+([('L2',None)] if len(retained)==2 else [])
            for level,run in units: inspect_unit(c,subject,task,level,run,required_copes_only=True)
        source = {task: partner_representations(task, maps(subject, task)) for task in ('sharedreward','trust')}
        if cohort == 'three_paradigm':
            source['socialdoors'] = doors_representations(maps(subject,'socialdoors','L1',1), maps(subject,'doors','L1',1))
        for family in families: arrays[family][i] = np.stack([source[t][family] for t in tasks])
        if (i+1)%25 == 0 or i+1 == len(subjects): print(f'  Characterization input maps {cohort}: {i+1}/{len(subjects)}', flush=True)
    for a in arrays.values(): a.flush()
    # Match source means/norms from the original run as an additional input check.
    historical=pd.read_csv(c.output('work/source_image_metrics.tsv'),sep='\t').fillna({'run':0})
    current=pd.DataFrame(diagnostics).fillna({'run':0})
    keys=['subject','task','level','run','cope']
    historical=historical.drop_duplicates(keys).set_index(keys)
    for row in current.to_dict('records'):
        key=tuple(row[k] for k in keys)
        require(key in historical.index,'source map absent from original load log')
        original=historical.loc[key]
        require(np.allclose([row['source_mean'],row['source_norm']],
                            [original.source_mean,original.source_norm],atol=1e-5,rtol=2e-5),
                'source-map mean/norm differs from original v4 load')
    write_tsv(out, f'work/features/{cohort}/source_metrics.tsv', diagnostics)
    # These hashes protect reconstructed inputs for all later bootstrap/permutation restarts.
    write_json(out, f'work/features/{cohort}/complete.json', {'fingerprint':fingerprint,
               'hashes':{f:sha256(p) for f,p in paths.items()}, 'shape':shape})
    return {f:np.load(p,mmap_mode='r') for f,p in paths.items()}, mask, reference


def verify_oof(out, scope, cohort, arrays, mask, reference, spec):
    assert_development(scope)
    subjects, tasks = scope['subjects'], TASKS[cohort]
    index = {s:i for i,s in enumerate(subjects)}
    records, checks = [], []
    for family, x in arrays.items():
        rows = scope['predictions'][scope['predictions'].family == family]
        for (training, task, mode, fold), group in rows.groupby(['train_tasks','test_task','scope','fold']):
            name = 'common' if mode == 'common' else 'lopo-'+task if mode == 'lopo' else 'task-'+training
            entry = scope['models'][(family+'_'+name, int(fold))]
            assert_development(scope, group.subject)
            require(all(scope['folds'][s] == fold for s in group.subject), 'OOF rescoring fold leak')
            if mode == 'lopo': require(task not in entry['training_tasks'], 'OOF LOPO task leak')
            w,b = coefficients(scope, entry, mask, reference)
            idx = [index[s] for s in group.subject]
            values = x[idx, tasks.index(task)].astype(float)@w.astype(float)+b
            margins = values[:,0]-values[:,1]
            original = group.margin.to_numpy(float)
            require(np.allclose(margins, original, atol=spec['oof_atol'], rtol=spec['oof_rtol']),
                    'saved fold maps do not reproduce original OOF margins; stop interpretation')
            require(np.array_equal(margins>0, original>0), 'OOF forced-choice classifications changed')
            checks.append({'cohort':cohort,'family':family,'train_tasks':training,'test_task':task,'scope':mode,
                           'fold':fold,'n':len(idx),'max_absolute_error':float(np.max(np.abs(margins-original)))})
            if mode == 'common':
                records.extend({'subject':s,'cohort':cohort,'family':family,'task':task,'fold':fold,
                                'positive_score':v[0],'negative_score':v[1],'margin':m}
                               for s,v,m in zip(group.subject, values, margins))
    write_tsv(out, f'work/{cohort}_oof_scores.tsv', records)
    return checks


def fit_binary(x, scope, subjects, flips=None):
    assert_development(scope, subjects)
    require(len(subjects) == len(x) and x.shape[2] == 2, 'resampling participant/map shape mismatch')
    y = np.broadcast_to(np.array([1,-1]), x.shape[:3]).copy()
    if flips is not None:
        require(np.asarray(flips).shape == (len(subjects),) and np.isin(flips,[-1,1]).all(), 'invalid participant flips')
        y *= np.asarray(flips,dtype=int)[:,None,None]
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        estimator = LinearSVC(**PARAMETERS, random_state=SEED).fit(x.reshape(-1,x.shape[-1]), y.reshape(-1))
    return estimator.coef_[0], float(estimator.intercept_[0])


def replicate_rng(key, index):
    return np.random.default_rng(np.random.SeedSequence([SEED, int(hashlib.sha256(key.encode()).hexdigest()[:8],16), index]))


def bootstrap(out, scope, cohort, family, x, mask, reference, spec, fingerprint):
    assert_development(scope)
    count, v = spec['bootstrap_samples'], x.shape[-1]
    key = cohort+'_'+family
    folder = out.output('work/bootstrap/'+key); folder.mkdir(parents=True, exist_ok=True)
    marker = folder/'checkpoint.json'
    state = json.loads(marker.read_text()) if marker.exists() else {'fingerprint':fingerprint,'n':count,'row_hashes':[]}
    require(state['fingerprint'] == fingerprint and state['n'] == count, 'bootstrap checkpoint contract changed')
    done = len(state['row_hashes'])
    require(done <= count, 'invalid bootstrap checkpoint count')
    samples = {}
    for kind in ('weight','haufe'):
        path = folder/(kind+'.npy')
        require(not done or path.exists(), 'bootstrap distribution missing')
        samples[kind] = (np.load(path,mmap_mode='r+') if path.exists() and done else
                         np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(count,v)))
        path.chmod(0o600)
        require(samples[kind].shape == (count,v), 'bootstrap cache shape mismatch')
    for i, expected in enumerate(state['row_hashes']):
        require(hashlib.sha256(samples['weight'][i].tobytes()+samples['haufe'][i].tobytes()).hexdigest() == expected,
                'bootstrap checkpoint data damaged')
    for i in range(done, count):
        assert_development(scope)
        indices = replicate_rng('bootstrap:'+key,i).integers(0,len(x),len(x))
        # Duplicate whole participants, retaining every task and both classes.
        selected = x[indices]
        subjects = [scope['subjects'][j] for j in indices]
        w,b = fit_binary(selected, scope, subjects)
        a = haufe(selected,w,b)
        samples['weight'][i], samples['haufe'][i] = w,a
        state['row_hashes'].append(hashlib.sha256(samples['weight'][i].tobytes()+samples['haufe'][i].tobytes()).hexdigest())
        if (i+1)%10 == 0 or i+1 == count:
            for data in samples.values(): data.flush()
            write_json(out, 'work/bootstrap/'+key+'/checkpoint.json',state)
            print(f'  Bootstrap {key}: {i+1}/{count}',flush=True)
    results = {}
    for kind,data in samples.items():
        # Chunk by voxel to bound memory while retaining the full private distribution.
        stats = {name:np.empty(v,np.float32) for name in ('mean','sd','proportion_positive','proportion_negative')}
        for start in range(0,v,4096):
            block = np.asarray(data[:,start:start+4096],float)
            stats['mean'][start:start+4096] = block.mean(0)
            stats['sd'][start:start+4096] = block.std(0,ddof=1)
            stats['proportion_positive'][start:start+4096] = (block>0).mean(0)
            stats['proportion_negative'][start:start+4096] = (block<0).mean(0)
        stable = (stats['proportion_positive']>=spec['sign_stability'])|(stats['proportion_negative']>=spec['sign_stability'])
        stats['sign_stable'] = np.where(stable, stats['mean'], 0)
        for name,values in stats.items(): save_vector(out,key+'_bootstrap_'+kind+'_'+name,values,mask,reference)
        results[kind] = stats
    return results


def permutation_accuracies(x, scope, cohort, flips, mode='common'):
    """Refit all five folds; training and held-out labels share each participant's flip."""
    assert_development(scope)
    tasks, subjects = TASKS[cohort], scope['subjects']
    folds = np.array([scope['folds'][s] for s in subjects])
    correct = np.zeros((len(subjects), len(tasks)),bool)
    for fold in range(1,6):
        train, test = folds != fold, folds == fold
        require(train.any() and test.any(), 'empty permutation fold')
        require(not set(np.array(subjects)[train]) & set(np.array(subjects)[test]), 'permutation participant leakage')
        groups = [tuple(range(len(tasks)))] if mode == 'common' else [tuple(j for j in range(len(tasks)) if j != k) for k in range(len(tasks))]
        for g, training_tasks in enumerate(groups):
            w,b = fit_binary(x[train][:,training_tasks],scope,list(np.array(subjects)[train]),np.asarray(flips)[train])
            targets = range(len(tasks)) if mode == 'common' else [g]
            for j in targets:
                if mode == 'lopo': require(j not in training_tasks,'permutation LOPO task leakage')
                margins = (x[test,j,0].astype(float)-x[test,j,1].astype(float))@w
                correct[test,j] = margins*np.asarray(flips)[test]>0
    return correct.mean(0)


def permutation(out, scope, cohort, family, x, spec, fingerprint, mode='common'):
    count = spec['permutations']; key = cohort+'_'+family+'_'+mode
    relative = 'work/permutation/'+key+'.json'; path = out.output(relative)
    state = json.loads(path.read_text()) if path.exists() else {'fingerprint':fingerprint,'n':count,'null':[]}
    require(state['fingerprint'] == fingerprint and state['n'] == count and len(state['null'])<=count, 'permutation checkpoint changed')
    for i in range(len(state['null']),count):
        # Deliberately shared flips between common/LOPO and all tasks in a cohort/family.
        flips = replicate_rng('permutation:'+cohort+'_'+family,i).choice([-1,1],len(x))
        state['null'].append(permutation_accuracies(x,scope,cohort,flips,mode).tolist())
        write_json(out,relative,state)
        if (i+1)%10 == 0 or i+1 == count: print(f'  Permutation {key}: {i+1}/{count}',flush=True)
    null = np.array(state['null']); require(np.isfinite(null).all() and np.all((null>=0)&(null<=1)), 'invalid permutation values')
    summary=[]
    for j,task in enumerate(TASKS[cohort]):
        row = scope['performance'].query('family == @family and scope == @mode and test_task == @task').iloc[0]
        summary.append({'cohort':cohort,'family':family,'scope':mode,'test_task':task,'n':len(x),
                        'permutations':count,'observed_accuracy':float(row.accuracy),'null_mean':float(null[:,j].mean()),
                        'null_sd':float(null[:,j].std(ddof=1)), 'null_mean_mc_se':float(null[:,j].std(ddof=1)/np.sqrt(count)),
                        'null_q025':float(np.quantile(null[:,j],.025)), 'null_q975':float(np.quantile(null[:,j],.975)),
                        'empirical_p':float((1+(null[:,j]>=row.accuracy).sum())/(1+count)),
                        'chance_centering_flag':abs(float(null[:,j].mean())-.5)>.05})
    return summary, null


def clusters(values, mask, reference, minimum=20):
    volume = reconstruct(values,mask); rows=[]
    # Face-connected components, independently by sign; peak = bootstrap-mean Haufe value.
    for sign in (-1,1):
        labels,n = label(volume*sign>0, structure=generate_binary_structure(3,1))
        for k in range(1,n+1):
            indices=np.argwhere(labels==k)
            if len(indices)<minimum: continue
            vals=volume[tuple(indices.T)]; peak=indices[np.argmax(np.abs(vals))]
            xyz=nib.affines.apply_affine(reference.affine,peak)
            rows.append({'sign':sign,'voxel_count':len(indices),'peak_absolute_haufe':float(np.max(np.abs(vals))),
                         'peak_signed_haufe':float(volume[tuple(peak)]),'x':xyz[0],'y':xyz[1],'z':xyz[2],
                         'mean_haufe':float(vals.mean()),'connectivity':6,'inferential_p':'not_applicable'})
    return rows
