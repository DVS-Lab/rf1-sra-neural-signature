"""Only primary SR↔Trust nulls; reuse matching original nulls, refit controls."""
import os,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from utils import write_json
from characterization_audit import require
from characterization_compute import replicate_rng,assert_development
from cross_valence_parallel import available_memory
from task_negative_design import MODES,ENDPOINTS,score_fold
_STATE=None

def initialize(scope,path,directions):
    global _STATE
    assert_development(scope); _STATE=(scope,np.load(path,mmap_mode='r'),np.asarray(directions),threadpool_limits(limits=1))

def one(mode,index):
    scope,x,directions,_=_STATE
    flips=replicate_rng('specificity-permutation',index).choice([-1,1],len(scope['subjects']))
    correct=[]
    for fold in range(1,6): correct.extend((score_fold(scope,x,directions[fold-1],fold,mode,flips,central_only=True)>0).mean(2))
    return np.asarray(correct).mean(0).tolist()

def original_null(base,count=500):
    frame=pd.read_csv(base.root/'results/revised/specificity/aggregate/permutation_nulls.tsv',sep='\t'); result=[]
    for a,b,_,_,_ in ENDPOINTS[:2]:
        z=frame[(frame['mode']=='outcome')&(frame.train==a)&(frame.test==b)].sort_values('iteration')
        require(len(z)==count and np.array_equal(z.iteration,np.arange(count)),'missing/duplicate original null index')
        result.append(z.accuracy.to_numpy())
    return np.stack(result,axis=1)

def run(out,scope,x,directions,key,requested=96,count=500):
    original=original_null(scope['c'],count); states={}; variants=MODES[1:]
    assert_development(scope)
    audit_path=out.output('work/permutations/original_reproduction.json')
    if audit_path.exists():
        check=json.loads(audit_path.read_text()); require(check['fingerprint']==key and check['passed'],'original null audit drift')
    else:
        initialize({k:scope[k] for k in ('subjects','guard','folds')},str(x.filename),directions)
        checks=[]
        for i in range(min(2,count)):
            value=one('original',i)
            require(np.allclose(value,original[i],atol=1e-12,rtol=0),'original permutation predictions differ')
            checks.append(dict(iteration=i,max_absolute_error=float(np.max(abs(np.asarray(value)-original[i])))))
        write_json(out,'work/permutations/original_reproduction.json',dict(fingerprint=key,passed=True))
        from utils import write_tsv
        write_tsv(out,'results/aggregate/original_null_reconstruction.tsv',checks)
    for mode in variants:
        path=out.output('work/permutations/'+mode+'.json')
        state=json.loads(path.read_text()) if path.exists() else dict(fingerprint=key,results={})
        require(state['fingerprint']==key,'permutation checkpoint drift')
        for i,v in state['results'].items(): require(str(int(i))==i and 0<=int(i)<count and len(v)==2 and np.isfinite(v).all() and np.all((np.asarray(v)>=0)&(np.asarray(v)<=1)),'invalid empirical null')
        states[mode]=state
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    memory=available_memory(); peak=int(len(x)*2*x.shape[-1]*80+512*1024**2)
    cap=int(.8*memory//peak) if memory else requested; require(requested>0 and cap>0,'insufficient workers/RAM')
    workers=min(requested,cpus,cap); write_json(out,'provenance/parallel.json',dict(requested=requested,workers=workers,estimated_worker_bytes=peak,available_memory_bytes=memory,blas_threads=1))
    jobs=((mode,i) for i in range(count) for mode in variants if str(i) not in states[mode]['results'])
    done=sum(len(z['results']) for z in states.values()); total=len(variants)*count
    print(f'Task-negative permutations: {done}/{total}; {workers} processes; original nulls reused',flush=True)
    def accept(mode,i,value):
        nonlocal done
        states[mode]['results'][str(i)]=value; done+=1
        write_json(out,'work/permutations/'+mode+'.json',states[mode])
        if done%10==0 or done==total:
            write_json(out,'provenance/permutation_progress.json',dict(completed=done,total=total,holdout_scored=False))
            print(f'  Task-negative permutations {done}/{total}',flush=True)
    minimal={k:scope[k] for k in ('subjects','guard','folds')}
    if workers==1:
        initialize(minimal,str(x.filename),directions)
        for mode,i in jobs: accept(mode,i,one(mode,i))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(minimal,str(x.filename),directions)) as pool:
            pending={}
            def fill():
                while len(pending)<2*workers:
                    try: mode,i=next(jobs)
                    except StopIteration: break
                    pending[pool.submit(one,mode,i)]=(mode,i)
            fill()
            while pending:
                ready,_=wait(pending,return_when=FIRST_COMPLETED)
                for f in ready:
                    mode,i=pending.pop(f); accept(mode,i,f.result())
                fill()
    return np.stack([original,*[np.array([states[m]['results'][str(i)] for i in range(count)]) for m in variants]],axis=1)
