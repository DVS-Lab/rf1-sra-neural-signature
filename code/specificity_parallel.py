"""Indexed synchronized participant permutations; restartable, RAM-bounded pool."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import os,json
import numpy as np
from threadpoolctl import threadpool_limits
from utils import write_json
from characterization_audit import require
from characterization_compute import replicate_rng
from cross_valence_parallel import available_memory
from specificity_design import MODES, ENDPOINTS, cv, endpoint_correct
_STATE=None

def initialize(scope,paths,keep):
    global _STATE
    _STATE=(scope,{k:np.load(p,mmap_mode='r') for k,p in paths.items()},keep,threadpool_limits(limits=1))

def one(mode,index):
    scope,arrays,keep,_=_STATE
    # Same flip for ALL a person's maps, tasks, masks and model variants, train AND test.
    flips=replicate_rng('specificity-permutation',index).choice([-1,1],len(scope['subjects']))
    return endpoint_correct(cv(scope,arrays,mode,keep,flips)).mean(0).tolist()

def run(out,scope,arrays,keep,key,requested=96,count=500):
    require(requested>0,'workers must be positive'); states={}
    for mode in MODES:
        p=out.output('work/permutations/'+mode+'.json')
        z=json.loads(p.read_text()) if p.exists() else dict(fingerprint=key,results={})
        require(z['fingerprint']==key,'permutation identity drift')
        for i,value in z['results'].items():
            require(str(int(i))==i and 0<=int(i)<count and len(value)==len(ENDPOINTS) and np.isfinite(value).all() and np.all((np.array(value)>=0)&(np.array(value)<=1)),'invalid permutation checkpoint')
        states[mode]=z
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    # Includes selected arrays, solver float64 copies and source/target scoring copies.
    peak=int(len(scope['subjects'])*2*arrays['outcome'].shape[-1]*80+512*1024**2); memory=available_memory()
    cap=int(.8*memory//peak) if memory else requested
    require(cap>=1,'insufficient RAM for one specificity worker'); workers=min(requested,cpus,cap)
    write_json(out,'provenance/parallel_execution.json',dict(requested_workers=requested,workers=workers,cpu_slots=cpus,
        available_memory_bytes=memory,estimated_peak_bytes_per_worker=peak,blas_threads_per_worker=1))
    jobs=((mode,i) for i in range(count) for mode in MODES if str(i) not in states[mode]['results'])
    done=sum(len(z['results']) for z in states.values()); total=count*len(MODES)
    print(f'Specificity permutations {done}/{total}; {workers} processes, requested {requested}',flush=True)
    def accept(mode,i,value):
        nonlocal done
        states[mode]['results'][str(i)]=value; done+=1
        write_json(out,'work/permutations/'+mode+'.json',states[mode])
        if done%10==0 or done==total:
            write_json(out,'provenance/permutation_progress.json',dict(completed=done,total=total,holdout_scored=False))
            print(f'  Specificity permutations {done}/{total}',flush=True)
    paths={k:str(a.filename) for k,a in arrays.items()}; minimal={k:scope[k] for k in ('subjects','guard','folds')}
    if workers==1:
        initialize(minimal,paths,keep)
        for mode,i in jobs: accept(mode,i,one(mode,i))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(minimal,paths,keep)) as pool:
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
    return np.array([[states[m]['results'][str(i)] for m in MODES] for i in range(count)])
