"""Deterministic indexed permutations shared across a bounded global process pool."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os,json
import numpy as np
from threadpoolctl import threadpool_limits
from utils import write_json
from characterization_audit import require
from characterization_compute import assert_development,replicate_rng
from cross_valence_parallel import available_memory
from final_stress_compute import fit_score,identifier
from final_stress_design import permutation_targets
_STATE=None

def initialize(scope,paths,keep):
    global _STATE
    assert_development(scope)
    _STATE=(scope,{k:np.load(v,mmap_mode='r') for k,v in paths.items()},keep,threadpool_limits(limits=1))

def one(p,iteration):
    scope,arrays,keep,_=_STATE
    flips=replicate_rng('final_stress:'+p['family'],iteration).choice([-1,1],len(scope['subjects']))
    p={**p,'test':permutation_targets(p)}; margins=[]
    for fold in range(1,6): margins.extend(fit_score(scope,arrays,p,fold,keep,flips)[-1])
    return float((np.array(margins)[:,0]>0).mean())

def run(out,scope,arrays,plans,keep,key,requested=96,count=500):
    ps=[p for p in plans if p['permutation']]; states={}; by={identifier(p):p for p in ps}
    if not ps: return []
    for name,p in by.items():
        path=out.output('work/permutations/'+name+'.json')
        state=json.loads(path.read_text()) if path.exists() else dict(fingerprint=key,plan=p,results={})
        require(state['fingerprint']==key and state['plan']==p,'permutation checkpoint changed')
        for i,value in state['results'].items(): require(str(int(i))==i and 0<=int(i)<count and 0<=value<=1,'invalid permutation checkpoint')
        states[name]=state
    require(requested>0,'workers must be positive')
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    peak=max(len(p['subjects'])*len(p['train'])*2*arrays[p['train'][0]].shape[-1]*40 for p in ps)+256*1024**2
    memory=available_memory(); cap=int(.8*memory//peak) if memory else requested
    require(cap>=1,'insufficient available RAM'); workers=min(cpus,requested,cap)
    write_json(out,'provenance/parallel_execution.json',dict(requested_workers=requested,workers=workers,cpu_slots=cpus,available_memory_bytes=memory,estimated_peak_bytes_per_worker=peak,blas_threads_per_worker=1))
    jobs=((name,i) for i in range(count) for name in by if str(i) not in states[name]['results'])
    done=sum(len(s['results']) for s in states.values()); total=len(ps)*count
    print(f'Permutations: {done}/{total}; {workers} processes (requested {requested}; RAM limited)',flush=True)
    def accept(name,i,value):
        nonlocal done
        states[name]['results'][str(i)]=value; done+=1
        write_json(out,'work/permutations/'+name+'.json',states[name])
        if done%10==0 or done==total:
            write_json(out,'provenance/permutation_progress.json',dict(completed=done,total=total,workers=workers,holdout_scored=False))
            print(f'  Permutations {done}/{total}',flush=True)
    paths={k:str(a.filename) for k,a in arrays.items() if any(k in p['train']+permutation_targets(p) for p in ps)}
    minimal={k:scope[k] for k in ('subjects','folds','guard')}
    if workers==1:
        initialize(minimal,paths,keep)
        for name,i in jobs: accept(name,i,one(by[name],i))
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(minimal,paths,keep)) as pool:
            pending={}
            def fill():
                while len(pending)<2*workers:
                    try: name,i=next(jobs)
                    except StopIteration: break
                    pending[pool.submit(one,by[name],i)]=(name,i)
            fill()
            while pending:
                ready,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in ready:
                    name,i=pending.pop(future); accept(name,i,future.result())
                fill()
    write_json(out,'provenance/permutation_progress.json',dict(completed=total,total=total,workers=workers,holdout_scored=False))
    return [(name,permutation_targets(by[name])[0],np.array([states[name]['results'][str(i)] for i in range(count)])) for name in by]
