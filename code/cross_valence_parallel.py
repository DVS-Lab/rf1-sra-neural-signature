"""Bounded process pool for deterministic, independently checkpointed permutations."""
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import multiprocessing as mp
import os
import json
import numpy as np
from threadpoolctl import threadpool_limits
from characterization_audit import require
from characterization_compute import assert_development,replicate_rng
from cross_valence_compute import fit_fold,score_fold
from cross_valence_design import permutation_plans
from utils import write_json

_WORKER=None


def available_memory():
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except ImportError:
        try: return int(os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE'))
        except (ValueError,OSError): return None


def worker_budget(scope,arrays,requested):
    require(isinstance(requested,int) and requested>0,'worker count must be a positive integer')
    cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count() or 1
    # Conservative estimate: selected float32 rows + several float64 solver copies,
    # process/import overhead. Shared read-only .npy pages are not copied per job.
    folds=np.array([scope['folds'][s] for s in scope['subjects']]); peak=0
    for p in permutation_plans():
        x=arrays[p['key']]; train=max(int((folds!=f).sum()) for f in range(1,6))
        peak=max(peak,train*len(p['train'])*2*x.shape[-1]*32+256*1024**2)
    memory=available_memory(); cap=int(.8*memory//peak) if memory is not None else requested
    require(cap>=1,'not enough currently available RAM for even one permutation worker')
    used=min(requested,cpus,cap)
    return dict(requested_workers=requested,cpu_slots=cpus,available_memory_bytes=memory,
                estimated_peak_bytes_per_worker=peak,workers=used,blas_threads_per_worker=1,
                scheduling='one whole five-fold permutation/model job per process; bounded shared pool across all training specifications',
                note='80% of available RAM limits worker count; estimate is conservative, not a measured guarantee')


def initialize(scope,paths):
    global _WORKER
    assert_development(scope)
    limiter=threadpool_limits(limits=1)
    arrays={k:np.load(path,mmap_mode='r',allow_pickle=False) for k,path in paths.items()}
    _WORKER=(scope,arrays,limiter)


def one_permutation(p,iteration):
    scope,arrays,_=_WORKER; assert_development(scope)
    x=arrays[p['key']]; n=len(x)
    flips=replicate_rng('cross_valence:partner_pair:labels',iteration).choice([-1,1],n)
    margins=np.empty((n,len(p['test'])))
    for fold in range(1,6):
        w,b,train,test=fit_fold(scope,x,p,fold,flips)
        margins[test]=score_fold(scope,x,p,w,b,train,test,flips)
    accuracy=(margins>0).mean(0)
    return [float((margins>0).mean(axis=1).mean())] if p['scope']=='pooled_cross_valence' else accuracy.tolist()


def run_permutations(out,scope,arrays,spec,fingerprint,requested_workers=96):
    assert_development(scope); plans=permutation_plans(); count=spec['permutations']
    states={}; plan_by_key={}
    for p in plans:
        key=p['family']+'_'+p['name']; plan_by_key[key]=p
        path=out.output('work/permutations/'+key+'.json')
        state=json.loads(path.read_text()) if path.exists() else dict(fingerprint=fingerprint,plan=p,results={})
        require(state['fingerprint']==fingerprint and state['plan']==p,'permutation checkpoint contract changed')
        require('results' in state,'checkpoint predates indexed parallel execution; do not mix computations')
        length=1 if p['scope']=='pooled_cross_valence' else len(p['test'])
        for i,value in state['results'].items():
            require(str(int(i))==i and 0<=int(i)<count,'invalid permutation index')
            require(len(value)==length and np.isfinite(value).all() and np.all((np.array(value)>=0)&(np.array(value)<=1)),'invalid permutation result')
        states[key]=state
    keys=list(states); total=count*len(plans); done=sum(len(s['results']) for s in states.values())
    budget=worker_budget(scope,arrays,requested_workers)
    write_json(out,'provenance/parallel_execution.json',budget)
    print(f'  Permutations: {done}/{total} complete; {budget["workers"]} workers, one BLAS thread each (requested {requested_workers})',flush=True)
    if budget['workers']<requested_workers:
        print(f'  Worker cap: CPU slots={budget["cpu_slots"]}; available RAM={budget["available_memory_bytes"]}; estimated bytes/worker={budget["estimated_peak_bytes_per_worker"]}',flush=True)
    jobs=((key,i) for i in range(count) for key in keys if str(i) not in states[key]['results'])
    def accept(key,i,value):
        nonlocal done
        states[key]['results'][str(i)]=value
        # Only the parent writes; atomic replacement prevents torn/racing checkpoints.
        write_json(out,'work/permutations/'+key+'.json',states[key]); done+=1
        if done%10==0 or done==total:
            write_json(out,'provenance/permutation_progress.json',dict(completed=done,total=total,workers=budget['workers'],holdout_scored=False))
            print(f'  Permutations: {done}/{total} complete',flush=True)
    if done<total:
        # Minimize worker state: no source paths, source configs, public writer, or held-out data.
        minimal={k:scope[k] for k in ('subjects','guard','folds')}
        paths={k:str(x.filename) for k,x in arrays.items() if k in {p['key'] for p in plans}}
        require(len(paths)==len({p['key'] for p in plans}),'permutation inputs must be verified private memory maps')
        if budget['workers']==1:
            initialize(minimal,paths)
            for key,i in jobs: accept(key,i,one_permutation(plan_by_key[key],i))
        else:
            # Spawn works on Linux and macOS and avoids forking initialized BLAS state.
            with ProcessPoolExecutor(max_workers=budget['workers'],mp_context=mp.get_context('spawn'),
                                     initializer=initialize,initargs=(minimal,paths)) as pool:
                pending={}
                def fill():
                    while len(pending)<2*budget['workers']:
                        try: key,i=next(jobs)
                        except StopIteration: break
                        pending[pool.submit(one_permutation,plan_by_key[key],i)]=(key,i)
                fill()
                try:
                    while pending:
                        ready,_=wait(pending,return_when=FIRST_COMPLETED)
                        for future in ready:
                            key,i=pending.pop(future); accept(key,i,future.result())
                        fill()
                except BaseException:
                    for future in pending: future.cancel()
                    raise
    result={}
    for key,state in states.items():
        require(len(state['results'])==count,'incomplete permutation set')
        state['null']=[state['results'][str(i)] for i in range(count)]
        write_json(out,'work/permutations/'+key+'.json',state)
        result[key]=np.array(state['null'])
    write_json(out,'provenance/permutation_progress.json',dict(completed=total,total=total,workers=budget['workers'],holdout_scored=False))
    return result
