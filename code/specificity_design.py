"""Fixed final specificity design; no candidate/validation/UGR execution paths."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import warnings
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from utils import Config
from characterization_audit import require, PARAMETERS, SEED
from characterization_compute import assert_development, fit_binary, replicate_rng
from cross_valence_compute import fold_indices
from revised_design import center

MODES = ('legacy', 'outcome', 'dmn_only', 'dmn_excluded', 'generic')
TASKS = ('sharedreward', 'trust', 'socialdoors', 'sr_friend_stranger', 'trust_friend_stranger')
DOMAINS = tuple(t+'_'+v for t in TASKS for v in ('positive', 'negative'))
CELLS = tuple((i,j) for i in range(10) for j in range(10) if (i<6)==(j<6))
# Six directed cross-task transfers and the unchanged within-Doors positive control.
ENDPOINTS = tuple((a,b) for a in range(3) for b in range(3) if a != b or a == 2) + ((3,4),(4,3))

@dataclass
class SpecificityConfig(Config):
    def output(self, relative):
        p=Path(relative)
        require(not p.is_absolute() and p.parts and p.parts[0] in ('work','results','reports','provenance') and '..' not in p.parts,'specificity output escape')
        return super().output('/'.join([p.parts[0],'revised','specificity',*p.parts[1:]]))

def output_config(base):
    return SpecificityConfig(**{k:getattr(base,k) for k in Config.__dataclass_fields__})

def generic_template(generic, train, task):
    # Equal conditions/people, source task only. No test subjects or class labels.
    w=np.asarray(generic[train,task],float).mean(0)
    w-=w.mean(); norm=np.linalg.norm(w)
    require(np.isfinite(norm) and norm>0,'degenerate generic task template')
    return w/norm

def fit_generic(x, scope, subjects, flips=None):
    """Same L2 squared-hinge C=1 objective, solved in two primal dimensions (w,b).

    No rescaling, parameter search, warning suppression or accuracy-based fallback.
    Use this solver for every generic observed/permuted fit, not just failures.
    """
    assert_development(scope,subjects)
    require(x.ndim==4 and len(x)==len(subjects) and x.shape[2:]==(2,1),
            'generic baseline requires participant/task/pair/one-feature input')
    require(np.isfinite(x).all(),'nonfinite generic projections')
    y=np.broadcast_to(np.array([1,-1]),x.shape[:3]).copy()
    if flips is not None:
        flips=np.asarray(flips)
        require(flips.shape==(len(subjects),) and np.isin(flips,[-1,1]).all(),'invalid generic participant flips')
        y*=flips.astype(int)[:,None,None]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        estimator=LinearSVC(**{**PARAMETERS,'dual':False},random_state=SEED).fit(x.reshape(-1,1),y.reshape(-1))
    return estimator.coef_[0],float(estimator.intercept_[0])


def cv(scope, arrays, mode, keep, flips=None):
    """Social 6x6 and friend–stranger 4x4 cells use fixed participant folds and fixed C=1 LinearSVC."""
    assert_development(scope); require(mode in MODES,'unknown specificity model')
    x=arrays['legacy' if mode=='legacy' else 'outcome']
    require(x.shape[:3]==(len(scope['subjects']),10,2),'invalid specificity features')
    chosen=None
    if mode in ('dmn_only','dmn_excluded'):
        chosen=keep if mode=='dmn_only' else ~keep
        require(chosen.any(),'empty fixed mask')
    result=np.zeros((len(x),10,10))
    for fold in range(1,6):
        train,test=fold_indices(scope,fold); ids=[scope['subjects'][i] for i in train]
        for source in range(10):
            targets=np.arange(6) if source<6 else np.arange(6,10)
            a=np.asarray(x[train,source:source+1]); b=np.asarray(x[test[:,None],targets[None,:]])
            if chosen is not None:
                a=center(a[...,chosen]); b=center(b[...,chosen])
            if mode=='generic':
                template=generic_template(arrays['generic'],train,source//2 if source<6 else (source-6)//2)
                a=(a.astype(float)@template)[...,None]
                b=(b.astype(float)@template)[...,None]
            fitter=fit_generic if mode=='generic' else fit_binary
            w,intercept=fitter(a,scope,ids,None if flips is None else flips[train])
            scores=b.astype(float)@w+intercept
            margins=scores[:,:,0]-scores[:,:,1]
            if flips is not None: margins*=flips[test,None]
            result[test[:,None],source,targets[None,:]]=margins
    require(np.isfinite(result).all(),'nonfinite specificity margins')
    return result

def endpoint_correct(margins):
    # Average four train/test-valence correctness values within each participant.
    return np.stack([(margins[:,2*a:2*a+2,2*b:2*b+2]>0).mean((1,2)) for a,b in ENDPOINTS],axis=1)

def interval(values, key, count=10000):
    values=np.asarray(values,float); rng=replicate_rng('specificity-bootstrap:'+key,0); draws=[]
    for start in range(0,count,200):
        idx=rng.integers(0,len(values),(min(200,count-start),len(values)))
        draws.extend(values[idx].mean(1))
    return tuple(float(v) for v in np.quantile(draws,[.025,.975]))
