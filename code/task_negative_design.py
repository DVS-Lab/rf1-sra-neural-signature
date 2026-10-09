"""Fixed final empirical deactivation control; no alternate templates or tuning."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from utils import Config
from characterization_audit import require
from characterization_compute import assert_development,fit_binary,correlation
from specificity_design import fit_generic
from cross_valence_compute import fold_indices

REFERENCE='monetary_doors_decision_task_negative'
MODES=('original','task_negative','residualized')
ENDPOINTS=(('sharedreward','trust',0,2,'primary'),('trust','sharedreward',2,0,'primary'),
           ('sr_friend_stranger','trust_friend_stranger',6,8,'secondary'),
           ('trust_friend_stranger','sr_friend_stranger',8,6,'secondary'),
           ('sharedreward','socialdoors',0,4,'secondary_shared_task_context'),
           ('trust','socialdoors',2,4,'secondary_shared_task_context'))

@dataclass
class TaskNegativeConfig(Config):
    def output(self,relative):
        p=Path(relative)
        require(not p.is_absolute() and p.parts and p.parts[0] in ('work','results','reports','provenance') and '..' not in p.parts,'task-negative output escape')
        target=super().output('/'.join([p.parts[0],'revised','task_negative_control',*p.parts[1:]]))
        namespace=self.root.resolve()/p.parts[0]/'revised/task_negative_control'
        require(target.is_relative_to(namespace),'task-negative namespace symlink escape')
        return target

def output_config(base):
    return TaskNegativeConfig(**{k:getattr(base,k) for k in Config.__dataclass_fields__})

def negative_template(raw):
    """Select negative task effects BEFORE centering; never use social labels."""
    raw=np.asarray(raw,dtype=float)
    require(raw.ndim==2 and len(raw)>0 and np.isfinite(raw).all(),'invalid raw decision COPEs')
    mean=raw.mean(0); negative=np.minimum(mean,0.)
    require(np.any(negative<0),'no negative decision effects; stop before interpreting classifiers')
    centered=negative-negative.mean(); norm=np.linalg.norm(centered)
    require(np.isfinite(norm) and norm>0,'constant/degenerate negative template; stop before classifiers')
    return mean,negative,centered/norm

def remove_direction(x,template):
    x=np.asarray(x,float); t=np.asarray(template,float)
    require(np.isfinite(x).all() and np.isfinite(t).all() and abs(t.mean())<1e-10 and np.isclose(t@t,1,atol=1e-10),'invalid projection geometry')
    result=x-(x@t)[...,None]*t
    require(np.max(abs(result@t)) <= 1e-9*max(1.,float(np.max(np.linalg.norm(x,axis=-1)))),'residual projection nonzero')
    return result

def train_template(scope,raw,fold):
    assert_development(scope); train,test=fold_indices(scope,fold)
    require(raw.shape[0]==len(scope['subjects']),'decision sample mismatch')
    return negative_template(raw[train])

def score_fold(scope,x,template,fold,mode,flips=None,central_only=False):
    """Four ordered train/test-valence cells per endpoint; same participant folds."""
    assert_development(scope); require(mode in MODES,'unknown empirical control')
    train,test=fold_indices(scope,fold); endpoints=ENDPOINTS[:2] if central_only else ENDPOINTS
    result=np.empty((len(test),len(endpoints),4)); cache={}
    for ei,(_,_,a,b,_) in enumerate(endpoints):
        for v in range(2):
            source=a+v
            if source not in cache:
                features=np.asarray(x[train,source:source+1],float)
                if mode=='residualized': features=remove_direction(features,template)
                if mode=='task_negative': features=(features@template)[...,None]
                fitter=fit_generic if mode=='task_negative' else fit_binary
                cache[source]=fitter(features,scope,[scope['subjects'][i] for i in train],None if flips is None else flips[train])
            w,intercept=cache[source]; target=np.asarray(x[test,b:b+2],float)
            if mode=='residualized': target=remove_direction(target,template)
            if mode=='task_negative': target=(target@template)[...,None]
            scores=target@w+intercept; margins=scores[:,:,0]-scores[:,:,1]
            if flips is not None: margins*=flips[test,None]
            result[:,ei,2*v:2*v+2]=margins
    require(np.isfinite(result).all(),'nonfinite task-negative predictions')
    return result

def old_margins(old):
    return np.stack([old[:,a:a+2,b:b+2].reshape(len(old),4) for _,_,a,b,_ in ENDPOINTS],axis=1)

def sanity(mean,negative,template,dmn):
    require(dmn.any() and (~dmn).any(),'empty anatomical reference region')
    magnitude=-negative; total=magnitude.sum()
    return dict(negative_fraction=float((mean<0).mean()),dmn_negative_fraction=float((mean[dmn]<0).mean()),
        non_dmn_negative_fraction=float((mean[~dmn]<0).mean()),dmn_mean_negative_magnitude=float(magnitude[dmn].mean()),
        non_dmn_mean_negative_magnitude=float(magnitude[~dmn].mean()),dmn_negative_magnitude_fraction=float(magnitude[dmn].sum()/total),
        mean_signed_cope=float(mean.mean()),template_norm=float(np.linalg.norm(template)),
        limited_negative_support=bool((mean<0).mean()<.01),dmn_negativity_enriched=bool((mean[dmn]<0).mean()>(mean[~dmn]<0).mean()))
