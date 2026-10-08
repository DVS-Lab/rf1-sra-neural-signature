"""Regression for real-scale 1-D nonconvergence and safe production recovery."""
import json
import warnings
from copy import deepcopy
import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from utils import PipelineError,sha256,write_json
from characterization_audit import digest
from characterization_compute import fit_binary
from specificity_design import fit_generic,output_config
from specificity_restart import prepare,REUSABLE
from specificity import checkpoint
from test_characterization import toy_scope


def test_primal_same_objective_as_converged_dual_and_permuted_labels():
    scope=toy_scope(40); rng=np.random.default_rng(4)
    x=rng.normal(size=(40,1,2,1))
    for flips in (None,rng.choice([-1,1],40)):
        a,b=fit_binary(x,scope,scope['subjects'],flips)
        w,c=fit_generic(x,scope,scope['subjects'],flips)
        np.testing.assert_allclose(x@a+b,x@w+c,atol=2e-4,rtol=2e-4)
    with pytest.raises(PipelineError,match='HOLDOUT LOCK'):
        fit_generic(x[:1],scope,['sub-held000'])


def test_large_projection_dual_failure_primal_matches_independent_objective():
    scope=toy_scope(140); rng=np.random.default_rng(44)
    x=rng.normal(500,1000,(140,1,2,1))
    # This scale provokes the production exception without needing private images.
    with pytest.raises(ConvergenceWarning): fit_binary(x,scope,scope['subjects'])
    for flips in (None,rng.choice([-1,1],140)):
        w,b=fit_generic(x,scope,scope['subjects'],flips)
        y=np.tile([1,-1],140)
        if flips is not None: y*=np.repeat(flips,2)
        design=np.column_stack([x.reshape(-1),np.ones(x.size)])
        # Independent reference changes optimization coordinates only, not the
        # original feature scale, C, or regularization (including intercept).
        scale=np.array([1/np.std(x),1.])
        def fun(beta):
            theta=beta*scale; residual=np.maximum(0,1-y*(design@theta))
            return .5*(theta@theta)+residual@residual
        def jac(beta):
            theta=beta*scale; residual=np.maximum(0,1-y*(design@theta))
            return (theta-2*design.T@(y*residual))*scale
        reference=minimize(fun,np.zeros(2),jac=jac,method='BFGS',options={'gtol':1e-8})
        actual=np.array([w[0],b])/scale
        assert fun(actual)-reference.fun<1e-5
        np.testing.assert_allclose(design@np.array([w[0],b]),design@(reference.x*scale),atol=3e-4)


def fixture_state(cfg):
    out=output_config(cfg)
    contract=json.loads((cfg.root/'config/specificity_solver_restart.json').read_text())
    old=dict(code=contract['old_code'],subjects=['sub-development'],qc='same',software={'sklearn':'same'},source='same')
    oldkey=digest(old); write_json(out,'work/identity.json',dict(fingerprint=oldkey,**old))
    for m in REUSABLE: checkpoint(out,oldkey,m,lambda:np.ones((2,2)))
    hashes={}
    for m in ('legacy','outcome','generic'):
        p=out.output('work/features/'+m+'.npy'); p.parent.mkdir(parents=True,exist_ok=True); np.save(p,np.ones((2,2)))
        hashes[m]=sha256(p)
    write_json(out,'work/features/complete.json',dict(fingerprint=oldkey,hashes=hashes))
    new={**old,'code':{'specificity_design.py':'primal'}}
    return out,oldkey,new


def test_restart_preserves_four_fits_and_features_and_resumes_new_generic(cfg):
    out,oldkey,new=fixture_state(cfg)
    old_files={str(p):sha256(p) for p in out.output('work').rglob('*') if p.is_file()}
    assert prepare(out,new,dry_run=True)==oldkey
    assert not out.output('work/solver_restart.json').exists()
    assert prepare(out,new)==oldkey
    for p,h in old_files.items(): assert sha256(p)==h
    for m in REUSABLE: checkpoint(out,oldkey,m,lambda:pytest.fail('completed model refit'))
    newkey=digest(new); checkpoint(out,newkey,'generic',lambda:np.ones((2,2))*2)
    write_json(out,'work/permutations/generic.json',dict(fingerprint=newkey,results={}))
    assert prepare(out,new)==oldkey
    checkpoint(out,newkey,'generic',lambda:pytest.fail('new generic unnecessarily refit'))
    with pytest.raises(PipelineError,match='implementation changed'):
        prepare(out,{**new,'code':{'specificity_design.py':'another repair'}})


@pytest.mark.parametrize('problem',['data','software','old_code','identity','feature','observed','generic','null'])
def test_restart_fails_closed_on_unrelated_drift_or_unreviewed_products(cfg,problem):
    out,oldkey,new=fixture_state(cfg)
    if problem=='data': new['qc']='changed'
    elif problem=='software': new['software']={'sklearn':'changed'}
    elif problem=='old_code':
        p=out.output('work/identity.json'); old=json.loads(p.read_text()); old.pop('fingerprint'); old['code']={}; old['fingerprint']=digest(old); p.write_text(json.dumps(old))
    elif problem=='identity':
        p=out.output('work/identity.json'); old=json.loads(p.read_text()); old['fingerprint']='bad'; p.write_text(json.dumps(old))
    elif problem=='feature': out.output('work/features/generic.npy').write_bytes(b'damaged')
    elif problem=='observed': out.output('work/observed/outcome.npy').write_bytes(b'damaged')
    elif problem=='generic': checkpoint(out,oldkey,'generic',lambda:np.ones((2,2)))
    elif problem=='null': write_json(out,'work/permutations/legacy.json',dict(fingerprint=oldkey,results={}))
    with pytest.raises(PipelineError): prepare(out,new)
    assert not out.output('work/solver_restart.json').exists()


def test_fresh_identity_and_regular_resume(cfg):
    out=output_config(cfg); identity=dict(code={'current':'hash'},subjects=['sub-development'])
    assert prepare(out,identity,dry_run=True)==digest(identity)
    assert not out.output('work/identity.json').exists()
    assert prepare(out,identity)==digest(identity)
    assert prepare(out,identity)==digest(identity)
