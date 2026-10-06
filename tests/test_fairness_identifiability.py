"""Exercise event-derived dependence, which random design columns cannot detect."""
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from final_stress_fairness import (
    CONDITIONS, TASK_EVS, build_events, contrast_matrix, design_qc, fsl_matrix, fsf_text,
)


def event_design():
    # Same marginal counts as the failed preview, but entirely invented timing.
    rng=np.random.default_rng(17)
    cells=[(name,32 if i<high else 16) for name,high in zip(CONDITIONS,(7,7,7,6)) for i in range(12)]
    rng.shuffle(cells)
    trials=[]; rows=[]; onset=4.
    for i,(name,endowment) in enumerate(cells):
        social,fair=name.split('_'); rt=float(rng.uniform(.5,2.5))
        trials.append(SimpleNamespace(trial_id=str(i),sociality=social,endowment=endowment,
            offer=(2 if endowment==32 else 1) if fair=='unfair' else endowment//2,
            missed=False,broad_onset=onset,broad_duration=5.+rt,
            response_onset=onset+3.+rt,response_time=rt))
        rows.append(dict(trial_id=str(i),trial_type='decision',onset=onset+3.))
        onset+=float(rng.uniform(9.,13.))
    ev,counts,_,failures=build_events(rows,SimpleNamespace(collapse_trials=lambda _:trials))
    assert not failures
    assert [counts[k] for k in CONDITIONS]==[12]*4
    assert (counts['endowment_high'],counts['endowment_low'])==(27,21)
    old={k:ev[k] for k in CONDITIONS}
    for name,sign in [('endowment_high',1),('endowment_low',-1)]:
        old[name]=[(a,b,1.) for a,b,c in ev['endowment_difference'] if c*sign>0]
    old.update({k:v for k,v in ev.items() if k not in TASK_EVS})
    return ev,old


def test_event_partition_dependence_and_signed_endowment_recovery():
    ev,old=event_design()
    times=np.arange(0,600,.1)
    def matrix(events):
        return np.column_stack([sum((amp*((times>=onset)&(times<onset+duration))
            for onset,duration,amp in values),np.zeros(len(times))) for values in events.values()])
    bad=matrix(old)[:,:6]; good=matrix(ev)[:,:5]
    np.testing.assert_allclose(bad[:,:4].sum(1),bad[:,4:].sum(1))
    assert np.linalg.matrix_rank(bad)==5
    assert np.linalg.matrix_rank(good)==5
    np.testing.assert_allclose(good[:,4],.5*(bad[:,4]-bad[:,5]))
    # Endowment imbalance must not contaminate the four adjusted condition means.
    beta=np.array([1.,2.,3.,4.,8.])
    fitted=np.linalg.lstsq(good,good@beta,rcond=None)[0]
    np.testing.assert_allclose(fitted,beta,atol=1e-10)
    np.testing.assert_allclose(contrast_matrix(5)@fitted,[1,2,3,4,-1,-1,0,8],atol=1e-10)


@pytest.mark.skipif(shutil.which('feat_model') is None or not os.environ.get('FSLDIR'),reason='FSL feat_model required')
def test_fsl_rendered_old_model_fails_and_signed_model_passes(tmp_path):
    ev,old=event_design()
    rng=np.random.default_rng(91)
    confounds=tmp_path/'confounds.txt'; np.savetxt(confounds,rng.normal(size=(400,5)))
    settings={'fmri(level)':1,'fmri(mixed_yn)':2,'fmri(tr)':1.615,'fmri(npts)':400,'fmri(ndelete)':0,
        'fmri(evs_vox)':0,'fmri(temphp_yn)':0,'fmri(templp_yn)':0,'fmri(paradigm_hp)':100,
        'fmri(critical_z)':5.3,'fmri(noise)':.66,'fmri(noisear)':.34}
    for name,events in [('old',old),('signed',ev)]:
        folder=tmp_path/name; folder.mkdir()
        paths={k:folder/(k+'.txt') for k in events}
        for key,values in events.items():
            np.savetxt(paths[key],np.asarray(values).reshape(-1,3),fmt='%.8f')
        fsf=folder/'design.fsf'
        fsf.write_text(fsf_text(settings,events,paths,folder/'output.feat'))
        result=subprocess.run(['feat_model',str(fsf.with_suffix('')),str(confounds)],capture_output=True,text=True,timeout=60)
        assert result.returncode==0,result.stdout+result.stderr
        x=fsl_matrix(folder/'design.mat'); c=fsl_matrix(folder/'design.con')
        metrics,_=design_qc(x,contrast_matrix(len(events)))
        if name=='old':
            assert metrics['failures'],'The duplicated all-trial signal must be rejected'
        else:
            assert not metrics['failures'],metrics
            np.testing.assert_allclose(c[:,:len(events)],contrast_matrix(len(events)))
            assert np.all(c[:,len(events):]==0)
            assert metrics['rank']==metrics['active_columns']
            print('Signed-endowment FSL check:',metrics)
