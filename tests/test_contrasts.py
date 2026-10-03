import numpy as np
import pytest
from signature_pilot import representations
from inventory import cope_path
from preflight import verify_l1, parse_fsf
from utils import PipelineError


def test_contrast_arithmetic():
    maps = {i: np.array([float(i*i)]) for i in range(1, 35)}
    sr = representations('sharedreward', maps)
    assert sr['reward'][0] == 9  # ((16-9)+(36-25))/2
    assert sr['reward'][1] == 3
    assert set(sr) == {'reward'}  # Full-trial slots must never be reinterpreted as decisions.
    trust = representations('trust', maps)
    assert trust['reward'][0] == 15
    assert trust['reward'][1] == 9
    assert trust['friend_stranger'][0] == 13
    assert trust['decision'][0] == 6.5 and trust['decision'][1] == 1
    from signature_pilot import doors_representations
    doors = doors_representations({3: 8, 4: 10}, {3: 2, 4: 7})
    assert doors == {'reward': (10, 7), 'decision': (8, 2)}
    ugr = representations('ugr', maps)
    assert ugr['ugr_pmod'][0] == 196 and ugr['ugr_pmod'][1] == 169
    assert ugr['ugr_constant'][0] == 37 and ugr['ugr_constant'][1] == 5


def test_doors_separate_l1(cfg):
    pos = cope_path(cfg, 'sub-fixture000', 'socialdoors', 4, 'L1', 1)
    neg = cope_path(cfg, 'sub-fixture000', 'doors', 4, 'L1', 1)
    assert pos != neg and 'L1_task-socialdoors' in str(pos) and 'L1_task-doors' in str(neg)
    assert str(pos).endswith('stats/cope4.nii.gz')
    with pytest.raises(PipelineError, match='separate L1'): cope_path(cfg, 'sub-fixture000', 'socialdoors', 4)


def test_current_template_vectors_not_just_names(cfg):
    spec = cfg.contrasts['sharedreward']
    from conftest import make_subject
    from inventory import feat_dir
    make_subject(cfg, 'sub-fixture000', secondary=False)
    values = parse_fsf(feat_dir(cfg, 'sub-fixture000', 'sharedreward', 'L1', 1)/'design.fsf')
    verify_l1(values, spec)
    values['fmri(con_real4.4)'] = '-1'
    with pytest.raises(PipelineError, match='weights'): verify_l1(values, spec)


def test_ugr_pmod_weights(cfg):
    assert cfg.contrasts['ugr']['copes'][13]['weights'] == {2: 1., 4: 1.}
    assert cfg.contrasts['ugr']['copes'][14]['weights'] == {6: 1., 8: 1.}


def test_ev_order_drift_fails(cfg):
    spec = cfg.contrasts['trust']
    values = parse_fsf(cfg.repos['trust'] / spec['template'])
    values['fmri(evtitle5)'] = 'C_def'
    with pytest.raises(PipelineError, match='EV ordering'): verify_l1(values, spec)


def test_l2_mean_contract(cfg):
    from preflight import verify_l2
    spec = cfg.contrasts['sharedreward']
    values = parse_fsf(cfg.repos['aging'] / 'templates/L2_task-sharedreward_model-fulltrial_type-act.fsf')
    verify_l2(values, spec)
    values['fmri(evg2.1)'] = '-1'
    with pytest.raises(PipelineError, match='intercept-only'): verify_l2(values, spec)
