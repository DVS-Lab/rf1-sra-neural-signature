"""Final development stress tests: fixed definitions, separate from frozen candidates."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from utils import Config, PipelineError
from characterization_audit import require
from revised_design import center

FAMILIES=('social_context','friend_stranger_context')
VISUAL_IDS=(22,23,24,32,36,40,47,48)

@dataclass
class StressConfig(Config):
    def output(self,relative):
        p=Path(relative)
        if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0] not in ('work','results','reports','provenance'):
            raise PipelineError('Stress-test outputs must stay in their namespace')
        return super().output('/'.join([p.parts[0],'revised','final_stress_tests',*p.parts[1:]]))

def output_config(base):
    return StressConfig(**{k:getattr(base,k) for k in Config.__dataclass_fields__})

def partners(task,maps):
    pairs=((1,2),(3,4),(5,6)) if task=='sharedreward' else ((4,5),(6,7),(8,9))
    return center(np.stack([(maps[a]+maps[b])/2 for a,b in pairs]))

def context(partner_maps,family):
    c,f,s=partner_maps
    return center([(f+s)/2,c] if family=='social_context' else [f,s])

def ugr_context(maps):
    return {'ugr_context':center([(maps[5]+maps[7])/2,(maps[1]+maps[3])/2]),
            'ugr_high':center([maps[5],maps[1]]),'ugr_low':center([maps[7],maps[3]])}

def trust_norm(maps):
    return {'trust_norm':center([(maps[6]+maps[8])/2,(maps[7]+maps[9])/2]),
            'trust_computer_norm':center([maps[4],maps[5]])}

def reduced(x,keep):
    return center(np.asarray(x)[...,keep])

def plan(name,family,train,test,subjects,visual=False,permutation=False,group='context'):
    return dict(name=name,family=family,train=list(train),test=list(test),subjects=list(subjects),
                visual=visual,permutation=permutation,group=group)

def make_plans(available):
    """Matched subjects for every transfer family; original participant folds retained."""
    result=[]
    def intersection(keys): return sorted(set.intersection(*(set(available.get(k,[])) for k in keys)))
    for family in FAMILIES:
        keys=[d+'_'+family for d in ('sr_full','sr_outcome','trust')]
        ids=intersection(keys)
        for visual in (False,True):
            for i,key in enumerate(keys):
                # Includes full-trial reference on the identical phase-available sample.
                result.append(plan('phase_'+family+'_'+str(i),family,[key],keys,ids,visual,
                                   not visual and i in (1,2),group='phase'))
            result.append(plan('phase_'+family+'_pooled',family,keys[1:],keys[1:],ids,visual,group='phase'))
            baseline=[keys[0],keys[2]]
            result.append(plan('architecture_'+family,family,baseline,baseline,intersection(baseline),visual,group='architecture'))
    for source in ('trust','sr_outcome'):
        keys=[source+'_social_context','ugr_context']
        for visual in (False,True):
            result.append(plan(source+'_to_ugr','social_context',keys[:1],['ugr_context','ugr_high','ugr_low'],
                               intersection(keys),visual,not visual and source=='trust',group='ugr'))
    norm=['trust_norm','ugr_norm']
    ids=intersection(norm)
    result.extend([plan('trust_to_ugr_norm','social_norm_violation',[norm[0]],[norm[1],'ugr_computer_norm'],ids,
                        permutation=True,group='norm'),
                   plan('ugr_to_trust_norm','social_norm_violation',[norm[1]],[norm[0]],ids,permutation=True,group='norm'),
                   plan('computer_norm_control','computer_norm_violation',['trust_computer_norm'],['ugr_computer_norm'],ids,group='norm')])
    return result

def permutation_targets(p):
    if p['group']=='phase':
        return [p['test'][2] if p['train'][0].startswith('sr_outcome') else p['test'][1]]
    return p['test'][:1]

def usable(scope,ids):
    # Computational minimum inherited from v4; not a power guarantee.
    return len(ids)>=10 and {scope['folds'][s] for s in ids}==set(range(1,6))
