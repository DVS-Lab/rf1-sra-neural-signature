"""Prespecified final development follow-up; no validation or mask-selection path."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from utils import Config, PipelineError
from revised_design import center, partner_representations, doors_representations

FAMILIES = ('social_context', 'friend_stranger_context')
TASKS = ('sharedreward', 'trust', 'socialdoors')

@dataclass
class CrossValenceConfig(Config):
    def output(self, relative):
        p = Path(relative)
        if p.is_absolute() or not p.parts or p.parts[0] not in ('work','results','reports','provenance') or '..' in p.parts:
            raise PipelineError('Cross-valence outputs must stay in their own namespace')
        return super().output('/'.join([p.parts[0], 'revised', 'cross_valence', *p.parts[1:]]))


def output_config(base):
    return CrossValenceConfig(**{k:getattr(base,k) for k in Config.__dataclass_fields__})


def domains(cohort):
    return tuple(t+'_'+v for t in TASKS[:2 if cohort=='partner_pair' else 3] for v in ('positive','negative'))


def representations(task, maps, monetary=None):
    if task == 'socialdoors':
        return {'social_context': np.stack([center([maps[k],monetary[k]]) for k in (1,2)]),
                'social_context_collapsed': doors_representations(maps,monetary)['social_context']}
    if task not in TASKS[:2]: raise PipelineError('Unknown partner task')
    positive_keys, negative_keys = ((2,4,6),(1,3,5)) if task=='sharedreward' else ((5,7,9),(4,6,8))
    pos, neg = (np.stack([maps[k] for k in keys]) for keys in (positive_keys,negative_keys))
    context=(pos+neg)/2
    return {'social_context': np.stack([center([v[1:].mean(0),v[0]]) for v in (pos,neg)]),
            'friend_stranger_context': np.stack([center(v[1:]) for v in (pos,neg)]),
            'social_context_collapsed': partner_representations(task,maps)['social_context'],
            'friend_stranger_context_collapsed': center(context[1:])}


def plans(cohort):
    """Each training specification is fit once per fold and tested on all named domains."""
    names=domains(cohort); result=[]
    for family in FAMILIES if cohort=='partner_pair' else FAMILIES[:1]:
        for i,name in enumerate(names):
            result.append(dict(family=family, name=name, key=family, train=[i], test=list(range(len(names))),
                               test_names=list(names), scope='matrix'))
        if cohort!='partner_pair': continue
        for valence, train, test in [('positive',[0,2],[1,3]),('negative',[1,3],[0,2])]:
            result.append(dict(family=family,name='pooled_'+valence,key=family,train=train,test=test,
                               test_names=[names[i] for i in test],scope='pooled_cross_valence'))
        if family=='friend_stranger_context':
            for name,train in [('sharedreward',[0]),('trust',[1]),('common',[0,1])]:
                result.append(dict(family=family,name='collapsed_'+name,key=family+'_collapsed',train=train,
                                   test=[0,1],test_names=list(TASKS[:2]),scope='collapsed'))
    return result


def permutation_plans():
    selected=[]
    for p in plans('partner_pair'):
        if p['scope']=='matrix':
            i=p['train'][0]; j={0:3,1:2,2:1,3:0}[i]
            p={**p,'test':[j],'test_names':[domains('partner_pair')[j]]}
            selected.append(p)
        elif p['family']=='social_context' and p['scope']=='pooled_cross_valence': selected.append(p)
        elif p['family']=='friend_stranger_context' and p['name']=='collapsed_common': selected.append(p)
    return selected


def training_names(p,cohort):
    names=TASKS[:2] if p['key'].endswith('_collapsed') else domains(cohort)
    return [names[i] for i in p['train']]
