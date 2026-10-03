"""Frozen v4 contrast definitions and separate output namespaces."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
from utils import Config, PipelineError

VERSION = 'v4_social_reward_context_closeness'
COHORTS = {'partner_pair': ('sharedreward', 'trust'),
           'three_paradigm': ('sharedreward', 'trust', 'socialdoors', 'doors')}
FAMILIES = {'partner_pair': ('partner_reward', 'social_reward', 'social_context',
                            'closeness_positive', 'closeness_negative', 'closeness_reward', 'valence'),
            'three_paradigm': ('social_reward', 'social_context')}
DESCRIPTIONS = {
    'partner_reward': 'Computer / friend / stranger identity from reward-minus-negative-outcome maps',
    'social_reward': 'Mean human reward-minus-negative-outcome map versus computer/monetary counterpart',
    'social_context': 'Human versus computer/monetary context, averaging positive and negative outcomes equally',
    'closeness_positive': 'Friend versus stranger during reward/reciprocation',
    'closeness_negative': 'Friend versus stranger during punishment/defection',
    'closeness_reward': 'Friend versus stranger reward-minus-negative-outcome maps',
    'valence': 'Positive versus negative outcome maps, averaging computer/friend/stranger equally',
}


@dataclass
class RevisedConfig(Config):
    scope: str = 'inventory'

    def output(self, relative):
        tree, *rest = str(relative).split('/')
        return super().output('/'.join([tree, 'revised', self.scope, *rest]))


def revised_config(base, scope='inventory'):
    if scope != 'inventory' and scope != 'samples' and not any(
            scope == f'{policy}/{cohort}' for policy in ('tsnr_coverage_fd', 'prior_four_metric_policy')
            for cohort in COHORTS):
        raise PipelineError('Unknown revised output scope')
    fields = {key: deepcopy(getattr(base, key)) for key in Config.__dataclass_fields__}
    # Aging's fitted maps live in aging, but their saved provenance references
    # harmonized RF1 BOLD inputs in the original sharedreward repository.
    fields['repos'] = {key: value for key, value in fields['repos'].items()
                       if key in ('linux2', 'sharedreward', 'aging', 'trust', 'socdoors')}
    specs = fields['contrasts']
    specs.pop('ugr')
    specs['trust']['copes'] = {k: value for k, value in specs['trust']['copes'].items() if k in range(4, 10)}
    for task in ('socialdoors', 'doors'):
        specs[task]['copes'] = {1: {'name': 'win', 'weights': {1: 1.}},
                                2: {'name': 'loss', 'weights': {2: 1.}},
                                4: specs[task]['copes'][4]}
    return RevisedConfig(**fields, scope=scope)


def center(values):
    values = np.asarray(values, dtype=np.float32)
    return values - values.mean(axis=-1, keepdims=True, dtype=np.float64).astype(np.float32)


def partner_representations(task, maps):
    keys = ((2, 4, 6), (1, 3, 5)) if task == 'sharedreward' else ((5, 7, 9), (4, 6, 8))
    positive = np.stack([maps[k] for k in keys[0]])  # C, F, S
    negative = np.stack([maps[k] for k in keys[1]])
    reward = positive - negative
    context = (positive + negative) / 2
    return {key: center(value) for key, value in {
        'partner_reward': reward,
        'social_reward': [reward[1:].mean(0), reward[0]],
        'social_context': [context[1:].mean(0), context[0]],
        'closeness_positive': positive[1:],
        'closeness_negative': negative[1:],
        'closeness_reward': reward[1:],
        'valence': [positive.mean(0), negative.mean(0)],
    }.items()}


def doors_representations(social, monetary):
    return {'social_reward': center([social[4], monetary[4]]),
            'social_context': center([(social[1]+social[2])/2, (monetary[1]+monetary[2])/2])}
