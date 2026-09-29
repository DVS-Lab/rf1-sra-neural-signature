"""Configuration, read-only source paths, guarded output writers, and holdout policy."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineError(RuntimeError):
    """A scientific or mechanical contract could not be established."""


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_yaml(path):
    with Path(path).open() as stream:
        return yaml.safe_load(stream)


@dataclass
class Config:
    root: Path
    repos: dict
    bids: Path
    fmriprep: Path
    exclusions: Path
    templateflow: Path
    paths: dict
    analysis: dict
    contrasts: dict

    @property
    def protected(self):
        return [*self.repos.values(), self.bids, self.fmriprep, self.exclusions, self.templateflow]

    def output(self, relative):
        """Only this checkout's output trees; resolve symlinks before every write."""
        lexical = Path(relative)
        if lexical.is_absolute() or not lexical.parts or lexical.parts[0] not in {
            'work', 'results', 'reports', 'provenance'
        }:
            raise PipelineError('Output must be within a designated project output tree')
        target = (self.root / lexical).resolve()
        if not target.is_relative_to(self.root.resolve()):
            raise PipelineError('Output escapes project root')
        for source in self.protected:
            source = source.resolve()
            if target == source or target.is_relative_to(source):
                raise PipelineError('Refusing output inside a protected source')
        if target.exists() and target.is_file() and target.stat().st_nlink > 1:
            raise PipelineError('Refusing a hard-linked output')
        return target


def load_config(config_dir=None, root=None):
    root = Path(root or PROJECT_ROOT).resolve()
    folder = Path(config_dir or root / 'config')
    paths = read_yaml(folder / 'paths.yaml')
    analysis = read_yaml(folder / 'analysis.yaml')
    contrasts = read_yaml(folder / 'contrast_spec.yaml')
    parent = Path(os.environ.get('PROJECTS_ROOT', paths['projects_root'])).expanduser().resolve()
    repos = {key: Path(os.environ.get('RF1_' + key.upper() + '_ROOT', parent / value)).expanduser().resolve()
             for key, value in paths['repositories'].items()}
    def source(name, env, base=None):
        value = Path(os.environ.get(env, paths[name])).expanduser()
        return ((base / value) if base is not None else value).resolve()
    c = Config(root, repos, source('bids', 'BIDS_ROOT', repos['linux2']),
               source('fmriprep', 'FMRIPREP_ROOT', repos['linux2']),
               source('source_exclusions', 'SOURCEDATA_EXCLUSIONS_ROOT'),
               source('templateflow', 'TEMPLATEFLOW_HOME'), paths, analysis, contrasts)
    required = {'session': '01', 'holdout_n': 50, 'folds': 5, 'seed': 20260928,
                'space': 'MNI152NLin6Asym', 'normalization': 'spatial_mean_center', 'coverage': .95}
    if any(analysis.get(k) != v for k, v in required.items()):
        raise PipelineError('Primary analysis settings differ from the frozen pilot contract')
    svm = analysis['classifier']
    if (svm['C'] != 1.0 or svm['dual'] is not True or svm['class_weight'] is not None
            or svm['max_iter'] < 10000):
        raise PipelineError('Invalid primary LinearSVC settings')
    for p in c.protected:
        if root == p or root.is_relative_to(p) or p.is_relative_to(root):
            raise PipelineError('Analysis checkout and source directories must be disjoint')
    for tree in ['work', 'results', 'reports', 'provenance']:
        c.output(tree)
    return c


@contextmanager
def atomic_output(c, relative):
    target = c.output(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    # NIfTI writers require a recognizable extension.
    suffix = ''.join(target.suffixes)
    fd, temporary = tempfile.mkstemp(prefix='.writing-', suffix=suffix, dir=target.parent)
    os.close(fd)
    try:
        yield Path(temporary)
        c.output(relative)  # Re-check symlink/path state immediately before replacement.
        os.replace(temporary, target)
        if Path(relative).parts[0] == 'work':
            target.chmod(0o600)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_tsv(c, relative, data, columns=None):
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=columns)
    with atomic_output(c, relative) as path:
        frame.to_csv(path, sep='\t', index=False)


def write_json(c, relative, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)): return [clean(v) for v in x]
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, Path): return str(x)
        return x
    with atomic_output(c, relative) as path:
        path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def write_text(c, relative, text):
    with atomic_output(c, relative) as path:
        path.write_text(text)


def subject_id(value):
    value = str(value)
    if not re.fullmatch(r'sub-[A-Za-z0-9]+', value):
        raise PipelineError('Invalid BIDS participant identifier')
    return value


@dataclass(frozen=True)
class DevelopmentGuard:
    development: frozenset
    holdout: frozenset

    def __post_init__(self):
        if not self.development or self.development & self.holdout or len(self.holdout) != 50:
            raise PipelineError('Invalid locked development/holdout partition')

    def check(self, subjects, development_subjects):
        """Explicit development list is mandatory at every participant-data boundary."""
        allowed = set(development_subjects)
        requested = set(subjects)
        if not allowed or not allowed <= self.development or allowed & self.holdout:
            raise PipelineError('HOLDOUT LOCK: invalid explicit development list')
        if not requested <= allowed or requested & self.holdout:
            raise PipelineError('HOLDOUT LOCK: participant is not authorized for modeling')
        for subject in requested:
            subject_id(subject)
