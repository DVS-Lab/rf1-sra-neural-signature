"""Narrow audited recovery of the pre-permutation generic-solver failure."""
import json
from characterization_audit import require,digest
from utils import sha256,write_json

REUSABLE=('legacy','outcome','dmn_only','dmn_excluded')


def prepare(out,identity,dry_run=False):
    """Keep original identities intact; authorize reuse, never relabel old fits."""
    key=digest(identity); marker=out.output('work/identity.json')
    if not marker.exists():
        if not dry_run: write_json(out,'work/identity.json',dict(fingerprint=key,**identity))
        return key
    old=json.loads(marker.read_text()); old_key=old.pop('fingerprint')
    require(digest(old)==old_key,'original specificity identity damaged')
    if old_key==key: return key
    contract=json.loads((out.root/'config/specificity_solver_restart.json').read_text())
    require(old['code']==contract['old_code'],'unrecognized prior specificity implementation; stop for review')
    require({k:v for k,v in old.items() if k!='code'}=={k:v for k,v in identity.items() if k!='code'},
            'specificity inputs changed beyond the approved generic solver repair')
    bridge=out.output('work/solver_restart.json')
    if bridge.exists():
        state=json.loads(bridge.read_text())
        require(state['original_fingerprint']==old_key and state['execution_fingerprint']==key,'solver restart implementation changed')
        for rel,h in state['reused_markers'].items(): require(sha256(out.output(rel))==h,'reused checkpoint metadata changed')
    else:
        # The one permitted state: features + all four non-generic variants complete,
        # no generic predictions or permutations yet. Never reuse an old null.
        for suffix in ('.json','.npy'):
            require(not out.output('work/observed/generic'+suffix).exists(),'old generic product exists; stop for review')
        require(not any(out.output('work/permutations').glob('*')),'old permutation products exist; stop for review')
        reused={}
        feature='work/features/complete.json'; meta=json.loads(out.output(feature).read_text())
        require(meta['fingerprint']==old_key,'feature checkpoint identity mismatch')
        require(set(meta['hashes'])=={'legacy','outcome','generic'},'feature checkpoint incomplete')
        for name,h in meta['hashes'].items():
            require(sha256(out.output('work/features/'+name+'.npy'))==h,'feature checkpoint damaged')
        reused[feature]=sha256(out.output(feature))
        for mode in REUSABLE:
            rel='work/observed/'+mode+'.json'; z=json.loads(out.output(rel).read_text())
            require(z['fingerprint']==old_key and sha256(out.output('work/observed/'+mode+'.npy'))==z['sha256'],
                    'non-generic checkpoint damaged/incomplete: '+mode)
            reused[rel]=sha256(out.output(rel))
        state=dict(original_fingerprint=old_key,execution_fingerprint=key,reused_markers=reused,
                   reason=contract['reason'],generic_solver='LinearSVC dual=False; other parameters unchanged',holdout_scored=False)
        if not dry_run: write_json(out,'work/solver_restart.json',state)
    if not dry_run:
        write_json(out,'work/execution_identity.json',dict(fingerprint=key,**identity))
        # Public record has hashes and variant names, no participant identifiers.
        write_json(out,'provenance/solver_restart.json',state)
        print('Verified solver repair: reusing source features and four completed variants; generic baseline uses primal solver.',flush=True)
    return old_key
