"""Descriptive concept-class profiles, classified from the actual saved transcript."""
import argparse
import csv
import json
from pathlib import Path
import re

import torch
from analyze_workspace_inventory import canonical
from fit_ouro_family import save_json,digest


def profile(out):
    # Fixed transparent analysis vocabulary; this is not an independent hypothesis test.
    universe=['elephant','plant','herbivore','grass','leaf','bark','tree',
              'mammal','vegetation','food','fruit','root']
    capture=json.loads((out/'capture/demo-animal.json').read_text())['result']
    transcript=capture['prompt']['text']+capture['generated']['raw_text']
    mentioned={canonical(w) for w in re.findall(r"[^\W\d_]+",transcript)}
    explicit=set(universe)&mentioned;unspoken=set(universe)-mentioned
    dictionary=json.loads((out/'dictionary-0.json').read_text())
    words={i:canonical(t) for i,t in zip(dictionary['token_ids'],dictionary['decoded'])}
    rows=[]
    for path in sorted((out/'inventory').glob('*.pt')):
        data=torch.load(path,map_location='cpu',weights_only=True);r=data['records']['demo-animal'];site=data['site']
        def counts(ids):
            values=[words.get(int(i),'') for i in ids]
            return dict(explicit=sum(w in explicit for w in values),unspoken=sum(w in unspoken for w in values))
        top=counts(r['top8_ids'].flatten())
        for k,snapshot in r['snapshots'].items():
            for theta in [0,.01,.05]:
                mask=r['valid'][:,:k] & (snapshot['coefficients']>=theta*r['state_norm'][:,None])
                selected=counts(r['token_ids'][:,:k][mask])
                rows.append(dict(site=site,loop=site//48+1,physical=site%48,k=k,theta=theta,cells=len(mask),
                    explicit=selected['explicit'],unspoken=selected['unspoken'],top8_explicit=top['explicit'],top8_unspoken=top['unspoken']))
    with (out/'animal-concept-profiles.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save_json(out/'animal-concept-profile-method.json',dict(script_sha256=digest(Path(__file__)),
        interpretation='Descriptive post-run analysis, not a preregistered confirmation; unspoken candidates are not proof of hidden representation',
        universe=universe,explicit=sorted(explicit),unspoken=sorted(unspoken),
        classification='Exact case/plural-normalized word presence in this saved prompt and completion; no imported paper class labels',
        sport_excluded_reason='No final answer within the fixed generation budget'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in rows if r['k']==25 and r['theta']==.05]
    fig,axes=plt.subplots(2,1,figsize=(11,7),constrained_layout=True,sharex=True)
    for axis,prefix,title in [(axes[0],'','Sparse inventory (k=25, θ=0.05)'),(axes[1],'top8_','Native J-lens top-8')]:
        for name,label in [('explicit','Named in actual prompt/completion'),('unspoken','Unspoken candidate concepts')]:
            axis.plot([r['site'] for r in selected],[r[prefix+name]/r['cells'] for r in selected],label=label)
        for loop in range(4):axis.axvspan(loop*48+24,loop*48+40,color='grey',alpha=.12)
        axis.set(ylabel='Matching slots per cell',title=title);axis.legend()
    axes[1].set_xlabel('Virtual source layer (zero based)')
    fig.suptitle('Animal prompt: classes derived from our own transcript (tree is explicitly named)')
    for ext in ['png','pdf']:fig.savefig(out/f'animal-concept-profiles.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);profile(p.parse_args().output)
