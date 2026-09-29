"""Recompute the complete inventory threshold grid and retained-answer counts."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from fit_ouro_family import save_json, digest
from workspace_audit_analysis import wilson


def rate(values):
    n=len(values);k=sum(values)
    return dict(k=k,n=n,rate=k/n if n else None,ci95=wilson(k,n) if n else None)


def paired_cluster(rows,left,right,draws=10000):
    groups=defaultdict(list)
    for group,values in rows:groups[group].append(float(values[left])-float(values[right]))
    if not groups:return None
    sums=np.array([sum(v) for v in groups.values()]);sizes=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(42);draw=rng.integers(len(groups),size=(draws,len(groups)))
    boot=sums[draw].sum(1)/sizes[draw].sum(1)
    return dict(difference=float(sums.sum()/sizes.sum()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                clusters=len(groups),n=int(sizes.sum()),exploratory=True)


PLURALS={'elephants':'elephant','plants':'plant','leaves':'leaf','fruits':'fruit','grasses':'grass',
         'trees':'tree','mammals':'mammal','roots':'root','herbivores':'herbivore','balls':'ball',
         'players':'player','teams':'team','fans':'fan','crowds':'crowd','matches':'match'}


def canonical(text):
    word=text.strip().casefold()
    return PLURALS.get(word,word)


def analyze(out):
    manifest=json.loads((out/'manifest.json').read_text());fingerprint=digest(out/'manifest.json')
    protocol=json.loads((out/'protocol.json').read_text())
    captures={}
    for p in sorted((out/'capture').glob('*.json')):
        r=json.loads(p.read_text());assert r['manifest_sha256']==fingerprint
        captures[r['spec']['id']]=r
    assert len(captures)==len(manifest['tasks'])
    report_rows=[];report_valid=[];sources=[]
    for r in captures.values():
        if r['spec']['family']!='report':continue
        result=r['result'];category=r['spec']['category']
        sources.append(dict(category=category,first_token=result['source_text'],source_in_category=result['source_in_category'],
            completed_word=result['greedy_source']['complete_first_word']))
        for trial in result['trials']:
            scores={}
            for name in ['baseline','transfer','random']:
                v=trial['conditions'].get(name,{})
                for metric in ['top1','top5']:
                    scores[name+'/'+metric]=bool(v.get('whole',{}).get(metric,False))
                scores[name+'/completed_word']=v.get('completed_word_hit',False)
            report_rows.append((category,scores))
            if result['source_in_category']:report_valid.append((category,scores))
    report_summary=dict(sources=sources,source_valid_count=sum(s['source_in_category'] for s in sources),
        all_attempts={k:rate([v[k] for _,v in report_rows]) for k in report_rows[0][1]},
        valid_source_subset={k:rate([v[k] for _,v in report_valid]) for k in report_rows[0][1]},
        contrasts={m:paired_cluster(report_rows,'transfer/'+m,'random/'+m) for m in ['top1','top5','completed_word']})
    ablations=[];all_ablation=[]
    for p in sorted((out/'ablation').glob('*.json')):
        record=json.loads(p.read_text());assert record['manifest_sha256']==fingerprint
        r=record['result'];assert r['sham_exact']
        values={name:bool(v.get('whole',{}).get('top1',False)) for name,v in r['conditions'].items()}
        group=record['spec']['category']+':'+record['spec']['argument']
        all_ablation.append((group,values))
        if r['baseline_correct']:ablations.append((group,values))
    assert len(all_ablation)==64
    ablation_summary=dict(all_attempts={k:rate([v[k] for _,v in all_ablation]) for k in all_ablation[0][1]},
        baseline_correct={k:rate([v[k] for _,v in ablations]) for k in all_ablation[0][1]},
        contrasts={k:paired_cluster(ablations,k,'random_42') for k in ['pursuit_all','pursuit_loop2','pursuit_loop4','argument_all']},
        paper_counts=protocol['ablation']['paper_counts'])
    dictionary=json.loads((out/'dictionary-0.json').read_text())
    for rank in range(1,4):assert json.loads((out/f'dictionary-{rank}.json').read_text())==dictionary
    words={i:canonical(t) for i,t in zip(dictionary['token_ids'],dictionary['decoded'])};allowed=set(words)
    layers=[];counts=defaultdict(Counter);overlaps=defaultdict(list);explained=defaultdict(list)
    max_kkt=0.;max_refit_steps=0;basis_ranks=Counter();max_orth=0.;cells=Counter()
    inventory_files=sorted((out/'inventory').glob('*.pt'));assert len(inventory_files)==188
    for path in inventory_files:
        meta=json.loads(path.with_suffix('.json').read_text());assert meta['manifest_sha256']==fingerprint
        assert digest(path)==meta['result']['artifact_sha256']
        data=torch.load(path,map_location='cpu',weights_only=True);site=data['site']
        for name,r in data['records'].items():
            max_kkt=max(max_kkt,r['max_kkt']);max_refit_steps=max(max_refit_steps,r['max_refit_steps'])
            if not name.startswith('demo-'):
                basis_ranks.update(r['basis_rank'].tolist());max_orth=max(max_orth,r['basis_orthogonality_error']);continue
            cells[name]+=len(r['state_norm']);selected=r['token_ids'].numpy();valid=r['valid'].numpy()
            top8=r['top8_ids'].tolist()
            topcounts=Counter(words[t] for row in top8 for t in row if t in allowed)
            counts[(name,'top8',site//48+1)].update(topcounts)
            for k,snap in r['snapshots'].items():
                coefficients=snap['coefficients'].numpy();norm=r['state_norm'].numpy()
                explained[(name,k)].extend((1-snap['residual_fraction']).tolist())
                for theta in [0,.01,.05]:
                    mask=valid[:,:k] & (coefficients>=theta*norm[:,None])
                    counter=Counter(words[int(t)] for t in selected[:,:k][mask])
                    counts[(name,f'k{k}/theta{theta:g}',site//48+1)].update(counter)
                    layers.append(dict(demo=name,site=site,loop=site//48+1,physical_layer=site%48,k=k,theta=theta,
                        cells=len(norm),occupied_atoms=int(mask.sum()),mean_atoms=float(mask.sum()/len(norm)),
                        variance_explained=float((1-snap['residual_fraction']).mean()),
                        tree=counter['tree'],mammal=counter['mammal'],vegetation=counter['vegetation'],
                        elephant=counter['elephant'],top8_tree=topcounts['tree'],top8_mammal=topcounts['mammal'],top8_vegetation=topcounts['vegetation']))
                    if k==25 and theta==.05:
                        for i in range(len(norm)):
                            support=set(int(t) for t in selected[i,:k][mask[i]])
                            ranked={t for t in top8[i] if t in allowed}
                            union=support|ranked
                            overlaps[name].append(len(support&ranked)/len(union) if union else 1.)
    with (out/'occupancy-grid.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(layers[0]));writer.writeheader();writer.writerows(layers)
    save_json(out/'concept-counts.json',{'/'.join(map(str,key)):dict(value.most_common()) for key,value in counts.items()})
    inventory_summary=dict(dictionary_size=len(allowed),layers=188,cells=dict(cells),
        explained={f'{demo}/k{k}':float(np.mean(v)) for (demo,k),v in explained.items()},
        primary_jaccard={k:float(np.mean(v)) for k,v in overlaps.items()},max_normalized_kkt=max_kkt,
        max_refit_steps=max_refit_steps,basis_ranks=dict(basis_ranks),max_basis_orthogonality_error=max_orth,
        plural_merges=PLURALS,scope=protocol['inventory']['scope'])
    summary=dict(report=report_summary,ablation=ablation_summary,inventory=inventory_summary)
    save_json(out/'summary.json',summary)
    def cell(v):return f"{v['k']}/{v['n']} ({v['rate']:.1%})" if v['n'] else '0/0'
    lines=['# Validated scoring, sparse inventory and subspace ablation','',
        'Development follow-up to the completed paper audit. The report wrapper was chosen using previously observed outcomes; this is not independent confirmation. Earlier results are preserved.','',
        '## Verbal report','',
        f"Fixed native-chat `Answer:` wrapper; category-valid first-token sources: {report_summary['source_valid_count']}/14. Invalid sources remain in the all-attempt denominator. All generation extends exact token IDs; no truncation or text re-tokenization.",'',
        '| Metric | Transfer | Random delta | Baseline |','|---|---:|---:|---:|']
    for metric in ['top1','top5','completed_word']:
        lines.append('| '+metric+' | '+' | '.join(cell(report_summary['all_attempts'][c+'/'+metric]) for c in ['transfer','random','baseline'])+' |')
    lines+=['','Teacher-forced sequence log probabilities are stored separately and never treated as answer accuracy. Continuation edits act only at the original prompt positions. A completed first word requires a boundary or termination, so a split prefix is never credited as a full word.','',
        '## Ten-direction subspace ablation','',
        f"Baseline correct: {cell(ablation_summary['all_attempts']['baseline'])}. Every condition uses that same baseline-correct set; all 64 prompts were executed. All 64 sham passes reproduce baseline logits exactly.",'',
        '| Condition | Answers kept here | Paper Ouro answers kept |','|---|---:|---:|']
    for name,v in ablation_summary['baseline_correct'].items():
        if name in ['baseline','sham']:continue
        paper=protocol['ablation']['paper_counts'].get(name)
        lines.append(f"| {name} | {cell(v)} | {str(paper)+'/42' if paper is not None else '—'} |")
    lines+=['','Paper reference: [Appendix E, Table 11 and Appendix B, Table 8](https://arxiv.org/html/2609.01924v1). These comparisons use the released flexible-generalization prompts and our frozen physical-layer indexing; unresolved author-code details prevent claiming exact procedural identity.','',
        'The pursuit basis is selected from clean states, refitted with nonnegative coefficients, and frozen before the intervention pass. Random controls use ten vocabulary atoms shared over positions at each layer, with three fixed seeds. Paired cluster-bootstrap intervals and all-attempt results are in `summary.json`. Intervals are exploratory, not corrected for the full study.','',
        '## Sparse inventory','',
        f"Two exact paper-quoted prompts (animal and hidden sport), all prefill and up to 128 greedy continuation tokens, all 188 fitted source sites. Dictionary: {len(allowed):,} allowed word tokens. All 12 (k, threshold) combinations are in `occupancy-grid.csv`; per-cell selections, coefficient snapshots and residual fractions are in the layer artifacts.",'',
        '| Prompt | Cells | Variance explained at k=25 | At k=50 | Primary support/top8 Jaccard |','|---|---:|---:|---:|---:|']
    for demo in sorted(cells):
        lines.append(f"| {demo} | {cells[demo]:,} | {inventory_summary['explained'][demo+'/k25']:.1%} | {inventory_summary['explained'][demo+'/k50']:.1%} | {inventory_summary['primary_jaccard'][demo]:.3f} |")
    lines+=['',f"Largest normalized NNLS KKT residual: {max_kkt:.3g}; maximum refit iterations: {max_refit_steps}; largest basis orthogonality error: {max_orth:.3g}. Solver failures abort a run. Rank-deficient spans are recorded without inventing extra directions.",'',
        'The paper’s 21.5% explained-variance result averages eight demonstrations. Our two-prompt values have a different population and are not an exact replication of that aggregate. Top8 uses native unembedding including normalization; pursuit uses the stated linear J-transpose atom dictionary.','',
        '## Remaining gaps','',
        'Exact author prompt wrappers, word filtering, solver settings and all eight demonstration transcripts remain unresolved. The word filter and explicit plural merges are disclosed here. Capacity under pursuit and a larger frozen independent confirmation set remain separate work. These results characterize the fitted instrument and causal response, not consciousness or a complete inventory of all stored concepts.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot(out,layers,ablation_summary)


def plot(out,layers,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,8),constrained_layout=True)
    for demo,label in [('demo-animal','Animal prompt'),('demo-sport','Hidden sport prompt')]:
        rows=[r for r in layers if r['demo']==demo and r['k']==25 and r['theta']==.05]
        axes[0].plot([r['site'] for r in rows],[r['mean_atoms'] for r in rows],label=label)
    for loop in range(4):
        axes[0].axvspan(48*loop+24,48*loop+40,color='grey',alpha=.12)
    axes[0].set(xlabel='Virtual source layer (zero based)',ylabel='Mean retained atoms per cell',title='Sparse inventory: k=25, coefficient ≥ 0.05 × state norm')
    axes[0].legend()
    names=['pursuit_all','pursuit_loop2','pursuit_loop4','random_42','random_43','random_44','argument_all']
    values=[summary['baseline_correct'][n]['rate'] for n in names]
    axes[1].barh(names,values,color=['#b44']*3+['#478']*3+['#777'])
    axes[1].set(xlim=(0,1),xlabel='Fraction of baseline-correct answers retained',title='Frozen clean-state ablation; identical trial set for every condition')
    for ext in ['png','pdf']:fig.savefig(out/f'inventory-and-ablation.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    analyze(p.parse_args().output)
