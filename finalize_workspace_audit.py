"""Post-run archival regression, report subsets and standalone figures."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_workspace import cluster_contrast,rates
from fit_ouro_family import ROOT,save_json,digest
from workspace_audit_analysis import score,cell


def main():
    out=ROOT/'results/workspace-audit';old=ROOT/'results/workspace-full'
    s=json.loads((out/'summary.json').read_text());m=json.loads((out/'manifest.json').read_text())
    records=[json.loads(f.read_text()) for f in sorted((out/'records').glob('*.json'))]
    assert {r['id'] for r in records}=={t['id'] for t in m['tasks']}
    verified=0;subsets=defaultdict(list);layers=defaultdict(list);coverage=[]
    for record in records:
        spec=record['spec'];new=record['result'];family=spec['family']
        if spec['profile']!='primary':continue
        if family in ('flexible','probe','report'):
            previous=json.loads((old/'records'/(record['id']+'.json')).read_text())['result']
            trials=previous['trials'] if family!='probe' else [previous]
            by_destination={trial['destination']:trial for trial in trials}
            for trial in new['trials']:
                if family=='flexible':
                    coverage.append((spec['category']+':'+spec['argument'],{c:score(trial['conditions'].get(c,{}))
                        for c in ['transfer','loops_2','loops_3','loops_4','loops_23','loops_24','loops_34','loops_1234']}))
                prev=by_destination.get(trial['destination'])
                if prev is None or not prev['eligible']:continue
                for a,b in [('baseline','baseline'),('swap','sustained'),('transfer','projection_transfer')]:
                    for target in ['47','95','143','191']:
                        for word in ['answer','swap_answer']:
                            assert trial['conditions'][a]['scores'][target][word]['whole']==prev['conditions'][b]['scores'][target][word],(record['id'],a,target,word)
                    verified+=1
            if family=='report':
                for trial in new['trials']:
                    base=trial['conditions']['baseline']['scores']['191']['swap_answer']['whole']
                    for metric in ['top1','top5']:
                        values={c:score(trial['conditions'].get(c,{}),metric=metric) for c in ['baseline','transfer','random_42','swap']}
                        subsets['all/'+metric].append((spec['category'],values))
                        if new['source_is_category_member']:subsets['valid_source/'+metric].append((spec['category'],values))
                        if base['rank'] is not None and base['rank']>10:
                            subsets['outside_top10/'+metric].append((spec['category'],values))
                            if new['source_is_category_member']:subsets['valid_source_outside_top10/'+metric].append((spec['category'],values))
        if family=='probe' and not new['latent']['concept_present_in_prompt']:
            for key,row in new['latent']['readouts'].items():
                if row['intermediate'].get('eligible'):layers[key].append(row['intermediate']['hit5'])
    audit=dict(archived_condition_score_comparisons_exact=verified,records=len(records),
        loop_coverage_sensitivity={k:cluster_contrast(coverage,k,'transfer')
            for k in ['loops_2','loops_3','loops_4','loops_23','loops_24','loops_34','loops_1234']},
        report_subsets={k:rates(v) for k,v in subsets.items()},
        report_subset_contrasts={k:cluster_contrast(v,'transfer','baseline') for k,v in subsets.items()},
        per_site_latent_hit5={k:dict(n=len(v),rate=float(np.mean(v))) for k,v in layers.items()},
        finalizer_sha256=digest(Path(__file__)))
    save_json(out/'additional-validation.json',audit)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    families=[('flexible','top1','Flexible steering'),('probe','top1','Two-hop steering'),('report','top5','Verbal report (top-5)')]
    paper=[73/192,26/90,106/133]
    for ax,(family,metric,label),reference in zip(axes,families,paper):
        values=s['rates'][f'{family}/whole/{metric}'];names=['baseline','swap','transfer','random_42']
        heights=[values[n]['rate']*100 for n in names]
        ax.bar(['Unedited','Coordinate\nexchange','Projection\ntransfer','Random'],heights,color=['#9ca3af','#b45309','#2563eb','#64748b'])
        ax.axhline(reference*100,color='#047857',linestyle='--',label='Paper reported rate')
        for i,n in enumerate(names):
            v=values[n];lo,hi=v['wilson95'];ax.errorbar(i,100*v['rate'],yerr=[[max(0,100*(v['rate']-lo))],[max(0,100*(hi-v['rate']))]],color='black',capsize=3)
            ax.text(i,max(hi*100,heights[i])+2,f'{int(v["sum"])}/{v["n"]}',ha='center',fontsize=9)
        ax.set_title(label);ax.set_ylim(0,100);ax.set_ylabel('Success (%)');ax.legend(fontsize=8)
    fig.suptitle('Ouro reproduction audit — fixed primary settings; whole-token scoring')
    fig.savefig(out/'paper-comparison.png',dpi=180);fig.savefig(out/'paper-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),layout='constrained')
    strengths=[.25,.5,.75,1,1.25,1.5,2]
    for family,metric,label in families:
        values=s['rates'][f'{family}/whole/{metric}']
        axes[0].plot(strengths,[100*values['transfer' if a==1 else f'strength_{a:g}']['rate'] for a in strengths],marker='o',label=label)
    axes[0].set(xlabel='Transfer strength',ylabel='Success (%)',title='Fixed strength sweep');axes[0].legend(fontsize=8)
    for mode,color in [('own','#2563eb'),('raw','#9ca3af'),('final','#b45309')]:
        xs=sorted(int(k.split(':')[0]) for k in layers if k.endswith(':'+mode))
        # Disjoint lines prevent connecting absent control-layer spans.
        for loop in range(4):
            xx=[x for x in xs if x//48==loop]
            axes[1].plot(xx,[100*float(np.mean(layers[f'{x}:{mode}'])) for x in xx],color=color,label=mode if loop==0 else None)
    for boundary in [47.5,95.5,143.5]:axes[1].axvline(boundary,color='#d1d5db',linestyle=':')
    axes[1].set(xlabel='Virtual layer (zero-based)',ylabel='Intermediate hit@5 (%)',title='Prompt-absent latent readout by layer')
    axes[1].legend();fig.savefig(out/'sensitivity-and-readout.png',dpi=180);fig.savefig(out/'sensitivity-and-readout.pdf');plt.close(fig)
    extra=['# Additional validation','',f'{verified} archived condition records have exactly equal whole-token scores at all four loop ends. '
           'This checks the unedited baseline, coordinate exchange, and projection transfer on every overlapping eligible primary trial.', '',
           '## Verbal-report subsets','',
           '| Subset | Transfer success | Baseline success |','|---|---:|---:|']
    for k,v in audit['report_subsets'].items():extra.append(f'| {k} | {cell(v["transfer"])} | {cell(v["baseline"])} |')
    extra+=['','## All loop-subset contrasts against loops 2–4','',
        'Exploratory comparisons of the complete loop-subset grid. A benefit over each single loop does not prove every remaining loop is necessary.','',
        '| Condition minus loops 2–4 | Difference | 95% semantic-cluster interval |','|---|---:|---|']
    for k,v in audit['loop_coverage_sensitivity'].items():
        lo,hi=v['ci95'];extra.append(f'| {k} | {100*v["difference"]:+.1f} pp | [{100*lo:.1f}, {100*hi:.1f}] pp |')
    extra+=['','All subsets are descriptive post-run diagnostics. They do not replace the full attempted-trial primary denominator.','',
       '![Paper comparison](paper-comparison.png)','',
       'Error bars above are descriptive Wilson intervals; paired semantic-cluster intervals are in the main report. '
       'Paper and local denominators differ for verbal report.','',
       '![Sensitivity and readout](sensitivity-and-readout.png)']
    (out/'VALIDATION.md').write_text('\n'.join(extra)+'\n')
    print('Validated',verified,'archived condition records; figures and report subsets written.')


if __name__=='__main__':main()
