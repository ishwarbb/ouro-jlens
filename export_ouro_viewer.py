"""Export existing immutable research artifacts into the local viewer format."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import torch
from transformers import AutoTokenizer
from ouro_viewer import Store,DEFAULT_DATA
from ouro_viewer_engine import MODEL,inventory_record,rounded
from fit_ouro_family import ROOT,digest


def export(data_dir,refresh=False):
    torch.set_num_threads(2)
    store=Store(data_dir);source=ROOT/'results/workspace-inventory-v2'
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True)
    runs={};vocab={}
    for name,title in [('demo-animal','The largest land animal'),('demo-sport','A sport without a name')]:
        capture=json.loads((source/'capture'/f'{name}.json').read_text());r=capture['result'];g=r['generated']
        run_id='research-'+name.removeprefix('demo-')
        if (store.path(run_id)/'run.json').exists() and not refresh:continue
        ids=r['seq_ids'];prompt_length=len(r['prompt']['seq_ids'])
        notes=['Recorded research run. Four loop-end targets were not captured; their cells are left empty.',
               'Sparse coefficients are shown relative to the residual norm. Values are rounded for display.']
        if g['limit_reached']:notes.append('This continuation reached the 128-token budget.')
        if g['unclosed_thought']:notes.append('The sport trace ends inside a thought block and has no final answer.')
        created=json.loads((source/'manifest.json').read_text())['created_unix']
        runs[name]=dict(schema_version=1,id=run_id,title=title,kind='research',created_at=datetime.fromtimestamp(created,timezone.utc).isoformat(),
            model=MODEL,input=dict(mode='chat',prompt=capture['spec']['user'],system='',prefill='',max_new_tokens=128),
            prompt_text=r['prompt']['text'],prompt_length=prompt_length,layers=[],
            tokens=[dict(id=t,text=tokenizer.decode([t]),position=i,segment='prompt' if i<prompt_length else 'completion',special=t in tokenizer.all_special_ids) for i,t in enumerate(ids)],
            generation=dict(raw_text=g['raw_text'],answer_text=g['text'],output_tokens=len(g['token_ids']),limit_reached=g['limit_reached'],
                unclosed_thought=g['unclosed_thought'],had_thought=g['had_thought'],stopped=g['stopped_on_special']),
            notes=notes,provenance=dict(source='workspace-inventory-v2',manifest_sha256=digest(source/'manifest.json'),
                validation='Independently verified reconstruction and subspace geometry',top_k=8,inventory_k=[8,16,25,50],
                routing='own loop end',recorded_source_layers=188,missing_sites=[47,95,143,191]))
        vocab[name]=set(ids)
    if not runs:
        print('Research runs already exported');return
    for path in sorted((source/'inventory').glob('*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True);site=saved['site']
        for name,run in runs.items():
            r=saved['records'][name]
            row=dict(site=site,physical=site%48,loop=site//48+1,target=(site//48+1)*48-1,
                jlens=dict(ids=r['top8_ids'].tolist(),probabilities=rounded(r['top8_probabilities'])),
                logit=dict(ids=r['raw_top8_ids'].tolist(),probabilities=rounded(r['raw_top8_probabilities'])),inventory=inventory_record(r))
            run['layers'].append(row)
            for array in [r['top8_ids'],r['raw_top8_ids'],r['token_ids']]:vocab[name].update(t for t in array.flatten().tolist() if t>=0)
    for name,run in runs.items():
        run['vocabulary']={str(t):tokenizer.decode([t]) for t in sorted(vocab[name])}
        path=store.path(run['id']);path.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'capture'/f'{name}.pt',path/'states.pt')
        store.save(run);print('Exported',run['id'],len(run['tokens']),'tokens',len(run['layers']),'layers',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--refresh-research',action='store_true')
    args=parser.parse_args();export(args.data,args.refresh_research)
