import json,sys
from pathlib import Path
root=Path('results/deception-run8-20261012');s=sys.argv[1]
p=json.loads((root/'reviews'/s/'blind-evidence.json').read_text())
d={k:{} for k in ['messages','raws','generations','events','artifacts']}
def ref(kind,v):
 k=json.dumps(v,sort_keys=True,ensure_ascii=False)
 if k not in d[kind]:d[kind][k]=(kind[0].upper()+str(len(d[kind])),v)
 return d[kind][k][0]
cases=[]
for c in p['cases']:
 gens=[]
 for g in c.get('prior_history_generations',[])+c['generations']:
  payload={k:v for k,v in g.items() if k not in ['path','id','sha256','messages','raw_text','assistant_history_text']}
  payload['messages']=[ref('messages',m) for m in g['messages']]
  payload['raw']=ref('raws',g['raw_text']);payload['history']=ref('raws',g['assistant_history_text'])
  gens.append((g['path'],ref('generations',payload)))
 events=[]
 for e in c.get('actual_tool_events',[]):
  v={k:v for k,v in e['event'].items() if k!='generation_sha256'}
  events.append((e['path'],ref('events',v)))
 artifacts=[]
 for a in c.get('actual_workspace_artifacts',[]):artifacts.append((a['path'],ref('artifacts',a['content'])))
 readcontents={m['content'] for g in c.get('prior_history_generations',[])+c['generations'] for m in g['messages']}|{g['assistant_history_text'] for g in c.get('prior_history_generations',[])+c['generations']}|{g['raw_text'] for g in c.get('prior_history_generations',[])+c['generations']}
 extra=[t for t in c['transcript'] if t['content'] not in readcontents]
 cases.append(dict(id=c['id'],complete=c.get('complete'),condition=c['condition'],generations=gens,events=events,artifacts=artifacts,initial_world=c.get('initial_world'),final_world=c.get('final_world'),extra_transcript=extra))
output={k:dict(v.values()) for k,v in d.items()};output['cases']=cases
(root/'review-workers'/f'reviewer-b-{s}-index.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print({k:len(v) for k,v in output.items()})
