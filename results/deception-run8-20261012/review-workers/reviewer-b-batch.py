import json,sys
from pathlib import Path
root=Path('results/deception-run8-20261012');study=sys.argv[1];start=int(sys.argv[2]);end=int(sys.argv[3]);p=json.loads((root/'reviews'/study/'blind-evidence.json').read_text())
seen=set()
for n,c in enumerate(p['cases'][start:end],start):
 print('\nCASE',n,c['id'],'COMPLETE',c.get('complete'),'TRUTH',c.get('truth'))
 print('INITIAL',json.dumps(c.get('initial_world')),'FINAL',json.dumps(c.get('final_world')))
 gs=c.get('prior_history_generations',[])+c['generations']; generated_history={g['assistant_history_text'] for g in gs}
 for j,g in enumerate(gs):
  print('GEN',j,g['path'],'complete',g['complete'],'capped',g['capped'],'prior',g['prior_history'])
  for m in g['messages']:
   if m['role']=='assistant' and m['content'] in generated_history: continue
   key=json.dumps(m,sort_keys=True)
   if key not in seen:
    seen.add(key);print('INCOMING',json.dumps(m,ensure_ascii=False))
  print('EXACT RAW',json.dumps(g['raw_text'],ensure_ascii=False))
 for e in c.get('actual_tool_events',[]):
  print('ACTUAL EVENT',json.dumps(e['event'],ensure_ascii=False))
 artseen=set()
 for art in c.get('actual_workspace_artifacts',[]):
  key=json.dumps(art['content'],sort_keys=True)
  if key not in artseen:artseen.add(key);print('ACTUAL ARTIFACT CONTENT',json.dumps(art['content'],ensure_ascii=False))
 readcontents={m['content'] for g in gs for m in g['messages']}|{g['assistant_history_text'] for g in gs}|{g['raw_text'] for g in gs}
 for t in c['transcript']:
  if t['content'] not in readcontents:print('ADDITIONAL TRANSCRIPT',json.dumps(t,ensure_ascii=False))
