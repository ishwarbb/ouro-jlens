import json,copy
from pathlib import Path
from run8_review import validate_annotations
ROOT=Path('results/deception-run8-20261012')
def packet(study): return json.loads((ROOT/'reviews'/study/'blind-evidence.json').read_text())
def save(study,rows):
 a=json.loads((ROOT/'reviews'/study/'annotation-template.json').read_text());a.update(reviewer_id='run8-independent-a',lens_evidence_seen=False,full_raw_and_history_reviewed=True,cases=rows)
 p=ROOT/'review-submissions/reviewer-a'/f'{study}.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
 checked=validate_annotations(ROOT/'reviews'/study/'blind-evidence.json',p)
 progpath=ROOT/'review-workers/reviewer-a.json';prog=json.loads(progpath.read_text());prog.setdefault('studies_saved',[])
 if study not in prog['studies_saved']:prog['studies_saved'].append(study)
 prog.setdefault('counts',{})[study]=len(checked);prog['cases_reviewed']=sum(prog['counts'].values());prog['current_stage']=f'{study} validated; continuing next study';progpath.write_text(json.dumps(prog,indent=2)+'\n')
 print(str(p),'validated',len(checked))
def span(g,text=None):
 t=g['raw_text'] if text is None else text;s=0 if text is None else g['raw_text'].index(t)
 return dict(generation_path=g['path'],start=s,end=s+len(t),text=t)
def intern(study):
 p=packet(study);texts=[];idx={};items=[];obj=[];oi={}
 def ti(t):
  if t not in idx:idx[t]=len(texts);texts.append(t)
  return idx[t]
 def oref(o):
  t=json.dumps(o,sort_keys=True,ensure_ascii=False)
  if t not in oi:oi[t]=len(obj);obj.append(o)
  return oi[t]
 for c in p['cases']:
  d={'id':c['id'],'family':c.get('family'),'phase':c.get('phase'),'complete':c.get('complete')}
  d['generations']=[{'turn':g['turn'],'raw':ti(g['raw_text']),'messages':[(m['role'],ti(m['content'])) for m in g['messages']],'complete':g['complete'],'capped':g['capped']} for g in c.get('prior_history_generations',[])+c['generations']]
  d['transcript']=[(m['role'],ti(m.get('raw_text',m['content']))) for m in c['transcript']]
  d['events']=[oref({k:v for k,v in e['event'].items() if k not in ['generation_sha256','artifact_hashes']}) for e in c['actual_tool_events']]
  d['artifacts']=[(a['path'].split('/workspace/')[-1],oref(a['content'])) for a in c.get('actual_workspace_artifacts',[])]
  d['worlds']=[oref(c[k]) for k in ['initial_world','final_world'] if k in c]
  items.append(d)
 out=dict(texts=texts,objects=obj,cases=items); (ROOT/'review-workers'/f'reviewer-a-{study}-intern.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n');return out
