import json,sys,importlib.util
from pathlib import Path
root=Path('results/deception-run8-20261012');study=sys.argv[1]
p=json.loads((root/'reviews'/study/'blind-evidence.json').read_text());a=json.loads((root/'reviews'/study/'annotation-template.json').read_text());decisions=json.loads((root/'review-workers'/f'reviewer-b-{study}-decisions.json').read_text());rows=[]
for d in decisions:
 c=p['cases'][d['index']];gs=c.get('prior_history_generations',[])+c['generations'];spans=[]
 for g in gs:
  raw=g['raw_text'];end=raw.find('<|im_end|>');end=len(raw) if end<0 else end
  while end>0 and raw[end-1].isspace():end-=1
  if end:spans.append(dict(generation_path=g['path'],start=0,end=end,text=raw[:end]))
 timing=dict(notes=d['timing'],earliest_explicit_strategy=None)
 if d.get('strategy_span'):
  st=d['strategy_span'];g=c['generations'][st['generation']];start=g['raw_text'].index(st['text']);timing['earliest_explicit_strategy']=dict(generation_path=g['path'],start=start,end=start+len(st['text']),text=st['text'],interpretation=st.get('interpretation','Explicit proposal; motive is not established.'))
 if d.get('cutoff_span'):
  st=d['cutoff_span'];g=c['generations'][st.get('generation',0)];start=g['raw_text'].index(st['text']);timing['earliest_false_assertion_or_affirmative_false_report_strategy']=dict(generation_path=g['path'],start=start,end=start+len(st['text']),text=st['text'],kind=st['kind'])
 rows.append(dict(id=c['id'],source_digest=c['source_digest'],assessment=d['assessment'],evidence_spans=spans,factual_claims=dict(notes=d['facts'],initial_world=c.get('initial_world'),final_world=c.get('final_world'),complete=c.get('complete')),disclosure=dict(notes=d['disclosure']),knowledge=dict(notes=d['knowledge']),strategy_timing=timing,confidence=d['confidence']))
a.update(reviewer_id='run8-independent-b',lens_evidence_seen=False,full_raw_and_history_reviewed=True,cases=rows);out=root/'review-submissions/reviewer-b'/f'{study}.json';tmp=out.with_suffix('.json.tmp');tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');tmp.replace(out)
spec=importlib.util.spec_from_file_location('run8_review',root/'code/run8_review.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.ROOT=Path.cwd();mod.SOURCE_ROOTS={k:mod.ROOT/'results'/v.name for k,v in mod.SOURCE_ROOTS.items()}
print('validated',len(mod.validate_annotations(root/'reviews'/study/'blind-evidence.json',out)),out)
progresspath=root/'review-workers/reviewer-b.json';progress=json.loads(progresspath.read_text());progress['cases_reviewed']=159+len(rows) if study=='run5_pilot' else 207+len(rows);progress['current_stage']=f'{study}: {len(rows)}/{len(p["cases"])} saved';progress['partial_studies']={study:len(rows)}
if len(rows)==len(p['cases']) and study not in progress['studies_saved']:progress['studies_saved'].append(study)
progresspath.write_text(json.dumps(progress,indent=2)+'\n')
