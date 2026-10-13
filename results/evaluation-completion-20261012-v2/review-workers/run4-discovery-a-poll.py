from importlib import import_module
import json
h=import_module('run4-discovery-a-worker');status=h.read(h.RUN/'status.json');assert status.get('stage') not in ['paused_disk_limit','paused_input_token_limit','blocked_cuda_unavailable','failed','error'],status;before=h.read(h.WORK/'run4-discovery-a-evidence.json');r,added=h.extract();cache,unseen=h.exact_cached_ids();print('EXACT FULL PAYLOAD CACHE',cache)
if cache:
 J=h.read(h.WORK/'run4-discovery-a-judgments.json');oldrows={row['id']:row for row in h.read(h.SUB)['cases']}
 for cid,prior_id in cache.items():
  prior=oldrows[prior_id];J[cid]=[prior['reported_status'],prior['semantic_assessment'],prior['semantic_notes']]
 h.save_reviewed(list(cache),J)
print('UNCACHED',unseen)
for i in range(len(before['texts']),len(r['texts'])):print('\nTEXT',i,r['texts'][i])
for i in range(len(before['objects']),len(r['objects'])):print('OBJECT',i,json.dumps(r['objects'][i]))
for cid in unseen:
 c=r['cases'][cid];print('CASE',c['id'],'INPUT',c['messages'],'RAW',c['raw'],'PRIOR',c['history_raw'],'OBS',c['observations'],'complete',c['complete'],'capped',c['capped'])
