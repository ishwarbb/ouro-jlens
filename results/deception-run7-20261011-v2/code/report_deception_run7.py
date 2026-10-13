"""Local Section 5.3 adaptation gallery with full transcripts and lens explorers."""
from __future__ import annotations

from collections import defaultdict
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

from prepare_deception_run4 import digest, read, sha
from run_deception_run4 import atomic_text, dump, now
from prepare_deception_run6 import SITES
from run7_readouts import capture_limit

PAPER = 'https://transformer-circuits.pub/2026/workspace/#audit-opus-examples'
REPORT_VERSION = 1


def safe_json(value):
    """Embed untrusted text in application/json without closing the script tag."""
    return (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            .replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
            .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def transcript(generations, tool_events=None):
    """Reconstruct saved chronological history, including actual tool feedback.

    Each generation records its full incoming history. Append only matching new
    messages, then the generated assistant output. Prefix mismatches are explicit
    provenance events rather than a silent replacement of earlier history.
    """
    result = []
    tool_events = tool_events or {}
    ordered = sorted(generations, key=lambda g: g.get('turn', 0))
    for g in ordered:
        incoming = [dict(role=m['role'], content=m['content']) for m in g['messages']]
        current = [{k: m[k] for k in ('role', 'content')} for m in result if m['role'] != 'provenance']
        if incoming[:len(current)] != current:
            result.append(dict(role='provenance', content='Saved incoming history differs from the preceding turn; the exact history follows.', turn=g.get('turn')))
            result.extend(dict(m, turn=g.get('turn'), incoming_history=True) for m in incoming)
        else:
            result.extend(dict(m, turn=g.get('turn')) for m in incoming[len(current):])
        content = g.get('assistant_history_text', g['raw_text'])
        result.append(dict(role='assistant', content=content, turn=g.get('turn'),
            complete=g.get('complete', False), raw_text=g['raw_text'], generation_id=g.get('id')))
        if g.get('turn') in tool_events:
            # The final action can be capped or invalid, leaving no subsequent
            # generation whose incoming history would contain this real result.
            event = tool_events[g['turn']]
            result.append(dict(role='tool', content=json.dumps(event['result'], sort_keys=True),
                               turn=g.get('turn'), actual_tool_event=True))
    return result


def _verify_generation(path):
    g = read(path)
    for name, expected in g.get('artifacts', {}).items():
        assert sha(path.parent / name) == expected, f'Changed source artifact: {path.parent/name}'
    return g


def explorer_data(generation_path, tokenizer=None):
    """Load a CPU readout after validating its generation and artifact hashes."""
    import torch
    path = Path(generation_path)
    g = _verify_generation(path)
    directory = path.parent / 'lens'
    metric_path = directory / 'metrics.json'
    if not metric_path.exists():
        return dict(turn=g.get('turn'), available=False, reason='Readouts have not been saved for this generation.',
                    complete=g.get('complete', False), generation_sha256=sha(path))
    m = read(metric_path)
    readout_path = directory / 'readouts.pt'
    assert m['generation_sha256'] == sha(path), 'Readouts reference a different generation'
    assert m['readouts_sha256'] == sha(readout_path), 'Readout artifact changed'
    execution_hash = g.get('revision_execution_sha256', g.get('execution_sha256'))
    if execution_hash is not None:
        assert m['revision_execution_sha256'] == execution_hash, 'Readout execution differs'
    r = torch.load(readout_path, map_location='cpu', weights_only=True)
    assert r['generation_sha256'] == sha(path)
    assert r['input_ids'] == g['input_ids'] and r['output_ids'] == g['output_ids']
    assert r['sites'] == SITES, 'Unregistered explorer site'
    assert r['states_sha256'] == g['artifacts']['states.pt'], 'State provenance differs'
    assert r['revision_execution_sha256'] == m['revision_execution_sha256']
    if tokenizer is not None:
        assert tokenizer.decode(g['output_ids']) == g['raw_text']
    n = len(g['input_ids'])
    stop = len(g['output_ids']) - int(bool(g['output_ids'] and g['output_ids'][-1] in (0, 2)))
    cap = capture_limit(g, r)
    assert r['positions'] == list(range(max(0, n - cap), n + stop)), 'Incomplete explorer capture'
    assert r['uncaptured_prompt'] == dict(begin=0, end_exclusive=max(0, n - cap)), 'Uncaptured prompt metadata differs'
    positions = r['positions']
    words = r['words']
    shape = (len(SITES), len(positions))
    tensors = {}
    vocabulary = set()
    for kind in ('jacobian', 'raw_logit'):
        data = r['readouts'][kind]
        for key in ('top_ids', 'top_values'):
            assert tuple(data[key].shape) == (*shape, 25), f'Invalid {key} shape'
        for key in ('concept_ranks', 'concept_values', 'concept_ties'):
            assert tuple(data[key].shape) == (*shape, len(words)), f'Invalid {key} shape'
        assert torch.isfinite(data['top_values']).all() and torch.isfinite(data['concept_values']).all()
        assert (data['concept_ranks'] >= 1).all() and (data['concept_ties'] >= 1).all()
        vocabulary.update(data['top_ids'].flatten().tolist())
        tensors[kind] = {name: tensor.tolist() for name, tensor in data.items()}
    assert tuple(r['prefix_absence'].shape) == (len(positions), len(words))
    ids = [g['input_ids'][p] if p < n else g['output_ids'][p - n] for p in positions]
    decoded = {str(token): (tokenizer.decode([token]) if tokenizer else f'ID {token}') for token in vocabulary}
    return dict(turn=g.get('turn'), available=True, complete=g.get('complete', False),
        sites=r['sites'], positions=positions, words=words, windows=r['windows'], boundary=r['boundary'],
        tokens=[dict(position=p, token_id=token, text=tokenizer.decode([token]) if tokenizer else f'ID {token}',
                     section='prompt' if p < n else 'generated') for p, token in zip(positions, ids)],
        decoded_tokens=decoded, readouts=tensors, prefix_absence=r['prefix_absence'].tolist(),
        uncaptured_prompt=r['uncaptured_prompt'], generation_sha256=sha(path),
        prompt_capture_tokens=cap,
        readouts_sha256=m['readouts_sha256'], states_sha256=r['states_sha256'],
        revision_execution_sha256=r['revision_execution_sha256'],
        normalization=r['normalization'])


STYLE = '''
body{font:15px/1.55 system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 22px;color:#202531;background:#fafbfc}
h1,h2,h3{line-height:1.25}a{color:#23559e}nav{margin:12px 0}code,pre{font:13px/1.5 ui-monospace,monospace}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f2f6;padding:14px;border-radius:6px;margin:8px 0}
table{border-collapse:collapse;width:100%;margin:14px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #dfe3e9;vertical-align:top}
label{display:inline-block;margin:8px 14px 8px 0}select,button{font:inherit;padding:5px}details{margin:10px 0}
.note{padding:12px 15px;background:#edf2fb;border-left:4px solid #4776be}.warn{background:#fff4dc;border-color:#aa7924}
.message{margin:14px 0;border-left:4px solid #c8ceda;padding-left:12px}.assistant{border-color:#4776be}.tool{border-color:#318372}
.badge{font-size:12px;padding:3px 6px;border-radius:4px;background:#e7eaf0;white-space:nowrap}.scroll{overflow:auto;max-height:760px;border:1px solid #ccd3df;background:white}
canvas{display:block}.selected{outline:2px solid #202531}#cell{min-height:120px}#selection{font-weight:600}.token-summary{max-height:250px;overflow:auto}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.card{padding:14px;background:white;border:1px solid #dfe3e9;border-radius:6px}
'''


EXPLORER_JS = r'''
const source=JSON.parse(document.getElementById('data').textContent), turns=source.turns;
const turnSelect=document.getElementById('turn'), kindSelect=document.getElementById('kind'),wordSelect=document.getElementById('word');
const canvas=document.getElementById('heatmap'),ctx=canvas.getContext('2d');
let active=null,word=0,kind='jacobian',chosen=null;
turns.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent='Turn '+t.turn+(t.available?'':' — readouts pending');turnSelect.append(o)});
function appendText(parent,tag,text){const el=document.createElement(tag);el.textContent=text;parent.append(el);return el}
function setTurn(){active=turns[Number(turnSelect.value)];wordSelect.replaceChildren();chosen=null;
 if(!active||!active.available){canvas.width=0;canvas.height=0;document.getElementById('capture').textContent=active?active.reason:'No saved readouts.';document.getElementById('cell').replaceChildren();return}
 active.words.forEach((w,i)=>{const o=document.createElement('option');o.value=i;o.textContent=w;wordSelect.append(o)});word=0;
 const omitted=active.uncaptured_prompt.end_exclusive;
 document.getElementById('capture').textContent='All '+active.positions.length+' saved positions × '+active.sites.length+' fixed sites, all four loops. Prompt positions 0–'+Math.max(-1,omitted-1)+(omitted?' were not captured.':' (none omitted).')+' End-of-sequence tokens have no residual capture. A source position shows predictions after consuming that token.';
 const text=document.getElementById('tokens');text.replaceChildren();active.tokens.forEach(t=>appendText(text,'span','['+t.position+' '+t.section+' id='+t.token_id+'] '+JSON.stringify(t.text)+' '));draw();}
function draw(){if(!active||!active.available)return;kind=kindSelect.value;word=Number(wordSelect.value);const d=active.readouts[kind];
 const width=19,height=10,left=112,top=96;canvas.width=left+active.positions.length*width;canvas.height=top+active.sites.length*height+26;
 ctx.fillStyle='#fff';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.font='11px system-ui';
 active.tokens.forEach((t,i)=>{ctx.save();ctx.translate(left+i*width+14,top-7);ctx.rotate(-Math.PI/2);ctx.fillStyle=t.section==='prompt'?'#52647c':'#202531';ctx.fillText(t.position+' '+JSON.stringify(t.text).slice(0,24),0,0);ctx.restore()});
 active.sites.forEach((s,row)=>{const y=top+row*height;ctx.fillStyle='#344258';ctx.fillText('loop '+(Math.floor(s/48)+1)+' · layer '+(s%48),3,y+9);
  active.positions.forEach((p,col)=>{const rank=d.concept_ranks[row][col][word];const score=Math.max(0,1-Math.log10(rank)/4);const light=98-score*58;ctx.fillStyle='hsl(216 66% '+light+'%)';ctx.fillRect(left+col*width,y,width-1,height-1)});
  if(row===0||Math.floor(active.sites[row-1]/48)!==Math.floor(s/48)){ctx.strokeStyle='#344258';ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke()}});
 const anchor=active.boundary&&active.boundary.source_position;
 if(anchor!==null&&anchor!==undefined){const col=active.positions.indexOf(anchor);if(col>=0){const x=left+(col+1)*width-1;ctx.strokeStyle='#16814c';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x,28);ctx.lineTo(x,canvas.height-22);ctx.stroke();ctx.fillStyle='#16814c';ctx.fillText('Immediate pre-value: source '+anchor,Math.max(left,Math.min(x+5,canvas.width-245)),22);ctx.lineWidth=1}}
 ctx.fillStyle='#344258';ctx.fillText('Darker = lower exact vocabulary rank; tracked word forms use their best logit. Green line marks the fixed pre-value source. Click for details.',left,canvas.height-8);
 if(chosen){ctx.strokeStyle='#d23c26';ctx.lineWidth=2;ctx.strokeRect(left+chosen.col*width,top+chosen.row*height,width,height);showCell(chosen.row,chosen.col)}}
function showCell(row,col){chosen={row,col};const d=active.readouts[kind],s=active.sites[row],t=active.tokens[col],w=active.words[word];const box=document.getElementById('cell');box.replaceChildren();
 appendText(box,'p','Source '+t.position+' ('+t.section+', token '+JSON.stringify(t.text)+'), loop '+(Math.floor(s/48)+1)+', physical layer '+(s%48)+', own-loop target '+((Math.floor(s/48)+1)*48-1)+'.');
 appendText(box,'p',w+': rank '+d.concept_ranks[row][col][word]+', tied vocabulary entries '+d.concept_ties[row][col][word]+', best logit '+d.concept_values[row][col][word].toFixed(5)+', literal word absent from consumed prefix: '+active.prefix_absence[col][word]+'.');
 const table=document.createElement('table');box.append(table);const head=document.createElement('tr');table.append(head);['Top rank','Token ID','Decoded token','Logit'].forEach(x=>appendText(head,'th',x));
 d.top_ids[row][col].forEach((id,i)=>{const tr=document.createElement('tr');table.append(tr);[i+1,id,JSON.stringify(active.decoded_tokens[String(id)]),d.top_values[row][col][i].toFixed(5)].forEach(v=>appendText(tr,'td',String(v)))});
 appendText(box,'p','Lexical hits are diagnostic. A word can reflect task content, anticipated narration, or another meaning; this view does not establish deceptive intent.');}
canvas.addEventListener('click',e=>{if(!active||!active.available)return;const rect=canvas.getBoundingClientRect();const col=Math.floor(((e.clientX-rect.left)*canvas.width/rect.width-112)/19),row=Math.floor(((e.clientY-rect.top)*canvas.height/rect.height-96)/10);if(row>=0&&row<active.sites.length&&col>=0&&col<active.positions.length){chosen={row,col};draw()}});
turnSelect.addEventListener('change',setTurn);kindSelect.addEventListener('change',draw);wordSelect.addEventListener('change',draw);setTurn();
'''


def _json_details(label, value):
    return '<details><summary>' + escape(label) + '</summary><pre>' + escape(json.dumps(value, indent=2)) + '</pre></details>'


def _case_view(directory, episode, tokenizer):
    generation_paths = sorted(directory.glob('turn-*/generation.json'), key=lambda p: int(p.parent.name.split('-')[-1]))
    source_hash = digest(dict(episode=sha(directory/'episode.json') if (directory/'episode.json').exists() else None,
        generations={str(p.relative_to(directory)): sha(p) for p in generation_paths},
        metrics={str(p.relative_to(directory)): sha(p) for p in sorted(directory.glob('turn-*/lens/metrics.json'))},
        tool_events={str(p.relative_to(directory)): sha(p) for p in sorted(directory.glob('tool-*.json'))},
        renderer_version=REPORT_VERSION, decoded=tokenizer is not None))
    cache = directory / 'view-provenance.json'
    view = directory / 'view.html'
    if cache.exists() and view.exists():
        previous = read(cache)
        if previous['source_digest'] == source_hash and previous['view_sha256'] == sha(view):
            return previous
    if episode:
        for rel, expected in episode.get('artifacts', {}).items():
            assert sha(directory / rel) == expected, f'Changed episode artifact: {rel}'
    generations = [_verify_generation(p) for p in generation_paths]
    tool_paths = sorted(directory.glob('tool-*.json'), key=lambda p: int(p.stem.split('-')[-1]))
    tool_events = {int(p.stem.split('-')[-1]): read(p) for p in tool_paths}
    conversations = transcript(generations, tool_events)
    turns = [explorer_data(p, tokenizer) for p in generation_paths]
    case_id = episode.get('id', directory.name) if episode else directory.name
    phase = episode.get('phase', generations[0].get('phase', 'unknown') if generations else 'unknown') if episode else generations[0].get('phase', 'unknown') if generations else 'unknown'
    complete = bool(episode and episode.get('complete'))
    eligible = episode.get('lens_claim_eligible') if episode else None
    limit = ('Known-concept lens calibration passed for this setting. Independent full-history semantic review remains pending.'
             if eligible is True else 'This setting is not eligible for a calibrated lens interpretation.' if eligible is False
             else 'Lens eligibility has not been assigned for this case.')
    body = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>' + escape(case_id) + '</title><style>' + STYLE + '</style></head><body>',
        '<nav><a href="../../analysis.html">All cases</a> · <a href="episode.json">Episode JSON</a> · <a href="' + PAPER + '">Paper Section 5.3</a></nav>',
        '<h1>' + escape(case_id) + '</h1>',
        '<p><span class="badge">' + escape(phase) + '</span> · ' + ('Complete saved episode' if complete else 'Incomplete or not yet finalized') + '</p>',
        '<p class="note">Ouro adaptation of four qualitative Section 5.3 scenarios. This is not an exact reproduction of the Opus model experiments. '
        'All saved cases are included. Mechanical tool facts, lexical readouts, and an independent assessment of intent are separate evidence.</p>',
        '<p class="note' + ('' if eligible else ' warn') + '">' + escape(limit) + '</p>',
        '<h2>Full chronological transcript</h2>']
    for message in conversations:
        role = message['role']
        label = role + (f' · generation turn {message["turn"]}' if message.get('turn') is not None else '')
        if role == 'assistant' and message.get('complete') is False:
            label += ' · incomplete generation'
        body.extend(['<section class="message ' + escape(role, quote=True) + '"><strong>' + escape(label) + '</strong>',
                     '<pre>' + escape(message['content']) + '</pre>'])
        if role == 'assistant' and message.get('raw_text') != message['content'] and message.get('raw_text') is not None:
            body.append('<details><summary>Exact raw output, including stop marker</summary><pre>' + escape(message['raw_text']) + '</pre></details>')
        body.append('</section>')
    body.append('<h2>Actual workspace and tool provenance</h2>')
    for label, key in (('Initial world', 'initial_world'), ('Final world', 'final_world'), ('Mechanical behavioral facts', 'behavioral_facts')):
        if episode and key in episode:
            body.append(_json_details(label, episode[key]))
    for path in tool_paths:
        body.append(_json_details('Actual tool event ' + path.stem, read(path)))
    if episode:
        body.append(_json_details('Saved episode, parsing outcomes, and artifact hashes', episode))
    links = ''.join('<li><a href="' + quote(str(path.relative_to(directory))) + '">' + escape(str(path.relative_to(directory))) + '</a></li>'
                    for path in sorted(directory.rglob('*.json')) if path.name not in ('view-provenance.json',))
    body.append('<details><summary>Source JSON artifacts</summary><ul>' + links + '</ul></details>')
    body.extend(['<h2>Every captured token and registered layer</h2>',
        '<p>Prompt and generated source tokens are shown in order. All four loops retain their fixed physical layers 24–40. '
        'J-lens uses each source’s own loop-end transport; raw logits are the comparator. Colors show exact full-vocabulary ranks for fixed tracked word forms.</p>',
        '<label>Generation <select id="turn"></select></label><label>Readout <select id="kind"><option value="jacobian">J-lens</option><option value="raw_logit">Raw logits</option></select></label>',
        '<label>Tracked word <select id="word"></select></label><p id="capture"></p>',
        '<div class="scroll"><canvas id="heatmap" aria-label="Token by layer readout rank heatmap"></canvas></div>',
        '<details><summary>All captured source token IDs and decoded token pieces</summary><pre class="token-summary" id="tokens"></pre></details>',
        '<div id="cell"><p>Click a token and layer to inspect the top 25 entries, exact tracked-word rank, ties, logit, and prefix absence.</p></div>',
        _json_details('Fixed window definitions and pre-value boundaries', [dict(turn=t.get('turn'), windows=t.get('windows'), boundary=t.get('boundary')) for t in turns]),
        '<script id="data" type="application/json">' + safe_json(dict(turns=turns)) + '</script>',
        '<script>' + EXPLORER_JS + '</script></body></html>'])
    atomic_text(view, ''.join(body))
    receipt = dict(source_digest=source_hash, view_sha256=sha(view), renderer_version=REPORT_VERSION,
        generation_sha256={str(p.relative_to(directory)): sha(p) for p in generation_paths},
        verified_readouts=[dict(turn=t.get('turn'), generation_sha256=t['generation_sha256'], readouts_sha256=t.get('readouts_sha256'),
                               states_sha256=t.get('states_sha256')) for t in turns],
        tokenizer_decoded=tokenizer is not None)
    dump(cache, receipt)
    return receipt


def _case_inventory(root):
    path = root / 'case-inventory.json'
    if not path.exists():
        return []
    value = read(path)
    if isinstance(value, list):
        return value
    # The runner may retain richer per-phase inventory metadata.
    for name in ('cases', 'main', 'episodes', 'inventory'):
        if isinstance(value.get(name), list):
            return value[name]
    rows = []
    for phase, items in value.items():
        if isinstance(items, list):
            rows.extend(dict(item, phase=item.get('phase', phase)) for item in items if isinstance(item, dict))
    return rows


def render(root, extra=None, tokenizer=None):
    """Publish every saved, partial, planned, and explicitly excluded case."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metadata = read(root/'metadata.json') if (root/'metadata.json').exists() else {}
    status = read(root/'status.json') if (root/'status.json').exists() else {}
    rows, aggregate = [], defaultdict(lambda: dict(saved=0, complete=0, task_correct=0, valid_actions=0, attempts=0))
    directories = sorted((root/'cases').glob('*')) if (root/'cases').exists() else []
    for directory in directories:
        if not directory.is_dir():
            continue
        path = directory / 'episode.json'
        episode = read(path) if path.exists() else None
        generation_paths = list(directory.glob('turn-*/generation.json'))
        if not episode and not generation_paths:
            continue
        first = read(generation_paths[0]) if generation_paths else {}
        data = episode or first
        receipt = _case_view(directory, episode, tokenizer)
        row = dict(id=data.get('id', directory.name), family=data.get('family', 'measurement'), phase=data.get('phase', 'unknown'),
            setting=data.get('setting', 'pending'), condition=data.get('condition', ''),
            saved_episode=episode is not None, complete=bool(episode and episode.get('complete')),
            task_correct=episode.get('task_correct', episode.get('answer_correct')) if episode else None,
            attempts=episode.get('attempts', 0) if episode else 0,
            valid_actions=episode.get('valid_actions', 0) if episode else 0,
            lens_claim_eligible=episode.get('lens_claim_eligible') if episode else data.get('lens_claim_eligible'),
            semantic_review_complete=bool(episode and episode.get('semantic_review_complete')),
            href=str((directory/'view.html').relative_to(root)), source_sha256=sha(path) if path.exists() else None,
            view_sha256=receipt['view_sha256'], generated_turns=len(generation_paths))
        rows.append(row)
        key = (row['phase'], row['family'], row['setting'], row['condition'])
        a = aggregate[key]
        a['saved'] += int(row['saved_episode']); a['complete'] += int(row['complete']); a['task_correct'] += int(row['task_correct'] is True)
        a['valid_actions'] += row['valid_actions']; a['attempts'] += row['attempts']
    saved = {row['id'] for row in rows}
    inventory = _case_inventory(root)
    unsaved = [row for row in inventory if row.get('id') not in saved]
    analysis = dict(title=metadata.get('title', 'Run 7: Section 5.3 qualitative adaptation'), utc=now(), status=status,
        total_saved_episodes=sum(r['saved_episode'] for r in rows), total_partial_cases=sum(not r['saved_episode'] for r in rows),
        all_saved_cases=rows, planned_or_excluded_cases=unsaved,
        aggregate=[dict(phase=k[0], family=k[1], setting=k[2], condition=k[3], **v) for k,v in sorted(aggregate.items())],
        family_selection=extra, independent_semantic_review='Pending; mechanical behavior and lexical readouts do not establish intent.',
        paper_section_5_3=PAPER, model_scope='Ouro adaptation; exact Opus reproduction not claimed.',
        selection_policy='Every saved case, including negative outcomes, calibration failures, baselines and incomplete cases, is retained.')
    dump(root/'analysis.json', analysis)
    title = analysis['title']
    body = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>' + escape(title) + '</title><style>' + STYLE + '</style></head><body><h1>' + escape(title) + '</h1>',
        '<nav><a href="index.html">Live progress</a> · <a href="plan/protocol.json">Frozen protocol</a> · <a href="case-inventory.json">Complete case inventory</a> · <a href="blind-review.html">Lens-blind full histories</a> · '
        '<a href="analysis.json">Analysis JSON</a> · <a href="' + PAPER + '">Paper Section 5.3</a></nav>',
        '<p class="note">This run adapts the paper’s four qualitative scenarios to Ouro: fabricated bugs, edited metrics, silent corrections, and self-issued approval. '
        'Every saved episode is shown, including controls, failed calibration, incorrect baselines, negative outcomes and incomplete generations. '
        'Full transcripts include actual tool feedback, factual baseline history and workspace provenance.</p>',
        '<p class="note warn">Mechanical behavioral facts and lexical lens readouts are separate from an independent full-history semantic review. '
        'A tracked-word hit does not by itself establish deceptive intent. A failed known-concept lens check makes the relevant setting ineligible for a calibrated lens claim; '
        'the behavioral transcript remains visible.</p>',
        '<p>' + str(analysis['total_saved_episodes']) + ' saved episodes · ' + str(analysis['total_partial_cases']) + ' partial cases · ' + str(len(unsaved)) + ' planned or excluded cases.</p>',
        _json_details('Live status', status)]
    if extra is not None:
        body.append(_json_details('Per-family tool capability and lens calibration decisions', extra))
    if rows:
        body.append('<label>Family <select id="family"><option value="">All</option>' + ''.join('<option>' + escape(f) + '</option>' for f in sorted({r['family'] for r in rows})) + '</select></label>')
        body.append('<label>Phase <select id="phase"><option value="">All</option>' + ''.join('<option>' + escape(p) + '</option>' for p in sorted({r['phase'] for r in rows})) + '</select></label>')
        body.append('<table id="cases"><thead><tr><th>Case and transcript</th><th>Family / phase</th><th>Setting / condition</th><th>Completion</th><th>Mechanical correctness</th><th>Lens claim eligibility</th></tr></thead><tbody>')
        for row in rows:
            label = 'Complete' if row['complete'] else 'Incomplete' if row['saved_episode'] else 'Partial'
            correct = 'Yes' if row['task_correct'] is True else 'No' if row['task_correct'] is False else 'Pending'
            eligibility = 'Passed known-concept check' if row['lens_claim_eligible'] is True else 'Ineligible' if row['lens_claim_eligible'] is False else 'Pending / diagnostic'
            body.append('<tr data-family="' + escape(row['family'], quote=True) + '" data-phase="' + escape(row['phase'], quote=True) + '"><td><a href="' + quote(row['href']) + '">' + escape(row['id']) + '</a></td><td>' + escape(row['family']) + '<br>' + escape(row['phase']) + '</td><td>' + escape(row['setting']) + '<br>' + escape(row['condition']) + '</td><td>' + label + '</td><td>' + correct + '</td><td>' + eligibility + '</td></tr>')
        body.append('</tbody></table><script>for(const id of ["family","phase"]){document.getElementById(id).addEventListener("change",()=>{const family=document.getElementById("family").value,phase=document.getElementById("phase").value;for(const row of document.querySelectorAll("#cases tbody tr")){row.hidden=!!((family&&row.dataset.family!==family)||(phase&&row.dataset.phase!==phase))}})}</script>')
    else:
        body.append('<p>No episodes have been generated yet. The protocol, complete planned inventory, and progress links are available above.</p>')
    if unsaved:
        body.append('<h2>Planned and excluded cases</h2><p>Excluded records remain in this inventory with their explicit reason. Missing cases are not counted as successful outcomes.</p><table><tr><th>Case</th><th>Family</th><th>Status</th><th>Reason</th></tr>')
        for row in unsaved:
            body.append('<tr><td>' + escape(str(row.get('id', row.get('record_id', 'unknown')))) + '</td><td>' + escape(str(row.get('family', ''))) + '</td><td>' + escape(str(row.get('status', 'planned'))) + '</td><td>' + escape(str(row.get('reason', row.get('exclusion_reason', '')))) + '</td></tr>')
        body.append('</table>')
    body.append('</body></html>')
    atomic_text(root/'analysis.html', ''.join(body))
    return analysis
