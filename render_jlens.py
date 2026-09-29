"""Render the fitted-lens summary and held-out token readouts as static HTML."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
out = ROOT/'results/jlens'
summaries = [json.loads(p.read_text()) for p in out.glob('fit-summary-*.json')]
summary = max(summaries, key=lambda x:x['n_prompts'])
readouts = json.loads((out/'heldout-readouts.json').read_text())
assert readouts['n_fitting_prompts'] == summary['n_prompts']
esc = lambda x:html.escape(str(x))
labels = {str(s['site']):f"Pass {s['recurrent_pass']}, layer {s['physical_layer']}" for s in summary['source_labels']}
seconds = sum(s['invocation_seconds'] for s in summaries if s['n_prompts'] <= summary['n_prompts'])
parts = ['<!doctype html><meta charset="utf-8"><title>Ouro Jacobian lens pilot</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;font:16px/1.5 system-ui;color:#17212b}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border-bottom:1px solid #d8dfe5;text-align:left;padding:10px;vertical-align:top}code,pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8}pre{padding:14px}details{border:1px solid #ccd4dc;border-radius:6px;padding:12px;margin:14px 0}summary{cursor:pointer;font-weight:600}.note{background:#fff5d9;padding:14px}</style><h1>Ouro-2.6B Jacobian lens</h1><p>Historical four-site pilot; see the standalone Ouro workspace explorer for the full fit.</p>']
parts.append(f'<p><strong>{summary["n_prompts"]} fitting passages · 64 tokens each · 4 fitted sites · {seconds:.1f} seconds of fitting</strong></p>')
parts.append('<p class="note">This is a small pilot lens, fitted on neutral text. It is not an honesty classifier and does not establish lens quality across tasks. Only the four listed sites are fitted; the adapter supports all 192 layer/pass positions.</p>')
parts.append('<p>Each row below lists the top vocabulary tokens after transporting an activation into the final-layer basis. The output row uses the model’s actual final readout. Token rankings are qualitative checks; their logits are not honesty scores.</p>')
parts.append('<h2>Fitted sites</h2><table><tr><th>Virtual site (zero-based)</th><th>Pass and physical layer (one-based)</th><th>Matrix shape</th></tr>')
for site,label in labels.items(): parts.append(f'<tr><td>{site}</td><td>{label}</td><td>2048 × 2048</td></tr>')
parts.append('</table><p>Target: pass 4, layer 48, before final normalization. Readout applies the final normalization exactly once. Weights were frozen and KV caching disabled during fitting.</p>')
for result in readouts['results']:
    parts.append('<h2>Held-out readout</h2><pre>'+esc(result['prompt'])+'</pre><table><tr><th>Readout</th><th>Top 10 next-token readings</th></tr>')
    rows = [(labels[site],tokens) for site,tokens in result['lens_top10'].items()]
    rows.append(('Model output',result['model_top10']))
    for label,tokens in rows:
        cells=' &nbsp; '.join('<code>'+esc(repr(t['token']))+'</code>' for t in tokens)
        parts.append('<tr><td>'+esc(label)+'</td><td>'+cells+'</td></tr>')
    parts.append('</table>')
parts.append('<details><summary>Full fitting summary</summary><pre>'+esc(json.dumps(summary,indent=2))+'</pre></details>')
parts.append('<p><a href="https://github.com/ishwarbb/ouro-jlens/tree/main/results/jlens">Verification and machine-readable readouts on GitHub</a></p>')
text='\n'.join(parts)
(ROOT/'jlens.html').write_text(text)
(ROOT/'site').mkdir(exist_ok=True)
(ROOT/'site/jlens.html').write_text(text)
print(ROOT/'site/jlens.html')
