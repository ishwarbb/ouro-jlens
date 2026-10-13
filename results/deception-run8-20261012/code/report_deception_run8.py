"""Run 8 gallery, interface comparison and deferred-work/review navigation."""
from __future__ import annotations
from html import escape
import json
from pathlib import Path

from prepare_deception_run4 import read
from run_deception_run4 import atomic_text,dump,now
from report_deception_run7 import render as render_captures,STYLE


def render(root,extra=None,tokenizer=None):
    root=Path(root)
    result=render_captures(root,extra=extra,tokenizer=tokenizer)
    episodes=[read(path) for path in (root/'cases').glob('*/episode.json')]
    groups={}
    for ep in episodes:
        key=(ep['phase'],ep.get('interface','diagnostic'),ep.get('family','measurement'))
        row=groups.setdefault(key,dict(saved=0,complete=0,correct=0))
        row['saved']+=1;row['complete']+=int(ep['complete'])
        row['correct']+=int(ep.get('task_correct',ep.get('answer_correct',False)))
    result['interface_groups']=[dict(phase=k[0],interface=k[1],family=k[2],**v) for k,v in sorted(groups.items())]
    result['review_backlog']=read(root/'review-backlog.json') if (root/'review-backlog.json').exists() else None
    result['deferred_work']=read(root/'plan/deferred-work.json') if (root/'plan/deferred-work.json').exists() else None
    dump(root/'analysis.json',result)
    nav=('<p class="note"><a href="comparison.html">Example-control comparison</a> · '
         '<a href="deferred-work.html">Deferred-work ledger</a> · '
         '<a href="reviews.html">Independent review packages</a> · '
         '<a href="plan/tasks.json">All registered tasks</a></p>')
    page=root/'analysis.html';html=page.read_text()
    marker=html.index('</h1>')+len('</h1>');atomic_text(page,html[:marker]+nav+html[marker:])
    comparison=extra.get('example_comparison') if isinstance(extra,dict) else None
    if comparison is None and (root/'run8-summary.json').exists():comparison=read(root/'run8-summary.json')['example_comparison']
    body=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Run8 example controls</title><style>'+STYLE+'</style><h1>Run8: example controls</h1>',
        '<nav><a href="analysis.html">All transcripts and readouts</a> · <a href="plan/protocol.json">Frozen protocol</a></nav>',
        '<p class="note">The primary interface specifies fields and types without example argument values. '
        'The matched comparison adds the original numeric examples to the same instructions. '
        'Compare the two interfaces within each record, condition and repetition.</p>',
        '<p>First edit matching the original dummy rule, any temporary defect and a persistent final defect are separate measures. '
        'A match does not establish copying intent or concealed strategy. Incomplete cases remain visible; paired conclusions use complete sixteen-episode record blocks.</p>']
    if comparison is None:body.append('<p>The 96 comparison episodes are registered and awaiting execution.</p>')
    else:body.append('<pre>'+escape(json.dumps(comparison,indent=2))+'</pre>')
    body.append('<h2>Saved phases and interfaces</h2><table><tr><th>Phase</th><th>Interface</th><th>Family</th><th>Saved</th><th>Complete</th><th>Artifact-correct</th></tr>')
    for row in result['interface_groups']:
        body.append('<tr>'+''.join('<td>'+escape(str(row[k]))+'</td>' for k in ('phase','interface','family','saved','complete','correct'))+'</tr>')
    body.append('</table>');atomic_text(root/'comparison.html',''.join(body))
    ledger=result['deferred_work'] or dict(items=[])
    body=['<!doctype html><meta charset="utf-8"><title>Run8 deferred work</title><style>'+STYLE+'</style>',
        '<h1>Previously deferred evaluations</h1><nav><a href="analysis.html">Run8 cases</a> · <a href="reviews.html">Review prerequisites</a></nav>',
        '<p class="note">This ledger distinguishes fresh Run8 stages, completed evidence that needs review, and older conditional execution. '
        'No failed old gate is overridden. Completed studies are not regenerated.</p>']
    for row in ledger['items']:
        body.append('<details><summary>'+escape(row['id'])+' · '+escape(row['status'])+'</summary><pre>'+escape(json.dumps(row,indent=2))+'</pre></details>')
    atomic_text(root/'deferred-work.html',''.join(body))
    reviewroot=root/'reviews'
    body=['<!doctype html><meta charset="utf-8"><title>Run8 independent review</title><style>'+STYLE+'</style>',
        '<h1>Independent full-history reviews</h1><nav><a href="plan/review-sources.json">Frozen source inventory</a> · <a href="review-backlog.json">Review backlog</a></nav>',
        '<p class="note">Review the raw conversations and actual artifacts before inspecting lens readouts. '
        'Copy an annotation template to a new file before editing it; the prepared template is frozen. '
        'Two distinct reviewers and unresolved disagreement handling are required. A wrong action or a keyword does not establish intent.</p><ul>']
    for folder in sorted(reviewroot.glob('*')):
        if folder.is_dir() and (folder/'blind-review.html').exists():
            relative=str(folder.relative_to(root))
            body.append('<li><a href="'+escape(relative,quote=True)+'/blind-review.html">'+escape(folder.name)+'</a> · '
                '<a href="'+escape(relative,quote=True)+'/annotation-template.json">Annotation template</a> · '
                '<a href="'+escape(relative,quote=True)+'/blind-evidence.json">Raw evidence</a></li>')
    body.append('</ul><p>New Run8 main and comparison review packets are exported after generation completes. '
        'Ingested reviews are saved in Run8 receipts; they do not rewrite the older gate files.</p>')
    atomic_text(root/'reviews.html',''.join(body))
    return result
