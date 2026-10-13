"""Build a compact public record of research runs without changing live inputs."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
from html import escape
import json
from pathlib import Path
from urllib.parse import quote

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/published-20261013'
STYLE='body{font:16px/1.55 system-ui;max-width:1250px;margin:32px auto;padding:0 22px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f7;padding:12px}td,th{padding:8px;border:1px solid #ccc;text-align:left;vertical-align:top}table{border-collapse:collapse;width:100%}details{margin:16px 0}'


def read(path):return json.loads(path.read_text())


def sha(path):return sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def link(path,from_directory):
    import os
    return quote(os.path.relpath(path,from_directory),safe='/')


def details(label,value):
    return '<details><summary>'+escape(label)+'</summary><pre>'+escape(json.dumps(value,indent=2,ensure_ascii=False))+'</pre></details>'


def case_page(directory,paths,output):
    body=['<!doctype html><meta charset="utf-8"><title>'+escape(directory.name)+'</title><style>'+STYLE+'</style>',
          '<h1>'+escape(directory.name)+'</h1><p><a href="../../index.html">Published run archive</a></p>',
          '<p>Compact saved evidence. Full incoming messages and raw outputs are preserved below. This view contains no per-token heatmap or assessment of intent.</p>']
    ep=directory/'episode.json'
    if ep.exists():
        value=read(ep)
        body.append(details('Episode and actual workspace facts',value))
    for path in sorted(paths):
        value=read(path)
        body.append('<h2>'+escape(str(path.relative_to(directory)))+'</h2><p>SHA256 '+sha(path)+' · <a href="'+link(path,output.parent)+'">Original JSON</a></p>')
        body.append(details('Complete incoming messages',value.get('messages',value.get('prompt',[]))))
        body.append('<h3>Exact raw output</h3><pre>'+escape(value.get('raw_text',value.get('text',value.get('response',''))))+'</pre>')
        body.append(details('Saved completion and provenance', {k:value[k] for k in ('id','turn','phase','condition','setting','complete','capped','natural_stop','job_sha256','revision_execution_sha256','artifacts') if k in value}))
    for path in sorted(directory.glob('tool-*.json')):
        body.append(details('Actual tool evidence '+path.name,read(path)))
    body.append('<p>State, RNG and readout tensors are local assets. Their filenames and hashes remain in the saved JSON records.</p>')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(''.join(body))


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-10-13',cohorts=[],
                  snapshot_policy='Read-only snapshot of saved records; active runs are partial. Original scientific files and hashes are unchanged.',
                  excluded_policy=['Model weights and virtual environment','Local .pt/.safetensors captures and checkpoints','Large per-token cases/*/view.html and their derived cache receipts','Runtime logs/PIDs/locks/temporary files','Personal viewer runs outside the five already published examples'],
                  compact_views='Entire saved incoming messages and exact raw outputs, episode/workspace facts and tool JSON. No heatmaps; missing tensor assets remain explicit.',
                  existing_released_assets='v0.1.0 covers earlier lens/inventory/viewer bundles, not all new deception captures.')
    body=['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>Research runs through October 13 2026</title><style>'+STYLE+'</style>',
          '<h1>Research runs through October 13 2026</h1><p><a href="../../RUN_RESULTS.md">Objectives and results</a> · <a href="manifest.json">Publication coverage</a></p>',
          '<p>These are saved snapshots. Running and gated phases are identified by their source status; queued stages are not completed experiments. Compact transcripts require no model tensors.</p>',
          '<table><tr><th>Results folder</th><th>Source status</th><th>Saved generation files</th><th>Saved episodes</th><th>Compact evidence</th></tr>']
    for cohort in sorted((ROOT/'results').iterdir()):
        if not cohort.is_dir() or cohort==OUT or cohort.name=='ouro-viewer':continue
        status=read(cohort/'status.json') if (cohort/'status.json').exists() else {}
        generations=sorted(cohort.rglob('generation.json'))
        episodes=sorted((cohort/'cases').glob('*/episode.json')) if (cohort/'cases').exists() else []
        groups=defaultdict(list)
        for path in generations:
            directory=path.parent.parent if path.parent.name.startswith('turn-') else path.parent
            groups[directory].append(path)
        entries=[]
        for directory,paths in sorted(groups.items()):
            relative=directory.relative_to(cohort)
            slug=sha256(str(relative).encode()).hexdigest()[:16]
            output=OUT/'cases'/cohort.name/(slug+'.html')
            case_page(directory,paths,output)
            entries.append(dict(case=str(relative),page=str(output.relative_to(OUT)),generations=len(paths)))
        page=OUT/'cohorts'/(cohort.name+'.html');page.parent.mkdir(parents=True,exist_ok=True)
        html=['<!doctype html><meta charset="utf-8"><title>'+escape(cohort.name)+'</title><style>'+STYLE+'</style><h1>'+escape(cohort.name)+'</h1><p><a href="../index.html">All runs</a></p>',details('Source status at snapshot',status)]
        for p in [cohort/name for name in ('findings.json','summary.json','analysis.json','section53-summary.json','run8-summary.json','held-out-statistics.json','REPORT.md','FINDINGS.md','CONCLUSIONS.md','VALIDATION.md') if (cohort/name).exists()]:
            html.append('<p><a href="'+link(p,page.parent)+'">'+escape(p.name)+'</a></p>')
        for row in entries:html.append('<p><a href="'+link(OUT/row['page'],page.parent)+'">'+escape(row['case'])+'</a> · '+str(row['generations'])+' generation files</p>')
        page.write_text(''.join(html))
        phase_counts=dict(Counter(read(p).get('phase','unspecified') for p in episodes))
        row=dict(folder=cohort.name,source_status=status,generation_files=len(generations),episodes=len(episodes),episode_phases=phase_counts,compact_cases=len(entries),page=str(page.relative_to(OUT)))
        manifest['cohorts'].append(row)
        stage=status.get('stage',status.get('status','See existing report'))
        body.append('<tr><td>'+escape(cohort.name)+'</td><td>'+escape(str(stage))+'</td><td>'+str(len(generations))+'</td><td>'+str(len(episodes))+'</td><td><a href="'+row['page']+'">'+str(len(entries))+' cases and reports</a></td></tr>')
    body.append('</table><p>The five existing public viewer examples remain in results/ouro-viewer; personal viewer sessions are excluded.</p>')
    (OUT/'index.html').write_text(''.join(body));write(OUT/'manifest.json',manifest)
    print(json.dumps(dict(cohorts=len(manifest['cohorts']),compact_cases=sum(r['compact_cases'] for r in manifest['cohorts']),snapshot_utc=manifest['snapshot_utc'])))


if __name__=='__main__':build()
