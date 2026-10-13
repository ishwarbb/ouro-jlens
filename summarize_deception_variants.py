"""Make a browsable screen report without treating lexical flags as judgments."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import re


def normalized(text):
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args();root=a.directory
    protocol=json.loads((root/"protocol.json").read_text())
    reviews=json.loads((root/"reviews.json").read_text()) if (root/"reviews.json").exists() else {}
    rows=[]
    for job in protocol["jobs"]:
        path=root/job["id"]/"generation.json"
        if not path.exists():continue
        d=json.loads(path.read_text())
        d["advantageous_claim_verbatim"]=normalized(d["advantageous_claim"]) in normalized(d["recipient_text"])
        d["review"]=reviews.get(d["id"],{})
        rows.append(d)
    counts={}
    for condition in ["accuracy","incentive","instructed"]:
        for form in ["native","reasoning"]:
            subset=[d for d in rows if d["condition"]==condition and d["form"]==form]
            counts[f"{condition}/{form}"]=dict(n=len(subset),capped=sum(d["capped"] for d in subset),
                degenerate=sum(d["degenerate"] for d in subset),lexical_candidates=sum(bool(d["candidate_patterns"]) for d in subset),
                exact_claim_candidates=sum(d["advantageous_claim_verbatim"] for d in subset),
                reviewed=sum(bool(d["review"]) for d in subset))
    summary=dict(completed=len(rows),planned=len(protocol["jobs"]),counts=counts,
        exact_claim_candidates=[d["id"] for d in rows if d["advantageous_claim_verbatim"]],
        reviews=reviews,note="All automated flags are triage only. Negation, quotation, role compliance and task confusion require manual review.")
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    # Compact review queue preserves all raw outputs in the per-case artifacts.
    queue=[dict(id=d["id"],condition=d["condition"],scenario=d["scenario"],form=d["form"],
                exact_claim=d["advantageous_claim_verbatim"],candidate=bool(d["candidate_patterns"]),
                record=d["authoritative_record"],recipient=d["recipient_text"],explanation=d["explanation_text"],
                capped=d["capped"],degenerate=d["degenerate"]) for d in rows]
    (root/"review-queue.json").write_text(json.dumps(queue,indent=2,ensure_ascii=False)+"\n")
    html=["<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Ouro broad deception screen</title>",
          "<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 22px;color:#172438;background:#f7f9fc}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:left;border-bottom:1px solid #cdd6e0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf1f6;padding:14px;font:14px/1.5 ui-monospace,monospace}article{background:white;border:1px solid #d7dfe9;border-radius:8px;padding:18px;margin:16px 0}summary{cursor:pointer}input,select{padding:10px;margin:5px;border:1px solid #adbccc;border-radius:5px}.note{padding:15px;background:#e7eff9}a{color:#245ca0}</style>",
          f"<h1>Ouro: deception screen</h1><p>{len(rows)} / {len(protocol['jobs'])} generations saved across {protocol['scenarios']} synthetic scenarios.</p>",
          "<p class='note'>Accuracy controls, conflicting incentives, and explicit contradictory instructions are separate conditions. Automated text matches nominate cases only; they are not deception diagnoses. Exact input/output IDs are retained. Selected J-lens results are exploratory and cannot estimate detector accuracy.</p>",
          "<p><a href='protocol.json'>Frozen prompts and sampling protocol</a> · <a href='summary.json'>Summary JSON</a> · <a href='reviews.json'>Manual reviews</a> · <a href='lens/index.html'>Selected J-lens cases</a></p>",
          "<table><tr><th>Condition / format</th><th>Saved</th><th>Capped</th><th>Degenerate</th><th>Exact claim text matches</th><th>Manually reviewed</th></tr>"]
    for key,d in counts.items():html.append(f"<tr><td>{key}</td><td>{d['n']}</td><td>{d['capped']}</td><td>{d['degenerate']}</td><td>{d['exact_claim_candidates']}</td><td>{d['reviewed']}</td></tr>")
    html.extend(["</table><p>Exact matches can still occur inside a denial or quote. They must be read in context.</p>",
        "<label>Search <input id='q' placeholder='scenario, claim, review'></label><select id='condition'><option value=''>All conditions</option><option>accuracy</option><option>incentive</option><option>instructed</option></select><select id='form'><option value=''>Both formats</option><option>native</option><option>reasoning</option></select><label><input id='flag' type='checkbox'>Exact claim text matches only</label>"])
    for d in sorted(rows,key=lambda d:(not bool(d["review"]),not d["advantageous_claim_verbatim"],d["id"])):
        label=d["review"].get("classification","Not manually reviewed")
        html.extend([f"<article data-condition='{d['condition']}' data-form='{d['form']}' data-flag='{int(d['advantageous_claim_verbatim'])}' id='{d['id']}'>",
            f"<h2>{d['id']}</h2><p><b>{escape(label)}</b> · {d['output_tokens']} tokens · capped={d['capped']} · degenerate={d['degenerate']}</p>",
            f"<p><b>Authoritative record:</b> {escape(d['authoritative_record'])}</p>",
            f"<p><b>Recipient message:</b></p><pre>{escape(d['recipient_text'])}</pre>",
            f"<details><summary>Reasoning / explanation</summary><pre>{escape(d['explanation_text'])}</pre></details>",
            f"<details><summary>Full raw response and manual review</summary><pre>{escape(d['raw_text'])}</pre><pre>{escape(json.dumps(d['review'],indent=2,ensure_ascii=False))}</pre></details>",
            f"<p><a href='{d['id']}/generation.json'>Exact prompt, token IDs and response</a></p></article>"])
    html.append("<script>function filter(){let q=document.querySelector('#q').value.toLowerCase(),c=document.querySelector('#condition').value,f=document.querySelector('#form').value,k=document.querySelector('#flag').checked;document.querySelectorAll('article').forEach(a=>a.hidden=(c&&a.dataset.condition!==c)||(f&&a.dataset.form!==f)||(k&&a.dataset.flag!=='1')||!a.textContent.toLowerCase().includes(q))}document.querySelectorAll('input,select').forEach(e=>e.addEventListener('input',filter));</script></html>")
    (root/"index.html").write_text("\n".join(html))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
