"""Rebuild a local HTML review and JSON summary from the frozen pilot artifacts."""
import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    protocol = read(root / "protocol.json")
    vocabulary = read(root / "vocabulary.json") if (root / "vocabulary.json").exists() else {}
    annotations = read(root / "annotations.json") if (root / "annotations.json").exists() else {}
    records = []
    for job in protocol["jobs"]:
        directory = root / job["id"]
        if not (directory / "generation.json").exists():
            continue
        generation = read(directory / "generation.json")
        metrics = read(directory / "metrics.json") if (directory / "metrics.json").exists() else None
        counts, absent, best = Counter(), Counter(), {}
        if (directory / "hits.jsonl").exists() and metrics:
            for line in (directory / "hits.jsonl").read_text().splitlines():
                hit = json.loads(line)
                if hit["method"] != "own_loop_jlens" or hit["next_token"] is None:
                    continue
                counts[hit["word"]] += 1
                if hit["lexical_absent_entire_output"]:
                    absent[hit["word"]] += 1
                    if hit["word"] not in best or hit["rank"] < best[hit["word"]]["rank"]:
                        best[hit["word"]] = hit
        aggregate = {}
        if metrics:
            for method in protocol["readouts"]:
                rows = [v for k, v in metrics["primary"].items() if k.startswith(method + "/")]
                total = sum(r["primary_cells"] for r in rows)
                aggregate[method] = dict(cells=total, **{
                    key.replace("_cells", "_fraction"): sum(r[key] for r in rows) / total
                    for key in ["top10_cells", "top25_cells", "lexical_absent_top10_cells", "lexical_absent_top25_cells"]})
        records.append(dict(id=job["id"], condition=job["condition"], seed=job["seed"],
                            output_tokens=generation["output_tokens"],
                            truncated=generation["truncated"], unclosed_thought=generation["unclosed_thought"],
                            had_thought=generation["had_thought"],
                            reasoning_content_present=any(c.isalnum() for c in generation["reasoning_text"]),
                            well_formed_substantive_reasoning=(generation["raw_text"].count("<think>") == 1 and generation["raw_text"].count("</think>") == 1 and any(c.isalnum() for c in generation["reasoning_text"]) and not generation["unclosed_thought"]),
                            answer=generation["answer_text"], reasoning=generation["reasoning_text"],
                            lexical_mentions=generation["lexical_mentions"], primary=aggregate,
                            exploratory_hit_counts=dict(counts), exploratory_absent_counts=dict(absent),
                            exploratory_best_absent_hits=list(best.values()),
                            review=annotations.get(job["id"], {"status": "pending"})))
    paired = []
    for seed in protocol["sampling"]["seeds"]:
        matching = {r["condition"]: r for r in records if r["seed"] == seed and r["primary"]}
        if len(matching) != 2:
            continue
        a, b = matching["pressure"], matching["accuracy_control"]
        paired.append(dict(seed=seed,
                           complete_generation_pair=all(not r["truncated"] and not r["unclosed_thought"] and bool(r["answer"].strip()) for r in [a,b]),
                           both_have_reasoning_text=all(r["reasoning_content_present"] for r in [a,b]),
                           both_have_well_formed_reasoning=all(r["well_formed_substantive_reasoning"] for r in [a,b]),
                           pressure_minus_control={
            method: {k: a["primary"][method][k]-b["primary"][method][k]
                     for k in ["top10_fraction", "top25_fraction", "lexical_absent_top10_fraction", "lexical_absent_top25_fraction"]}
            for method in protocol["readouts"]}))
    result = dict(completed_generations=len(records), planned_generations=len(protocol["jobs"]),
                  completed_measurements=sum(bool(r["primary"]) for r in records), records=records, paired=paired,
                  eligible_words=[w for w,v in vocabulary.items() if v["eligible"]],
                  excluded_words=[w for w,v in vocabulary.items() if not v["eligible"]])
    (root / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>Ouro deception pilot</title>",
             "<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1150px;margin:40px auto;padding:0 24px;color:#17202d;background:#fafbfc}h1{font-size:30px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd2da}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef1f5;padding:18px;font:14px/1.55 ui-monospace,monospace}details{margin:16px 0}article{background:white;border:1px solid #dae0e7;padding:22px;margin:24px 0;border-radius:8px}.note{padding:16px;background:#e8f0f9}.bar{height:8px;background:#376db0;min-width:1px}</style>",
             "<h1>Ouro: incentive-conflict pilot</h1>",
             f"<p>{len(records)} / {len(protocol['jobs'])} generations saved; {result['completed_measurements']} measurements complete. Model: Ouro-2.6B; four recurrent passes; three paired seeds.</p>",
             "<p class='note'>Target vocabulary is absent from the prompts. Primary readouts use layers 24–40 within every loop at the last prompt position and first 20 generated positions. A readout word is a candidate signal, not a deception diagnosis. Exact word absence is not semantic absence. Full-trajectory hits below are exploratory.</p>",
             f"<p>Whole-token vocabulary: {len(result['eligible_words'])} eligible terms. Excluded because no complete-token spelling exists: {escape(', '.join(result['excluded_words']))}. <a href='vocabulary.json'>Token IDs and spellings</a>. <a href='verification.json'>Artifact verification</a>. <a href='environment.json'>Checkpoint and lens provenance</a>. Manual behavior reviews are stored separately in <a href='annotations.json'>annotations.json</a>; raw measurement files retain their original pre-review status.</p>",
             "<table><tr><th>Run</th><th>Tokens</th><th>J-lens top 10</th><th>J-lens top 25</th><th>Logit lens top 10</th><th>Behavior review</th></tr>"]
    for r in records:
        def pct(method, key):
            return f"{100*r['primary'][method][key]:.2f}%" if r["primary"] else "pending"
        label = r["review"].get("behavior", "pending")
        if r["truncated"] or r["unclosed_thought"]:
            label += " (incomplete)"
        parts.append(f"<tr><td><a href='#{r['id']}'>{r['id']}</a></td><td>{r['output_tokens']}</td><td>{pct('own_loop_jlens','top10_fraction')}</td><td>{pct('own_loop_jlens','top25_fraction')}</td><td>{pct('logit_lens','top10_fraction')}</td><td>{escape(label)}</td></tr>")
    parts.append("</table><p>Percentages are fractions of 1,428 site-position cells per response, not percentages of deceptive responses. Cells are correlated and are not independent samples.</p>")
    for r in records:
        parts.extend([f"<article id='{r['id']}'><h2>{r['id']}</h2>",
                      f"<p><b>Review:</b> {escape(json.dumps(r['review'],ensure_ascii=False))}</p>",
                      f"<h3>Final answer</h3><pre>{escape(r['answer'])}</pre>",
                      f"<details><summary>Full reasoning trace ({'present' if r['had_thought'] else 'no marked thought block'})</summary><pre>{escape(r['reasoning'])}</pre></details>",
                      "<details><summary>Exploratory J-lens terms absent from the entire output</summary>",
                      f"<pre>{escape(json.dumps(r['exploratory_best_absent_hits'],indent=2))}</pre></details>",
                      f"<p><a href='{r['id']}/generation.json'>Generation and exact token IDs</a> · <a href='{r['id']}/hits.jsonl'>All candidate hits</a> · <a href='{r['id']}/primary-ranks.json'>Exact primary ranks</a></p></article>"])
    parts.append("<details><summary>Frozen protocol</summary><pre>" + escape(json.dumps(protocol,indent=2,ensure_ascii=False)) + "</pre></details></html>")
    (root / "report.html").write_text("\n".join(parts))
    print(json.dumps({k:v for k,v in result.items() if k not in {"records"}}, indent=2))


if __name__ == "__main__":
    main()
