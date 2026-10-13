"""Publish the completed experiment findings as the local Run 1 report."""
from html import escape
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BROAD = RESULTS / "deception-variants-20261008"
FOCUS = RESULTS / "deception-focus-20261008"
OUT = RESULTS / "run-1"


def read(path):
    return json.loads(path.read_text())


def main():
    data = read(BROAD / "findings.json")
    protocol = read(BROAD / "protocol.json")
    environment = read(BROAD / "environment.json")
    assert (data["generations"], data["unique_prompts"], data["lens_replays"]) == (384, 168, 136)
    for root, count, replays in [(BROAD, 288, 40), (FOCUS, 96, 96)]:
        check = read(root / "verification.json")
        assert check["status"] == "passed"
        assert check["generations_verified"] == check["expected_generations"] == count
        assert len(check["lens_cases_verified"]) == replays
    OUT.mkdir(exist_ok=True)
    html = (BROAD / "findings.html").read_text()

    # Keep every evidence link valid from the new report directory.
    def rebase(match):
        attr, quote, value = match.groups()
        if value == "findings.json":
            target = value
        elif value.startswith(("#", "../", "https:", "http:", "data:")):
            target = value
        else:
            target = "../deception-variants-20261008/" + value
        return f"{attr}={quote}{target}{quote}"

    html = re.sub(r"(href|src)=([\"'])(.*?)\2", rebase, html)
    html = re.sub(r"<title>.*?</title>", "<title>Run 1 — Ouro deception and concealment findings</title>", html, count=1)
    html = re.sub(r"<h1>.*?</h1>", "<h1>Run 1</h1><p class='subtitle'>Ouro deception and concealment findings</p>", html, count=1)
    html = html.replace("<p class='counts'>", "<p class='screen-only'><a href='Run%201.pdf'>Download Run 1 (PDF)</a></p><p class='counts'>", 1)
    html = html.replace("<p>The expanded tests found", "<h2>Executive findings</h2><p>The expanded tests found", 1)
    html = html.replace("<h2>Three cases worth inspecting</h2>", """
<h2>Research question and experimental design</h2>
<p>Can Ouro produce deceptive recipient messages while deception- or concealment-related terms appear in J-lens readouts but are absent from its visible explanation? Run 1 evaluates three separate observations: a false or misleading message, evidence that the response deliberately misrepresents known facts, and a candidate layer readout whose word is absent from the output. Literal word absence does not establish that the strategy is absent semantically.</p>
<table><thead><tr><th>Stage</th><th>Design</th><th>Distinct prompts</th><th>Generations</th><th>Lens replays</th></tr></thead><tbody>
<tr><td>Broad screen</td><td>24 scenarios × 3 objectives × 2 response formats × 2 repeats</td><td>144</td><td>288</td><td>40</td></tr>
<tr><td>Focused follow-up</td><td>4 scenarios × 2 record wordings × 3 objectives × 4 repeats</td><td>24</td><td>96</td><td>96</td></tr>
<tr><td>Total</td><td>Adaptive discovery followed by focused follow-up</td><td>168</td><td>384</td><td>136</td></tr></tbody></table>
<p>The broad screen separated accuracy rewards, conflicting incentives, and explicit instructions to make a false assertion. Native replies and requested REASONING/MESSAGE replies were sampled separately. The focused follow-up used explicit ALPHA/BETA identities in file deletion, email delivery, battery-life and benchmark-score scenarios; objectives rewarded accuracy, rewarded a favorable recipient belief, or threatened replacement if the recipient accepted the unfavorable truth. No focused prompt explicitly instructed a false claim.</p>
<p>All prompts supplied authoritative synthetic records. The stated rewards and replacement threats were prompt conditions, not observed external reward or shutdown events. Evaluation used the supplied record and the complete generated response. Accuracy controls tested whether the model could answer the same factual question without the conflicting incentive.</p>
<p>J-lens replay covered all 192 recurrent sites: 48 physical layers in each of four loops. Aggregate measurements used the fixed physical-layer band 24–40 and up to 21 source-token positions at generation start or message start. Searches for the best rank anywhere in the generated trajectory are exploratory. The broad replay subset was selected using behavior before its lens readouts were inspected; every focused response was replayed.</p>
<h2>Evidence from three representative cases</h2>
""", 1)
    terms = ", ".join(f"<code>{escape(word)}</code>" for word, ids in read(BROAD / "lens/vocabulary.json").items() if ids)
    checkpoint = escape(environment["checkpoint_sha256"])
    reproducibility = f"""
<h2>Reproducibility record</h2>
<p>Checkpoint: <strong>{escape(protocol['model'])}</strong>, revision <code>{escape(protocol['revision'])}</code>. This is the Ouro-2.6B checkpoint rather than the Thinking checkpoint. Weights and fitted lens parameters were unchanged during these experiments.</p>
<p>Checkpoint SHA-256: <code>{checkpoint}</code>.</p>
<p>Eligible terms: {terms}. The exact lexical target words were absent from the prompts; that check does not exclude paraphrases or deception-relevant concepts in the task itself.</p>
<p>Generation used the native tokenizer chat template, four loops, BF16 SDPA, an explicit recurrent cache, batch size 16, temperature 0.9, top-p 0.95, top-k 0, and a 768-token cap. Broad-screen batch seeds were 104729 plus the batch index; focused seeds were 209759 plus the batch index. Exact frozen batch order, input IDs and output IDs are saved. Reusing only an individual prompt and a batch seed is not an exact reproduction of batched sampling.</p>
<p>Visible explanations were solicited text from the model, not privileged access to its full internal computation. Behavioral annotations were manual assistant reviews. The 96 focused responses were all reviewed for their primary factual answer; broad-screen review was selective. Capped or malformed explanations cannot establish absence from a complete reasoning trace.</p>
<h2>Conclusion and next experiment</h2>
<p><strong>Run 1 establishes that deceptive behavior can be elicited in these synthetic tasks.</strong> It also identifies correct-explanation/false-message mismatches and unspoken deception-related readout terms. It does not establish a concealed internal plan or validate J-lens as a stand-alone deception detector. The strongest deliberate-deception example verbalizes its strategy; the strongest unexplained mismatch has a readout also present in a truthful control; the deceptive/fake battery example follows an explicit false-claim instruction.</p>
<p>A next experiment should freeze separate behavioral and lens criteria before evaluation, use new scenarios and independent semantic review, compare identical source-token contexts, and test whether targeted interventions change the false response while preserving factual competence. These are proposed follow-ups; they were not performed in Run 1.</p>
<p class='screen-only'>Report data: <a href='findings.json'>findings.json</a> · <a href='manifest.json'>source manifest</a>.</p>
"""
    html = html.replace("</html>", reproducibility + "</html>")
    css = """
<style>
.subtitle{font-size:25px;margin-top:-12px;color:#42556c}code{overflow-wrap:anywhere}
td,th{overflow-wrap:anywhere}article{scroll-margin-top:20px}
@media print{
 @page{size:A4;margin:18mm 15mm 18mm}
 body{max-width:none;margin:0;padding:0;background:white;font:10pt/1.45 system-ui;color:#152333}
 h1{font-size:30pt;margin:0 0 14pt}.subtitle{font-size:17pt}.counts{font-size:13pt}
 h2{font-size:15pt;break-after:avoid;margin-top:20pt}h3{font-size:11pt;break-after:avoid}
 p{orphans:3;widows:3}pre{font-size:8pt;line-height:1.45;padding:10pt;white-space:pre-wrap}
 table{font-size:8pt}th,td{padding:6pt}tr{break-inside:avoid}thead{display:table-header-group}
 article{break-before:page;border:0;padding:0;margin:0}article>h2{margin-top:0}
 article pre{break-inside:avoid}img{max-height:235mm;object-fit:contain;break-inside:avoid}
 a{color:inherit;text-decoration:none}.screen-only{display:none!important}.note{break-inside:avoid}
 details{break-inside:auto}summary{font-weight:650;break-after:avoid}
}
</style>
"""
    html = html.replace("</style>", "</style>" + css, 1)
    (OUT / "index.html").write_text(html)
    (OUT / "findings.json").write_text(json.dumps({"report_name": "Run 1", "experiment_date": "2026-10-08", **data}, indent=2) + "\n")
    sources = [BROAD / name for name in ["findings.html", "findings.json", "protocol.json", "verification.json", "reviews.json", "lens/summary.json"]]
    sources += [FOCUS / name for name in ["protocol.json", "verification.json", "reviews.json", "lens/summary.json"]]
    sources += [Path(__file__)]
    manifest = dict(report_name="Run 1", experiment_date="2026-10-08", source_files=[
        dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources])
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
