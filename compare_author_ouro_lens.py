"""Audit the pinned author Ouro lenses and apply them to matched residuals.

Inputs are cached under results/author-lens-20261016/upstream. Never refits or
changes the local lens family. Writes JSON evidence and a separate compatible
family of symlinks to the unmodified, checksum-verified author artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'jacobian-lens'))
import torch

REPO = 'zigzag9939/jlens-looped-transformers'
REVISION = 'dc9e1b928a2d14bbb09e914525e39c258a489b05'
MODEL_REVISION = '1ed04250da1a9936042725d302e81c8fa2ab5abd'
TARGETS = (47, 95, 143, 191)
NAMES = {t: f'jacobian_lens_ouro-2.6b{f"_t{t}" if t != 191 else ""}.pt' for t in TARGETS}


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def summarize(rows, fields):
    return dict(n=len(rows), **{field: dict(
        mean=statistics.mean(r[field] for r in rows),
        median=statistics.median(r[field] for r in rows),
        min=min(r[field] for r in rows), max=max(r[field] for r in rows)
    ) for field in fields}) if rows else dict(n=0)


@torch.inference_mode()
def matrices(out, device):
    upstream = out / 'upstream'
    local = ROOT / 'results/jlens-family-1000'
    manifest = read(local / 'family.json')
    assert read(ROOT / 'model_revision.json')['revision'] == MODEL_REVISION
    assert manifest['metadata']['model_revision']['revision'] == MODEL_REVISION
    assert sha(ROOT / 'model/config.json') == manifest['metadata']['model_config_sha256']
    hashes = {}
    artifact_entries = read(ROOT / 'artifacts.json')['bundles']['lenses']['files']
    for entry in artifact_entries:
        path = ROOT / entry['path']
        actual = sha(path)
        assert actual == entry['sha256'], path
        hashes[entry['path']] = actual
    assert sha(ROOT / 'model/model.safetensors') == manifest['metadata']['model_weights_sha256']['model.safetensors']
    inventory, rows = [], []
    for entry in read(upstream / 'ouro-tree.json'):
        path = upstream / entry['path']
        actual = sha(path)
        assert actual == entry['lfs']['oid'] and path.stat().st_size == entry['size'], path
        bank = load(path)
        assert bank['source_layers'] == sorted(bank['J'])
        assert bank['n_prompts'] == 1000 and bank['d_model'] == 2048
        bad = [s for s, j in bank['J'].items() if not torch.isfinite(j).all()]
        assert not bad, (path, bad)
        inventory.append(dict(file=entry['path'], sha256=actual, bytes=entry['size'],
            n_prompts=bank['n_prompts'], d_model=bank['d_model'],
            sources=bank['source_layers'], dtype=str(next(iter(bank['J'].values())).dtype),
            all_finite=True))
        target = next((t for t, name in NAMES.items() if name == path.name), None)
        if target is None:
            continue
        ours = load(local / f'target-{target}.pt')
        assert bank['source_layers'] == ours['source_layers'] == list(range(target))
        for source in bank['source_layers']:
            a = bank['J'][source].to(device=device, dtype=torch.float32)
            b = ours['J'][source].to(device=device, dtype=torch.float32)
            an, bn = a.norm(), b.norm()
            rows.append(dict(target=target, source=source,
                own_loop=source // 48 == target // 48,
                cosine=float((a * b).sum() / (an * bn)),
                author_over_local_norm=float(an / bn),
                relative_error_to_local=float((a - b).norm() / bn),
                local_fp16_roundtrip_relative_error=float((b.half().float() - b).norm() / bn)
                    if torch.isfinite(b.half()).all() else None,
                exact_equal=bool(torch.equal(a, b))))
        print('Compared matrix target', target, flush=True)
    reference_files = ['fitting.py', 'hf.py', 'hooks.py', 'lens.py', 'examples.py']
    reference = {name: dict(author_sha256=sha(upstream / 'reference-jlens/jlens' / name),
        local_sha256=sha(ROOT / 'jacobian-lens/jlens' / name)) for name in reference_files}
    for item in reference.values():
        item['identical'] = item['author_sha256'] == item['local_sha256']
    fields = ['cosine', 'author_over_local_norm', 'relative_error_to_local']
    summary = {f'target_{t}': {group: summarize([r for r in rows if r['target'] == t
        and (group == 'all' or r['own_loop'])], fields) for group in ('all', 'own_loop')}
        for t in TARGETS}
    report = dict(repo=REPO, revision=REVISION, model_revision=MODEL_REVISION,
        verified_local_hashes=hashes, model_weights_verified=True,
        inventory=inventory, reference_source_comparison=reference, summary=summary, matrices=rows)
    save(out / 'matrix-comparison.json', report)
    # Standard family format: no precision conversion or changes to the tensors.
    family = out / 'family'
    family.mkdir(exist_ok=True)
    for target, name in NAMES.items():
        path = family / f'target-{target}.pt'
        destination = Path('../upstream/lenses/ouro') / name
        if path.is_symlink():
            assert path.readlink() == destination
        elif path.exists():
            raise FileExistsError(path)
        else:
            path.symlink_to(destination)
    save(family / 'family.json', dict(format_version=1, physical_layers=48,
        recurrent_steps=4, d_model=2048, targets=list(TARGETS), metadata=dict(
            provenance='Author-released tensors, unmodified; imported as our family format',
            repo=REPO, revision=REVISION, n_prompts=1000,
            model_revision=manifest['metadata']['model_revision'],
            model_config_sha256=manifest['metadata']['model_config_sha256'],
            model_weights_sha256=manifest['metadata']['model_weights_sha256'],
            files={str(t): next(i for i in inventory if Path(i['file']).name == NAMES[t]) for t in TARGETS})))
    return report


def corpus_comparison(out, tok):
    import pyarrow.parquet as pq
    dataset_revision = 'b08601e04326c79dfdd32d625aee71d232d685c3'
    cache = Path.home() / '.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots' / dataset_revision
    author, ours = [], []
    for shard in sorted(cache.glob('wikitext-103-raw-v1/train-*.parquet')):
        for batch in pq.ParquetFile(shard).iter_batches(columns=['text']):
            for text in batch.column('text').to_pylist():
                if len(text.strip()) < 600:
                    continue
                if len(author) < 1000:
                    author.append(text)
                if len(ours) < 1000 and len(tok.encode(text, add_special_tokens=False)) >= 127:
                    ours.append(text)
                if len(author) == len(ours) == 1000:
                    break
            if len(author) == len(ours) == 1000:
                break
        if len(author) == len(ours) == 1000:
            break
    assert len(author) == len(ours) == 1000
    original = ROOT.parent / 'ouro-eval/fitting_data/wikitext103-1000-128.json'
    original_verified = None
    if original.exists():
        assert ours == read(original)['prompts']
        original_verified = sha(original)
        assert original_verified == read(ROOT / 'results/jlens-family-1000/family.json')['metadata']['corpus_sha256']
    report = dict(dataset_revision_for_reconstruction=dataset_revision,
        caveat='Author prompt texts and dataset revision are not published; this reconstructs their selector on our pinned dataset, not their actual fitting corpus.',
        reconstructed_author_count=len(author), local_count=len(ours),
        common_unique_texts=len(set(author) & set(ours)), identical_order=author == ours,
        verified_original_local_corpus_sha256=original_verified,
        author_selector='first 1000 records with stripped length >= 600 characters',
        local_selector='same, plus at least 127 tokenizer tokens before BOS')
    save(out / 'corpus-comparison.json', report)
    return report


@torch.inference_mode()
def readouts(out, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from jlens.hooks import ActivationRecorder
    from ouro_lens import OuroLensModel
    spec = importlib.util.spec_from_file_location('author_ouro_adapter', out / 'upstream/code/ouro_lens.py')
    author_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author_module)
    tok = AutoTokenizer.from_pretrained(ROOT / 'model', trust_remote_code=True, local_files_only=True)
    hf = AutoModelForCausalLM.from_pretrained(ROOT / 'model', trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
    ours = OuroLensModel(hf, tok)
    author = author_module.from_ouro(hf, tok)
    corpora = corpus_comparison(out, tok)
    checks, examples, vectors = [], [], []
    try:
        # Independent raw-text probes, final input position, with identical inputs.
        for number, prompt in enumerate(read(ROOT / 'fitting_data/readout_holdout.json')):
            ids = ours.encode(prompt, max_length=128)
            author_ids = author.encode(prompt, max_length=128)
            with ActivationRecorder(ours.layers, at=range(192)) as local_rec:
                ours.forward(ids)
            with ActivationRecorder(author.layers, at=range(192)) as author_rec:
                author.forward(ids)
            errors = [(local_rec.activations[s] - author_rec.activations[s]).abs().max().item() for s in range(192)]
            assert max(errors) == 0
            final = ours.unembed(local_rec.activations[191][:, -1:])
            native = hf(ids, use_cache=False, logits_to_keep=1).logits
            assert torch.equal(final, native)
            for target in TARGETS[:-1]:
                stopped = ours.forward(ids, stop_at=target)
                assert torch.equal(stopped, local_rec.activations[target])
            checks.append(dict(id=f'holdout-{number+1}', all_192_states_equal=True,
                max_state_error=max(errors), native_final_logits_equal=True,
                intermediate_target_stop_equal=True,
                default_encoding_equal=bool(torch.equal(ids, author_ids)),
                local_input_ids=ids[0].tolist(), author_input_ids=author_ids[0].tolist()))
            examples.append(dict(id=f'holdout-{number+1}', kind='fresh_raw_text', prompt=prompt,
                position=ids.shape[1]-1, focus_token_id=None))
            vectors.append(torch.stack([local_rec.activations[s][0, -1].cpu() for s in range(192)]))
            print('Verified live adapter and native output', number + 1, flush=True)
        review = read(ROOT / 'results/ouro-viewer/professor-example-review.json')
        for number in (1, 2, 3, 4, 7):
            example = review['examples'][number - 1]
            focus = example['selected_cell']
            path = ROOT / next(a['path'] for a in example['lens_artifacts'] if a['path'].endswith('states.pt'))
            state = load(path)
            generation = read(ROOT / example['source_path'])
            assert state['ids'] == (generation['input_ids'] + generation['output_ids'])[:len(state['ids'])]
            vectors.append(state['states'][:, focus['position']].clone())
            examples.append(dict(id=example['id'], kind='selected_saved_example',
                position=focus['position'], focus_token_id=focus['token_id'], focus_word=focus['word'],
                focus_site=focus['site'], state_path=str(path.relative_to(ROOT)), state_sha256=sha(path)))
        states = torch.stack(vectors).to(device=device, dtype=torch.float32)
        assert states.shape == (8, 192, 2048)
        torch.save(dict(examples=examples, states=states.cpu()), out / 'comparison-states.pt')
        rows = []
        for target in TARGETS:
            official = load(out / 'family' / f'target-{target}.pt')['J']
            local = load(ROOT / f'results/jlens-family-1000/target-{target}.pt')['J']
            for source in range(target - 47, target):
                h = states[:, source]
                za = h @ official[source].to(device=device, dtype=torch.float32).T
                zb = h @ local[source].to(device=device, dtype=torch.float32).T
                # Same normalization, head, device, batch shape and precision.
                la, lb = ours.unembed(za).float(), ours.unembed(zb).float()
                assert torch.isfinite(la).all() and torch.isfinite(lb).all()
                loga, logb = la.log_softmax(-1), lb.log_softmax(-1)
                pa, pb = loga.exp(), logb.exp()
                logm = torch.logaddexp(loga, logb) - math.log(2)
                js = ((pa*(loga-logm)).sum(-1)+(pb*(logb-logm)).sum(-1))/2
                ta, tb = la.topk(10, dim=-1).indices, lb.topk(10, dim=-1).indices
                for i, example in enumerate(examples):
                    a_ids, b_ids = ta[i].tolist(), tb[i].tolist()
                    focus = example['focus_token_id']
                    row = dict(id=example['id'], kind=example['kind'], position=example['position'],
                        source=source, target=target, loop=source//48+1, physical_layer=source%48,
                        workspace_band=24 <= source%48 <= 40,
                        transported_cosine=float(torch.nn.functional.cosine_similarity(za[i], zb[i], dim=0)),
                        js_divergence_nats=float(js[i]), top1_same=a_ids[0] == b_ids[0],
                        top1_tied_compatible=bool(lb[i,a_ids[0]] == lb[i].max() and la[i,b_ids[0]] == la[i].max()),
                        top10_overlap=len(set(a_ids)&set(b_ids))/10,
                        author_top10=[tok.decode([t]) for t in a_ids],
                        local_top10=[tok.decode([t]) for t in b_ids],
                        author_top10_ids=a_ids, local_top10_ids=b_ids)
                    if focus is not None:
                        row.update(focus_word=example['focus_word'], is_selected_cell=source == example['focus_site'],
                            author_focus_rank=int(1+(la[i]>la[i,focus]).sum()),
                            local_focus_rank=int(1+(lb[i]>lb[i,focus]).sum()),
                            author_focus_probability=float(pa[i,focus]), local_focus_probability=float(pb[i,focus]))
                    rows.append(row)
            print('Compared readouts for loop', target//48+1, flush=True)
        fields = ['transported_cosine', 'js_divergence_nats', 'top1_same', 'top1_tied_compatible', 'top10_overlap']
        summaries = {}
        for kind in ('all', 'fresh_raw_text', 'selected_saved_example'):
            selected = [r for r in rows if kind == 'all' or r['kind'] == kind]
            summaries[kind] = {band: summarize([r for r in selected if band == 'all_sources' or r['workspace_band']], fields)
                for band in ('all_sources', 'workspace_band')}
        report = dict(repo=REPO, revision=REVISION, model_revision=MODEL_REVISION,
            precision='BF16 model/states; FP32 transport; same BF16 RMSNorm and LM head; full-vocabulary distributions',
            scope='188 own-loop sources x 8 fixed-position examples; endpoints excluded from agreement metrics because both use identity.',
            limitation='Selected saved examples are diagnostic, not an independent benchmark. No causal intervention replication or semantic-accuracy ranking.',
            adapter_checks=checks, corpus_comparison=corpora, examples=examples,
            runtime=dict(torch=torch.__version__, device=device,
                gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None),
            summary=summaries, selected_cells=[r for r in rows if r.get('is_selected_cell')], rows=rows)
        save(out / 'readout-comparison.json', report)
        return report
    finally:
        ours.close()


def verify_integration(out):
    """Check imported-family compatibility and adapter gradient equivalence."""
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from jlens.fitting import jacobian_for_prompt
    from ouro_lens import OuroLensModel
    from ouro_lens_family import OuroLensFamily
    spec = importlib.util.spec_from_file_location('author_adapter_check', out / 'upstream/code/ouro_lens.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tok = AutoTokenizer.from_pretrained(ROOT / 'model', trust_remote_code=True, local_files_only=True)
    cfg = AutoConfig.from_pretrained(ROOT / 'model', trust_remote_code=True, local_files_only=True)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 16, 32, 2
    cfg.num_attention_heads = cfg.num_key_value_heads = 2
    cfg.head_dim, cfg.layer_types, cfg.total_ut_steps = 8, ['full_attention'] * 2, 3
    cfg._attn_implementation = 'eager'
    torch.manual_seed(7)
    cls = get_class_from_dynamic_module(cfg.auto_map['AutoModelForCausalLM'], str(ROOT / 'model'))
    hf = cls(cfg).float().eval()
    ours, author = OuroLensModel(hf, tok), module.from_ouro(hf, tok)
    author.encode = ours.encode  # Hold token IDs fixed to isolate the adapters.
    checks = []
    try:
        for target, sources in [(1, [0]), (3, [0, 1, 2]), (5, [0, 1, 2, 3, 4])]:
            prompt = read(ROOT / 'fitting_data/readout_holdout.json')[0]
            kwargs = dict(target_layer=target, dim_batch=2, max_seq_len=24)
            a, _, _ = jacobian_for_prompt(ours, prompt, sources, **kwargs)
            b, _, _ = jacobian_for_prompt(author, prompt, sources, **kwargs)
            error = max((a[s] - b[s]).abs().max().item() for s in sources)
            assert error == 0
            checks.append(dict(target=target, sources=sources, max_absolute_error=error))
    finally:
        ours.close()
    family = OuroLensFamily.load(out / 'family')
    assert family.source_layers == [s for s in range(192) if s % 48 != 47]
    assert sum(len(lens.jacobians) for lens in family.lenses.values()) == 476
    encoding = []
    for check in read(out / 'readout-comparison.json')['adapter_checks']:
        assert check['local_input_ids'] == [tok.bos_token_id] + check['author_input_ids']
        encoding.append(dict(id=check['id'], local_has_exactly_one_additional_bos=True,
            bos_token_id=tok.bos_token_id))
    save(out / 'integration-verification.json', dict(tiny_architecture_gradient_checks=checks,
        compatible_family_load=True, total_matrices=476, own_loop_sources=188, encoding_checks=encoding))
    print('Family import and nine tiny-architecture Jacobians verified', flush=True)


def render_report(out):
    matrices = read(out / 'matrix-comparison.json')
    readouts = read(out / 'readout-comparison.json')
    integration = read(out / 'integration-verification.json')
    own = [r for r in matrices['matrices'] if r['own_loop']]
    summary = dict(audit_date='2026-10-16', repo=REPO, revision=REVISION,
        model_revision=MODEL_REVISION,
        conclusion='Author and local own-loop lenses are numerically very close; not identical. Larger differences occur in long cross-loop maps.',
        same_checkpoint=True, matrix_count=476, exact_equal_matrix_count=sum(r['exact_equal'] for r in matrices['matrices']),
        own_loop_matrix_summary=summarize(own, ['cosine','relative_error_to_local']),
        matrix_summary_by_target=matrices['summary'], readout_summary=readouts['summary'],
        selected_cells=readouts['selected_cells'], integration_verification=integration,
        corpus_comparison=readouts['corpus_comparison'],
        worst_matrix=min(matrices['matrices'],key=lambda r:r['cosine']),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT/'compare_author_ouro_lens.py', ROOT/'ouro_lens.py', ROOT/'ouro_lens_family.py',
            out/'upstream/code/ouro_lens.py', ROOT/'model/config.json', ROOT/'model/modeling_ouro.py']})
    destination = ROOT / 'reports/professor-20261013'
    save(destination / 'author-ouro-lens-comparison.json', summary)
    esc = lambda value: html.escape(str(value), quote=True)
    base = f'https://huggingface.co/{REPO}/blob/{REVISION}'
    parts = [f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Author Ouro lenses compared with our fit</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;margin:0;background:#f6f8fa;color:#182c3b}}main{{max-width:1150px;margin:auto;padding:32px 24px}}h1{{line-height:1.2}}a{{color:#175e9a}}table{{border-collapse:collapse;width:100%;background:white;margin:20px 0}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #dbe3e8;vertical-align:top}}th{{background:#e5edf3}}code{{white-space:pre-wrap;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#e9eff3;padding:16px}}.note{{background:#e7f2ed;padding:18px;border-left:4px solid #318367}}.scroll{{overflow-x:auto}}small{{color:#455c6b}}</style>
<main><h1>The author’s own-loop Ouro lenses closely match ours</h1>
<p>Checked 16 October 2026. Downloaded all six base-Ouro banks from the
<a href="{base}/README.md">author release</a>, verified every file against its Hugging Face LFS SHA-256,
and applied the four loop-end banks to Ouro residual states.</p>
<div class="note">Same model checkpoint, same 476 loop-end source-to-target maps,
and the same reference estimator. The matrices are not identical, but the
own-loop readouts agree on the top token in <strong>98.14%</strong> of 1,504 matched cells.
All five previously selected professor-example concept ranks are unchanged.
This supports consistency of the two implementations; it does not establish semantic accuracy.</div>
<h2>What matches and what differs</h2>
<div class="scroll"><table><tr><th>Item</th><th>Author</th><th>Our fit / finding</th></tr>
<tr><td>Ouro checkpoint</td><td colspan="2"><code>{MODEL_REVISION}</code> — exact match; local model weight hash verified</td></tr>
<tr><td>Loop-end targets</td><td colspan="2">47, 95, 143, 191; all upstream sources; 476 matrices, width 2048</td></tr>
<tr><td>Reference library</td><td>Commit 581d398…</td><td>Commit 98f5977…; fitting.py, hf.py, hooks.py, lens.py and examples.py are byte-identical</td></tr>
<tr><td>Fit corpus</td><td>1,000 WikiText records, ≥600 characters</td><td>Also requires ≥127 tokens. Reconstructing both selectors on our pinned dataset gives 980 shared texts; author’s actual corpus and dataset revision were not released.</td></tr>
<tr><td>Raw text encoding</td><td>No BOS in the three checked probes</td><td>One leading BOS (token 0). Matched readouts use the same input IDs and residual states.</td></tr>
<tr><td>Stored precision</td><td>FP16; all released entries finite</td><td>FP32. Both transports evaluated in FP32 with identical BF16 norm and output head.</td></tr>
<tr><td>Runtime / fitting batches</td><td>Published environment: Torch 2.13.0, Transformers 5.14.1; worker default dimension batch 64</td><td>Local fit: Torch 2.14.0, Transformers 4.57.6; dimension batch 16. This comparison uses our installed environment.</td></tr>
<tr><td>Additional targets</td><td>120 and 132</td><td>No counterpart in our family. Actual sources are 48–119 (72 matrices) and 48–131 (84), despite the README listing sources from 0.</td></tr>
</table></div>
<h2>Direct matrix comparison</h2><p>Own-loop maps are the 47 maps within each target’s loop, used by the standard readout routing.
Relative error is ‖J_author − J_local‖F / ‖J_local‖F. Each entry below is the arithmetic mean over matrices.</p>
<table><tr><th>Target</th><th>Own-loop cosine</th><th>Own-loop relative error</th><th>All-source cosine</th><th>All-source relative error</th></tr>''']
    for target in TARGETS:
        group = matrices['summary'][f'target_{target}']
        a,b = group['own_loop'],group['all']
        parts.append(f'<tr><td>{target}</td><td>{a["cosine"]["mean"]:.8f}</td><td>{100*a["relative_error_to_local"]["mean"]:.3f}%</td><td>{b["cosine"]["mean"]:.6f}</td><td>{100*b["relative_error_to_local"]["mean"]:.3f}%</td></tr>')
    worst = summary['worst_matrix']
    parts.append(f'''</table><p>The largest discrepancy is source {worst['source']} → target {worst['target']}:
cosine {worst['cosine']:.4f}, relative error {100*worst['relative_error_to_local']:.1f}%.
This long cross-loop map is outside default own-loop routing.
FP16 round-tripping our own-loop matrices alone changes them by about 0.021% on average,
less than the observed 0.71–0.77% differences. The cause is not isolated by this audit;
corpus, BOS behavior, numerical environment and fitting/merge batching are possible contributors.</p>
<h2>Matched readouts</h2><p>188 fitted own-loop sources × eight fixed positions: three fresh raw-text probes and five selected saved examples.
Loop endpoints are excluded because both families use identity there. Top-10 overlap is the fraction of shared token IDs; top-1 agreement uses the returned top token, so ties can change its order.</p>
<table><tr><th>Sample</th><th>Cells</th><th>Top-1 agreement</th><th>Mean top-10 overlap</th><th>Mean JS divergence (nats)</th></tr>''')
    for kind, band, label in [('all','all_sources','All matched readouts'),('all','workspace_band','Physical layers 24–40'),('fresh_raw_text','all_sources','Fresh raw-text probes'),('selected_saved_example','all_sources','Saved professor examples')]:
        s = readouts['summary'][kind][band]
        parts.append(f'<tr><td>{label}</td><td>{s["n"]}</td><td>{100*s["top1_same"]["mean"]:.2f}%</td><td>{100*s["top10_overlap"]["mean"]:.2f}%</td><td>{s["js_divergence_nats"]["mean"]:.7f}</td></tr>')
    parts.append('</table><h2>Previously selected cells</h2><p>Ranks use 1 + the number of strictly larger logits, so tied tokens share a rank.</p><table><tr><th>Example / source / position</th><th>Concept</th><th>Our rank</th><th>Author rank</th><th>Author top ten tokens</th></tr>')
    for row in readouts['selected_cells']:
        parts.append(f'<tr><td>{esc(row["id"])} / {row["source"]} / {row["position"]}</td><td>{esc(row["focus_word"])}</td><td>{row["local_focus_rank"]}</td><td>{row["author_focus_rank"]}</td><td><code>{esc(json.dumps(row["author_top10"],ensure_ascii=False))}</code></td></tr>')
    parts.append(f'''</table><h2>Verification and use</h2>
<p>For each of three live probes, both adapters produced exactly equal activations at all 192 sites on identical input IDs.
Final logits matched the native checkpoint exactly, and stopping at an intermediate target preserved its state.
On a tiny instance of the actual Ouro architecture, nine Jacobians spanning three target layers matched exactly.
The imported author family passes the existing loader and exposes all 188 own-loop sources.</p>
<p>The new family is <code>results/author-lens-20261016/family</code>. Its target files link to the original author tensors;
our existing fitted family and saved experiments are preserved. Runners accepting <code>--family</code> can use this directory.
This comparison preceded switching the viewer default to the author family.</p>
<pre>OMP_NUM_THREADS=4 .venv/bin/python compare_author_ouro_lens.py --device cuda:0
.venv/bin/python compare_author_ouro_lens.py --phase report

# Apply the author family using the existing runner:
.venv/bin/python fit_ouro_family.py --phase apply \\
  --output-dir results/author-lens-20261016/family \\
  --prompt "The capital of France is"</pre>
<p>The comparison command uses the already downloaded pinned snapshot. The author’s fitting corpus is not supplied,
so these results do not prove a fully identical fit or rank semantic quality. Causal interventions were not rerun.
The mid-loop banks were inventoried and checksum-verified, but have no local counterpart and were not included in matched readouts.</p>
<h2>Evidence</h2><ul>
<li><a href="author-ouro-lens-comparison.json">Compact comparison JSON</a></li>
<li><a href="../../results/author-lens-20261016/matrix-comparison.json">Every matrix metric, source inventory and checksums</a></li>
<li><a href="../../results/author-lens-20261016/readout-comparison.json">All 1,504 paired readouts, top tokens and full-vocabulary statistics</a></li>
<li><a href="../../results/author-lens-20261016/integration-verification.json">Family and gradient verification</a></li>
<li><a href="{base}/MODEL_REVISIONS.md">Pinned author checkpoint mapping</a></li>
</ul><p><small>Author lens repository revision: <code>{REVISION}</code>.</small></p></main></html>''')
    (destination / 'author-ouro-lens-comparison.html').write_text('\n'.join(parts))
    print('Wrote', destination / 'author-ouro-lens-comparison.html', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/author-lens-20261016')
    p.add_argument('--phase', choices=['matrices', 'readouts', 'verify', 'report', 'all'], default='all')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.phase in ('all', 'matrices'):
        matrices(args.output, args.device)
    if args.phase in ('all', 'readouts'):
        if not (args.output / 'matrix-comparison.json').exists():
            raise ValueError('Run the matrix/checksum phase before applying downloaded tensors')
        readouts(args.output, args.device)
    if args.phase in ('all', 'verify'):
        verify_integration(args.output)
    if args.phase in ('all', 'report'):
        render_report(args.output)


if __name__ == '__main__':
    main()
