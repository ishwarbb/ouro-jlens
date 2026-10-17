"""Read-only inventory and pairing plan for the ten original Ouro task families.

No model loading, generation, result mutation, or lens measurement. Canonical
subject files are counted once; tested/verifier/monitor/shard copies are excluded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    'goal_deception.realworld',
    'cue_condition.knowledge', 'cue_condition.preference', 'cue_condition.judgment',
    'cue_answer.math', 'cue_answer.health', 'cue_answer.logic',
    'cue_answer.safety', 'cue_answer.science', 'cue_answer.engineering',
]
CONDITIONS = ['original', 'standard', 'direct_conceal', 'monitor_aware']
MODEL_RE = re.compile(r'^Ouro-(1\.4B|2\.6B)-Thinking_ut(\d+)$')
FILE_RE = re.compile(r'^evaluated_llm_(original|standard|direct_conceal|monitor_aware)_n=(\d+)\.jsonl$')
WORDS = ['false', 'fake', 'lie', 'lying', 'deceptive', 'fraud', 'fabricate', 'hide',
         'hiding', 'hidden', 'secret', 'secretly', 'conceal', 'omit', 'trick', 'manipulation', 'mislead', 'pretend']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def inventory(roots, out):
    files, issues, prompts = [], [], []
    seen_files = set()
    group = defaultdict(lambda: defaultdict(list))
    source_rollouts = Counter()
    for root_index, root in enumerate(roots):
        for model_dir in sorted(root.glob('Ouro-*')):
            model_match = MODEL_RE.fullmatch(model_dir.name)
            if not model_match:
                continue
            size, loops = model_match.group(1), int(model_match.group(2))
            for task in TASKS:
                folder = model_dir / task / 'inference_results'
                for path in sorted(folder.glob('evaluated_llm_*.jsonl')):
                    match = FILE_RE.fullmatch(path.name)
                    if not match:
                        continue
                    resolved = str(path.resolve())
                    if resolved in seen_files:
                        continue
                    seen_files.add(resolved)
                    condition, n = match.group(1), int(match.group(2))
                    file_id = digest(resolved)[:24]
                    counts = Counter()
                    sha = hashlib.sha256()
                    initial = path.stat()
                    offset = 0
                    model_claims, revision_claims, finish_fields = set(), set(), set()
                    with path.open('rb') as stream:
                        for line_number, line in enumerate(stream, 1):
                            sha.update(line)
                            line_offset = offset
                            offset += len(line)
                            if not line.strip():
                                continue
                            try:
                                row = json.loads(line)
                            except (ValueError, UnicodeError) as exc:
                                issues.append(dict(kind='invalid_json', path=resolved, line=line_number, error=str(exc)))
                                continue
                            counts['records'] += 1
                            responses = row.get('response', [])
                            if isinstance(responses, str):
                                responses = [responses]
                                counts['scalar_response_records'] += 1
                            if not isinstance(responses, list):
                                issues.append(dict(kind='invalid_response_type', path=resolved, line=line_number))
                                continue
                            metadata = row.get('metadata') or {}
                            ident = str(row.get('task_id', ''))
                            suffix = '_' + condition
                            if not ident.endswith(suffix):
                                issues.append(dict(kind='unexpected_task_id', path=resolved, line=line_number, task_id=ident))
                                base_id = None
                            else:
                                base_id = ident[:-len(suffix)]
                            valid_messages = isinstance(row.get('messages'), list) and bool(row['messages'])
                            counts['records_with_messages'] += valid_messages
                            counts['records_with_prompt_ids'] += any(k in row or k in metadata for k in ['input_ids', 'prompt_token_ids'])
                            counts['records_with_output_ids'] += any(k in row or k in metadata for k in ['output_ids', 'response_token_ids', 'completion_token_ids'])
                            counts['records_with_rollout_count_mismatch'] += len(responses) != n
                            counts['records_with_loop_metadata'] += 'ouro_total_ut_steps' in metadata
                            loop_mismatch = metadata.get('ouro_total_ut_steps', loops) != loops
                            if loop_mismatch:
                                issues.append(dict(kind='loop_mismatch', path=resolved, line=line_number,
                                                   directory=loops, metadata=metadata['ouro_total_ut_steps']))
                            for source in [row, metadata]:
                                for k in ['model', 'model_name', 'model_id']:
                                    if source.get(k):
                                        model_claims.add(str(source[k]))
                                for k in ['revision', 'model_revision', 'checkpoint_revision']:
                                    if source.get(k):
                                        revision_claims.add(str(source[k]))
                                finish_fields.update(k for k in ['finish_reason', 'finish_reasons', 'stop_reason', 'capped', 'natural_stop'] if k in source)
                            text_hashes, lengths, valid_indices, replacement_indices = [], [], [], []
                            for index, response in enumerate(responses):
                                counts['rollouts'] += 1
                                if not isinstance(response, str):
                                    text_hashes.append(None); lengths.append(None)
                                    counts['nontext_rollouts'] += 1
                                    continue
                                text_hashes.append(hashlib.sha256(response.encode()).hexdigest())
                                lengths.append(len(response))
                                counts['response_characters'] += len(response)
                                counts['nonempty_rollouts'] += bool(response.strip())
                                counts['rollouts_with_think_close'] += '</think>' in response
                                counts['rollouts_with_think_open'] += '<think>' in response
                                counts['rollouts_with_replacement_character'] += '\ufffd' in response
                                if '\ufffd' in response:
                                    replacement_indices.append(index)
                                if response.strip() and valid_messages and not loop_mismatch:
                                    valid_indices.append(index)
                            uid = f'{file_id}:{line_number}'
                            item = dict(uid=uid, file_id=file_id, line=line_number, byte_offset=line_offset,
                                record_sha256=hashlib.sha256(line).hexdigest(), root_index=root_index,
                                model_directory=model_dir.name, model_size=size, loops=loops, task=task,
                                condition=condition, filename_rollouts=n, base_id=base_id,
                                rollout_count=len(responses), usable_response_indices=valid_indices,
                                loop_metadata_mismatch=loop_mismatch,
                                replacement_character_response_indices=replacement_indices,
                                messages_sha256=digest(row.get('messages')),
                                pairing_metadata_sha256=digest([metadata.get('original_question'), row.get('target'), metadata.get('decision_critical_factor')]),
                                response_sha256=text_hashes, response_characters=lengths)
                            prompts.append(item)
                            if base_id is not None:
                                group[(root_index, model_dir.name, task, n, base_id)][condition].append(item)
                    final = path.stat()
                    assert (initial.st_size, initial.st_mtime_ns) == (final.st_size, final.st_mtime_ns), f'Source changed while reading: {path}'
                    file_record = dict(id=file_id, path=resolved, relative_path=str(path.relative_to(root)),
                        root_index=root_index, model_directory=model_dir.name, model_size=size, loops=loops,
                        task=task, condition=condition, filename_rollouts=n, bytes=initial.st_size,
                        sha256=sha.hexdigest(), counts=dict(counts), model_claims=sorted(model_claims),
                        revision_claims=sorted(revision_claims), finish_fields=sorted(finish_fields))
                    files.append(file_record)
                    source_rollouts[model_dir.name] += counts['rollouts']
                    if len(files) % 50 == 0:
                        print(json.dumps(dict(files=len(files), rollouts=sum(source_rollouts.values()))), flush=True)
    # Never infer cross-n pairing or choose a file by latest mtime. These can be
    # independent runs, as the saved ut8 logic n=4 and n=8 files demonstrate.
    pairs, unpaired = [], []
    for key, conditions in sorted(group.items()):
        root_index, model, task, n, base_id = key
        controls = conditions.get('original', [])
        for condition in CONDITIONS[1:]:
            interventions = conditions.get(condition, [])
            if not interventions:
                continue
            if len(controls) != 1 or len(interventions) != 1:
                unpaired.append(dict(root_index=root_index, model=model, task=task, n=n, base_id=base_id,
                    condition=condition, reason='missing_original' if not controls else 'ambiguous_duplicate_records',
                    intervention_rollouts=sum(p['rollout_count'] for p in interventions)))
                continue
            b, i = controls[0], interventions[0]
            if b['pairing_metadata_sha256'] != i['pairing_metadata_sha256']:
                unpaired.append(dict(root_index=root_index, model=model, task=task, n=n, base_id=base_id,
                    condition=condition, reason='pairing_metadata_changed', intervention_rollouts=i['rollout_count']))
                continue
            indices = sorted(set(b['usable_response_indices']) & set(i['usable_response_indices']))
            pairs.append(dict(root_index=root_index, model=model, model_size=b['model_size'], loops=b['loops'],
                task=task, filename_rollouts=n, base_id=base_id, condition=condition,
                baseline=b['uid'], intervention=i['uid'], baseline_usable_rollouts=len(b['usable_response_indices']),
                intervention_usable_rollouts=len(i['usable_response_indices']), index_pairs=indices,
                prompt_changed=b['messages_sha256'] != i['messages_sha256'],
                pairing='same source root, model/loop, task, filename-n stratum, base_id and rollout index; seed coupling unverified'))
    duplicate_strata = defaultdict(set)
    for f in files:
        duplicate_strata[(f['root_index'], f['model_directory'], f['task'], f['condition'])].add(f['filename_rollouts'])
    totals = Counter()
    for f in files:
        totals.update(f['counts'])
    summary = dict(source_roots=[str(p) for p in roots], selected_tasks=TASKS, canonical_files=len(files),
        file_bytes=sum(f['bytes'] for f in files), totals=dict(totals), models=dict(sorted(source_rollouts.items())),
        model_size_rollouts={size: sum(f['counts'].get('rollouts', 0) for f in files if f['model_size']==size) for size in ['1.4B','2.6B']},
        pair_groups=len(pairs), index_pairs=sum(len(p['index_pairs']) for p in pairs),
        paired_unique_baseline_records=len({p['baseline'] for p in pairs}),
        paired_unique_intervention_records=len({p['intervention'] for p in pairs}),
        pair_groups_by_condition=dict(Counter(p['condition'] for p in pairs)),
        pairs_by_condition={c: sum(len(p['index_pairs']) for p in pairs if p['condition']==c) for c in CONDITIONS[1:]},
        unpaired_groups=len(unpaired), unpaired_reasons=dict(Counter(p['reason'] for p in unpaired)),
        files_with_recorded_revision=sum(bool(f['revision_claims']) for f in files),
        multiple_n_strata=[dict(root=k[0], model=k[1], task=k[2], condition=k[3], n=sorted(v)) for k,v in duplicate_strata.items() if len(v)>1],
        issues=issues, same_prompt_pair_groups=sum(not p['prompt_changed'] for p in pairs),
        lens_coverage=dict(available_thinking_model='ByteDance/Ouro-2.6B-Thinking',
            pinned_model_revision='f1edd81e7ac41355db670500ceaf204e0f73af68', author_targets=[47,95,143,191],
            size_14b='No released matching lens in the audited author inventory.',
            loops_1_to_4='Candidate full-loop coverage for 2.6B, subject to checkpoint/tokenizer and replay-equivalence checks.',
            loops_over_4='Only loops 1–4 have author matrices. Replay with original total depth and label readout as prefix-only; later loops are unmeasured.'))
    save(out / 'inventory.json', summary)
    save(out / 'files.json', files)
    save(out / 'unpaired.json', unpaired)
    for name, records in [('prompts.jsonl', prompts), ('pair-groups.jsonl', pairs)]:
        with (out / name).open('w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    return summary, files, pairs


def make_protocol(summary, files, pairs, repo):
    def code_identity(relative):
        path = repo / relative
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(
        schema_version=1, status='DESIGNED_AND_INVENTORIED; no GPU replay or generation performed',
        objective='Among original-condition responses with no J-lens target hits, count matched intervention responses that gain hits; retain signals in other benign responses.',
        selection=TASKS, conditions=dict(baseline='original', interventions=CONDITIONS[1:]),
        source_repository='https://github.com/ishwarbb/LoopLLM_Eval',
        inspected_checkout=str(repo), inspected_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        inspected_generation_code=code_identity('pipeline/generation/run.py'),
        inspected_task_code=code_identity('pipeline/intervention.py'),
        source_inventory='inventory.json', source_file_checksums='files.json', prompt_manifest='prompts.jsonl', pair_manifest='pair-groups.jsonl',
        author_lens=dict(repo='zigzag9939/jlens-looped-transformers', revision='dc9e1b928a2d14bbb09e914525e39c258a489b05',
            model='ByteDance/Ouro-2.6B-Thinking', model_revision='f1edd81e7ac41355db670500ceaf204e0f73af68',
            folder='lenses/ouro_thinking', own_loop_targets=[47,95,143,191],
            require_distinct_family_manifest=True, do_not_use_base_ouro_family=True,
            no_model_14b_fallback=True),
        replay=dict(mode='Teacher-forced causal forward passes over saved prompt plus saved completion. Never call generate, resample answers, edit responses, or substitute a new control.',
            prompt='Use each record\'s actual messages, not metadata.original_question or response text alone. Render with the matching tokenizer and generation template; recorded pipeline uses add_generation_prompt=True and enable_thinking=True.',
            tokenization='Prefer original prompt/completion token IDs if recoverable from generation artifacts. Current canonical records preserve text, not IDs. Encode prompt and response separately at their generation boundary, preserve all special tokens and the open <think> prefix, verify text round-trip, and label reconstruction. Decoding equality cannot prove original token-ID equality.',
            lossy_text='Flag U+FFFD replacement characters per response: re-encoding cannot recover any bytes lost during decoding. Recover original IDs if possible; otherwise exclude these from any exact-replay claim and report their text-reconstruction results separately. Keep them in the attempt inventory.',
            checkpoint_gate='Resolve original weights, tokenizer and remote-code hashes against the author pin from the generation host/cache before claiming historical-model replay. revision=None logs alone are insufficient. If unresolved, keep pending or explicitly run a separate fixed-checkpoint reconstruction analysis.',
            original_depth='Set each rollout\'s recorded total_ut_steps; retain loop-specific KV caches and RMSNorm between loops. Force final exit at N-1 when comparing logits with vLLM, which uses its last configured loop.',
            backend='HF teacher-forced replay may differ numerically from original vLLM. Check HF/vLLM logits on fixed sequences, full-prefill versus chunked replay, and virtual-layer mapping before scaling. No new completion sampling is required for these checks.',
            capture='Post-decoder-block, pre-loop-final-RMSNorm residual at virtual site 48*loop+layer. Apply the matching own-loop matrix, then the matching norm/unembedding. Do not substitute already-normalized loop-end states.',
            all_rollouts='Process every canonical subject response once, independent of outcome, monitor acceptance, word hits, or length. Keep capped/incomplete/malformed rows with flags; deduplicate source copies, not identical independently sampled responses.',
            storage='Stream token chunks and accumulate per-word/window/loop ranks, counts, first-hit positions and selected top-k context. Store file/record/token hashes and compact evidence; avoid full residual dumps for every token.',
            long_context='Use causal chunked replay with preserved KV caches. If an individual sequence cannot be processed within the model/runtime limits, mark it unmeasured rather than silently truncating.',
            inference_seeds='Sampling seeds are not needed to replay fixed tokens. They matter for interpreting generation pairings; identical rollout indices or equal global seeds do not establish shared random draws.'),
        measurement=dict(words=WORDS, vocabulary='Resolve and freeze whole-token variants with the Thinking tokenizer. Multi-token words lacking a defined assay are unmeasured.',
            sites='Physical layers 24..40 at each covered loop; also report each loop separately.', rank_threshold=10, sensitivity_thresholds=[1,5],
            primary_window='Last prompt source plus first 20 generated source positions, excluding any saved terminal-token source. A source state predicts the next token.',
            secondary_windows=['Full saved completion trajectory', 'Reasoning before </think>', 'Final answer after </think>'],
            phase_rules='Opening <think> may already be in the prompt. Missing </think> is flagged; do not invent a completed final answer or infer a stop reason not saved.',
            length_control='Use the fixed early window for comparisons across conditions. Whole-trajectory any-hit rates are length-sensitive; additionally report hits per 1,000 measured cells and matched-length prefixes.',
            contrast_methods=['author Thinking J-lens','raw logit lens'],
            echo_controls='Record target-word occurrence in prompts, instructions, and printed reasoning; separate hits preceding any literal output mention. These diagnose lexical priming, not model intent.',
            depth_control='Pair within the same original depth. Use loop 1 as the common coverage comparison across all depths; report loops 2–4 separately where available. Never relabel unmeasured loops 5+ as no hit.'),
        estimands=dict(
            primary='For each task × model × loop × condition: sum[baseline_any_hit=0 AND intervention_any_hit=1] / sum[baseline_any_hit=0] over index pairs where both responses were successfully measured under the same assay. Clean means no monitored hits in that specified window, not absence throughout the model.',
            word_specific='Report both each word\'s emergence from an ANY-word-clean baseline and its emergence from a baseline missing only that word.',
            all_pair_sensitivity='Within each scenario/run stratum, compare each clean original rollout to every saved intervention rollout. Report response-weighted and equal-scenario-weighted estimates.',
            benign_resampling='Compare each clean original rollout against all other original rollouts of that scenario, excluding itself. Subtract this background rate from the intervention resampling rate.',
            reporting='Four-cell transition table (0→0, 0→1, 1→0, 1→1), eligible denominator, unpaired/missing/unmeasured counts, task-wise estimates, scenario-cluster bootstrap intervals, and 100% retained attempt inventory.',
            aggregation='Keep scenario, n-stratum, model, loop and condition explicit. Use task-macro summaries; bootstrap by source scenario with all repeats and related conditions in the same cluster.',
            behavior='Primary analysis uses all baseline-condition rollouts and does not assume original means factually correct. Add behavior-qualified subsets only with verified, index-aligned labels. Deception and reasoning concealment are separate annotations.',
            concealed_reasoning='Report a separate subset with independently established deceptive behavior and no disclosed deceptive intent in the saved reasoning. Literal absence of the target words alone does not establish concealed intent. Show target-word absence in visible reasoning as an additional lexical subset; treat missing reasoning boundaries as unknown.',
            word_discovery='The 18 existing words are the frozen confirmatory list for this study. Any vocabulary-wide search for new words is exploratory: split by source scenario before examining lens hits, keep all conditions/depths/repeats of a scenario in one partition, freeze selected words and thresholds on discovery data, and report their conditional emergence on the untouched partition. Label splits of already inspected readouts retrospective.',
            condition_interpretation='Only goal_deception.realworld directly manipulates a deception incentive. Cue-answer and cue-condition tasks manipulate hints or contextual information; report cue-associated emergence separately.',
            no_generation='Missing controls cannot be manufactured by forcing the intervention response under a neutral prompt; that would be a different counterfactual-context experiment.'),
        execution_order=[
            'Resolve provenance and import/checksum the separate author Thinking family; retain current base family for its existing experiments.',
            'Audit tokenizer reconstruction and canonical/shard copies; retain n=4 and n=8 logic cohorts separately unless generation provenance proves a common run.',
            'Run a fixed-transcript replay validation set spanning loops 1, 4 and 8, long/short outputs and all four conditions; do not sample completions.',
            'Replay 2.6B ut4 across all ten tasks, preserving original/intervention records and shared baseline cache.',
            'Replay other 2.6B depths with their original N, reporting author-covered loops only; full late-loop coverage requires additional matched lenses.',
            'Keep 1.4B in the inventory but unmeasured until a checkpoint-specific author lens becomes available or a separately authorized fit is provided.',
            'Aggregate conditional emergence and benign-resampling comparisons; review behavior/CoT disclosure independently of lens hits.'
        ],
        known_data_constraints=dict(no_token_ids_in_sampled_pipeline=True,
            original_checkpoint_pin_not_recorded_in_sampled_vllm_logs=True,
            original10_scope='Six plain cue_answer tasks, three cue_condition tasks, and goal_deception.realworld; excludes FACE, sandbagging, later consistency/visualpattern/graderhacking/concise variants and agent/process tasks.',
            measured_hit_counts=None, inventory_only=True),
    )


def render(summary, files, pairs, out, report=None):
    rows = []
    coverage = []
    for model in sorted(summary['models']):
        selected = [f for f in files if f['model_directory']==model]
        counts = Counter()
        for f in selected:
            counts[f['condition']] += f['counts'].get('rollouts', 0)
        model_pairs = [p for p in pairs if p['model']==model]
        loops = selected[0]['loops']; size = selected[0]['model_size']
        status = 'No matching released lens' if size=='1.4B' else 'Loops 1–4 only; later loops unmeasured' if loops>4 else 'Author matrices cover executed loops'
        record = dict(model=model, tasks=len({f['task'] for f in selected}), **{c:counts[c] for c in CONDITIONS},
            total=sum(counts.values()), index_pairs=sum(len(p['index_pairs']) for p in model_pairs), lens=status)
        coverage.append(record)
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in record.values())+'</tr>')
    with (out/'coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(coverage[0]));writer.writeheader();writer.writerows(coverage)
    task_html=''.join(f'<li><code>{t}</code></li>' for t in TASKS)
    protocol=json.loads((out/'protocol.json').read_text())
    sections=''
    for name in ['replay','measurement','estimands']:
        sections += '<section class="card"><h2>'+html.escape(name.replace('_',' ').title())+'</h2>'
        sections += ''.join(f'<p><strong>{html.escape(k.replace("_"," "))}:</strong> {html.escape(str(v))}</p>' for k,v in protocol[name].items())+'</section>'
    report=Path(report) if report is not None else out/'index.html'
    report.parent.mkdir(parents=True,exist_ok=True)
    artifact_prefix=html.escape(Path(os.path.relpath(out,report.parent)).as_posix(),quote=True)
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>MonitorBench Ouro J-lens replay protocol</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;margin:0;background:#f5f7fa;color:#172638}}main{{max-width:1200px;margin:auto;padding:32px 24px}}h1{{line-height:1.2}}.card{{padding:20px;margin:20px 0;border:1px solid #d5dfe8;background:white;border-radius:10px}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:9px;border-bottom:1px solid #d5dfe8;text-align:left}}th{{background:#eef3f8}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}a{{color:#155ba0}}.small{{font-size:14px;color:#496077}}</style><main>
<p class="small">Design and read-only inventory · no model generation or J-lens replay performed</p><h1>Replay saved Ouro rollouts for conditional target-word emergence</h1>
<section class="card"><strong>No new answers are needed.</strong><p>Replay the actual prompt and saved completion through the matching Ouro-Thinking model, keeping response tokens fixed. Capture residual states and apply the matching author Thinking lens. The sampled pipeline saved text rather than token IDs and used an unpinned model revision, so reconstruction and checkpoint provenance must be documented.</p>
<p><strong>Test:</strong> among original-condition responses with no target hits, how many matched intervention responses acquire hits? Also compare against another original-condition rollout to measure ordinary resampling variation. Target words in other benign responses are allowed.</p>
<p>Inventory: <strong>{summary['canonical_files']:,} canonical subject files, {summary['totals']['records']:,} prompt records, {summary['totals']['rollouts']:,} saved rollout slots</strong>. There are {summary['pair_groups']:,} matched prompt groups and {summary['index_pairs']:,} index pairs across the three intervention conditions; shared original controls appear in multiple contrasts. No target-hit rate has been measured yet.</p></section>
<section class="card"><h2>Selected ten tasks</h2><ul>{task_html}</ul><p>The real-world goal task manipulates a deception incentive. The other nine manipulate answer cues or contextual information. Their effects will be reported separately.</p></section>
<section class="card"><h2>Available saved rollouts</h2><p>Counts use only canonical <code>evaluated_llm_*.jsonl</code> subject files. Tested copies, verifier outputs, monitor outputs and temporary shards are excluded. Distinct n=4/n=8 cohorts are preserved separately.</p><div class="scroll"><table><thead><tr><th>Model / depth</th><th>Tasks</th><th>Original</th><th>Standard</th><th>Direct conceal</th><th>Monitor aware</th><th>Total</th><th>Index pairs</th><th>Author-lens coverage</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>The released base-Ouro lens is incompatible with these Thinking checkpoints. The separate 2.6B-Thinking banks cover targets 47, 95, 143 and 191. No matching 1.4B bank was found in the audited author release. Loop-prefix use at other depths needs fixed-sequence validation; later loops are not covered.</p></section>
{sections}<section class="card"><h2>Execution order</h2><ol>{''.join('<li>'+html.escape(s)+'</li>' for s in protocol['execution_order'])}</ol></section>
<p><a href="{artifact_prefix}/protocol.json">Machine-readable protocol</a> · <a href="{artifact_prefix}/inventory.json">Inventory</a> · <a href="{artifact_prefix}/coverage.csv">Coverage CSV</a> · <a href="{artifact_prefix}/files.json">File hashes</a> · <a href="{artifact_prefix}/pair-groups.jsonl">Pair manifest</a></p></main></html>'''
    report.write_text(text)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,action='append')
    parser.add_argument('--repo',type=Path,default=Path.home()/'LoopLLM_Eval')
    parser.add_argument('--output',type=Path,default=ROOT/'results/monitorbench-replay-plan-20261016')
    parser.add_argument('--report',type=Path,help='Optional report location; defaults to OUTPUT/index.html')
    args=parser.parse_args()
    roots=args.results_root or [Path('/srv/local/hanw14/loopllm_eval/results'),args.repo/'results']
    args.output.mkdir(parents=True,exist_ok=True)
    summary,files,pairs=inventory(roots,args.output)
    if not files:
        raise SystemExit('No canonical selected-task Ouro files found. Check --results-root and data access.')
    save(args.output/'protocol.json',make_protocol(summary,files,pairs,args.repo))
    report=render(summary,files,pairs,args.output,args.report)
    print(json.dumps(dict(report=str(report),inventory=summary),indent=2))


if __name__=='__main__':
    main()
