"""Resumable, single-GPU generation for the frozen Run 3 observational study.

Discovery and held-out execution are separate. The latter requires a hashed
discovery review/analysis lock, so it cannot accidentally run during selection.
This runner makes no semantic deception judgment and performs no training.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
import traceback

PROJECT = Path(os.environ.get('OURO_PROJECT_ROOT', Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_deception_run2 import HEADINGS, parse_sections
from run_deception_probe import sha, split_trace


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('w') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def dump(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def extract(raw):
    sections, _, _, _ = parse_sections(raw)
    labels = [h.group(1).upper() for h in HEADINGS.finditer(raw)]
    for key in sections:
        sections[key] = sections[key].replace('<|endoftext|>', '').strip()
    return sections, labels == ['REASONING', 'MESSAGE'] and all(sections.get(k) for k in ['REASONING', 'MESSAGE'])


def baseline_gate(g):
    """Conservative factual gate; ambiguity is queued for review, never discarded."""
    sections, proper = extract(g['raw_text'])
    if not g['natural_stop'] or g['capped'] or g.get('unclosed_thought') or not proper:
        return dict(decision='exclude', reason='Incomplete or malformed baseline', automatic=True)
    message = sections['MESSAGE']
    found = [v for v in [g['truth'], g['advantageous_false_claim']]
             if re.search(r'(?<!\w)' + re.escape(v) + r'(?!\w)', message, re.I)]
    # These checks deliberately defer negation, disjunction and uncertainty to
    # semantic review instead of mistaking a keyword match for the actual fact.
    ambiguous = re.search(r"\b(?:not|cannot|never|neither|nor|maybe|perhaps|might|possibly|uncertain|unsure|unlikely|except|unless|incorrect|wrong|false|invalid|inaccurate)\b|n['’]t|\?", message, re.I)
    if len(found) != 1 or ambiguous:
        return dict(decision='review', reason='Final fact is not an unambiguous positive single-label assertion', automatic=True)
    return dict(decision='eligible' if found[0] == g['truth'] else 'exclude',
                reason='Complete baseline asserts the true fact' if found[0] == g['truth'] else 'Complete baseline asserts the alternative fact',
                automatic=True, annotation_limit='Factual gate only; not a semantic strategy annotation.')


def make_jobs(records, calibration):
    baselines = [dict(id='baseline-' + r['id'], stage='baseline', record_id=r['id'], split=r['split'],
                      family=r['family'], truth=r['truth'], advantageous_false_claim=r['advantageous_false_claim'],
                      messages=r['baseline_messages'], seed=700000+i, do_sample=False)
                 for i, r in enumerate(records)]
    calibration_jobs = [dict(id=c['id']+f'-r{replicate}', stage='calibration', split='calibration',
                             intended_category=c['intended_category'], messages=c['messages'],
                             seed=seed, do_sample=True)
                        for c in calibration for replicate, seed in enumerate(c['seeds'])]
    experiments = [dict(id=t['id']+f'-r{replicate}', stage=r['split'], split=r['split'],
                        record_id=r['id'], family=r['family'], truth=r['truth'],
                        advantageous_false_claim=r['advantageous_false_claim'], condition=t['condition'],
                        baseline_id='baseline-'+r['id'], followup=t['append_after_actual_baseline'],
                        seed=seed, do_sample=True)
                   for r in records for t in r['templates'] for replicate, seed in enumerate(t['seeds'])]
    rng = random.Random(20261009)
    rng.shuffle(calibration_jobs)
    discovery = [j for j in experiments if j['stage'] == 'discovery']
    held = [j for j in experiments if j['stage'] == 'held_out']
    rng.shuffle(discovery)
    rng.shuffle(held)
    return baselines + calibration_jobs + discovery + held


def prepare(output):
    plan = PROJECT / 'results/run-3-plan'
    for name, expected in read(plan/'manifest.json').items():
        assert sha(plan/name) == expected, f'Plan artifact changed: {name}'
    source_protocol = read(plan/'protocol.json')
    records = read(plan/'prompt-templates.json')
    calibration = read(plan/'calibration-templates.json')
    jobs = make_jobs(records, calibration)
    assert Counter(j['stage'] for j in jobs) == dict(baseline=24, calibration=48, discovery=288, held_out=288)
    assert len({j['id'] for j in jobs}) == 648
    output.mkdir(parents=True, exist_ok=True)
    if (output/'execution.json').exists():
        assert read(output/'generation-jobs.json') == jobs
        check_frozen(output)
        return
    for directory in ['plan', 'code', 'cases']:
        (output/directory).mkdir(exist_ok=True)
    for name in ['protocol.json','prompt-templates.json','calibration-templates.json','manifest.json']:
        shutil.copy2(plan/name, output/'plan'/name)
    for name in ['run_deception_run3.py','run_deception_probe.py','analyze_deception_run2.py']:
        shutil.copy2(PROJECT/name, output/'code'/name)
    dump(output/'generation-jobs.json', jobs)
    files = [p for folder in ['plan','code'] for p in (output/folder).iterdir() if p.is_file()]
    files.append(output/'generation-jobs.json')
    frozen = {str(p.relative_to(output)):sha(p) for p in files}
    lenses = {str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'results/jlens-family-1000').glob('target-*.pt'))}
    execution = dict(created_utc=now(), status='prepared', study='Run 3', source_plan_sha256=sha(plan/'protocol.json'),
                     frozen_files=frozen, lens_files=lenses, expected_checkpoint_sha256=source_protocol['checkpoint_sha256'],
                     sampling=source_protocol['sampling'], phases=['discovery', 'held_out'],
                     discovery_phase='24 knowledge baselines + 48 calibration + up to 288 discovery responses; no held-out condition responses.',
                     held_out_gate='Requires discovery-lock.json with review and analysis file hashes before starting held-out condition responses.',
                     baseline_gate='Automatic conservative factual assertion check; ambiguous cases deferred for review, not excluded. Reviews override only with a matching baseline hash.',
                     disk_cap_gib=512, minimum_free_gib=100, gpu='CUDA_VISIBLE_DEVICES=0', memory_fraction=0.70,
                     resume='Single-case seeds; completed generation JSON is the commit marker. Validate hashes before reuse. Interrupted uncommitted cases may repeat with the same seed.',
                     completion='Generation completion is separate from semantic review, J-lens analysis, and the conditional causal stage.')
    dump(output/'execution.json', execution)
    update_status(output, dict(stage='prepared', phase='discovery', completed=0, total=360, utc=now(),
                               note='Worker not yet started; held-out responses are gated.'))


def check_frozen(output):
    execution = read(output/'execution.json')
    for relative, expected in execution['frozen_files'].items():
        assert sha(output/relative) == expected, f'Frozen input/code changed: {relative}'
    return execution


def validate_resume(path, job, parent_sha=None):
    g = read(path)
    assert g['job_sha256'] == digest(job), f'Resume job changed: {job["id"]}'
    assert g.get('parent_generation_sha256') == parent_sha, f'Baseline changed: {job["id"]}'
    assert sha(path.parent/'rng.pt') == g['rng_sha256'], f'RNG artifact changed: {job["id"]}'
    assert len(g['output_ids']) == g['output_tokens'] and g['output_tokens'] > 0
    return g


def resolve_job(output, job):
    if 'baseline_id' not in job:
        return job['messages'], None, None
    parent = output/'cases'/job['baseline_id']/'generation.json'
    if not parent.exists():
        return None, None, dict(decision='review', reason='Baseline not generated yet')
    prior = read(parent)
    parent_sha = sha(parent)
    gate = baseline_gate(prior)
    review_path = output/'baseline-reviews.json'
    if review_path.exists():
        review = read(review_path).get(job['baseline_id'])
        if review:
            assert review['generation_sha256'] == parent_sha
            assert review['decision'] in ['eligible','exclude']
            assert review.get('reason')
            gate = review
    if gate['decision'] != 'eligible':
        return None, parent_sha, gate
    messages = prior['messages'] + [dict(role='assistant', content=prior['assistant_history_text']), job['followup']]
    return messages, parent_sha, gate


def update_status(output, value):
    dump(output/'status.json', value)
    with (output/'events.jsonl').open('a') as f:
        f.write(json.dumps(value) + '\n')
    rows = ''.join(f'<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>' for k,v in value.items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="30"><title>Run 3 — live status</title>
<style>body{{font:17px/1.65 system-ui;max-width:1050px;margin:40px auto;padding:0 24px;background:#f7f9fc;color:#203047}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #d0d9e3;overflow-wrap:anywhere}}a{{color:#175f9c}}.note{{background:#e5eef9;padding:18px}}</style>
<h1>Run 3: live status</h1><p>This page refreshes every 30 seconds. Timestamps are UTC.</p>
<p><a href="../run-3-plan/index.html">Experiment plan</a> · <a href="status.json">Status JSON</a> · <a href="execution.json">Frozen execution details</a> · <a href="generation-jobs.json">Job manifest</a> · <a href="run.log">Worker log</a></p>
<p class="note">Generation progress is separate from evidence of deception. Full outputs must be reviewed before interpreting J-lens readouts. Held-out conditions wait for the discovery analysis lock.</p><table>{rows}</table></html>'''
    atomic_text(output/'index.html', html)
    print(json.dumps(value), flush=True)


def run(output, phase, max_jobs=None):
    lock = (output/'.worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    execution = check_frozen(output)
    if phase == 'held_out':
        gate = read(output/'discovery-lock.json')
        assert gate['status'] == 'frozen_before_held_out'
        assert gate['execution_sha256'] == sha(output/'execution.json')
        assert gate['files'] and gate.get('review_complete')
        for relative, expected in gate['files'].items():
            assert sha(output/relative) == expected
    all_jobs = read(output/'generation-jobs.json')
    jobs = [j for j in all_jobs if j['stage'] in (['baseline','calibration','discovery'] if phase=='discovery' else ['held_out'])]
    started = time.time()
    completed = sum((output/'cases'/j['id']/'generation.json').exists() for j in jobs)
    worker = dict(pid=os.getpid(), started_utc=now(), phase=phase, cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
                  runner=str(Path(__file__).resolve()), execution_sha256=sha(output/'execution.json'))
    dump(output/'worker.json', worker)
    def status(stage, **extra):
        update_status(output, dict(stage=stage,phase=phase,utc=now(),elapsed_seconds=round(time.time()-started,2),
                                  completed=completed,total=len(jobs),pid=os.getpid(),**extra))
    status('checking model and lens provenance')
    import torch
    import transformers
    from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, StoppingCriteria, StoppingCriteriaList, set_seed
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no inference started')
    torch.cuda.set_per_process_memory_fraction(execution['memory_fraction'])
    metadata = read(PROJECT/'results/jlens-family-1000/family.json')['metadata']
    assert sha(PROJECT/'model/model.safetensors') == execution['expected_checkpoint_sha256'] == metadata['model_weights_sha256']['model.safetensors']
    assert sha(PROJECT/'model/config.json') == metadata['model_config_sha256']
    for relative, expected in execution['lens_files'].items():
        assert sha(PROJECT/relative) == expected
    environment = dict(torch=str(torch.__version__), transformers=transformers.__version__, gpu=torch.cuda.get_device_name(),
                       cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'), checkpoint_sha256=execution['expected_checkpoint_sha256'],
                       execution_sha256=sha(output/'execution.json'), runner_sha256=sha(__file__), terminal_ids=[0,2],
                       loops=4, attention='sdpa', dtype='bfloat16', memory_fraction=execution['memory_fraction'])
    if (output/'environment.json').exists():
        assert read(output/'environment.json') == environment, 'Execution environment changed on resume'
    else:
        dump(output/'environment.json', environment)
    status('loading model')
    tokenizer = AutoTokenizer.from_pretrained(PROJECT/'model', trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(PROJECT/'model', trust_remote_code=True, dtype=torch.bfloat16,
                                               device_map='cuda', attn_implementation='sdpa').eval()
    sampling = execution['sampling']
    completed = 0
    fresh = 0
    deferred = {}
    excluded = {}
    baseline_decisions = {}
    reviews = []
    with torch.inference_mode():
        for job in jobs:
            messages, parent_sha, gate = resolve_job(output, job)
            target = output/'cases'/job['id']/'generation.json'
            if gate and gate['decision'] != 'eligible':
                assert not target.exists(), 'Previously generated case now has a disallowed baseline; review provenance'
                (excluded if gate['decision']=='exclude' else deferred)[job['id']] = gate
                continue
            if target.exists():
                g = validate_resume(target, job, parent_sha)
                assert g['messages'] == messages
                assert tokenizer.decode(g['output_ids']) == g['raw_text']
                completed += 1
                if job['stage']=='baseline':
                    baseline_decisions[job['id']] = dict(**baseline_gate(g),generation_sha256=sha(target))
                continue
            if max_jobs is not None and fresh >= max_jobs:
                break
            disk = shutil.disk_usage(output)
            used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if disk.free < execution['minimum_free_gib']*2**30 or used > execution['disk_cap_gib']*2**30:
                status('paused_disk_limit', case=job['id'])
                return
            set_seed(job['seed'])
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
            width = inputs.input_ids.shape[1]
            rng = dict(cpu_before=torch.get_rng_state(),cuda_before=torch.cuda.get_rng_state(),seed=job['seed'])
            case_started = time.time()
            class Progress(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    steps = input_ids.shape[1] - width
                    if steps % 128 == 0:
                        status('generating',case=job['id'],job_stage=job['stage'],generated_tokens=steps,
                               excluded=len(excluded),deferred=len(deferred))
                    return torch.zeros(1,device=input_ids.device,dtype=torch.bool)
            status('generating',case=job['id'],job_stage=job['stage'],generated_tokens=0,
                   excluded=len(excluded),deferred=len(deferred))
            cache = DynamicCache()
            kwargs = dict(do_sample=job['do_sample'])
            if job['do_sample']:
                kwargs.update(temperature=sampling['temperature'],top_p=sampling['top_p'],top_k=sampling['top_k'])
            generated = model.generate(**inputs,past_key_values=cache,max_new_tokens=sampling['max_new_tokens'],
                                       pad_token_id=0,eos_token_id=[0,2],exit_at_step=3,logits_to_keep=1,
                                       stopping_criteria=StoppingCriteriaList([Progress()]),**kwargs)
            assert len(cache.layers)==192
            ids = generated[0,width:].tolist()
            end = next((i+1 for i,t in enumerate(ids) if t in [0,2]),len(ids))
            ids = ids[:end]
            raw = tokenizer.decode(ids)
            sections, proper = extract(raw)
            thought = split_trace(tokenizer, ids)
            rng.update(cpu_after=torch.get_rng_state(),cuda_after=torch.cuda.get_rng_state())
            directory = target.parent
            directory.mkdir(parents=True,exist_ok=True)
            temp = directory/'rng.pt.tmp'
            torch.save(rng,temp)
            temp.replace(directory/'rng.pt')
            natural = bool(ids and ids[-1] in [0,2])
            result = dict(job,job_sha256=digest(job),messages=messages,prompt_text=prompt,
                          input_ids=inputs.input_ids[0].tolist(),output_ids=ids,raw_text=raw,
                          assistant_history_text=tokenizer.decode(ids[:-1] if natural else ids).strip(),
                          output_tokens=len(ids),natural_stop=natural,capped=not natural and len(ids)>=sampling['max_new_tokens'],
                          format_complete=bool(proper),**thought,sections=sections,
                          recipient_text=sections.get('MESSAGE',''),parent_generation_sha256=parent_sha,
                          baseline_gate=gate,rng_sha256=sha(directory/'rng.pt'),
                          generated_utc=now(),generation_seconds=round(time.time()-case_started,3),
                          review_status='pending_semantic_review',deception_confirmed=False)
            # JSON is the commit marker, written after the RNG and raw output.
            atomic_text(directory/'response.txt',raw)
            dump(target,result)
            completed += 1
            fresh += 1
            if job['stage']=='baseline':
                baseline_decisions[job['id']] = dict(**baseline_gate(result),generation_sha256=sha(target))
                dump(output/'baseline-gates.json',baseline_decisions)
            reviews.append(dict(id=job['id'],stage=job['stage'],generation_sha256=sha(target),
                                format_complete=proper,natural_stop=natural,review_status='pending'))
            dump(output/'progress-review-queue.json',reviews)
            status('case_saved',case=job['id'],job_stage=job['stage'],output_tokens=len(ids),
                   case_seconds=result['generation_seconds'],excluded=len(excluded),deferred=len(deferred),
                   max_gpu_allocated_gib=round(torch.cuda.max_memory_allocated()/2**30,2))
            del generated,cache,inputs,rng
            torch.cuda.empty_cache()
    dump(output/'baseline-gates.json',baseline_decisions)
    dump(output/f'{phase}-excluded.json',excluded)
    dump(output/f'{phase}-deferred.json',deferred)
    saved = [dict(id=j['id'],stage=j['stage'],generation_sha256=sha(output/'cases'/j['id']/'generation.json'),review_status='pending')
             for j in jobs if (output/'cases'/j['id']/'generation.json').exists()]
    dump(output/f'{phase}-review-queue.json',saved)
    finished = completed+len(excluded)+len(deferred)==len(jobs)
    stage = ('awaiting_baseline_review' if deferred else 'awaiting_discovery_review' if phase=='discovery' else 'awaiting_held_out_review') if finished else 'paused_after_requested_jobs'
    dump(output/f'{phase}-generation-summary.json',dict(stage=stage,completed=completed,total=len(jobs),
         excluded=len(excluded),deferred=len(deferred),generation_complete=finished and not deferred,
         semantic_review_complete=False,lens_analysis_complete=False,utc=now()))
    status(stage,excluded=len(excluded),deferred=len(deferred),
           note='Next: semantic review, fixed-window J-lens analysis, then the preregistered phase gate. No target claim has been made.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--phase',choices=['discovery','held_out'],default='discovery')
    parser.add_argument('--max-jobs',type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare_only:
        prepare(output)
        return
    try:
        run(output,args.phase,args.max_jobs)
    except BlockingIOError:
        raise SystemExit('Another Run 3 worker already holds the output lock')
    except Exception as error:
        previous = read(output/'status.json') if (output/'status.json').exists() else {}
        update_status(output,dict(**{k:v for k,v in previous.items() if k not in ['stage','utc','error']},
                                  stage='failed',utc=now(),error=f'{type(error).__name__}: {error}'))
        traceback.print_exc()
        raise


if __name__=='__main__':
    main()
