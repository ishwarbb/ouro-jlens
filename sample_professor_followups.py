"""Small exploratory replication of the four existing score-switch prompts.

Eight samples per prompt, fixed seeds and 256-token cap. All attempts saved;
no instruction to hide reasoning, no rejection sampling, no lens-based selection.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'jacobian-lens'))
from replay_professor_followups import save,sha
from ouro_lens_config import DEFAULT_FAMILY


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--item',type=int,choices=range(4),required=True)
    args=p.parse_args()
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache,set_seed
    from run_deception_variants import extract_sections
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/professor-followups-20261016/fresh'
    out.mkdir(exist_ok=True)
    source=ROOT/f'results/deception-run2-history/history-i{args.item}-conflict-r2/generation.json'
    original=json.loads(source.read_text())
    seed=2026101600+args.item
    plan=dict(item=args.item,batch_size=8,seed=seed,temperature=0.9,top_p=0.95,top_k=0,max_new_tokens=256,
        source=str(source.relative_to(ROOT)),source_sha256=sha(source),messages=original['messages'],
        input_ids=original['input_ids'],model_revision=json.loads((ROOT/'model_revision.json').read_text()),
        author_lens_path=str(DEFAULT_FAMILY.relative_to(ROOT)),author_lens_manifest_sha256=sha(DEFAULT_FAMILY/'family.json'),
        selection='Exploratory replication on existing prompts; all eight attempts retained. Behavior screened before lens readout.',
        code_sha256=sha(Path(__file__)))
    receipt=out/f'item-{args.item}-plan.json'
    if receipt.exists():raise FileExistsError(receipt)
    save(receipt,plan)
    tok=AutoTokenizer.from_pretrained(ROOT/'model',trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/'model',trust_remote_code=True,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    set_seed(seed)
    ids=torch.tensor([original['input_ids']]*8,device='cuda')
    cache=DynamicCache();started=time.monotonic()
    with torch.inference_mode():
        generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=cache,
            max_new_tokens=256,do_sample=True,temperature=0.9,top_p=0.95,top_k=0,
            pad_token_id=0,eos_token_id=[0,2],exit_at_step=3,logits_to_keep=1)
    assert len(cache.layers)==192
    torch.save(dict(generated=generated.cpu(),input_ids=ids.cpu()),out/f'item-{args.item}-batch.pt')
    for i,row in enumerate(generated[:,ids.shape[1]:].tolist()):
        terminal=next((j for j,t in enumerate(row) if t in (0,2)),None)
        output=row[:terminal+1] if terminal is not None else row
        case=f'fresh-history-i{args.item}-s{i}'
        data=dict(id=case,source_prompt_sha256=sha(source),batch_seed=seed,batch_index=i,
            messages=original['messages'],prompt_text=original['prompt_text'],input_ids=original['input_ids'],
            output_ids=output,raw_text=tok.decode(output),natural_stop=terminal is not None,
            capped=terminal is None,degenerate=False,output_tokens=len(output),
            correct_answer=original['correct_answer'],wrong_answer=original['wrong_answer'],options=original['options'],
            **extract_sections(tok,output))
        save(out/case/'generation.json',data)
        print(case,data['natural_stop'],repr(data['raw_text']),flush=True)
    save(out/f'item-{args.item}-complete.json',dict(samples=8,seconds=time.monotonic()-started,
        plan_sha256=sha(receipt),gpu=torch.cuda.get_device_name(),torch=torch.__version__))


if __name__=='__main__':main()
