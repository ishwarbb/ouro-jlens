"""J-lens replay of selected exact screen transcripts and matched controls.

Selection is frozen before reading any lens. Replay uses a full causal forward,
so BF16 states can differ slightly from the original padded cached generation.
It does not resample or change the recorded response.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"jacobian-lens"))
from run_deception_probe import WORDS,dump,sha,mentions


def main():
    p=argparse.ArgumentParser();p.add_argument("--screen",type=Path,required=True)
    p.add_argument("--selection",type=Path,required=True);a=p.parse_args()
    selected=json.loads(a.selection.read_text())
    out=a.screen/"lens";out.mkdir(exist_ok=True)
    protocol=dict(selection=selected,words=WORDS,sites=list(range(192)),workspace_layers=[24,40],
       readouts=["jlens","logit"],primary_windows=["last prompt position and first 20 generated positions", "position before recipient message and first 20 message positions"],
       secondary="All generated positions, with workspace-band results separated from other layers. Selection is exploratory, based on behavior, and is not used to estimate prevalence or detector accuracy.",
       source_position="Residual at a source token predicts the next token; EOS source is excluded.",
       replay="Exact saved token IDs; no BOS inserted; full causal BF16 forward. Not bit-exact online hidden states.")
    if (out/"protocol.json").exists():
        assert json.loads((out/"protocol.json").read_text())==protocol
    else:
        dump(out/"protocol.json",protocol);shutil.copy2(__file__,out/"runner.py")
    import torch
    from ouro_full_fit import load_model
    from ouro_lens_family import OuroLensFamily
    from ouro_viewer_inference import CachedDecoder
    from workspace_interventions import Vocabulary
    from jlens.hooks import ActivationRecorder
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    torch.cuda.set_per_process_memory_fraction(0.85)
    begin=time.time()
    def status(stage,**extra):
        value=dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=round(time.time()-begin,2),**extra)
        dump(out/"status.json",value);print(json.dumps(value),flush=True)
    status("loading")
    model=load_model();tok=model.tokenizer
    family=OuroLensFamily.load(ROOT/"results/jlens-family-1000");family.check_model(model)
    vocab=Vocabulary(tok)
    word_tokens={word:vocab.candidates(word) for group in WORDS.values() for word in group}
    token_words={t:w for w,ts in word_tokens.items() for t in ts}
    dump(out/"vocabulary.json",word_tokens)
    env=json.loads((a.screen/"environment.json").read_text())
    assert sha(ROOT/"model/model.safetensors")==env["checkpoint_sha256"]==family.metadata["model_weights_sha256"]["model.safetensors"]
    release=json.loads((ROOT/"artifacts.json").read_text())
    checked={}
    for f in release["bundles"]["lenses"]["files"]:
        actual=sha(ROOT/f["path"]);assert actual==f["sha256"];checked[Path(f["path"]).name]=actual
    dump(out/"environment.json",dict(screen_environment=env,selection_sha256=sha(a.selection),
         runner_sha256=sha(__file__),lens_files=checked,torch=str(torch.__version__),gpu=torch.cuda.get_device_name()))
    with torch.inference_mode():
        for case_index,case in enumerate(selected["case_ids"]):
            directory=out/case;directory.mkdir(exist_ok=True)
            if (directory/"metrics.json").exists():continue
            record=json.loads((a.screen/case/"generation.json").read_text())
            inputs=record["input_ids"];outputs=record["output_ids"];ids=inputs+outputs;prefix=len(inputs)
            status("capturing exact transcript",case=case,index=case_index,total=len(selected["case_ids"]),tokens=len(ids))
            with ActivationRecorder(model.layers,at=list(range(192))) as captured:
                model.forward(torch.tensor([ids],device="cuda"))
            states=torch.stack([captured.activations[s][0] for s in range(192)]).cpu()
            del captured
            torch.save(dict(ids=ids,states=states),directory/"states.pt")
            # Compare replay prefix prediction with cached single-case prefix.
            replay_logits=model.unembed(states[191,prefix-1].to("cuda")).float()
            with CachedDecoder(model) as decoder:
                cached_logits,_=decoder.forward(inputs)
            difference=(cached_logits-replay_logits).abs()
            check=dict(max_abs=float(difference.max()),mean_abs=float(difference.mean()),
                       argmax_agrees=int(cached_logits.argmax())==int(replay_logits.argmax()))
            if check["max_abs"]>1.0:raise RuntimeError(f"Large replay discrepancy: {check}")
            dump(directory/"replay-check.json",check)
            del decoder,cached_logits,replay_logits,difference
            raw=record["raw_text"];recipient=record["recipient_text"]
            start_char=raw.rfind(recipient) if recipient else -1
            msg_token=0
            if start_char>=0:
                for i in range(len(outputs)):
                    if len(tok.decode(outputs[:i+1]))>start_char:
                        msg_token=i;break
            message_source=prefix+msg_token-1
            stop=len(ids)-1 if record["natural_stop"] else len(ids)
            source_positions=list(range(prefix-1,stop))
            windows={"start":set(range(prefix-1,min(prefix+20,stop))),
                     "message":set(range(message_source,min(message_source+21,stop)))}
            raw_mentions=mentions(raw);trace_mentions=mentions(record["explanation_text"])
            top_ids=torch.empty((2,192,len(source_positions),25),dtype=torch.int32)
            top_values=torch.empty_like(top_ids,dtype=torch.float32)
            metrics={};best={};counts=Counter();hits=0
            stream=(directory/"hits.jsonl").open("w")
            for site in range(192):
                target=family.target_for(site)
                matrix=None if site==target else family.lenses[target].jacobians[site].to("cuda")
                workspace=24<=site%48<=40
                for start in range(prefix-1,stop,32):
                    end=min(start+32,stop);h=states[site,start:end].to("cuda").float()
                    for mi,method in enumerate(["jlens","logit"]):
                        transported=h if mi==1 or matrix is None else h@matrix.T
                        scores=model.unembed(transported).float()
                        vals,order=scores.topk(25,-1)
                        vals=vals.cpu();order=order.cpu()
                        top_ids[mi,site,start-prefix+1:end-prefix+1]=order.int()
                        top_values[mi,site,start-prefix+1:end-prefix+1]=vals
                        for row,(row_ids,row_values) in enumerate(zip(order.tolist(),vals.tolist())):
                            position=start+row
                            found=[(j+1,t,token_words[t]) for j,t in enumerate(row_ids) if t in token_words]
                            for name,positions in windows.items():
                                if not workspace or position not in positions:continue
                                key=f"{method}/{name}/loop-{site//48+1}"
                                m=metrics.setdefault(key,dict(cells=0,top10=0,top25=0,absent_raw_top10=0,absent_trace_top10=0))
                                m["cells"]+=1;m["top10"]+=any(rank<=10 for rank,_,_ in found);m["top25"]+=bool(found)
                                m["absent_raw_top10"]+=any(rank<=10 and w not in raw_mentions for rank,_,w in found)
                                m["absent_trace_top10"]+=any(rank<=10 and w not in trace_mentions for rank,_,w in found)
                            for ordinal,tid,word in found:
                                rank=1+sum(v>row_values[ordinal-1] for v in row_values)
                                hit=dict(method=method,site=site,loop=site//48+1,layer=site%48,position=position,
                                         generated_position=position-prefix,word=word,token_id=tid,ordinal=ordinal,rank=rank,
                                         workspace=workspace,start_window=position in windows["start"],message_window=position in windows["message"],
                                         absent_raw=word not in raw_mentions,absent_explanation=word not in trace_mentions,
                                         token=tok.decode([ids[position]]),next_token=tok.decode([ids[position+1]]) if position+1<len(ids) else None,
                                         context=tok.decode(ids[max(prefix,position-12):min(len(ids),position+13)]))
                                stream.write(json.dumps(hit,ensure_ascii=False)+"\n");hits+=1
                                if method=="jlens" and workspace:
                                    counts[word]+=1
                                    if word not in best or rank<best[word]["rank"]:best[word]=hit
                del matrix
            stream.close()
            torch.save(dict(ids=top_ids,values=top_values,position_start=prefix-1,methods=["jlens","logit"]),directory/"top25.pt")
            result=dict(id=case,source_generation_sha256=sha(a.screen/case/"generation.json"),
                        source_positions=source_positions,message_source=message_source,message_boundary_found=start_char>=0,
                        windows={k:sorted(v) for k,v in windows.items()},metrics=metrics,
                        workspace_word_counts=dict(counts),best_workspace_hits=list(best.values()),
                        lexical_mentions=dict(raw=raw_mentions,explanation=trace_mentions),
                        natural_stop=record["natural_stop"],capped=record["capped"],degenerate=record["degenerate"],
                        replay_check=check,total_hits=hits)
            dump(directory/"metrics.json",result)
            del states,top_ids,top_values
            torch.cuda.empty_cache()
            status("case complete",case=case,index=case_index+1,total=len(selected["case_ids"]),hits=hits)
    model.close();status("complete",cases=len(selected["case_ids"]))


if __name__=="__main__":main()
