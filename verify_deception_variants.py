"""Verify screen token provenance and replay measurements from saved artifacts."""
import argparse
import json
from pathlib import Path
from run_deception_probe import sha,mentions


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument("directory",type=Path);a=p.parse_args();root=a.directory
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(2)
    protocol=read(root/"protocol.json");env=read(root/"environment.json")
    assert env["protocol_sha256"]==sha(root/"protocol.json")
    assert env["runner_sha256"]==sha(root/"runner.py")
    tok=AutoTokenizer.from_pretrained(Path(__file__).parent/"model",trust_remote_code=True)
    rows=0;output_tokens=0;capped=[];degenerate=[]
    for batch_path in sorted((root/"batches").glob("batch-*.pt")):
        batch=torch.load(batch_path,weights_only=True)
        width=batch["input_ids"].shape[1]
        for row,case in enumerate(batch["job_ids"]):
            record=read(root/case/"generation.json")
            job=protocol["jobs"][record["order"]]
            assert job["id"]==case and all(record[k]==v for k,v in job.items())
            prompt=tok.apply_chat_template(job["messages"],tokenize=False,add_generation_prompt=True)
            assert prompt==record["prompt_text"]
            assert tok.encode(prompt,add_special_tokens=False)==record["input_ids"]
            assert batch["input_ids"][row][batch["attention_mask"][row].bool()].tolist()==record["input_ids"]
            ids=record["output_ids"]
            assert batch["generated"][row,width:width+len(ids)].tolist()==ids
            assert tok.decode(ids)==record["raw_text"]
            assert record["natural_stop"]==(ids[-1] in [0,2] and not record["degenerate"])
            assert not mentions(prompt)
            if record.get("knowledge_check_source"):
                from analyze_deception_run2 import parse_sections,choice
                source=Path(__file__).parent/record["knowledge_check_source"]
                assert sha(source)==record["knowledge_check_sha256"]
                prior=read(source);sections,_,proper,_=parse_sections(prior["raw_text"])
                assert prior["natural_stop"] and not prior["capped"] and proper
                assert choice(sections["CHECK"],prior["options"])==record["correct_answer"]
                assert choice(sections["MESSAGE"],prior["options"])==record["correct_answer"]
                assert record["messages"][:2]==prior["messages"]
                assert record["messages"][2]["content"]==prior["raw_text"].replace("<|im_end|>","").replace("<|endoftext|>","").strip()
            rows+=1;output_tokens+=len(ids)
            if record["capped"]:capped.append(case)
            if record["degenerate"]:degenerate.append(case)
    lens=root/"lens";lens_verified=[];lens_checks=[]
    if (lens/"protocol.json").exists():
        lprotocol=read(lens/"protocol.json");vocab=read(lens/"vocabulary.json")
        all_tokens=[t for ts in vocab.values() for t in ts]
        for case in lprotocol["selection"]["case_ids"]:
            directory=lens/case
            if not (directory/"metrics.json").exists():continue
            record=read(root/case/"generation.json");m=read(directory/"metrics.json")
            assert m["source_generation_sha256"]==sha(root/case/"generation.json")
            states=torch.load(directory/"states.pt",weights_only=True,mmap=True)
            assert states["ids"]==record["input_ids"]+record["output_ids"]
            assert states["states"].shape==(192,len(states["ids"]),2048)
            top=torch.load(directory/"top25.pt",weights_only=True)
            prefix=len(record["input_ids"])
            absent_raw=[t for w,ts in vocab.items() if w not in mentions(record["raw_text"]) for t in ts]
            absent_explanation=[t for w,ts in vocab.items() if w not in mentions(record["explanation_text"]) for t in ts]
            for method_index,method in enumerate(top["methods"]):
                for window,positions in m["windows"].items():
                    indices=[position-top["position_start"] for position in positions]
                    for loop in range(4):
                        values=top["ids"][method_index,loop*48+24:loop*48+41][:,indices]
                        expected=m["metrics"][f"{method}/{window}/loop-{loop+1}"]
                        assert expected["cells"]==17*len(indices)
                        for key,count,tokens in [("top10",10,all_tokens),("top25",25,all_tokens),("absent_raw_top10",10,absent_raw),("absent_trace_top10",10,absent_explanation)]:
                            assert expected[key]==int(torch.isin(values[:,:,:count],torch.tensor(tokens)).any(-1).sum())
            hit_count=0
            for line in (directory/"hits.jsonl").read_text().splitlines():
                h=json.loads(line);mi=top["methods"].index(h["method"]);pos=h["position"]-top["position_start"]
                assert int(top["ids"][mi,h["site"],pos,h["ordinal"]-1])==h["token_id"]
                vals=top["values"][mi,h["site"],pos]
                assert 1+int((vals>vals[h["ordinal"]-1]).sum())==h["rank"]
                assert h["token_id"] in vocab[h["word"]]
                hit_count+=1
            assert hit_count==m["total_hits"]
            lens_verified.append(case);lens_checks.append(m["replay_check"])
            del states,top
    result=dict(status="passed",generations_verified=rows,expected_generations=len(protocol["jobs"]),
                output_tokens=output_tokens,capped=capped,degenerate=degenerate,lens_cases_verified=lens_verified,
                max_replay_logit_difference=max((v["max_abs"] for v in lens_checks),default=None),
                replay_argmax_agreement=sum(v["argmax_agrees"] for v in lens_checks),
                note="Ranks verified against stored top-25 values; exact transcript and primary counts independently reconstructed. Replay is not bit-exact cached generation.")
    (root/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
