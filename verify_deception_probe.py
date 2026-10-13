"""Replay pilot aggregates from saved top-k IDs; verify artifact consistency."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    import torch
    torch.set_num_threads(2)
    protocol = load(root / "protocol.json")
    vocab = load(root / "vocabulary.json")
    all_tokens = [t for entry in vocab.values() for t in entry["token_ids"]]
    environment = load(root / "environment.json")
    assert hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest() == environment["protocol_sha256"]
    assert hashlib.sha256((root / "runner.py").read_bytes()).hexdigest() == environment["runner_sha256"]
    release = load(Path(__file__).parent / "artifacts.json")
    expected = {Path(f["path"]).name: f["sha256"] for f in release["bundles"]["lenses"]["files"]}
    assert environment["family_files"] == expected
    verified = []
    for job in protocol["jobs"]:
        directory = root / job["id"]
        if not (directory / "metrics.json").exists():
            continue
        generation = load(directory / "generation.json")
        metrics = load(directory / "metrics.json")
        state = torch.load(directory / "states.pt", weights_only=True, mmap=True)
        top = torch.load(directory / "top25.pt", weights_only=True)
        prefix = len(generation["prompt_record"]["seq_ids"])
        ids = generation["prompt_record"]["seq_ids"] + generation["output_ids"]
        assert state["ids"] == ids
        assert state["states"].shape == (192, len(ids), 2048)
        assert top["ids"].shape == (2, 192, len(ids)-prefix+1, 25)
        assert top["position_start"] == prefix-1
        absent_tokens = [t for w, entry in vocab.items() if w not in generation["lexical_mentions"]["raw_text"] for t in entry["token_ids"]]
        for method_index, method in enumerate(top["methods"]):
            for loop in range(4):
                selected = top["ids"][method_index, loop*48+24:loop*48+41, :21]
                actual = metrics["primary"][f"{method}/loop-{loop+1}"]
                assert actual["primary_cells"] == selected.shape[0] * selected.shape[1]
                for count in [10, 25]:
                    for label, tokens in [("", all_tokens), ("lexical_absent_", absent_tokens)]:
                        n = int(torch.isin(selected[:, :, :count], torch.tensor(tokens)).any(-1).sum())
                        assert actual[f"{label}top{count}_cells"] == n
        # Every recorded hit must identify the saved ordinal in top-25.
        total_hits = 0
        for line in (directory / "hits.jsonl").read_text().splitlines():
            hit = json.loads(line)
            mi = top["methods"].index(hit["method"])
            assert int(top["ids"][mi, hit["site"], hit["position"]-prefix+1, hit["rank"]-1]) == hit["token_id"]
            assert hit["token_id"] in vocab[hit["word"]]["token_ids"]
            total_hits += 1
        verified.append(dict(id=job["id"], total_saved_positions=len(ids), candidate_hits=total_hits,
                             primary_cells_per_method=sum(v["primary_cells"] for k,v in metrics["primary"].items() if k.startswith("own_loop_jlens/"))))
        del state, top
    result = dict(status="passed", generations_verified=len(verified), expected=len(protocol["jobs"]),
                  release_lens_hashes_match=True, protocol_and_runner_hashes_match=True, records=verified)
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
