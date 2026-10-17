"""Hugging Face revisions the released lenses were fitted against.

Every model load in this directory passes revision=revision(model_id), so a
re-run downloads exactly these checkpoints (and, for Ouro/Huginn, exactly
this trust_remote_code). Mirrors MODEL_REVISIONS.md at the repo root.
"""

REVISIONS = {
    "Qwen/Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    "ByteDance/Ouro-2.6B": "1ed04250da1a9936042725d302e81c8fa2ab5abd",
    "ByteDance/Ouro-2.6B-Thinking": "f1edd81e7ac41355db670500ceaf204e0f73af68",
    "tomg-group-umd/huginn-0125": "bb6621b65e90b6a4b9b29ef88dc83866d450470c",
}

# Commit of https://github.com/anthropics/jacobian-lens the code was run with.
JLENS_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e"


def revision(model_id):
    if model_id not in REVISIONS:
        raise KeyError(f"no pinned revision for {model_id!r}; the released "
                       f"lenses only match {sorted(REVISIONS)}")
    return REVISIONS[model_id]
