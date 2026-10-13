"""Check portable Run 9/10 plan metadata without dependencies, downloads or GPUs."""

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative):
    return json.loads((ROOT / relative).read_text())


def check_links(document):
    path = ROOT / document
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
        if target.startswith(("https://", "http://", "#")):
            continue
        require((path.parent / target.split("#")[0]).exists(),
                f"Missing link in {document}: {target}")


def main():
    run9 = load("plans/run9/protocol.json")
    run10 = load("plans/run10/protocol.json")
    for relative, data in [("plans/run9/protocol.json", run9),
                           ("plans/run10/protocol.json", run10)]:
        require(data["launched"] is False, f"{relative} must remain a plan")
        require(data["status"] == "planned_pending_implementation_and_execution_freeze",
                f"Unexpected plan status: {relative}")
        require(not data["portability"]["original_host_required"], "Host dependency")
        require(not data["portability"]["runners_implemented"], "Runner status mismatch")
        base = (ROOT / relative).parent
        for link in [data["documentation"], data["portability"]["handoff"], *data["sources"]]:
            if not link.startswith("https://"):
                require((base / link).exists(), f"Missing protocol source: {link}")

    models = run9["models"]
    require(len(models) == len({m["id"] for m in models}) == 5, "Model panel mismatch")
    for model in [*models, run10["model"]]:
        require(re.fullmatch(r"[0-9a-f]{40}", model["revision"]), "Unpinned model revision")
        directory = Path(model["model_directory"])
        require(not directory.is_absolute() and ".." not in directory.parts,
                "Model directory must be portable and repository-relative")
    require(models[0]["revision"] == load("model_revision.json")["revision"]
            == run10["model"]["revision"], "Ouro revision mismatch")

    stages = run9["stages"]
    require(set(stages) == {"A", "B", "C"}, "Run 10 work remains in Run 9")
    a, b, c = (stages[key] for key in "ABC")
    model_count = len(models)
    require(a["calibration_episodes_per_model"] == len(a["families"])
            * a["calibration_records_per_family"] * len(a["calibration_settings"]),
            "Calibration count mismatch")
    require(a["validation_episodes_per_model"] == len(a["families"])
            * a["validation_records_per_family"], "Validation count mismatch")
    require(a["all_models_maximum_episodes"] == model_count
            * (a["calibration_episodes_per_model"] + a["validation_episodes_per_model"]),
            "Stage A total mismatch")
    require(a["all_models_runtime_checks"] == model_count * a["runtime_checks_per_model"],
            "Runtime check count mismatch")
    require(b["records_per_model"] == len(b["templates"]) * b["records_per_template"],
            "Stage B record count mismatch")
    require(b["cells_per_record"] == len(b["interfaces"]) * len(b["incentives"])
            * len(b["audit_conditions"]) * b["draws_per_cell"], "Factorial mismatch")
    require(b["episodes_per_model"] == b["records_per_model"] * b["cells_per_record"],
            "Stage B per-model count mismatch")
    require(b["maximum_episodes"] == b["episodes_per_model"] * b["maximum_models"],
            "Stage B total mismatch")
    require(b["lens_discovery_records"] + b["lens_heldout_records"] == b["records_per_model"],
            "Lens split mismatch")
    require(b["maximum_knowledge_generations"] == b["records_per_model"] * b["maximum_models"],
            "Stage B factual baseline mismatch")
    require(c["episodes_per_model"] == c["records_per_family"] * c["family_count"]
            * c["incentive_levels"] * c["audit_levels"], "Stage C factorial mismatch")
    require(c["maximum_episodes"] == c["episodes_per_model"] * c["maximum_models"],
            "Stage C total mismatch")
    require(c["maximum_knowledge_generations"] == c["records_per_family"]
            * c["family_count"] * c["maximum_models"], "Stage C factual baseline mismatch")
    feasibility = c["feasibility"]
    require(feasibility["maximum_episodes"] == feasibility["records_per_family"]
            * feasibility["family_count"] * feasibility["maximum_models"],
            "Richer feasibility count mismatch")
    require(c["maximum_episodes_including_feasibility"] == c["maximum_episodes"]
            + feasibility["maximum_episodes"], "Stage C combined count mismatch")
    total_generations = (
        a["all_models_runtime_checks"]
        + (a["all_models_maximum_episodes"] + b["maximum_episodes"])
        * run9["episode_limits"]["stages_A_B"]["maximum_generations_per_episode"]
        + c["maximum_episodes_including_feasibility"]
        * run9["episode_limits"]["stage_C"]["maximum_generations_per_episode"]
        + b["maximum_knowledge_generations"] + c["maximum_knowledge_generations"]
    )
    require(total_generations == run9["resources"][
        "total_maximum_runtime_and_behavior_generations_excluding_lens"],
        "Run 9 generation ceiling mismatch")

    records, model = run10["records"], run10["model"]
    require(model["physical_layers"] * model["loops"] == model["virtual_sites"] == 192,
            "Ouro geometry mismatch")
    require(records["fresh_prompts"] == records["semantic_templates"]
            * records["prompts_per_template"], "Run 10 template count mismatch")
    conditions = run10["conditions"]
    require(len(conditions) == len({c["id"] for c in conditions}) == 9,
            "Run 10 condition IDs mismatch")
    trials = records["fresh_prompts"] * records["destinations_per_prompt"] * len(conditions)
    require(trials == records["logical_trial_settings"]
            == run10["budget"]["logical_trial_settings"], "Run 10 trial count mismatch")
    require(trials * run10["endpoints"]["secondary_max_new_tokens"]
            == run10["budget"]["maximum_secondary_generated_tokens"], "Run 10 token ceiling mismatch")
    lo, hi = run10["intervention"]["physical_layer_band_zero_based"]
    for loop, sites in run10["intervention"]["virtual_site_ranges"].items():
        offset = (int(loop) - 1) * model["physical_layers"]
        require(sites == [offset + lo, offset + hi], "Virtual site mapping mismatch")
    require(run10["lenses"]["targets"] == [model["physical_layers"] * i - 1
                                            for i in range(1, model["loops"] + 1)],
            "Loop-end targets mismatch")
    require(run10["preflight"]["engineering_prompts"]
            * run10["preflight"]["maximum_forwards_per_prompt"]
            == run10["preflight"]["maximum_forwards"]
            == run10["budget"]["maximum_preflight_forwards"], "Preflight count mismatch")

    for document in ["RUN9_PLAN.md", "RUN10_PLAN.md", "plans/README.md"]:
        check_links(document)
    print(f"Run 9: {model_count} candidate models; maximum {total_generations:,} behavior/runtime generations.")
    print(f"Run 10: {trials:,} trial settings; maximum {run10['budget']['maximum_secondary_generated_tokens']:,} secondary tokens.")
    print("Plan metadata, counts, revisions, site mapping and local links passed. No experiments launched.")


if __name__ == "__main__":
    main()
