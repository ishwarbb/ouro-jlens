import json
from jlens.examples import load_wikitext_prompts

prompts = load_wikitext_prompts(n_prompts=1000)
with open("prompts_1000.json", "w") as f:
    json.dump(prompts, f)
print(f"saved {len(prompts)} prompts")
