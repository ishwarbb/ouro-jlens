# Third-party sources

`jacobian-lens/` contains the unchanged model-independent `jlens` implementation,
experiment prompts, package metadata, README and Apache-2.0 license from the
reference repository through the `ishwarbb/jacobian-lens` fork, pinned at
`98f59779f89e092c418615e5e67c52fa5fd590cd`. The exact source and included paths are
recorded in `provenance/jacobian-lens.json`. Original notices remain in the files
and `jacobian-lens/LICENSE`.

- Reference implementation: https://github.com/anthropics/jacobian-lens
- Original workspace study: https://transformer-circuits.pub/2026/workspace/index.html
- Looped-model study: https://arxiv.org/html/2609.01924v1

Ouro-2.6B model weights and model code come from `ByteDance/Ouro-2.6B`, revision
`1ed04250da1a9936042725d302e81c8fa2ab5abd`, under its upstream Apache-2.0 license.
They are downloaded separately; the model is not vendored or included in release
bundles. The model revision is recorded in `model_revision.json`.

The full fit uses WikiText-103-raw-v1 through the `Salesforce/wikitext` dataset,
revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Corpus records are not bundled;
the preparation scripts download the pinned upstream corpus and preserve its
selection metadata. Historical frozen records may contain evaluation prompts
and generated excerpts; their source attribution is retained in the manifests
and reports. The pilot's ten neutral passages in `fitting_data/neutral_10.json`
were authored for this experiment.

The viewer is an independent implementation inspired by the J-Lens Visualizer,
Neuronpedia and WeZZard's Qwen J-lens viewer. Links appear in its in-app guide.
No model API keys, benchmark judge credentials, or third-party website assets
are included.
