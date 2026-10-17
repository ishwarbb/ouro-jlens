# HF model revisions the lenses in this directory were fitted against
Pass revision=<hash> to from_pretrained / snapshot_download when re-downloading.

Qwen/Qwen3.6-27B              6a9e13bd6fc8f0983b9b99948120bc37f49c13e9   -> jacobian_lens_qwen3.6-27b.pt, jacobian_lens_qwen_t24.pt
ByteDance/Ouro-2.6B           1ed04250da1a9936042725d302e81c8fa2ab5abd   -> jacobian_lens_ouro-2.6b*.pt, jacobian_lens_ouro_mid_t*.pt
ByteDance/Ouro-2.6B-Thinking  f1edd81e7ac41355db670500ceaf204e0f73af68   -> thinking/jacobian_lens_ouro-2.6b*.pt
tomg-group-umd/huginn-0125    bb6621b65e90b6a4b9b29ef88dc83866d450470c   -> jacobian_lens_huginn-0125_r16.pt, *huginn_ts*_t*.pt
