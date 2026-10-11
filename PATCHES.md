# Changes from upstream

Base: <https://github.com/multimodal-art-projection/YuE> at commit `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` (yue2-infer 0.1.6), `src/yue2/` only. Licensed Apache 2.0; the license and notices are unchanged. `MODEL_LICENSE` was refreshed from upstream `main` (it gained the individual creator permission on 2026-09-16, after the base commit). It covers the weights only.

Modified or added files (the diff is in `patches/fp16-turing.patch`):

- `src/yue2/dtype.py` (new): reads `YUE2_DTYPE` (`float16` default, `bfloat16`, `float32`).
- `src/yue2/pipeline.py`: the AR/NAR model is loaded with that dtype instead of hard-coded `torch.bfloat16`; the recorded `model_dtype` follows it.
- `src/yue2/fast.py`: the optional vLLM backend uses the same dtype.
- `pyproject.toml`: version `0.1.6+fp16turing1`.

Not changed: attention (the model already uses `torch.nn.functional.scaled_dot_product_attention`, so no FlashAttention is needed), the VAE (already float32), sampling, tokenization, or weights.

Upstream's own tests were not vendored.

To apply the patch to a fresh upstream checkout instead of using the vendored copy:

```bash
git clone https://github.com/multimodal-art-projection/YuE.git && cd YuE
git checkout 92a73cc7652fcc1f937855e4b765e0a0edd7ff2e
git apply /path/to/yue2-fp16-turing/patches/fp16-turing.patch
```
