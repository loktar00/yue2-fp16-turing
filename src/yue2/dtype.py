"""Model dtype selection (fp16-for-Turing patch, not part of upstream YuE2).

Upstream hard-codes bfloat16, which Turing GPUs (RTX 20xx, sm_75) cannot run.
This module reads the YUE2_DTYPE environment variable instead.

    YUE2_DTYPE=float16    default in this repo; works on Turing and newer
    YUE2_DTYPE=bfloat16   upstream behaviour; Ampere (RTX 30xx) and newer
    YUE2_DTYPE=float32    slowest and ~2x the weight memory; a fallback if
                          fp16 ever produces NaN / silent audio
"""
import os

DEFAULT = "float16"
CHOICES = ("float16", "bfloat16", "float32")


def model_dtype_name():
    value = os.environ.get("YUE2_DTYPE", DEFAULT).strip().lower()
    value = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32"}.get(value, value)
    if value not in CHOICES:
        raise ValueError("YUE2_DTYPE must be one of %s, got %r" % (", ".join(CHOICES), value))
    return value


def model_dtype():
    import torch
    return getattr(torch, model_dtype_name())
