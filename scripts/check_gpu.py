#!/usr/bin/env python3
"""Report whether this torch build can use your GPU and which dtype to use."""
import sys

try:
    import torch
except ImportError:
    sys.exit("torch is not installed; run scripts/install.sh")

print("torch", torch.__version__, "cuda", torch.version.cuda)
if not torch.cuda.is_available():
    sys.exit("CUDA is NOT available: check the NVIDIA driver and that the CUDA build of torch is installed")

props = torch.cuda.get_device_properties(0)
cap = (props.major, props.minor)
print("GPU: %s, sm_%d%d, %.1f GiB" % (props.name, cap[0], cap[1], props.total_memory / 2**30))
arch = "sm_%d%d" % cap
if arch not in torch.cuda.get_arch_list():
    print("WARNING: this torch build has no kernels for %s (has %s)" % (arch, " ".join(torch.cuda.get_arch_list())))
print("bf16 supported:", torch.cuda.is_bf16_supported(including_emulation=False))
print("recommended YUE2_DTYPE:", "bfloat16 or float16" if cap >= (8, 0) else "float16 (Turing/Volta have no bf16)")
