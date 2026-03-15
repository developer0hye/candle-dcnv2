#!/usr/bin/env python3
"""Generate invariant and edge-case test data for candle-dcnv2.

These tests verify mathematical properties, not just numerical agreement:
1. Zero offset + unit mask = standard conv2d (fundamental invariant)
2. Zero mask = bias-only output (modulation kills signal)
3. Large offsets = zero-padded boundary behavior
4. Integer offsets = no interpolation needed (exact sampling)
5. Single-element spatial (1x1 input with 1x1 kernel)
"""

import torch
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from safetensors.torch import save_file
import os

os.makedirs("test-data", exist_ok=True)
torch.manual_seed(123)  # Different seed from main golden tests


def save(name: str, tensors: dict[str, torch.Tensor]):
    save_file(tensors, f"test-data/{name}.safetensors")
    shapes = {k: list(v.shape) for k, v in tensors.items() if v.dim() > 0}
    print(f"  {name}: {shapes}")


print("=== Invariant Tests ===\n")

# --------------------------------------------------------------------------
# 1. ZERO OFFSET INVARIANT: deform_conv2d(offset=0, mask=1) == F.conv2d
#    This is the fundamental correctness proof. If this fails, the algorithm
#    is fundamentally broken.
# --------------------------------------------------------------------------
print("1. Zero-offset invariant (deform_conv2d == conv2d)")
for cfg_name, B, C_in, C_out, H, W, kH, kW, s, p in [
    ("zero_offset_3x3", 2, 3, 8, 8, 8, 3, 3, (1, 1), (1, 1)),
    ("zero_offset_5x5", 1, 16, 32, 12, 12, 5, 5, (1, 1), (2, 2)),
    ("zero_offset_stride2", 2, 3, 8, 16, 16, 3, 3, (2, 2), (1, 1)),
    ("zero_offset_1x1", 1, 64, 128, 8, 8, 1, 1, (1, 1), (0, 0)),
    ("zero_offset_no_pad", 2, 3, 8, 8, 8, 3, 3, (1, 1), (0, 0)),
]:
    inp = torch.randn(B, C_in, H, W)
    weight = torch.randn(C_out, C_in, kH, kW)
    bias = torch.randn(C_out)

    out_H = (H + 2 * p[0] - kH) // s[0] + 1
    out_W = (W + 2 * p[1] - kW) // s[1] + 1

    offset = torch.zeros(B, 2 * kH * kW, out_H, out_W)
    mask = torch.ones(B, kH * kW, out_H, out_W)

    dcn_out = deform_conv2d(inp, offset, weight, bias=bias, stride=s,
                             padding=p, mask=mask)
    conv_out = F.conv2d(inp, weight, bias=bias, stride=s, padding=p)

    max_diff = (dcn_out - conv_out).abs().max().item()
    print(f"    {cfg_name}: max_diff={max_diff:.2e} (should be ~0)")

    save(cfg_name, {
        "input": inp, "offset": offset, "weight": weight, "bias": bias,
        "mask": mask, "output": dcn_out, "conv2d_output": conv_out,
    })

# --------------------------------------------------------------------------
# 2. ZERO MASK: when mask=0, all sampled values are zeroed out.
#    Output should equal bias broadcast (or zero if no bias).
# --------------------------------------------------------------------------
print("\n2. Zero-mask invariant")
inp = torch.randn(2, 3, 8, 8)
weight = torch.randn(8, 3, 3, 3)
bias = torch.randn(8)
offset = torch.randn(2, 18, 8, 8)  # Non-zero offsets don't matter
mask = torch.zeros(2, 9, 8, 8)  # All zeros

dcn_out = deform_conv2d(inp, offset, weight, bias=bias, stride=(1, 1),
                         padding=(1, 1), mask=mask)
expected = bias.view(1, 8, 1, 1).expand_as(dcn_out).contiguous()
max_diff = (dcn_out - expected).abs().max().item()
print(f"    zero_mask: max_diff from bias-only={max_diff:.2e}")

save("zero_mask", {
    "input": inp, "offset": offset, "weight": weight, "bias": bias,
    "mask": mask, "output": dcn_out, "expected_bias_only": expected,
})

# Also test zero mask without bias
dcn_out_no_bias = deform_conv2d(inp, offset, weight, bias=None, stride=(1, 1),
                                 padding=(1, 1), mask=mask)
max_diff_no_bias = dcn_out_no_bias.abs().max().item()
print(f"    zero_mask_no_bias: max_abs={max_diff_no_bias:.2e} (should be 0)")

save("zero_mask_no_bias", {
    "input": inp, "offset": offset, "weight": weight,
    "mask": mask, "output": dcn_out_no_bias,
})

# --------------------------------------------------------------------------
# 3. LARGE OFFSETS: offsets that push sampling coordinates far outside the
#    input boundary. Zero-padding means all out-of-bounds samples = 0.
# --------------------------------------------------------------------------
print("\n3. Large offsets (boundary behavior)")
inp = torch.randn(1, 3, 8, 8)
weight = torch.randn(8, 3, 3, 3)
bias = torch.randn(8)
# Offsets of +1000 push all sampling coords far outside [0, H-1]
offset = torch.full((1, 18, 8, 8), 1000.0)
mask = torch.ones(1, 9, 8, 8)

dcn_out = deform_conv2d(inp, offset, weight, bias=bias, stride=(1, 1),
                         padding=(1, 1), mask=mask)
expected = bias.view(1, 8, 1, 1).expand_as(dcn_out)
max_diff = (dcn_out - expected).abs().max().item()
print(f"    large_offset: max_diff from bias-only={max_diff:.2e}")

save("large_offset", {
    "input": inp, "offset": offset, "weight": weight, "bias": bias,
    "mask": mask, "output": dcn_out,
})

# Also negative large offsets
offset_neg = torch.full((1, 18, 8, 8), -1000.0)
dcn_out_neg = deform_conv2d(inp, offset_neg, weight, bias=bias, stride=(1, 1),
                             padding=(1, 1), mask=mask)
max_diff_neg = (dcn_out_neg - expected).abs().max().item()
print(f"    large_negative_offset: max_diff from bias-only={max_diff_neg:.2e}")

save("large_negative_offset", {
    "input": inp, "offset": offset_neg, "weight": weight, "bias": bias,
    "mask": mask, "output": dcn_out_neg,
})

# --------------------------------------------------------------------------
# 4. INTEGER OFFSETS: when offsets are exact integers, bilinear interpolation
#    should be equivalent to direct pixel lookup (no blending).
# --------------------------------------------------------------------------
print("\n4. Integer offsets (exact sampling, no interpolation)")
inp = torch.randn(1, 3, 8, 8)
weight = torch.randn(8, 3, 3, 3)
bias = torch.randn(8)
# Shift all sampling positions by exactly (+1, +1)
offset = torch.zeros(1, 18, 8, 8)
for k in range(9):
    offset[0, 2 * k, :, :] = 1.0      # shift y by +1
    offset[0, 2 * k + 1, :, :] = 1.0  # shift x by +1
mask = torch.ones(1, 9, 8, 8)

dcn_out = deform_conv2d(inp, offset, weight, bias=bias, stride=(1, 1),
                         padding=(1, 1), mask=mask)

save("integer_offset", {
    "input": inp, "offset": offset, "weight": weight, "bias": bias,
    "mask": mask, "output": dcn_out,
})
print(f"    integer_offset: output shape={list(dcn_out.shape)}")

# --------------------------------------------------------------------------
# 5. SINGLE SPATIAL ELEMENT: minimal possible spatial size
# --------------------------------------------------------------------------
print("\n5. Minimal spatial size (1x1)")
inp = torch.randn(1, 3, 1, 1)
weight = torch.randn(8, 3, 1, 1)
bias = torch.randn(8)
offset = torch.randn(1, 2, 1, 1)
mask = torch.ones(1, 1, 1, 1)

dcn_out = deform_conv2d(inp, offset, weight, bias=bias, stride=(1, 1),
                         padding=(0, 0), mask=mask)

save("minimal_1x1", {
    "input": inp, "offset": offset, "weight": weight, "bias": bias,
    "mask": mask, "output": dcn_out,
})
print(f"    minimal_1x1: output shape={list(dcn_out.shape)}")

# --------------------------------------------------------------------------
# TOLERANCE REPORT: measure actual max diff for all existing golden tests
# --------------------------------------------------------------------------
print("\n=== Tolerance Report for All Golden Tests ===\n")
golden_cases = [
    ("basic_3x3", (1, 1), (1, 1)),
    ("dcnv1_no_mask", (1, 1), (1, 1)),
    ("no_bias", (1, 1), (1, 1)),
    ("no_padding", (1, 1), (0, 0)),
    ("stride_2", (2, 2), (1, 1)),
    ("large_input", (1, 1), (1, 1)),
    ("non_square_kernel", (1, 1), (1, 2)),
    ("non_square_input", (1, 1), (1, 1)),
    ("batch_1", (1, 1), (1, 1)),
]

# Regenerate with same seed to measure precision
torch.manual_seed(42)
for name, s, p in golden_cases:
    from safetensors.torch import load_file
    data = load_file(f"test-data/{name}.safetensors")
    # The stored output IS the PyTorch output, so diff should be 0
    # But let's re-compute to verify reproducibility
    bias_t = data.get("bias", None)
    mask_t = data.get("mask", None)
    recomputed = deform_conv2d(
        data["input"], data["offset"], data["weight"],
        bias=bias_t, stride=s, padding=p, mask=mask_t,
    )
    max_diff = (recomputed - data["output"]).abs().max().item()
    mean_diff = (recomputed - data["output"]).abs().mean().item()
    print(f"  {name:25s}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}")

print("\nDone!")
