#!/usr/bin/env python3
"""Generate golden test data for candle-dcnv2 using PyTorch torchvision."""

import torch
from torchvision.ops import deform_conv2d
from safetensors.torch import save_file
import os

os.makedirs("test-data", exist_ok=True)
torch.manual_seed(42)


def generate_case(
    name: str,
    B: int,
    C_in: int,
    C_out: int,
    H: int,
    W: int,
    kH: int,
    kW: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
    use_mask: bool = True,
    use_bias: bool = True,
):
    inp = torch.randn(B, C_in, H, W)
    out_H = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_W = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

    offset = torch.randn(B, 2 * kH * kW, out_H, out_W)
    weight = torch.randn(C_out, C_in, kH, kW)
    bias = torch.randn(C_out) if use_bias else None
    mask = (
        torch.sigmoid(torch.randn(B, kH * kW, out_H, out_W)) if use_mask else None
    )

    output = deform_conv2d(
        inp,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )

    tensors = {
        "input": inp,
        "offset": offset,
        "weight": weight,
        "output": output,
    }
    if bias is not None:
        tensors["bias"] = bias
    if mask is not None:
        tensors["mask"] = mask

    save_file(tensors, f"test-data/{name}.safetensors")
    print(f"  {name}: input={list(inp.shape)} -> output={list(output.shape)}")


print("Generating golden test data...")

generate_case(
    "basic_3x3", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True,
)

generate_case(
    "dcnv1_no_mask", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=False, use_bias=True,
)

generate_case(
    "no_bias", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=False,
)

generate_case(
    "no_padding", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(1, 1), padding=(0, 0), use_mask=True, use_bias=True,
)

generate_case(
    "stride_2", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(2, 2), padding=(1, 1), use_mask=True, use_bias=True,
)

generate_case(
    "large_input", B=1, C_in=64, C_out=128, H=32, W=32, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True,
)

generate_case(
    "non_square_kernel", B=2, C_in=3, C_out=8, H=10, W=10, kH=3, kW=5,
    stride=(1, 1), padding=(1, 2), use_mask=True, use_bias=True,
)

generate_case(
    "non_square_input", B=2, C_in=3, C_out=8, H=8, W=12, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True,
)

generate_case(
    "batch_1", B=1, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
    stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True,
)

print("Done!")
