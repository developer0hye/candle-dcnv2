#!/usr/bin/env python3
"""Extract deform_conv2d inputs/outputs from a real BiRefNet model.

Hooks into torchvision.ops.deform_conv2d to capture all arguments
and the output at each call site, then saves them as safetensors
for golden-test comparison in Rust.
"""

import os
import torch
import torchvision.ops
from safetensors.torch import save_file

os.makedirs("test-data", exist_ok=True)

# Monkey-patch torchvision.ops.deform_conv2d to capture calls
_original_deform_conv2d = torchvision.ops.deform_conv2d
_captured_calls: list[dict[str, torch.Tensor]] = []


def _hooked_deform_conv2d(input, offset, weight, bias=None, stride=(1, 1),
                           padding=(0, 0), dilation=(1, 1), mask=None):
    output = _original_deform_conv2d(
        input, offset, weight, bias=bias, stride=stride,
        padding=padding, dilation=dilation, mask=mask,
    )

    call_data = {
        "input": input.detach().cpu(),
        "offset": offset.detach().cpu(),
        "weight": weight.detach().cpu(),
        "output": output.detach().cpu(),
        "stride_h": torch.tensor(stride[0] if isinstance(stride, tuple) else stride),
        "stride_w": torch.tensor(stride[1] if isinstance(stride, tuple) else stride),
        "padding_h": torch.tensor(padding[0] if isinstance(padding, tuple) else padding),
        "padding_w": torch.tensor(padding[1] if isinstance(padding, tuple) else padding),
        "dilation_h": torch.tensor(dilation[0] if isinstance(dilation, tuple) else dilation),
        "dilation_w": torch.tensor(dilation[1] if isinstance(dilation, tuple) else dilation),
    }
    if bias is not None:
        call_data["bias"] = bias.detach().cpu()
    if mask is not None:
        call_data["mask"] = mask.detach().cpu()

    _captured_calls.append(call_data)
    print(f"  Captured call #{len(_captured_calls)}: "
          f"input={list(input.shape)}, weight={list(weight.shape)}, "
          f"offset={list(offset.shape)}, output={list(output.shape)}, "
          f"stride={stride}, padding={padding}, dilation={dilation}, "
          f"has_mask={mask is not None}, has_bias={bias is not None}")

    return output


# Apply the hook
torchvision.ops.deform_conv2d = _hooked_deform_conv2d
# Also patch the module-level reference used by DeformConv2d
torchvision.ops.deform_conv.deform_conv2d = _hooked_deform_conv2d

print("Loading BiRefNet model...")
from transformers import AutoModelForImageSegmentation

model = AutoModelForImageSegmentation.from_pretrained(
    "ZhengPeng7/BiRefNet", trust_remote_code=True
)
model.eval()

print("Running inference with random input...")
with torch.no_grad():
    dummy_input = torch.randn(1, 3, 1024, 1024)
    _ = model(dummy_input)

print(f"\nCaptured {len(_captured_calls)} deform_conv2d calls total.")

# Save first few calls (different layer configurations)
saved_configs = set()
save_count = 0
for i, call_data in enumerate(_captured_calls):
    inp_shape = tuple(call_data["input"].shape)
    w_shape = tuple(call_data["weight"].shape)
    config_key = (inp_shape, w_shape)

    # Save one per unique configuration, up to 5
    if config_key in saved_configs:
        continue
    saved_configs.add(config_key)
    save_count += 1

    filename = f"test-data/birefnet_dcn_{save_count}.safetensors"
    save_file(call_data, filename)
    print(f"Saved {filename}: input={list(inp_shape)}, weight={list(w_shape)}")

    if save_count >= 5:
        break

# Also save a specific call with its full data for detailed testing
if _captured_calls:
    save_file(_captured_calls[0], "test-data/birefnet_dcn_first.safetensors")
    print(f"Saved test-data/birefnet_dcn_first.safetensors (first call)")

print("Done!")
