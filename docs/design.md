# candle-dcnv2: Deformable Convolution v2 for Candle

**Date**: 2026-03-15
**Status**: Draft
**Author**: Yonghye Kwon (developer0hye)

## Goal

Add `deform_conv2d` (DCNv2) to Hugging Face Candle as a candle-nn op, then upstream via PR to the candle main repo. Forward-only (inference), implemented with high-level tensor operations for automatic multi-backend support.

## Motivation

`deform_conv2d` is absent from candle, blocking ports of major vision models:

- **BiRefNet** — SOTA background removal (Localization + Reconstruction modules)
- **Deformable DETR / RT-DETR / RF-DETR** — object detection
- **Mask2Former** — panoptic/instance/semantic segmentation
- **InternImage / DCNv4** — vision foundation models

One op unlocks all of these.

## Scope

### In scope

- `deform_conv2d()` stateless function (torchvision.ops.deform_conv2d equivalent)
- `DeformConv2d` nn module (torchvision.ops.DeformConv2d equivalent)
- Forward pass only (inference)
- CPU backend via tensor operations (runs on all backends: CPU, CUDA, Metal, WASM)
- Full interface: stride, padding, dilation, groups, offset_groups, mask
- First implementation supports: any stride/padding, groups=1, offset_groups=1, dilation=(1,1)
- Unsupported parameter combinations return explicit errors
- Golden tests against PyTorch torchvision.ops.deform_conv2d

### Out of scope

- Backward pass (gradient computation for training)
- Native backend kernels (CPU im2col, CUDA, Metal) — future optimization PR
- DCNv3/v4 variants
- crates.io publication (development repo only)

## Architecture

### Development workflow

```
developer0hye/candle-dcnv2 (independent repo)
  ├── src/lib.rs          — implementation + unit tests
  ├── scripts/            — Python golden test data generator
  └── test-data/          — .safetensors golden files

    ↓ verified, then copy to:

huggingface/candle PR
  └── candle-nn/src/ops/deform_conv.rs
```

### Public API

```rust
/// Performs Deformable Convolution v2.
///
/// When `mask` is `Some`, performs DCNv2 (modulated deformable convolution).
/// When `mask` is `None`, performs DCNv1 (deformable convolution without modulation).
///
/// `groups` and `offset_groups` are inferred from tensor shapes,
/// matching torchvision.ops.deform_conv2d behavior:
///   - groups = C_in / weight.shape[1]
///   - offset_groups = offset.shape[1] / (2 * kH * kW)
pub fn deform_conv2d(
    input: &Tensor,        // [B, C_in, H, W]
    offset: &Tensor,       // [B, 2 * offset_groups * kH * kW, out_H, out_W]
    weight: &Tensor,       // [C_out, C_in / groups, kH, kW]
    bias: Option<&Tensor>, // [C_out]
    mask: Option<&Tensor>, // [B, offset_groups * kH * kW, out_H, out_W]
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor>

/// nn module owning convolution weights.
///
/// Does not implement the standard `Module` trait because `forward` requires
/// additional inputs (`offset`, `mask`) beyond a single tensor.
pub struct DeformConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
}

impl DeformConv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        bias: bool,
        vb: VarBuilder,
    ) -> Result<Self>;

    /// `offset_groups` is inferred from `offset.shape[1] / (2 * kH * kW)`.
    pub fn forward(
        &self,
        input: &Tensor,
        offset: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}
```

### Input validation

The following checks are performed before computation. Violations produce `candle_core::Error::Msg` with a descriptive message.

- `input`, `offset`, `weight` must be 4D tensors
- `C_in` must be divisible by inferred `groups` (`C_in / weight.shape[1]`)
- `offset.shape[1]` must be divisible by `2 * kH * kW`
- If `mask` is `Some`, `mask.shape[1]` must equal `offset_groups * kH * kW`
- `offset` and `mask` spatial dims must equal expected `out_H` x `out_W`
- `weight.shape[0]` (C_out) must be divisible by `groups`
- `bias` (if `Some`) must have length `C_out`
- Inferred `groups` and `offset_groups` must be ≥ 1

### Supported data types

Initial implementation targets `f32`. `f64` should also work since all operations are dtype-agnostic in candle. `bf16` and `f16` are not explicitly tested but may work depending on backend support.

### Internal algorithm (forward only)

Same algorithm as PyTorch CPU kernel, expressed as tensor operations.

Output spatial dimensions:
```
out_H = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1
out_W = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1
```

**Step 1 — Deformable im2col (tensor ops)**

1. Generate base sampling grid from kernel positions and output spatial positions
2. Add offsets to base grid to get fractional sampling coordinates
3. Bilinear interpolation: floor/ceil indices, gather from input, weighted sum of 4 neighbors. Sampling coordinates outside `[0, H-1]` x `[0, W-1]` contribute zero (zero-padding boundary), matching PyTorch behavior.
4. Apply mask (modulation) if present
5. Produce columns tensor `[B, C_in * kH * kW, outH * outW]`

**Step 2 — Convolution via batched matmul**

1. Reshape weight to `[C_out, C_in * kH * kW]`
2. Batched matmul: `weight.matmul(columns)` → `[B, C_out, outH * outW]`
3. Reshape to `[B, C_out, outH, outW]`
4. Add bias (broadcast over B, H, W)

### First implementation constraints

| Parameter | Supported | Unsupported (returns error) |
|---|---|---|
| stride | any | — |
| padding | any | — |
| dilation | (1, 1) only | dilation > 1 |
| groups | 1 only | groups > 1 |
| offset_groups | 1 only | offset_groups > 1 |
| mask | Some or None | — |
| bias | Some or None | — |

Unsupported combinations produce `candle_core::Error` with a clear message indicating the limitation and that it will be supported in a future update.

## Testing

### Golden test approach

**Generator script** (`scripts/generate_golden.py`):

```python
import torch
from torchvision.ops import deform_conv2d
from safetensors.torch import save_file

torch.manual_seed(42)

def generate_case(name, B, C_in, C_out, H, W, kH, kW, stride, padding, dilation=(1,1), use_mask=True, use_bias=True):
    input = torch.randn(B, C_in, H, W)
    out_H = (H + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
    out_W = (W + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1
    offset = torch.randn(B, 2 * kH * kW, out_H, out_W)
    weight = torch.randn(C_out, C_in, kH, kW)
    bias = torch.randn(C_out) if use_bias else None
    mask = torch.sigmoid(torch.randn(B, kH * kW, out_H, out_W)) if use_mask else None

    output = deform_conv2d(input, offset, weight, bias=bias, stride=stride,
                           padding=padding, dilation=dilation, mask=mask)

    tensors = {"input": input, "offset": offset, "weight": weight, "output": output}
    if bias is not None:
        tensors["bias"] = bias
    if mask is not None:
        tensors["mask"] = mask

    save_file(tensors, f"test-data/{name}.safetensors")
```

**Test cases**:

| Case | Config | Purpose |
|---|---|---|
| `basic_3x3` | B=2, 3→8, 8x8, k=3, s=1, p=1, mask+bias | Full DCNv2 |
| `dcnv1_no_mask` | Same but mask=None | DCNv1 fallback |
| `no_bias` | bias=None | Bias-free path |
| `no_padding` | padding=(0,0), k=3 | Zero-padding path |
| `stride_2` | stride=(2,2) | Strided convolution |
| `large_input` | B=1, 64→128, 32x32, k=3 | Realistic size |
| `non_square_kernel` | k=(3,5) | Non-square kernel |
| `non_square_input` | 8x12 spatial dims | H != W bug detection |
| `batch_1` | B=1 | Single image |

**Rust test**:

```rust
fn assert_tensor_close(a: &Tensor, b: &Tensor, atol: f64) {
    let diff = a.sub(b).unwrap().abs().unwrap()
        .flatten_all().unwrap()
        .max(0).unwrap()
        .to_scalar::<f64>().unwrap();
    assert!(diff < atol, "max abs diff: {diff}, tolerance: {atol}");
}

#[test]
fn test_basic_3x3() -> Result<()> {
    let data = safetensors::load("test-data/basic_3x3.safetensors", &Device::Cpu)?;
    let result = deform_conv2d(
        &data["input"], &data["offset"], &data["weight"],
        Some(&data["bias"]), Some(&data["mask"]),
        (1, 1), (1, 1), (1, 1),
    )?;
    // Tolerance: 1e-4 for f32. Bilinear interpolation + matmul accumulates
    // floating-point error across C_in * kH * kW elements. Verify empirically
    // with golden data; tighten to 1e-5 if all cases pass.
    assert_tensor_close(&result, &data["output"], 1e-4);
    Ok(())
}
```

## PR Strategy

1. **Open candle issue**: "Feature request: `deform_conv2d` for vision models"
   - List models unblocked: BiRefNet, Deformable DETR, Mask2Former, InternImage
   - Link to torchvision reference implementation
   - Offer to implement

2. **Develop in `developer0hye/candle-dcnv2`**
   - Implement + golden tests pass
   - Benchmark basic performance

3. **Submit PR to `huggingface/candle`**
   - Target: `candle-nn/src/ops/deform_conv.rs`
   - Include: implementation, tests, golden test data, documentation
   - PR description: motivation, API, limitations, future optimization path

4. **Future follow-up PRs** (out of scope for this spec):
   - Extend: dilation > 1, groups > 1, offset_groups > 1
   - Native CPU kernel (im2col with raw pointers) for performance
   - CUDA / Metal kernels

## Dependencies

- `candle-core` 0.9.x — tensor operations, Device, DType
- `candle-nn` 0.9.x — VarBuilder, Module trait
- `safetensors` — golden test data loading (dev-dependency)
- `torchvision` ≥ 0.17 — golden test generation (Python, not a Rust dependency)

## Success Criteria

- All golden tests pass with atol=1e-4 against PyTorch output (tighten to 1e-5 if empirically achievable)
- PR accepted and merged into candle main repo
- BiRefNet-lite can use this op for inference (verified in candle-birefnet follow-up project)
