# candle-dcnv2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Deformable Convolution v2 (DCNv2) for Hugging Face Candle using high-level tensor operations, targeting candle-nn upstream PR.

**Architecture:** `deform_conv2d()` stateless function + `DeformConv2d` nn module, implemented via deformable im2col (tensor ops) + batched matmul. Forward-only (inference). Groups/offset_groups inferred from tensor shapes.

**Tech Stack:** Rust, candle-core 0.9, candle-nn 0.9, safetensors (test data), Python + PyTorch (golden test generation)

**Spec:** `docs/design.md`

---

## File Structure

```
candle-dcnv2/
├── Cargo.toml                  — dependencies (candle-core, candle-nn, safetensors)
├── src/
│   ├── lib.rs                  — pub mod, re-exports
│   ├── deform_conv2d.rs        — deform_conv2d() stateless function + internal algorithm
│   ├── module.rs               — DeformConv2d nn module struct
│   └── validation.rs           — input shape validation helpers
├── tests/
│   └── golden.rs               — golden tests against PyTorch output
├── scripts/
│   └── generate_golden.py      — Python script to generate test data
└── test-data/
    ├── basic_3x3.safetensors
    ├── dcnv1_no_mask.safetensors
    ├── no_bias.safetensors
    ├── no_padding.safetensors
    ├── stride_2.safetensors
    ├── large_input.safetensors
    ├── non_square_kernel.safetensors
    ├── non_square_input.safetensors
    └── batch_1.safetensors
```

---

## Chunk 1: Project Setup + Golden Test Data

### Task 1: Update Cargo.toml with correct dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Update Cargo.toml**

```toml
[package]
name = "candle-dcnv2"
version = "0.1.0"
edition = "2024"
description = "Deformable Convolution v2 (DCNv2) for Hugging Face Candle"
license = "Apache-2.0"
repository = "https://github.com/developer0hye/candle-dcnv2"
keywords = ["deep-learning", "deformable-convolution", "candle", "dcnv2"]
categories = ["science"]

[dependencies]
candle-core = "0.9"
candle-nn = "0.9"

[dev-dependencies]
anyhow = "1"
candle-core = { version = "0.9", features = ["default"] }
safetensors = "0.5"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: Compiles with no errors (downloads dependencies)

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -s -m "chore: update dependencies for golden tests"
```

### Task 2: Create golden test data generator

**Files:**
- Create: `scripts/generate_golden.py`

- [ ] **Step 1: Write the generator script**

```python
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

generate_case("basic_3x3", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True)

generate_case("dcnv1_no_mask", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=False, use_bias=True)

generate_case("no_bias", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=False)

generate_case("no_padding", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(1, 1), padding=(0, 0), use_mask=True, use_bias=True)

generate_case("stride_2", B=2, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(2, 2), padding=(1, 1), use_mask=True, use_bias=True)

generate_case("large_input", B=1, C_in=64, C_out=128, H=32, W=32, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True)

generate_case("non_square_kernel", B=2, C_in=3, C_out=8, H=10, W=10, kH=3, kW=5,
              stride=(1, 1), padding=(1, 2), use_mask=True, use_bias=True)

generate_case("non_square_input", B=2, C_in=3, C_out=8, H=8, W=12, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True)

generate_case("batch_1", B=1, C_in=3, C_out=8, H=8, W=8, kH=3, kW=3,
              stride=(1, 1), padding=(1, 1), use_mask=True, use_bias=True)

print("Done!")
```

- [ ] **Step 2: Run the script to generate test data**

Run: `cd candle-dcnv2 && python scripts/generate_golden.py`
Expected: 9 `.safetensors` files created in `test-data/`

- [ ] **Step 3: Add test-data to .gitignore check**

Verify `test-data/*.safetensors` files are NOT in `.gitignore` — they must be committed for CI.

- [ ] **Step 4: Commit**

```bash
git add scripts/generate_golden.py test-data/
git commit -s -m "test: add golden test data generator and test fixtures"
```

---

## Chunk 2: Input Validation

### Task 3: Create module structure and validation

**Files:**
- Modify: `src/lib.rs`
- Create: `src/validation.rs`

- [ ] **Step 1: Write failing test for validation**

Create `src/validation.rs` with test stubs and validation function signature:

```rust
// src/validation.rs
use candle_core::{Result, Tensor};

/// Validated parameters extracted from input tensor shapes.
pub struct DeformConv2dParams {
    pub batch_size: usize,
    pub in_channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub groups: usize,
    pub offset_groups: usize,
}

pub fn validate_and_extract(
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<DeformConv2dParams> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_valid_basic_shapes() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        // out_h = (8 + 2*1 - 1*(3-1) - 1)/1 + 1 = 8
        let offset = Tensor::zeros((2, 2 * 9, 8, 8), DType::F32, dev).unwrap();
        let mask = Tensor::zeros((2, 9, 8, 8), DType::F32, dev).unwrap();

        let params = validate_and_extract(
            &input, &offset, &weight, None, Some(&mask),
            (1, 1), (1, 1), (1, 1),
        ).unwrap();

        assert_eq!(params.batch_size, 2);
        assert_eq!(params.in_channels, 3);
        assert_eq!(params.out_channels, 8);
        assert_eq!(params.kernel_h, 3);
        assert_eq!(params.kernel_w, 3);
        assert_eq!(params.out_h, 8);
        assert_eq!(params.out_w, 8);
        assert_eq!(params.groups, 1);
        assert_eq!(params.offset_groups, 1);
    }

    #[test]
    fn test_input_not_4d() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input, &offset, &weight, None, None,
            (1, 1), (1, 1), (1, 1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_offset_shape_mismatch() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        // Wrong spatial dims: 4x4 instead of 8x8
        let offset = Tensor::zeros((2, 18, 4, 4), DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input, &offset, &weight, None, None,
            (1, 1), (1, 1), (1, 1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_dilation() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 4, 4), DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input, &offset, &weight, None, None,
            (1, 1), (1, 1), (2, 2),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_unsupported_groups() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 6, 8, 8), DType::F32, dev).unwrap();
        // weight shape implies groups=2: C_in(6) / weight.shape[1](3) = 2
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input, &offset, &weight, None, None,
            (1, 1), (1, 1), (1, 1),
        );
        assert!(result.is_err());
    }
}
```

- [ ] **Step 2: Update lib.rs**

```rust
// src/lib.rs
mod validation;
mod deform_conv2d;
mod module;

pub use deform_conv2d::deform_conv2d;
pub use module::DeformConv2d;
```

Create empty `src/deform_conv2d.rs` and `src/module.rs`:

```rust
// src/deform_conv2d.rs
// Placeholder — implemented in Task 5
```

```rust
// src/module.rs
// Placeholder — implemented in Task 7
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --lib validation`
Expected: FAIL — `todo!()` panics

- [ ] **Step 4: Implement validate_and_extract**

```rust
// src/validation.rs — replace the todo!() body
use candle_core::{bail, Result, Tensor};

pub struct DeformConv2dParams {
    pub batch_size: usize,
    pub in_channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub out_channels: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub groups: usize,
    pub offset_groups: usize,
}

pub fn validate_and_extract(
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<DeformConv2dParams> {
    let input_dims = input.dims();
    let weight_dims = weight.dims();
    let offset_dims = offset.dims();

    if input_dims.len() != 4 {
        bail!("deform_conv2d: input must be 4D [B, C_in, H, W], got {}D", input_dims.len());
    }
    if weight_dims.len() != 4 {
        bail!("deform_conv2d: weight must be 4D [C_out, C_in/groups, kH, kW], got {}D", weight_dims.len());
    }
    if offset_dims.len() != 4 {
        bail!("deform_conv2d: offset must be 4D, got {}D", offset_dims.len());
    }

    let batch_size = input_dims[0];
    let in_channels = input_dims[1];
    let in_h = input_dims[2];
    let in_w = input_dims[3];

    let out_channels = weight_dims[0];
    let kernel_h = weight_dims[2];
    let kernel_w = weight_dims[3];

    // Infer groups from tensor shapes (matches PyTorch)
    let channels_per_group = weight_dims[1];
    if in_channels == 0 || channels_per_group == 0 || in_channels % channels_per_group != 0 {
        bail!(
            "deform_conv2d: C_in ({in_channels}) must be divisible by weight.shape[1] ({channels_per_group})"
        );
    }
    let groups = in_channels / channels_per_group;

    if out_channels % groups != 0 {
        bail!(
            "deform_conv2d: C_out ({out_channels}) must be divisible by groups ({groups})"
        );
    }

    // Infer offset_groups from offset shape
    let offset_channels = offset_dims[1];
    let kernel_size = kernel_h * kernel_w;
    if kernel_size == 0 || offset_channels % (2 * kernel_size) != 0 {
        bail!(
            "deform_conv2d: offset.shape[1] ({offset_channels}) must be divisible by 2 * kH * kW ({})",
            2 * kernel_size
        );
    }
    let offset_groups = offset_channels / (2 * kernel_size);

    // Compute expected output dims
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (dil_h, dil_w) = dilation;

    let out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    if out_h == 0 || out_w == 0 {
        bail!("deform_conv2d: computed output size is zero (out_h={out_h}, out_w={out_w})");
    }

    // Validate offset spatial dims
    if offset_dims[0] != batch_size || offset_dims[2] != out_h || offset_dims[3] != out_w {
        bail!(
            "deform_conv2d: offset spatial dims ({}, {}) don't match expected ({out_h}, {out_w})",
            offset_dims[2], offset_dims[3]
        );
    }

    // Validate mask if present
    if let Some(m) = mask {
        let mask_dims = m.dims();
        if mask_dims.len() != 4 {
            bail!("deform_conv2d: mask must be 4D, got {}D", mask_dims.len());
        }
        let expected_mask_channels = offset_groups * kernel_size;
        if mask_dims[1] != expected_mask_channels {
            bail!(
                "deform_conv2d: mask.shape[1] ({}) must equal offset_groups * kH * kW ({expected_mask_channels})",
                mask_dims[1]
            );
        }
        if mask_dims[0] != batch_size || mask_dims[2] != out_h || mask_dims[3] != out_w {
            bail!(
                "deform_conv2d: mask spatial dims ({}, {}) don't match expected ({out_h}, {out_w})",
                mask_dims[2], mask_dims[3]
            );
        }
    }

    // Validate bias if present
    if let Some(b) = bias {
        let bias_dims = b.dims();
        if bias_dims.len() != 1 || bias_dims[0] != out_channels {
            bail!(
                "deform_conv2d: bias must be 1D with length C_out ({out_channels}), got shape {:?}",
                bias_dims
            );
        }
    }

    // First implementation constraints
    if dilation != (1, 1) {
        bail!(
            "deform_conv2d: dilation {:?} not yet supported, only (1, 1) is currently implemented",
            dilation
        );
    }
    if groups != 1 {
        bail!(
            "deform_conv2d: groups={groups} not yet supported, only groups=1 is currently implemented"
        );
    }
    if offset_groups != 1 {
        bail!(
            "deform_conv2d: offset_groups={offset_groups} not yet supported, only offset_groups=1 is currently implemented"
        );
    }

    Ok(DeformConv2dParams {
        batch_size,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        groups,
        offset_groups,
    })
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --lib validation`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/lib.rs src/validation.rs src/deform_conv2d.rs src/module.rs
git commit -s -m "feat: add input validation for deform_conv2d"
```

---

## Chunk 3: Core Algorithm — deform_conv2d

### Task 4: Write golden test harness (red)

**Files:**
- Create: `tests/golden.rs`

- [ ] **Step 1: Write the golden test harness**

```rust
// tests/golden.rs
use candle_core::{DType, Device, Result, Tensor};
use candle_dcnv2::deform_conv2d;
use std::collections::HashMap;

fn load_test_case(name: &str) -> HashMap<String, Tensor> {
    let path = format!("test-data/{name}.safetensors");
    let raw = safetensors::tensor::SafeTensors::deserialize(
        &std::fs::read(&path).unwrap_or_else(|_| panic!("missing test data: {path}"))
    ).unwrap();

    let device = &Device::Cpu;
    raw.tensors()
        .into_iter()
        .map(|(name, view)| {
            let tensor = Tensor::from_raw_buffer(
                view.data(),
                view.dtype().try_into().unwrap(),
                view.shape(),
                device,
            ).unwrap();
            (name.to_string(), tensor)
        })
        .collect()
}

fn assert_tensor_close(a: &Tensor, b: &Tensor, atol: f64) {
    let diff = a
        .to_dtype(DType::F64).unwrap()
        .sub(&b.to_dtype(DType::F64).unwrap()).unwrap()
        .abs().unwrap()
        .flatten_all().unwrap()
        .max(0).unwrap()
        .to_scalar::<f64>().unwrap();
    assert!(
        diff < atol,
        "max abs diff: {diff}, tolerance: {atol}"
    );
}

fn run_golden_test(name: &str, stride: (usize, usize), padding: (usize, usize)) -> Result<()> {
    let data = load_test_case(name);
    let bias = data.get("bias");
    let mask = data.get("mask");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        bias,
        mask,
        stride,
        padding,
        (1, 1),
    )?;

    assert_tensor_close(&result, &data["output"], 1e-4);
    Ok(())
}

#[test]
fn golden_basic_3x3() -> Result<()> {
    run_golden_test("basic_3x3", (1, 1), (1, 1))
}

#[test]
fn golden_dcnv1_no_mask() -> Result<()> {
    run_golden_test("dcnv1_no_mask", (1, 1), (1, 1))
}

#[test]
fn golden_no_bias() -> Result<()> {
    run_golden_test("no_bias", (1, 1), (1, 1))
}

#[test]
fn golden_no_padding() -> Result<()> {
    run_golden_test("no_padding", (1, 1), (0, 0))
}

#[test]
fn golden_stride_2() -> Result<()> {
    run_golden_test("stride_2", (2, 2), (1, 1))
}

#[test]
fn golden_large_input() -> Result<()> {
    run_golden_test("large_input", (1, 1), (1, 1))
}

#[test]
fn golden_non_square_kernel() -> Result<()> {
    run_golden_test("non_square_kernel", (1, 1), (1, 2))
}

#[test]
fn golden_non_square_input() -> Result<()> {
    run_golden_test("non_square_input", (1, 1), (1, 1))
}

#[test]
fn golden_batch_1() -> Result<()> {
    run_golden_test("batch_1", (1, 1), (1, 1))
}
```

- [ ] **Step 2: Run to verify all tests fail**

Run: `cargo test --test golden`
Expected: FAIL — `deform_conv2d` function not found or not implemented

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/golden.rs
git commit -s -m "test: add golden tests for deform_conv2d (red)"
```

### Task 5: Implement deform_conv2d (green)

**Files:**
- Modify: `src/deform_conv2d.rs`

This is the core algorithm. It follows the PyTorch CPU kernel structure but uses candle tensor operations.

- [ ] **Step 1: Implement deform_conv2d**

```rust
// src/deform_conv2d.rs
use candle_core::{DType, Device, Result, Tensor};
use crate::validation::{validate_and_extract, DeformConv2dParams};

/// Performs Deformable Convolution v2 (forward only).
///
/// When `mask` is `Some`, performs DCNv2 (modulated deformable convolution).
/// When `mask` is `None`, performs DCNv1.
///
/// `groups` and `offset_groups` are inferred from tensor shapes:
///   - groups = C_in / weight.shape[1]
///   - offset_groups = offset.shape[1] / (2 * kH * kW)
pub fn deform_conv2d(
    input: &Tensor,
    offset: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Result<Tensor> {
    let params = validate_and_extract(input, offset, weight, bias, mask, stride, padding, dilation)?;
    let DeformConv2dParams {
        batch_size, in_channels, in_h, in_w,
        out_channels, kernel_h, kernel_w, out_h, out_w, ..
    } = params;

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let device = input.device();
    let dtype = input.dtype();
    let kernel_size = kernel_h * kernel_w;

    // Step 1: Deformable im2col via tensor operations
    //
    // For each output position (out_y, out_x) and each kernel position (ky, kx):
    //   sample_y = out_y * stride_h - pad_h + ky + offset_h
    //   sample_x = out_x * stride_w - pad_w + kx + offset_w
    //
    // Then bilinear interpolate from input at (sample_y, sample_x).

    // Build base grid: [kH*kW, out_h, out_w] for y and x
    let mut base_y_data = vec![0f64; kernel_size * out_h * out_w];
    let mut base_x_data = vec![0f64; kernel_size * out_h * out_w];

    for ky in 0..kernel_h {
        for kx in 0..kernel_w {
            let k_idx = ky * kernel_w + kx;
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let flat = k_idx * (out_h * out_w) + oy * out_w + ox;
                    base_y_data[flat] = (oy * stride_h + ky) as f64 - pad_h as f64;
                    base_x_data[flat] = (ox * stride_w + kx) as f64 - pad_w as f64;
                }
            }
        }
    }

    // [1, kH*kW, out_h, out_w] — broadcast over batch
    let base_y = Tensor::from_vec(base_y_data, (1, kernel_size, out_h, out_w), device)?
        .to_dtype(dtype)?;
    let base_x = Tensor::from_vec(base_x_data, (1, kernel_size, out_h, out_w), device)?
        .to_dtype(dtype)?;

    // offset: [B, 2*kH*kW, out_h, out_w]
    // Split into offset_y [B, kH*kW, out_h, out_w] and offset_x [B, kH*kW, out_h, out_w]
    let offset_y = offset.narrow(1, 0, kernel_size)?;           // first kH*kW channels
    let offset_x = offset.narrow(1, kernel_size, kernel_size)?; // next kH*kW channels

    // Sampling coordinates: [B, kH*kW, out_h, out_w]
    let sample_y = base_y.broadcast_add(&offset_y)?;
    let sample_x = base_x.broadcast_add(&offset_x)?;

    // Step 1b: Bilinear interpolation
    // Floor/ceil for 4-corner sampling
    let y0 = sample_y.floor()?;
    let x0 = sample_x.floor()?;
    let y1 = (&y0 + 1.0)?;
    let x1 = (&x0 + 1.0)?;

    // Interpolation weights
    let wy1 = sample_y.sub(&y0)?;  // fractional part y
    let wx1 = sample_x.sub(&x0)?;  // fractional part x
    let wy0 = (1.0 - &wy1)?;
    let wx0 = (1.0 - &wx1)?;

    let w_tl = wy0.mul(&wx0)?; // top-left weight
    let w_tr = wy0.mul(&wx1)?; // top-right weight
    let w_bl = wy1.mul(&wx0)?; // bottom-left weight
    let w_br = wy1.mul(&wx1)?; // bottom-right weight

    // Clamp indices to valid range, then gather.
    // Out-of-bounds positions get weight=0 via masking.
    let y0_long = y0.to_dtype(DType::I64)?;
    let x0_long = x0.to_dtype(DType::I64)?;
    let y1_long = y1.to_dtype(DType::I64)?;
    let x1_long = x1.to_dtype(DType::I64)?;

    let h_max = (in_h as i64) - 1;
    let w_max = (in_w as i64) - 1;

    // Boundary masks: 1.0 where valid, 0.0 where out of bounds
    let valid_y0 = y0_long.ge(&Tensor::zeros((), DType::I64, device)?.broadcast_as(y0_long.shape())?)?
        .to_dtype(dtype)?;
    let valid_y0 = valid_y0.mul(
        &y0_long.le(&Tensor::new(h_max, device)?.broadcast_as(y0_long.shape())?)?.to_dtype(dtype)?
    )?;
    let valid_x0 = x0_long.ge(&Tensor::zeros((), DType::I64, device)?.broadcast_as(x0_long.shape())?)?
        .to_dtype(dtype)?;
    let valid_x0 = valid_x0.mul(
        &x0_long.le(&Tensor::new(w_max, device)?.broadcast_as(x0_long.shape())?)?.to_dtype(dtype)?
    )?;
    let valid_y1 = y1_long.ge(&Tensor::zeros((), DType::I64, device)?.broadcast_as(y1_long.shape())?)?
        .to_dtype(dtype)?;
    let valid_y1 = valid_y1.mul(
        &y1_long.le(&Tensor::new(h_max, device)?.broadcast_as(y1_long.shape())?)?.to_dtype(dtype)?
    )?;
    let valid_x1 = x1_long.ge(&Tensor::zeros((), DType::I64, device)?.broadcast_as(x1_long.shape())?)?
        .to_dtype(dtype)?;
    let valid_x1 = valid_x1.mul(
        &x1_long.le(&Tensor::new(w_max, device)?.broadcast_as(x1_long.shape())?)?.to_dtype(dtype)?
    )?;

    let mask_tl = valid_y0.mul(&valid_x0)?;
    let mask_tr = valid_y0.mul(&valid_x1)?;
    let mask_bl = valid_y1.mul(&valid_x0)?;
    let mask_br = valid_y1.mul(&valid_x1)?;

    // Clamp indices to [0, max] for safe indexing (masked positions contribute 0 anyway)
    let y0_safe = y0_long.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x0_safe = x0_long.clamp(0i64, w_max)?.to_dtype(DType::U32)?;
    let y1_safe = y1_long.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x1_safe = x1_long.clamp(0i64, w_max)?.to_dtype(DType::U32)?;

    // Flatten spatial to 1D index: idx = y * W + x
    let in_w_t = Tensor::new(in_w as u32, device)?.broadcast_as(y0_safe.shape())?;
    let idx_tl = (y0_safe.mul(&in_w_t)?.add(&x0_safe))?;
    let idx_tr = (y0_safe.mul(&in_w_t)?.add(&x1_safe))?;
    let idx_bl = (y1_safe.mul(&in_w_t)?.add(&x0_safe))?;
    let idx_br = (y1_safe.mul(&in_w_t)?.add(&x1_safe))?;

    // Gather from input for each channel
    // input: [B, C_in, H, W] → flatten spatial → [B, C_in, H*W]
    let input_flat = input.reshape((batch_size, in_channels, in_h * in_w))?;

    // idx shape: [B, kH*kW, out_h, out_w] → [B, kH*kW * out_h * out_w]
    let n_samples = kernel_size * out_h * out_w;
    let idx_tl_flat = idx_tl.reshape((batch_size, n_samples))?;
    let idx_tr_flat = idx_tr.reshape((batch_size, n_samples))?;
    let idx_bl_flat = idx_bl.reshape((batch_size, n_samples))?;
    let idx_br_flat = idx_br.reshape((batch_size, n_samples))?;

    // For each batch and channel, gather from the flattened spatial dimension.
    // We need: [B, C_in, kH*kW*out_h*out_w]
    // Expand idx to [B, C_in, n_samples] by repeating along C_in dim
    let idx_tl_exp = idx_tl_flat.unsqueeze(1)?.broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_tr_exp = idx_tr_flat.unsqueeze(1)?.broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_bl_exp = idx_bl_flat.unsqueeze(1)?.broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_br_exp = idx_br_flat.unsqueeze(1)?.broadcast_as((batch_size, in_channels, n_samples))?;

    let val_tl = input_flat.gather(&idx_tl_exp, 2)?;  // [B, C_in, n_samples]
    let val_tr = input_flat.gather(&idx_tr_exp, 2)?;
    let val_bl = input_flat.gather(&idx_bl_exp, 2)?;
    let val_br = input_flat.gather(&idx_br_exp, 2)?;

    // Reshape weights and masks: [B, kH*kW, out_h, out_w] → [B, 1, n_samples]
    let w_tl = w_tl.mul(&mask_tl)?.reshape((batch_size, 1, n_samples))?;
    let w_tr = w_tr.mul(&mask_tr)?.reshape((batch_size, 1, n_samples))?;
    let w_bl = w_bl.mul(&mask_bl)?.reshape((batch_size, 1, n_samples))?;
    let w_br = w_br.mul(&mask_br)?.reshape((batch_size, 1, n_samples))?;

    // Bilinear result: [B, C_in, n_samples]
    let sampled = (val_tl.broadcast_mul(&w_tl)?
        + val_tr.broadcast_mul(&w_tr)?
        + val_bl.broadcast_mul(&w_bl)?
        + val_br.broadcast_mul(&w_br)?)?;

    // Apply mask (modulation) if present
    // mask: [B, kH*kW, out_h, out_w] → [B, 1, kH*kW, out_h*out_w] → broadcast over C_in
    let sampled = if let Some(m) = mask {
        let m = m.reshape((batch_size, 1, kernel_size, out_h * out_w))?
            .reshape((batch_size, 1, n_samples))?;
        sampled.broadcast_mul(&m)?
    } else {
        sampled
    };

    // Reshape to columns: [B, C_in, kH*kW, out_h*out_w] → [B, C_in*kH*kW, out_h*out_w]
    let columns = sampled.reshape((batch_size, in_channels * kernel_size, out_h * out_w))?;

    // Step 2: Convolution via batched matmul
    // weight: [C_out, C_in, kH, kW] → [C_out, C_in*kH*kW]
    let weight_flat = weight.reshape((out_channels, in_channels * kernel_size))?;

    // [C_out, C_in*kH*kW] x [B, C_in*kH*kW, out_h*out_w] → [B, C_out, out_h*out_w]
    // Use broadcast_matmul: expand weight to [B, C_out, C_in*kH*kW]
    let output = weight_flat.broadcast_matmul(&columns)?;

    // Reshape: [B, C_out, out_h*out_w] → [B, C_out, out_h, out_w]
    let output = output.reshape((batch_size, out_channels, out_h, out_w))?;

    // Add bias
    let output = if let Some(b) = bias {
        let b = b.reshape((1, out_channels, 1, 1))?;
        output.broadcast_add(&b)?
    } else {
        output
    };

    Ok(output)
}
```

**Important implementation note:** The offset channel layout in PyTorch's `torchvision.ops.deform_conv2d` interleaves y/x offsets per kernel position: `[offset_h_0, offset_w_0, offset_h_1, offset_w_1, ...]`. The code above assumes `[all_offset_h, all_offset_x]` layout. Check PyTorch source and adjust the `narrow` split accordingly. The PyTorch CPU kernel indexes as:

```cpp
const int offset_idx = 2 * mask_idx;  // mask_idx = ky * kW + kx
offset_h = offset_ptr[offset_idx * (out_h * out_w) + ...]
offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) + ...]
```

This means in the channel dimension, offsets are ordered: `[h0, w0, h1, w1, h2, w2, ...]`. The `narrow` approach splitting first/second half is WRONG. Instead, use stride/slice to extract alternating channels:

```rust
// Correct: extract interleaved h/w offsets
// offset: [B, 2*kH*kW, out_h, out_w] where channels = [h0, w0, h1, w1, ...]
let offset_y = offset.index_select(
    &Tensor::from_vec((0..kernel_size).map(|i| (2 * i) as u32).collect::<Vec<_>>(), kernel_size, device)?,
    1,
)?;  // [B, kH*kW, out_h, out_w]
let offset_x = offset.index_select(
    &Tensor::from_vec((0..kernel_size).map(|i| (2 * i + 1) as u32).collect::<Vec<_>>(), kernel_size, device)?,
    1,
)?;  // [B, kH*kW, out_h, out_w]
```

Replace the `narrow` calls with the `index_select` approach above.

- [ ] **Step 2: Run golden tests**

Run: `cargo test --test golden`
Expected: All 9 golden tests PASS. If tolerance fails, check offset channel order first.

- [ ] **Step 3: Commit**

```bash
git add src/deform_conv2d.rs
git commit -s -m "feat: implement deform_conv2d forward pass"
```

### Task 6: Refactor if needed

- [ ] **Step 1: Review code for clarity**

Check for duplicated logic, overly long functions, unclear variable names. Extract helpers if the main function exceeds ~150 lines.

Candidate extractions:
- `fn bilinear_sample(input: &Tensor, sample_y: &Tensor, sample_x: &Tensor, ...) -> Result<Tensor>` — the bilinear interpolation block
- `fn build_base_grid(...) -> Result<(Tensor, Tensor)>` — base grid generation

- [ ] **Step 2: Run all tests after refactor**

Run: `cargo test`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/
git commit -s -m "refactor: extract bilinear_sample helper"
```

---

## Chunk 4: DeformConv2d Module + Final Polish

### Task 7: Implement DeformConv2d module

**Files:**
- Modify: `src/module.rs`

- [ ] **Step 1: Write failing test for the module**

Add to `src/module.rs`:

```rust
// src/module.rs
use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;
use crate::deform_conv2d::deform_conv2d;

pub struct DeformConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
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
        use_bias: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kh, kw) = kernel_size;
        let weight = vb.get((out_channels, in_channels / groups, kh, kw), "weight")?;
        let bias = if use_bias {
            Some(vb.get(out_channels, "bias")?)
        } else {
            None
        };
        Ok(Self { weight, bias, stride, padding, dilation })
    }

    pub fn forward(
        &self,
        input: &Tensor,
        offset: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        deform_conv2d(
            input,
            offset,
            &self.weight,
            self.bias.as_ref(),
            mask,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_module_forward() -> Result<()> {
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let module = DeformConv2d::new(3, 8, (3, 3), (1, 1), (1, 1), (1, 1), 1, true, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), dev)?;
        let offset = Tensor::randn(0f32, 1.0, (1, 18, 8, 8), dev)?;
        let mask = Tensor::ones((1, 9, 8, 8), DType::F32, dev)?;

        let output = module.forward(&input, &offset, Some(&mask))?;
        assert_eq!(output.dims(), &[1, 8, 8, 8]);
        Ok(())
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test --lib module`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/module.rs
git commit -s -m "feat: add DeformConv2d nn module"
```

### Task 8: Update lib.rs exports and README

**Files:**
- Modify: `src/lib.rs`
- Modify: `README.md`

- [ ] **Step 1: Finalize lib.rs**

```rust
// src/lib.rs
mod validation;
mod deform_conv2d;
mod module;

pub use deform_conv2d::deform_conv2d;
pub use module::DeformConv2d;
```

- [ ] **Step 2: Update README.md**

```markdown
# candle-dcnv2

Deformable Convolution v2 (DCNv2) implementation for [Hugging Face Candle](https://github.com/huggingface/candle).

## Usage

```rust
use candle_dcnv2::deform_conv2d;

let output = deform_conv2d(
    &input,       // [B, C_in, H, W]
    &offset,      // [B, 2*kH*kW, out_H, out_W]
    &weight,      // [C_out, C_in, kH, kW]
    Some(&bias),  // [C_out]
    Some(&mask),  // [B, kH*kW, out_H, out_W]
    (1, 1),       // stride
    (1, 1),       // padding
    (1, 1),       // dilation
)?;
```

## Current limitations

- `dilation`: only `(1, 1)`
- `groups`: only `1`
- `offset_groups`: only `1`
- Forward pass only (no backward/training)

These will be extended in future versions.

## License

Apache-2.0
```

- [ ] **Step 3: Run full test suite**

Run: `cargo test`
Expected: All unit tests + all golden tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs README.md
git commit -s -m "docs: update lib.rs exports and README"
```

### Task 9: Open candle issue + prepare PR

**Files:** None (GitHub operations)

- [ ] **Step 1: Open feature request issue on candle**

```bash
gh issue create --repo huggingface/candle \
  --title "Feature: deform_conv2d (Deformable Convolution v2) for vision models" \
  --body "$(cat <<'EOF'
## Motivation

`deform_conv2d` (DCNv2) is widely used in vision models but currently absent from candle:

- **BiRefNet** — SOTA background removal
- **Deformable DETR / RT-DETR / RF-DETR** — object detection
- **Mask2Former** — segmentation
- **InternImage / DCNv4** — vision foundation models

## Proposal

Add `deform_conv2d()` function and `DeformConv2d` module to `candle-nn`, matching [torchvision.ops.deform_conv2d](https://pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html).

**Implementation approach:**
- Forward-only (inference)
- High-level tensor operations (works on all backends: CPU, CUDA, Metal, WASM)
- Groups/offset_groups inferred from tensor shapes
- First version: groups=1, offset_groups=1, dilation=(1,1)

I have a working implementation with golden tests at https://github.com/developer0hye/candle-dcnv2 and can submit a PR.

**Reference:** [torchvision CPU kernel](https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/deform_conv2d_kernel.cpp)
EOF
)"
```

- [ ] **Step 2: Note the issue number for the PR**

- [ ] **Step 3: Prepare the PR** (copy code to candle fork, submit PR — details depend on candle's contribution workflow)
