//! Mathematical invariant tests for deform_conv2d.
//!
//! These verify structural properties that MUST hold regardless of inputs,
//! providing stronger guarantees than golden tests alone.

use candle_core::{DType, Device, Result, Tensor};
use candle_dcnv2::deform_conv2d;

fn load_test_case(name: &str) -> std::collections::HashMap<String, Tensor> {
    let path = format!("test-data/{name}.safetensors");
    let data = std::fs::read(&path).unwrap_or_else(|_| panic!("missing test data: {path}"));
    let tensors = safetensors::SafeTensors::deserialize(&data).unwrap();
    let device = &Device::Cpu;

    tensors
        .tensors()
        .into_iter()
        .map(|(name, view)| {
            let dtype = match view.dtype() {
                safetensors::Dtype::F32 => DType::F32,
                safetensors::Dtype::F64 => DType::F64,
                safetensors::Dtype::I64 => DType::I64,
                safetensors::Dtype::U32 => DType::U32,
                dt => panic!("unsupported dtype {dt:?} for tensor {name}"),
            };
            let tensor = Tensor::from_raw_buffer(view.data(), dtype, view.shape(), device).unwrap();
            (name.to_string(), tensor)
        })
        .collect()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f64 {
    a.to_dtype(DType::F64)
        .unwrap()
        .sub(&b.to_dtype(DType::F64).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f64>()
        .unwrap()
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, atol: f64) {
    let diff = max_abs_diff(actual, expected);
    assert!(diff < atol, "max abs diff: {diff}, tolerance: {atol}");
}

// =========================================================================
// 1. ZERO-OFFSET INVARIANT: deform_conv2d(offset=0, mask=1) == conv2d
//    This is the fundamental correctness proof. Deformable convolution with
//    no deformation must be equivalent to standard convolution.
// =========================================================================

fn run_zero_offset_test(name: &str, stride: (usize, usize), padding: (usize, usize)) -> Result<()> {
    let data = load_test_case(name);

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        stride,
        padding,
        (1, 1),
    )?;

    // Compare against PyTorch deform_conv2d output
    assert_tensor_close(&result, &data["output"], 1e-4);

    // Compare against PyTorch F.conv2d output (the key invariant)
    assert_tensor_close(&result, &data["conv2d_output"], 1e-4);

    // Also verify deform_conv2d and conv2d agree in PyTorch
    let pytorch_diff = max_abs_diff(&data["output"], &data["conv2d_output"]);
    assert!(
        pytorch_diff < 1e-4,
        "PyTorch deform_conv2d vs conv2d disagree: {pytorch_diff}"
    );

    Ok(())
}

#[test]
fn zero_offset_3x3() -> Result<()> {
    run_zero_offset_test("zero_offset_3x3", (1, 1), (1, 1))
}

#[test]
fn zero_offset_5x5() -> Result<()> {
    run_zero_offset_test("zero_offset_5x5", (1, 1), (2, 2))
}

#[test]
fn zero_offset_stride2() -> Result<()> {
    run_zero_offset_test("zero_offset_stride2", (2, 2), (1, 1))
}

#[test]
fn zero_offset_1x1_kernel() -> Result<()> {
    run_zero_offset_test("zero_offset_1x1", (1, 1), (0, 0))
}

#[test]
fn zero_offset_no_padding() -> Result<()> {
    run_zero_offset_test("zero_offset_no_pad", (1, 1), (0, 0))
}

// =========================================================================
// 2. ZERO-MASK INVARIANT: mask=0 zeroes out all sampled values.
//    With bias: output = bias broadcast. Without bias: output = 0.
// =========================================================================

#[test]
fn zero_mask_equals_bias_only() -> Result<()> {
    let data = load_test_case("zero_mask");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        (1, 1),
        (1, 1),
        (1, 1),
    )?;

    // Should match PyTorch output
    assert_tensor_close(&result, &data["output"], 1e-4);

    // Should equal bias broadcast over spatial dims
    assert_tensor_close(&result, &data["expected_bias_only"], 1e-4);
    Ok(())
}

#[test]
fn zero_mask_no_bias_equals_zero() -> Result<()> {
    let data = load_test_case("zero_mask_no_bias");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        None,
        Some(&data["mask"]),
        (1, 1),
        (1, 1),
        (1, 1),
    )?;

    // Should match PyTorch output
    assert_tensor_close(&result, &data["output"], 1e-4);

    // Should be all zeros
    let zeros = Tensor::zeros(result.shape(), result.dtype(), result.device())?;
    assert_tensor_close(&result, &zeros, 1e-6);
    Ok(())
}

// =========================================================================
// 3. LARGE-OFFSET BOUNDARY: offsets pushing all samples far out of bounds.
//    Zero-padding means all sampled values are 0, so output = bias only.
// =========================================================================

#[test]
fn large_positive_offset() -> Result<()> {
    let data = load_test_case("large_offset");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        (1, 1),
        (1, 1),
        (1, 1),
    )?;

    assert_tensor_close(&result, &data["output"], 1e-4);

    // All samples out of bounds -> sampled values are 0 -> output is bias-only
    let bias_broadcast = data["bias"]
        .reshape((1, 8, 1, 1))?
        .broadcast_as(result.shape())?;
    assert_tensor_close(&result, &bias_broadcast, 1e-4);
    Ok(())
}

#[test]
fn large_negative_offset() -> Result<()> {
    let data = load_test_case("large_negative_offset");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        (1, 1),
        (1, 1),
        (1, 1),
    )?;

    assert_tensor_close(&result, &data["output"], 1e-4);

    let bias_broadcast = data["bias"]
        .reshape((1, 8, 1, 1))?
        .broadcast_as(result.shape())?;
    assert_tensor_close(&result, &bias_broadcast, 1e-4);
    Ok(())
}

// =========================================================================
// 4. INTEGER OFFSET: when offsets are exact integers, bilinear interpolation
//    should be equivalent to direct pixel lookup (no blending artifacts).
// =========================================================================

#[test]
fn integer_offset_exact_sampling() -> Result<()> {
    let data = load_test_case("integer_offset");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        (1, 1),
        (1, 1),
        (1, 1),
    )?;

    assert_tensor_close(&result, &data["output"], 1e-4);
    Ok(())
}

// =========================================================================
// 5. MINIMAL SPATIAL SIZE: 1x1 input with 1x1 kernel
// =========================================================================

#[test]
fn minimal_1x1_spatial() -> Result<()> {
    let data = load_test_case("minimal_1x1");

    let result = deform_conv2d(
        &data["input"],
        &data["offset"],
        &data["weight"],
        Some(&data["bias"]),
        Some(&data["mask"]),
        (1, 1),
        (0, 0),
        (1, 1),
    )?;

    assert_eq!(result.dims(), &[1, 8, 1, 1]);
    assert_tensor_close(&result, &data["output"], 1e-4);
    Ok(())
}

// =========================================================================
// 6. OUTPUT SHAPE CORRECTNESS: verify output dimensions match the formula
//    out = (in + 2*pad - dil*(k-1) - 1) / stride + 1
// =========================================================================

#[test]
fn output_shape_formula() -> Result<()> {
    let dev = &Device::Cpu;

    // Test various configurations
    // (C_in, C_out, H, W, kH, kW, stride, pad)
    let configs: Vec<[usize; 8]> = vec![
        [3, 8, 8, 8, 3, 3, 1, 1],    // standard
        [3, 8, 8, 8, 3, 3, 2, 1],    // strided
        [3, 8, 8, 8, 3, 3, 1, 0],    // no padding
        [3, 8, 16, 12, 5, 3, 1, 0],  // non-square kernel, non-square input
        [3, 8, 7, 7, 3, 3, 2, 1],    // odd spatial, strided
        [64, 128, 1, 1, 1, 1, 1, 0], // 1x1
    ];

    for [c_in, c_out, h, w, kh, kw, s, p] in configs {
        let out_h = (h + 2 * p - kh) / s + 1;
        let out_w = (w + 2 * p - kw) / s + 1;

        let input = Tensor::zeros((1, c_in, h, w), DType::F32, dev)?;
        let weight = Tensor::zeros((c_out, c_in, kh, kw), DType::F32, dev)?;
        let offset = Tensor::zeros((1, 2 * kh * kw, out_h, out_w), DType::F32, dev)?;

        let result = deform_conv2d(&input, &offset, &weight, None, None, (s, s), (p, p), (1, 1))?;

        assert_eq!(
            result.dims(),
            &[1, c_out, out_h, out_w],
            "shape mismatch for config: c_in={c_in}, h={h}, w={w}, k=({kh},{kw}), s={s}, p={p}"
        );
    }
    Ok(())
}

// =========================================================================
// 7. TOLERANCE ANALYSIS: report actual max diffs (not just pass/fail)
//    This test prints detailed precision info for manual inspection.
// =========================================================================

#[test]
fn tolerance_report() -> Result<()> {
    let golden_cases = vec![
        ("basic_3x3", (1, 1), (1, 1)),
        ("dcnv1_no_mask", (1, 1), (1, 1)),
        ("no_bias", (1, 1), (1, 1)),
        ("no_padding", (1, 1), (0, 0)),
        ("stride_2", (2, 2), (1, 1)),
        ("large_input", (1, 1), (1, 1)),
        ("non_square_kernel", (1, 1), (1, 2)),
        ("non_square_input", (1, 1), (1, 1)),
        ("batch_1", (1, 1), (1, 1)),
    ];

    let mut max_across_all: f64 = 0.0;

    for (name, stride, padding) in &golden_cases {
        let data = load_test_case(name);
        let bias = data.get("bias");
        let mask = data.get("mask");

        let result = deform_conv2d(
            &data["input"],
            &data["offset"],
            &data["weight"],
            bias,
            mask,
            *stride,
            *padding,
            (1, 1),
        )?;

        let diff = max_abs_diff(&result, &data["output"]);
        max_across_all = max_across_all.max(diff);
        eprintln!("  {name:25}: max_abs_diff = {diff:.2e}");
    }

    eprintln!("  ----------------------------------------");
    eprintln!("  worst case across all   : {max_across_all:.2e}");

    // All must pass at 1e-4
    assert!(
        max_across_all < 1e-4,
        "worst-case diff {max_across_all:.2e} exceeds 1e-4"
    );

    // If all pass at 1e-5, the implementation has excellent precision
    if max_across_all < 1e-5 {
        eprintln!("  ** All tests pass at tighter tolerance 1e-5 **");
    }

    Ok(())
}
