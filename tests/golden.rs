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
            let tensor =
                Tensor::from_raw_buffer(view.data(), DType::F32, view.shape(), device).unwrap();
            (name.to_string(), tensor)
        })
        .collect()
}

fn assert_tensor_close(actual: &Tensor, expected: &Tensor, atol: f64) {
    let diff = actual
        .to_dtype(DType::F64)
        .unwrap()
        .sub(&expected.to_dtype(DType::F64).unwrap())
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .max(0)
        .unwrap()
        .to_scalar::<f64>()
        .unwrap();
    assert!(diff < atol, "max abs diff: {diff}, tolerance: {atol}");
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
