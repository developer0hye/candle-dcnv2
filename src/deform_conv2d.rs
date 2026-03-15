use candle_core::{DType, Result, Tensor};

use crate::validation::{DeformConv2dParams, validate_and_extract};

/// Performs Deformable Convolution v2 (forward only).
///
/// When `mask` is `Some`, performs DCNv2 (modulated deformable convolution).
/// When `mask` is `None`, performs DCNv1.
///
/// `groups` and `offset_groups` are inferred from tensor shapes:
///   - groups = C_in / weight.shape[1]
///   - offset_groups = offset.shape[1] / (2 * kH * kW)
#[allow(clippy::too_many_arguments)] // Matches torchvision.ops.deform_conv2d signature
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
    let params =
        validate_and_extract(input, offset, weight, bias, mask, stride, padding, dilation)?;
    let columns = deformable_im2col(input, offset, mask, &params, stride, padding)?;
    matmul_and_bias(weight, bias, &columns, &params)
}

/// Bilinear-interpolation-based im2col with deformable offsets.
///
/// Produces `[B, C_in * kH * kW, out_h * out_w]` columns tensor.
fn deformable_im2col(
    input: &Tensor,
    offset: &Tensor,
    mask: Option<&Tensor>,
    params: &DeformConv2dParams,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    let batch_size = params.batch_size;
    let in_channels = params.in_channels;
    let kernel_h = params.kernel_h;
    let kernel_w = params.kernel_w;
    let out_h = params.out_h;
    let out_w = params.out_w;

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let device = input.device();
    let dtype = input.dtype();
    let kernel_size = kernel_h * kernel_w;
    let n_samples = kernel_size * out_h * out_w;

    // Build base sampling grid: [kH*kW, out_h, out_w] for y and x coordinates
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

    // [1, kH*kW, out_h, out_w] — broadcasts over batch
    let base_y =
        Tensor::from_vec(base_y_data, (1, kernel_size, out_h, out_w), device)?.to_dtype(dtype)?;
    let base_x =
        Tensor::from_vec(base_x_data, (1, kernel_size, out_h, out_w), device)?.to_dtype(dtype)?;

    // PyTorch interleaves offsets as [h0, w0, h1, w1, ...] in channel dim.
    // Extract even indices (y offsets) and odd indices (x offsets).
    let y_indices = Tensor::from_vec(
        (0..kernel_size).map(|i| (2 * i) as u32).collect::<Vec<_>>(),
        kernel_size,
        device,
    )?;
    let x_indices = Tensor::from_vec(
        (0..kernel_size)
            .map(|i| (2 * i + 1) as u32)
            .collect::<Vec<_>>(),
        kernel_size,
        device,
    )?;
    let offset_y = offset.index_select(&y_indices, 1)?; // [B, kH*kW, out_h, out_w]
    let offset_x = offset.index_select(&x_indices, 1)?;

    // Sampling coordinates
    let sample_y = base_y.broadcast_add(&offset_y)?;
    let sample_x = base_x.broadcast_add(&offset_x)?;

    // Bilinear interpolation
    let sampled = bilinear_sample(input, &sample_y, &sample_x, params)?;

    // Apply modulation mask if present
    let sampled = if let Some(m) = mask {
        // mask: [B, kH*kW, out_h, out_w] -> [B, 1, n_samples]
        let m = m.reshape((batch_size, 1, n_samples))?;
        sampled.broadcast_mul(&m)?
    } else {
        sampled
    };

    // Reshape to columns: [B, C_in * kH * kW, out_h * out_w]
    sampled.reshape((batch_size, in_channels * kernel_size, out_h * out_w))
}

/// Sample `input` at fractional `(sample_y, sample_x)` coordinates using bilinear interpolation.
///
/// Returns `[B, C_in, kH*kW * out_h * out_w]` — sampled values for each channel.
fn bilinear_sample(
    input: &Tensor,
    sample_y: &Tensor,
    sample_x: &Tensor,
    params: &DeformConv2dParams,
) -> Result<Tensor> {
    let DeformConv2dParams {
        batch_size,
        in_channels,
        in_h,
        in_w,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        ..
    } = *params;

    let device = input.device();
    let dtype = input.dtype();
    let kernel_size = kernel_h * kernel_w;
    let n_samples = kernel_size * out_h * out_w;

    // Floor/ceil for 4-corner sampling
    let y0 = sample_y.floor()?;
    let x0 = sample_x.floor()?;
    let y1 = (&y0 + 1.0)?;
    let x1 = (&x0 + 1.0)?;

    // Interpolation weights (fractional parts)
    let wy1 = sample_y.sub(&y0)?;
    let wx1 = sample_x.sub(&x0)?;
    let wy0 = (1.0 - &wy1)?;
    let wx0 = (1.0 - &wx1)?;

    let w_tl = wy0.mul(&wx0)?;
    let w_tr = wy0.mul(&wx1)?;
    let w_bl = wy1.mul(&wx0)?;
    let w_br = wy1.mul(&wx1)?;

    // Convert to integer indices for gather
    let y0_i64 = y0.to_dtype(DType::I64)?;
    let x0_i64 = x0.to_dtype(DType::I64)?;
    let y1_i64 = y1.to_dtype(DType::I64)?;
    let x1_i64 = x1.to_dtype(DType::I64)?;

    let h_max = (in_h as i64) - 1;
    let w_max = (in_w as i64) - 1;
    let zero = Tensor::zeros(y0_i64.shape(), DType::I64, device)?;
    let h_max_t = Tensor::new(h_max, device)?.broadcast_as(y0_i64.shape())?;
    let w_max_t = Tensor::new(w_max, device)?.broadcast_as(x0_i64.shape())?;

    // Boundary validity masks: 1.0 where in-bounds, 0.0 where out-of-bounds
    let valid_y0 = y0_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&y0_i64.le(&h_max_t)?.to_dtype(dtype)?)?;
    let valid_x0 = x0_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&x0_i64.le(&w_max_t)?.to_dtype(dtype)?)?;
    let valid_y1 = y1_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&y1_i64.le(&h_max_t)?.to_dtype(dtype)?)?;
    let valid_x1 = x1_i64
        .ge(&zero)?
        .to_dtype(dtype)?
        .mul(&x1_i64.le(&w_max_t)?.to_dtype(dtype)?)?;

    let mask_tl = valid_y0.mul(&valid_x0)?;
    let mask_tr = valid_y0.mul(&valid_x1)?;
    let mask_bl = valid_y1.mul(&valid_x0)?;
    let mask_br = valid_y1.mul(&valid_x1)?;

    // Clamp indices to [0, max] for safe indexing (out-of-bounds masked to 0 anyway)
    let y0_safe = y0_i64.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x0_safe = x0_i64.clamp(0i64, w_max)?.to_dtype(DType::U32)?;
    let y1_safe = y1_i64.clamp(0i64, h_max)?.to_dtype(DType::U32)?;
    let x1_safe = x1_i64.clamp(0i64, w_max)?.to_dtype(DType::U32)?;

    // Flatten spatial indices: idx = y * W + x
    let in_w_t = Tensor::new(in_w as u32, device)?.broadcast_as(y0_safe.shape())?;
    let idx_tl = y0_safe.mul(&in_w_t)?.add(&x0_safe)?;
    let idx_tr = y0_safe.mul(&in_w_t)?.add(&x1_safe)?;
    let idx_bl = y1_safe.mul(&in_w_t)?.add(&x0_safe)?;
    let idx_br = y1_safe.mul(&in_w_t)?.add(&x1_safe)?;

    // input: [B, C_in, H, W] -> [B, C_in, H*W]
    let input_flat = input.reshape((batch_size, in_channels, in_h * in_w))?;

    // idx: [B, kH*kW, out_h, out_w] -> [B, n_samples]
    let idx_tl_flat = idx_tl.reshape((batch_size, n_samples))?;
    let idx_tr_flat = idx_tr.reshape((batch_size, n_samples))?;
    let idx_bl_flat = idx_bl.reshape((batch_size, n_samples))?;
    let idx_br_flat = idx_br.reshape((batch_size, n_samples))?;

    // Expand idx to [B, C_in, n_samples] for gather along spatial dim
    let idx_tl_exp =
        idx_tl_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_tr_exp =
        idx_tr_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_bl_exp =
        idx_bl_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;
    let idx_br_exp =
        idx_br_flat
            .unsqueeze(1)?
            .broadcast_as((batch_size, in_channels, n_samples))?;

    // Gather 4 corner values: [B, C_in, n_samples]
    let val_tl = input_flat.gather(&idx_tl_exp.contiguous()?, 2)?;
    let val_tr = input_flat.gather(&idx_tr_exp.contiguous()?, 2)?;
    let val_bl = input_flat.gather(&idx_bl_exp.contiguous()?, 2)?;
    let val_br = input_flat.gather(&idx_br_exp.contiguous()?, 2)?;

    // Combine weights with boundary masks: [B, kH*kW, out_h, out_w] -> [B, 1, n_samples]
    let w_tl = w_tl.mul(&mask_tl)?.reshape((batch_size, 1, n_samples))?;
    let w_tr = w_tr.mul(&mask_tr)?.reshape((batch_size, 1, n_samples))?;
    let w_bl = w_bl.mul(&mask_bl)?.reshape((batch_size, 1, n_samples))?;
    let w_br = w_br.mul(&mask_br)?.reshape((batch_size, 1, n_samples))?;

    // Bilinear interpolation: weighted sum of 4 corners -> [B, C_in, n_samples]
    let result = val_tl
        .broadcast_mul(&w_tl)?
        .add(&val_tr.broadcast_mul(&w_tr)?)?
        .add(&val_bl.broadcast_mul(&w_bl)?)?
        .add(&val_br.broadcast_mul(&w_br)?)?;

    Ok(result)
}

/// weight matmul + bias addition.
///
/// columns: `[B, C_in * kH * kW, out_h * out_w]`
/// Returns: `[B, C_out, out_h, out_w]`
fn matmul_and_bias(
    weight: &Tensor,
    bias: Option<&Tensor>,
    columns: &Tensor,
    params: &DeformConv2dParams,
) -> Result<Tensor> {
    let DeformConv2dParams {
        batch_size,
        in_channels,
        out_channels,
        kernel_h,
        kernel_w,
        out_h,
        out_w,
        ..
    } = *params;

    // weight: [C_out, C_in, kH, kW] -> [C_out, C_in*kH*kW]
    let weight_flat = weight.reshape((out_channels, in_channels * kernel_h * kernel_w))?;

    // [C_out, C_in*kH*kW] x [B, C_in*kH*kW, out_h*out_w] -> [B, C_out, out_h*out_w]
    let output = weight_flat.broadcast_matmul(columns)?;

    // Reshape to spatial: [B, C_out, out_h, out_w]
    let output = output.reshape((batch_size, out_channels, out_h, out_w))?;

    if let Some(b) = bias {
        let b = b.reshape((1, out_channels, 1, 1))?;
        output.broadcast_add(&b)
    } else {
        Ok(output)
    }
}
