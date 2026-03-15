use candle_core::{Result, Tensor, bail};

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
    let input_dims = input.dims();
    let weight_dims = weight.dims();
    let offset_dims = offset.dims();

    if input_dims.len() != 4 {
        bail!(
            "deform_conv2d: input must be 4D [B, C_in, H, W], got {}D",
            input_dims.len()
        );
    }
    if weight_dims.len() != 4 {
        bail!(
            "deform_conv2d: weight must be 4D [C_out, C_in/groups, kH, kW], got {}D",
            weight_dims.len()
        );
    }
    if offset_dims.len() != 4 {
        bail!(
            "deform_conv2d: offset must be 4D, got {}D",
            offset_dims.len()
        );
    }

    let batch_size = input_dims[0];
    let in_channels = input_dims[1];
    let in_h = input_dims[2];
    let in_w = input_dims[3];

    let out_channels = weight_dims[0];
    let kernel_h = weight_dims[2];
    let kernel_w = weight_dims[3];

    // Infer groups from tensor shapes (matches PyTorch convention)
    let channels_per_group = weight_dims[1];
    if in_channels == 0 || channels_per_group == 0 || in_channels % channels_per_group != 0 {
        bail!(
            "deform_conv2d: C_in ({in_channels}) must be divisible by weight.shape[1] ({channels_per_group})"
        );
    }
    let groups = in_channels / channels_per_group;

    if out_channels % groups != 0 {
        bail!("deform_conv2d: C_out ({out_channels}) must be divisible by groups ({groups})");
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

    // Compute expected output spatial dimensions
    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (dil_h, dil_w) = dilation;

    let out_h = (in_h + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

    if out_h == 0 || out_w == 0 {
        bail!("deform_conv2d: computed output size is zero (out_h={out_h}, out_w={out_w})");
    }

    // Validate offset spatial dims match expected output
    if offset_dims[0] != batch_size || offset_dims[2] != out_h || offset_dims[3] != out_w {
        bail!(
            "deform_conv2d: offset spatial dims ({}, {}) don't match expected ({out_h}, {out_w})",
            offset_dims[2],
            offset_dims[3]
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
                mask_dims[2],
                mask_dims[3]
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn valid_basic_shapes() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        // out_h = (8 + 2*1 - 1*(3-1) - 1)/1 + 1 = 8
        let offset = Tensor::zeros((2, 2 * 9, 8, 8), DType::F32, dev).unwrap();
        let mask = Tensor::zeros((2, 9, 8, 8), DType::F32, dev).unwrap();

        let params = validate_and_extract(
            &input,
            &offset,
            &weight,
            None,
            Some(&mask),
            (1, 1),
            (1, 1),
            (1, 1),
        )
        .unwrap();

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
    fn input_not_4d() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();

        let result =
            validate_and_extract(&input, &offset, &weight, None, None, (1, 1), (1, 1), (1, 1));
        assert!(result.is_err());
    }

    #[test]
    fn offset_spatial_mismatch() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        // Wrong spatial dims: 4x4 instead of 8x8
        let offset = Tensor::zeros((2, 18, 4, 4), DType::F32, dev).unwrap();

        let result =
            validate_and_extract(&input, &offset, &weight, None, None, (1, 1), (1, 1), (1, 1));
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_dilation() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        // dilation=2: out_h = (8+2-2*(3-1)-1)/1+1 = 5
        let offset = Tensor::zeros((2, 18, 5, 5), DType::F32, dev).unwrap();

        let result =
            validate_and_extract(&input, &offset, &weight, None, None, (1, 1), (1, 1), (2, 2));
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_groups() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 6, 8, 8), DType::F32, dev).unwrap();
        // weight shape implies groups=2: C_in(6) / weight.shape[1](3) = 2
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();

        let result =
            validate_and_extract(&input, &offset, &weight, None, None, (1, 1), (1, 1), (1, 1));
        assert!(result.is_err());
    }

    #[test]
    fn mask_channel_mismatch() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();
        // Wrong mask channels: 5 instead of 9
        let mask = Tensor::zeros((2, 5, 8, 8), DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input,
            &offset,
            &weight,
            None,
            Some(&mask),
            (1, 1),
            (1, 1),
            (1, 1),
        );
        assert!(result.is_err());
    }

    #[test]
    fn bias_length_mismatch() {
        let dev = &Device::Cpu;
        let input = Tensor::zeros((2, 3, 8, 8), DType::F32, dev).unwrap();
        let weight = Tensor::zeros((8, 3, 3, 3), DType::F32, dev).unwrap();
        let offset = Tensor::zeros((2, 18, 8, 8), DType::F32, dev).unwrap();
        // Wrong bias length: 4 instead of 8
        let bias = Tensor::zeros(4, DType::F32, dev).unwrap();

        let result = validate_and_extract(
            &input,
            &offset,
            &weight,
            Some(&bias),
            None,
            (1, 1),
            (1, 1),
            (1, 1),
        );
        assert!(result.is_err());
    }
}
