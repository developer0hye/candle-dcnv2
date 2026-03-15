use candle_core::{Result, Tensor};

/// Performs Deformable Convolution v2 (forward only).
///
/// When `mask` is `Some`, performs DCNv2 (modulated deformable convolution).
/// When `mask` is `None`, performs DCNv1.
///
/// `groups` and `offset_groups` are inferred from tensor shapes:
///   - groups = C_in / weight.shape[1]
///   - offset_groups = offset.shape[1] / (2 * kH * kW)
pub fn deform_conv2d(
    _input: &Tensor,
    _offset: &Tensor,
    _weight: &Tensor,
    _bias: Option<&Tensor>,
    _mask: Option<&Tensor>,
    _stride: (usize, usize),
    _padding: (usize, usize),
    _dilation: (usize, usize),
) -> Result<Tensor> {
    todo!()
}
