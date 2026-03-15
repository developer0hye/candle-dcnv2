use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::deform_conv2d::deform_conv2d;

/// Deformable Convolution v2 module owning convolution weights.
///
/// Does not implement the standard `Module` trait because `forward` requires
/// additional inputs (`offset`, `mask`) beyond a single tensor.
pub struct DeformConv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl DeformConv2d {
    pub fn new(
        _in_channels: usize,
        _out_channels: usize,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
        _groups: usize,
        _use_bias: bool,
        _vb: VarBuilder,
    ) -> Result<Self> {
        todo!()
    }

    /// `offset_groups` is inferred from `offset.shape[1] / (2 * kH * kW)`.
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
