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
    #[allow(clippy::too_many_arguments)] // Matches torchvision.ops.DeformConv2d signature
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
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn module_forward_produces_correct_shape() -> Result<()> {
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

    #[test]
    fn module_forward_without_bias() -> Result<()> {
        let dev = &Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, dev);

        let module = DeformConv2d::new(3, 8, (3, 3), (1, 1), (1, 1), (1, 1), 1, false, vb)?;

        let input = Tensor::randn(0f32, 1.0, (1, 3, 8, 8), dev)?;
        let offset = Tensor::randn(0f32, 1.0, (1, 18, 8, 8), dev)?;

        let output = module.forward(&input, &offset, None)?;
        assert_eq!(output.dims(), &[1, 8, 8, 8]);
        Ok(())
    }
}
