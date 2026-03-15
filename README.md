# candle-dcnv2

Deformable Convolution v2 (DCNv2) implementation for [Hugging Face Candle](https://github.com/huggingface/candle).

Implemented with high-level tensor operations — works on all backends (CPU, CUDA, Metal, WASM).

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

Or use the `DeformConv2d` module which loads weights via `VarBuilder`:

```rust
use candle_dcnv2::DeformConv2d;

let module = DeformConv2d::new(
    in_channels, out_channels, (3, 3),
    (1, 1), (1, 1), (1, 1), 1, true, vb,
)?;
let output = module.forward(&input, &offset, Some(&mask))?;
```

## Current limitations

- `dilation`: only `(1, 1)`
- `groups`: only `1`
- `offset_groups`: only `1`
- Forward pass only (no backward/training)

These will be extended in future versions.

## License

Apache-2.0
