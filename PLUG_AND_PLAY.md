# PLUG_AND_PLAY — Grad

> Gradient-based optimization for ternary systems

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-grad = { git = "https://github.com/SuperInstance/ternary-grad" }
```

Use in your code:

```rust
use ternary_grad::TernaryAdam;

let mut optimizer = TernaryAdam::new(0.01);
optimizer.step(&gradients);
```

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
