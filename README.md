# ternary-grad

**Ternary gradient descent: the training infrastructure that makes {-1, 0, +1} neural networks learnable.**

You can't backpropagate through `sign(x)` — the gradient is zero everywhere (and undefined at 0). The **Straight-Through Estimator** (STE, Bengio 2013) solves this: during the forward pass, quantize to {-1, 0, +1}. During backward pass, pretend the quantization didn't happen and pass the gradient through unchanged.

This crate provides STE plus ternary-aware optimizers (SGD, Adam), gradient clipping, and learning rate schedules — everything you need to train a ternary neural network.

---

## The Straight-Through Estimator

```
Forward:  x → sign(x) → {-1, 0, +1}     (the actual quantization)
Backward: ∂L/∂x ≈ ∂L/∂sign(x)            (pretend sign didn't happen)
```

More precisely, the STE gradient:
```rust
fn ste_gradient(x: f64) -> f64 {
    if x.abs() <= 1.0 { 1.0 }  // pass through
    else { 0.0 }                // clip outside [-1, 1]
}
```

This creates a "gradient highway" for weights near {-1, 0, +1} and silences gradients for weights that have drifted too far from the quantization points.

---

## Architecture

- **`straight_through(x, threshold)`** — Quantize: x → {-1, 0, +1}
- **`ste_gradient(x)`** — STE: d(sign(x))/dx ≈ 1 if |x| ≤ 1
- **`ternary_sgd_update()`** — SGD with STE gradient modifier
- **`TernaryAdam`** — Adam optimizer with STE bias correction
- **`clip_ternary_gradient()`** — Clamp gradient magnitude
- **`cosine_lr()`** / **`step_lr()`** — Learning rate schedules
- **`quantization_error()`** — L2 distance between float and ternary weights
- **`ternary_accuracy()`** — Fraction of weights correctly quantized

---

## Quick Start

```rust
use ternary_grad::{straight_through, TernaryAdam, cosine_lr};

let mut weights = vec![0.8, -0.3, 0.01, -1.2, 0.5];
let grads = vec![0.1, -0.2, 0.3, -0.1, 0.2];
let mut optimizer = TernaryAdam::new(0.01, weights.len());

// Training step
optimizer.step(&mut weights, &grads);

// Quantize for inference
let ternary: Vec<i8> = weights.iter().map(|&w| straight_through(w, 0.5)).collect();
```

---

## Ecosystem

- **ternary-tnn** — Ternary neural network layers (uses this for training)
- **ternary-llm** — Ternary language model (uses Adam from here)
- **ternary-attention** — Ternary attention mechanisms
- **ternary-cookbook** — Working demos

## License

MIT
