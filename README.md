# ternary-grad

Ternary gradient descent for the SuperInstance {-1, 0, +1} ecosystem.

## Features

- **StraightThroughEstimator (STE)** — `forward` quantizes to trit; `backward` passes gradient unchanged (identity)
- **TernaryParam** — latent float + quantized trit + gradient accumulator; `sync_trit` re-quantizes
- **clip_gradients** — L2 norm clipping across a layer; `clip_grad_scalar` for scalars
- **TernarySgd** — SGD with optional momentum; updates latent, re-quantizes after each step
- **TernaryAdam** — Adam with bias-corrected moments in float space; trit read path only
- **StepLrScheduler** — multiply LR by `factor` every `step_size` steps
- **CosineAnnealingScheduler** — cosine decay between `max_lr` and `min_lr`
- **WarmupCosineScheduler** — linear warmup then cosine decay
- `sparsity` / `active_count` — measure trit utilization in a layer

## Usage

```rust
use ternary_grad::{TernaryParam, TernaryAdam, clip_gradients};

let mut layer: Vec<TernaryParam> = (0..64).map(|i| TernaryParam::new(i as f32 * 0.01)).collect();
let mut adam = TernaryAdam::with_defaults(0.001, layer.len());

// Training loop:
for p in &mut layer { p.grad = /* backprop result */ 0.1; }
clip_gradients(&mut layer, 1.0);
adam.step(&mut layer);
```

## Tests

23 tests covering quantization, STE forward/backward, SGD, Adam, gradient clipping, LR schedulers, and sparsity metrics.
