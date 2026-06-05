# ternary-grad

**Ternary gradient descent: the training infrastructure that makes {-1, 0, +1} neural networks learnable.**

You can't backpropagate through `sign(x)` — the gradient is zero almost everywhere (and undefined at 0). The **Straight-Through Estimator** (STE, Bengio et al., 2013) solves this: during the forward pass, quantize latent weights to {-1, 0, +1}. During the backward pass, pretend the quantization didn't happen and pass the gradient through unchanged.

This crate provides STE plus ternary-aware optimizers (SGD, Adam), gradient clipping, learning rate schedules, quantization diagnostics, and weight decay — everything you need to train a ternary neural network from scratch.

---

## Motivation

### Why Ternary Weights?

Neural network weights are typically 32-bit floats. For inference at scale, this is expensive:
- **Memory**: 32× more parameters per byte than 1-bit weights
- **Bandwidth**: moving weights dominates energy in transformer inference
- **Multiplication**: ternary weights reduce multiply-accumulate to simple addition/subtraction (or no-op for 0)

Ternary {-1, 0, +1} offers a compelling middle ground:
- **vs binary** {-1, +1}: the 0 state adds expressivity and natural sparsity
- **vs full precision**: ~16× memory reduction, hardware-friendly ops
- **vs pruning**: ternary quantization is deterministic and structured

Research demonstrates ternary networks achieve within 1–3% accuracy of full-precision on ImageNet and language tasks (Li et al., 2016; Zhu et al., 2017; Ma et al., 2024 — BitNet 1.58-bit).

### The Training Problem

The quantization function is:

```
q(w) =  1   if w > threshold
        0   if |w| ≤ threshold
       -1   if w < -threshold
```

The derivative `dq/dw` is zero everywhere (and undefined at the thresholds). Standard backpropagation gets no signal.

**The Straight-Through Estimator (STE)** is the key trick: define a "fake" gradient for backprop:

```
Forward:   q = quantize(w)        // actual ternary value
Backward:  ∂L/∂w ≈ ∂L/∂q         // pretend quantize is identity
```

This is a biased estimator, but empirically it works. The crate adds a refinement: only pass gradients through when `|w| ≤ 1.0`, creating a "funnel" that pushes weights toward the quantization points {-1, 0, +1}.

---

## Architecture

### `straight_through(x, threshold) -> i8`

Quantize a single float to {-1, 0, +1}:

```rust
use ternary_grad::straight_through;

assert_eq!(straight_through(2.0, 0.5), 1);
assert_eq!(straight_through(-2.0, 0.5), -1);
assert_eq!(straight_through(0.1, 0.5), 0);   // within threshold → 0
assert_eq!(straight_through(0.5, 0.5), 0);   // at threshold → 0
```

### `straight_through_batch(xs, threshold) -> Vec<i8>`

Vectorized quantization for inference:

```rust
use ternary_grad::straight_through_batch;

let xs = vec![-2.0, -0.1, 0.1, 2.0];
assert_eq!(straight_through_batch(&xs, 0.5), vec![-1, 0, 0, 1]);
```

### `ste_gradient(x) -> f64`

The STE gradient function used during backpropagation:

```rust
use ternary_grad::ste_gradient;

assert_eq!(ste_gradient(0.5), 1.0);   // pass gradient through
assert_eq!(ste_gradient(0.0), 1.0);   // pass gradient through
assert_eq!(ste_gradient(2.0), 0.0);   // clip: weight is far from quantization point
assert_eq!(ste_gradient(-2.0), 0.0);  // clip
```

This creates a gradient "funnel": weights inside `[-1, 1]` receive full gradients and are pushed toward {-1, 0, +1}. Weights outside this range have their gradients clipped to zero, preventing them from drifting further away.

### `ternary_sgd_update(weights, grads, lr, _threshold)`

Standard SGD update on latent (float) weights. The threshold parameter is reserved for future STE-aware variants; currently this is standard SGD:

```rust
use ternary_grad::ternary_sgd_update;

let mut w = vec![0.8, -0.8];
let g = vec![1.0, -1.0];
ternary_sgd_update(&mut w, &g, 0.1, 0.5);
// w[0] decreases toward 0, w[1] increases toward 0
```

### `TernaryAdam`

Adam optimizer adapted for ternary training. Maintains first and second moment estimates with bias correction:

```rust
use ternary_grad::TernaryAdam;

let mut adam = TernaryAdam::new(0.05, 2);
let mut w = vec![5.0, -3.0];
let target = vec![1.0, -1.0];

for _ in 0..500 {
    let g: Vec<f64> = w.iter()
        .zip(target.iter())
        .map(|(&wi, &ti)| wi - ti)
        .collect();
    adam.step(&mut w, &g);
}
// w converges near [1.0, -1.0] — the ternary quantization points
```

Hyperparameters:
- `lr`: learning rate (default: provided at construction)
- `beta1`: first-moment decay (default: 0.9)
- `beta2`: second-moment decay (default: 0.999)
- `eps`: numerical stability (default: 1e-8)

### `clip_ternary_gradient(grads, max_val)`

Clamp gradient magnitudes to prevent exploding gradients during ternary training:

```rust
use ternary_grad::clip_ternary_gradient;

let mut g = vec![-10.0, -0.5, 0.5, 10.0];
clip_ternary_gradient(&mut g, 1.0);
assert_eq!(g, vec![-1.0, -0.5, 0.5, 1.0]);
```

### Learning Rate Schedules

**Cosine annealing** — smoothly decay LR following a cosine curve:

```rust
use ternary_grad::cosine_lr;

let lr0 = cosine_lr(0.1, 0, 100);    // max: 0.1
let lr50 = cosine_lr(0.1, 50, 100);  // ~0.05
let lr100 = cosine_lr(0.1, 100, 100);// ~0.0
```

**Step decay** — drop LR by factor `gamma` every `step_size` steps:

```rust
use ternary_grad::step_lr;

assert_eq!(step_lr(0.1, 0, 10, 0.5), 0.1);
assert_eq!(step_lr(0.1, 10, 10, 0.5), 0.05);
assert_eq!(step_lr(0.1, 20, 10, 0.5), 0.025);
```

### Training Diagnostics

**Quantization error** — L₂ distance between latent weights and their ternary quantization:

```rust
use ternary_grad::{quantization_error, straight_through};

let w = vec![-1.0, 0.0, 1.0];
assert_eq!(quantization_error(&w, 0.5), 0.0);  // already ternary

let w = vec![0.3];
assert!(quantization_error(&w, 0.5) > 0.0);   // 0.3 → 0, error = 0.09
```

**Ternary accuracy** — fraction of weights that quantize "cleanly" (within threshold of their target):

```rust
use ternary_grad::ternary_accuracy;

let w = vec![0.9, -0.9, 0.1, 1.1, -1.1];
let acc = ternary_accuracy(&w, 0.5);
// Weights near {-1, 0, +1} count as correct
```

### `ternary_weight_decay(weights, grads, wd)`

L₂ regularization applied to gradients (before quantization):

```rust
use ternary_grad::ternary_weight_decay;

let mut w = vec![1.0, -1.0];
let mut g = vec![0.1, 0.1];
ternary_weight_decay(&mut w, &mut g, 0.01);
// g becomes [0.11, 0.09] — pulls weights toward 0
```

---

## Usage Examples

### Complete Training Loop

```rust
use ternary_grad::{straight_through_batch, TernaryAdam, cosine_lr,
                   quantization_error, ternary_accuracy, clip_ternary_gradient};

let mut weights: Vec<f64> = (0..100)
    .map(|i| (i as f64 / 50.0) - 1.0)  // initialize in [-1, 1]
    .collect();

let mut optimizer = TernaryAdam::new(0.01, weights.len());
let total_steps = 1000;

for step in 0..total_steps {
    // Compute gradients (example: mean-squared error toward target)
    let target: Vec<f64> = weights.iter()
        .map(|&w| straight_through(w, 0.5) as f64)
        .collect();
    let mut grads: Vec<f64> = weights.iter()
        .zip(target.iter())
        .map(|(&w, &t)| w - t)
        .collect();

    // Clip gradients
    clip_ternary_gradient(&mut grads, 1.0);

    // Update with scheduled learning rate
    let lr = cosine_lr(0.01, step, total_steps);
    optimizer.lr = lr;
    optimizer.step(&mut weights, &grads);

    if step % 100 == 0 {
        let qe = quantization_error(&weights, 0.5);
        let acc = ternary_accuracy(&weights, 0.5);
        println!("Step {}: QE={:.4}, Acc={:.2}%", step, qe, acc * 100.0);
    }
}

// Final inference: quantize to ternary
let ternary_weights = straight_through_batch(&weights, 0.5);
println!("Final weights: {:?}", ternary_weights);
```

### Convergence Check with Adam

```rust
use ternary_grad::TernaryAdam;

let mut adam = TernaryAdam::new(0.05, 2);
let mut w = vec![5.0, -3.0];
let target = vec![1.0, -1.0];

for _ in 0..500 {
    let g: Vec<f64> = w.iter()
        .zip(target.iter())
        .map(|(&wi, &ti)| wi - ti)
        .collect();
    adam.step(&mut w, &g);
}

assert!((w[0] - 1.0).abs() < 0.5);
assert!((w[1] - (-1.0)).abs() < 0.5);
```

---

## Mathematical Background

### The Straight-Through Estimator

For a quantization function `q(w)` with zero derivative almost everywhere, STE defines a surrogate gradient:

```
∂L/∂w ≈ ∂L/∂q · g_ste(w)
```

where `g_ste(w)` is the STE gradient function. This crate uses:

```
g_ste(w) = 1  if |w| ≤ 1
           0  otherwise
```

This is the "clipped identity" STE. Alternatives include:
- **Full identity**: `g_ste(w) = 1` everywhere (risky; weights can diverge)
- **Tanh STE**: `g_ste(w) = 1 - tanh²(w)` (smooth variant)
- **Sigmoid STE**: used in some binary network implementations

The clipped identity is preferred for ternary because it creates a natural basin of attraction around [-1, 1], funneling weights toward quantization points.

### Adam in the Ternary Setting

Standard Adam (Kingma & Ba, 2015) maintains:

```
m_t = β₁·m_{t-1} + (1-β₁)·g_t          // first moment
v_t = β₂·v_{t-1} + (1-β₂)·g_t²         // second moment
m̂_t = m_t / (1-β₁^t)                   // bias correction
v̂_t = v_t / (1-β₂^t)
w_t = w_{t-1} - α · m̂_t / (√v̂_t + ε)
```

In ternary training, `g_t` is the STE-modified gradient. Adam's adaptive learning rates help because:
- Some weights converge quickly to {-1, 0, +1} and need small updates
- Others start far from quantization points and need larger initial steps
- The second moment estimates naturally reduce step sizes as weights stabilize

### Quantization Error and the Ternary Gap

Define the **ternary gap** as:

```
Δ(w) = min( |w - (-1)|, |w - 0|, |w - 1| )
```

The per-weight quantization error is `Δ(w)²`, and the mean across all weights is the `quantization_error()` metric. During training, we want:

```
lim_{t→∞} Δ(w_t) = 0   for all weights
```

The STE gradient, combined with an optimizer like Adam, empirically achieves this for most weights in a network.

---

## Research Connections

### BitNet and 1.58-Bit Neural Networks

Microsoft's BitNet (Ma et al., 2024) demonstrated that large language models can be trained with weights in {-1, 0, +1} from scratch, achieving competitive performance with full-precision models at dramatically reduced inference cost. The key insight: with sufficient scale, the quantization noise averages out across layers. This crate provides the foundational gradient machinery for such experiments.

### Ternary Neural Networks (TNNs)

Li et al. (2016) introduced Ternary Weight Networks (TWN), using learned scaling factors `α` such that `W ≈ α · W_ternary`. Zhu et al. (2017) extended this to Trained Ternary Quantization (TTQ) with learned asymmetric thresholds. The `quantization_error()` and `ternary_accuracy()` metrics in this crate map directly to the training objectives in these papers.

### Hardware Implications

Ternary weights enable specialized hardware:
- **Add-only MACs**: multiplication by {-1, 0, +1} reduces to addition, subtraction, or no-op
- **Sparse computation**: the 0 state means many operations can be skipped
- **Huawei's ternary chip**: 7nm ternary silicon demonstrated 60% power reduction vs. binary equivalents

### Connection to Active Inference

In the free energy principle, precision-weighted prediction errors drive learning. Ternary precision (high / medium / low) naturally maps to {-1, 0, +1} weight states. The `ternary-active-inference` crate explores this connection, using `ternary-grad` for learning precision parameters.

### Sparse Coding and Compressed Sensing

The 0-state in ternary weights induces structured sparsity. This connects to:
- **LASSO regression**: L1 regularization pushes weights toward zero
- **Compressed sensing**: sparse signal recovery from underdetermined systems
- **Neural architecture search**: ternary masks as structured pruning

---

## Ecosystem

- **ternary-tnn** — Ternary neural network layers (uses this for training)
- **ternary-llm** — Ternary language model (uses Adam from here)
- **ternary-attention** — Ternary attention mechanisms
- **ternary-cookbook** — Working demos and tutorials
- **ternary-belief** — Belief propagation on ternary factor graphs
- **ternary-free-energy** — Free energy minimization utilities

## References

- Bengio, Y., Léonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons for conditional computation. *arXiv:1308.3432*.
- Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR*.
- Li, F., et al. (2016). Ternary weight networks. *arXiv:1605.04711*.
- Zhu, C., et al. (2017). Trained ternary quantization. *ICLR*.
- Ma, S., et al. (2024). The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. *arXiv:2402.17764*.

## License

MIT
