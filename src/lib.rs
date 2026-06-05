//! # ternary-grad
//! Ternary gradient descent: straight-through estimation, ternary optimizers.
//! For training {-1, 0, +1} neural networks.

/// Straight-through estimator: forward = sign, backward = identity
pub fn straight_through(x: f64, threshold: f64) -> i8 {
    if x > threshold { 1 }
    else if x < -threshold { -1 }
    else { 0 }
}

/// Batch straight-through for a vector
pub fn straight_through_batch(xs: &[f64], threshold: f64) -> Vec<i8> {
    xs.iter().map(|&x| straight_through(x, threshold)).collect()
}

/// STE gradient: d(sign(x))/dx ≈ 1 if |x| ≤ 1, else 0
pub fn ste_gradient(x: f64) -> f64 {
    if x.abs() <= 1.0 { 1.0 } else { 0.0 }
}

/// Ternary weight update with STE: update latent weights directly, STE applies to quantization.
pub fn ternary_sgd_update(weights: &mut [f64], grads: &[f64], lr: f64, _threshold: f64) {
    for (w, &g) in weights.iter_mut().zip(grads.iter()) {
        *w -= lr * g;
    }
}

/// Ternary Adam optimizer state
#[derive(Debug, Clone)]
pub struct TernaryAdam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub m: Vec<f64>,  // first moment
    pub v: Vec<f64>,  // second moment
    pub t: usize,     // timestep
}

impl TernaryAdam {
    pub fn new(lr: f64, size: usize) -> Self {
        Self {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            m: vec![0.0; size],
            v: vec![0.0; size],
            t: 0,
        }
    }

    pub fn step(&mut self, weights: &mut [f64], grads: &[f64]) {
        assert_eq!(weights.len(), grads.len());
        assert_eq!(weights.len(), self.m.len());
        self.t += 1;
        for i in 0..weights.len() {
            let g = grads[i];
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t as i32));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t as i32));
            weights[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

/// Gradient clipping in trit space: clamp gradient magnitude
pub fn clip_ternary_gradient(grads: &mut [f64], max_val: f64) {
    for g in grads.iter_mut() {
        *g = g.clamp(-max_val, max_val);
    }
}

/// Cosine annealing learning rate schedule
pub fn cosine_lr(base_lr: f64, current_step: usize, total_steps: usize) -> f64 {
    let progress = current_step as f64 / total_steps as f64;
    base_lr * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
}

/// Step LR: decay by gamma every step_size steps
pub fn step_lr(base_lr: f64, current_step: usize, step_size: usize, gamma: f64) -> f64 {
    let decay = gamma.powi((current_step / step_size) as i32);
    base_lr * decay
}

/// Ternary weight decay: L2 regularization scaled for ternary
pub fn ternary_weight_decay(weights: &mut [f64], grads: &mut [f64], wd: f64) {
    for (g, w) in grads.iter_mut().zip(weights.iter()) {
        *g += wd * w;
    }
}

/// Quantization error: L2 distance between float weights and their ternary quantization
pub fn quantization_error(weights: &[f64], threshold: f64) -> f64 {
    weights.iter()
        .map(|&w| {
            let q = straight_through(w, threshold) as f64;
            (w - q).powi(2)
        })
        .sum::<f64>()
        / weights.len() as f64
}

/// Ternary accuracy: fraction of weights correctly quantized
pub fn ternary_accuracy(weights: &[f64], threshold: f64) -> f64 {
    let correct = weights.iter()
        .filter(|&&w| {
            let q = straight_through(w, threshold);
            (w - q as f64).abs() < threshold
        })
        .count();
    correct as f64 / weights.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ste_positive() { assert_eq!(straight_through(2.0, 0.5), 1); }
    #[test]
    fn ste_negative() { assert_eq!(straight_through(-2.0, 0.5), -1); }
    #[test]
    fn ste_zero() { assert_eq!(straight_through(0.1, 0.5), 0); }
    #[test]
    fn ste_exact_threshold() { assert_eq!(straight_through(0.5, 0.5), 0); }

    #[test]
    fn ste_batch() {
        let xs = vec![-2.0, -0.1, 0.1, 2.0];
        let result = straight_through_batch(&xs, 0.5);
        assert_eq!(result, vec![-1, 0, 0, 1]);
    }

    #[test]
    fn ste_gradient_near_zero() {
        assert_eq!(ste_gradient(0.5), 1.0);
        assert_eq!(ste_gradient(0.0), 1.0);
    }

    #[test]
    fn ste_gradient_far_from_zero() {
        assert_eq!(ste_gradient(2.0), 0.0);
        assert_eq!(ste_gradient(-2.0), 0.0);
    }

    #[test]
    fn sgd_update_moves_toward_zero() {
        // Weights within [-1,1] where STE gradient = 1
        let mut w = vec![0.8, -0.8];
        let g = vec![1.0, -1.0];
        ternary_sgd_update(&mut w, &g, 0.1, 0.5);
        assert!(w[0] < 0.8);
        assert!(w[1] > -0.8);
    }

    #[test]
    fn adam_converges() {
        let mut adam = TernaryAdam::new(0.05, 2);
        let mut w = vec![5.0, -3.0];
        let target = vec![1.0, -1.0];
        for _ in 0..500 {
            let g: Vec<f64> = w.iter().zip(target.iter()).map(|(&wi, &ti)| wi - ti).collect();
            adam.step(&mut w, &g);
        }
        assert!((w[0] - 1.0).abs() < 0.5, "w[0]={}", w[0]);
        assert!((w[1] - (-1.0)).abs() < 0.5, "w[1]={}", w[1]);
    }

    #[test]
    fn gradient_clipping() {
        let mut g = vec![-10.0, -0.5, 0.5, 10.0];
        clip_ternary_gradient(&mut g, 1.0);
        assert_eq!(g, vec![-1.0, -0.5, 0.5, 1.0]);
    }

    #[test]
    fn cosine_lr_decreases() {
        let lr0 = cosine_lr(0.1, 0, 100);
        let lr50 = cosine_lr(0.1, 50, 100);
        let lr100 = cosine_lr(0.1, 100, 100);
        assert!(lr0 > lr50);
        assert!(lr50 > lr100);
        assert!(lr100 >= 0.0); // cosine reaches 0 at t=T
    }

    #[test]
    fn step_lr_decay() {
        assert_eq!(step_lr(0.1, 0, 10, 0.5), 0.1);
        assert_eq!(step_lr(0.1, 10, 10, 0.5), 0.05);
        assert_eq!(step_lr(0.1, 20, 10, 0.5), 0.025);
    }

    #[test]
    fn quantization_error_zero_for_ternary() {
        let w = vec![-1.0, 0.0, 1.0];
        assert_eq!(quantization_error(&w, 0.5), 0.0);
    }

    #[test]
    fn quantization_error_positive_for_float() {
        let w = vec![0.3];
        assert!(quantization_error(&w, 0.5) > 0.0);
    }
}
