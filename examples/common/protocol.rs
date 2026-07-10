//! Task-free plumbing of the **full-batch grokking protocol** shared by the
//! `grokking` and `state-tracking` examples: chunkwise/stepwise final-position
//! logits over a `MambaVocabNet`, accuracy, a cheap device-side grad-NaN
//! guard, and the PR console/CSV formatting. The task-specific pieces
//! (dataset, config, the training loop itself) stay in each example.

use super::diagnostics::{StatePr, WeightPr};
use burn::module::{ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::prelude::*;
use burn_mamba::prelude::*;

/// The SSD path used by chunkwise forwards: the recompute-backward serial
/// algorithm (the memory-saving custom backward), with the family following
/// the model and `chunk_len` supplied by the task (grokking passes `Some(2)`
/// matching its two-token sequences; `None` = the library optimum).
pub fn ssd_path(model: &MambaVocabNet, chunk_len: Option<usize>) -> MambaSsdPath {
    match model {
        MambaVocabNet::Mamba3(_) => {
            MambaSsdPath::Mamba3(Mamba3SsdPath::SerialRecalculated(chunk_len))
        }
        _ => MambaSsdPath::Mamba2(Mamba2SsdPath::SerialRecalculated(chunk_len)),
    }
}

/// Final-position logits `[n, vocab]` for a batch of token sequences `[n, s]`,
/// either chunkwise (`forward()`) or token-by-token (`step()`; identical by
/// the library's parity contract).
pub fn final_logits(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    stepwise: bool,
    chunk_len: Option<usize>,
) -> Tensor<2> {
    let [_b, s] = inputs_bs.dims();
    if stepwise {
        let mut caches = None;
        let mut logits = None;
        for t in 0..s {
            let x_b = inputs_bs.clone().narrow(1, t, 1).squeeze_dim::<1>(1);
            let (logits_bc, new_caches) = model.step(x_b, caches, None, None);
            caches = Some(new_caches);
            logits = Some(logits_bc);
        }
        logits.expect("at least one token")
    } else {
        let (logits_bsc, _caches) =
            model.forward(inputs_bs.clone(), None, ssd_path(model, chunk_len));
        logits_bsc.narrow(1, s - 1, 1).squeeze_dim::<2>(1)
    }
}

/// [`final_logits`] via the chunkwise path, additionally returning each
/// (virtual) layer's **attached** state moments — the state-PR penalty input
/// (gradients flow through the moments into the model).
pub fn final_logits_with_moments(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    chunk_len: Option<usize>,
) -> (Tensor<2>, Vec<StateMoments>) {
    let [_b, s] = inputs_bs.dims();
    let (logits_bsc, _caches, moments) =
        model.forward_with_state_moments_grad(inputs_bs.clone(), None, ssd_path(model, chunk_len));
    (logits_bsc.narrow(1, s - 1, 1).squeeze_dim::<2>(1), moments)
}

/// Fraction of examples whose final-position argmax matches the label.
pub fn accuracy(
    model: &MambaVocabNet,
    inputs_bs: &Tensor<2, Int>,
    labels_b: &Tensor<1, Int>,
    stepwise: bool,
    chunk_len: Option<usize>,
) -> f64 {
    let logits_bc = final_logits(model, inputs_bs, stepwise, chunk_len);
    let [b, _classes] = logits_bc.dims();
    let pred_b = logits_bc.argmax(1).reshape([b]);
    scalar_f32(pred_b.equal(labels_b.clone()).float().mean()) as f64
}

/// Read a single-element float tensor back to the host.
pub fn scalar_f32(t: Tensor<1>) -> f32 {
    t.into_data().to_vec::<f32>().unwrap()[0]
}

/// `1` if a gradient tensor holds any NaN or Inf, else `0` — kept device-side
/// so many can be summed with a single host sync.
fn grad_bad_indicator<const D: usize>(g: Tensor<D>) -> Tensor<1> {
    g.clone().is_nan().any().int().float() + g.is_inf().any().int().float()
}

/// Visitor that sums [`grad_bad_indicator`] over every parameter whose gradient
/// is present in `grads` — one device-side scalar, so the per-step healthy path
/// costs a single sync.
struct GradBadCount<'a> {
    grads: &'a GradientsParams,
    acc: Option<Tensor<1>>,
}

impl ModuleVisitor for GradBadCount<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let Some(g) = self.grads.get::<D>(param.id) else {
            return;
        };
        let bad = grad_bad_indicator(g);
        self.acc = Some(match self.acc.take() {
            Some(acc) => acc + bad,
            None => bad,
        });
    }
}

/// Number of parameters with a non-finite (NaN/Inf) gradient in `grads`.
pub fn nonfinite_grad_count(model: &MambaVocabNet, grads: &GradientsParams) -> f32 {
    let mut v = GradBadCount { grads, acc: None };
    model.visit(&mut v);
    v.acc.map(scalar_f32).unwrap_or(0.0)
}

/// Visitor that records the first parameter (in visitation order) whose gradient
/// is non-finite — its rank/shape and id, enough to locate the matrix. Only run
/// on the failing step (per-parameter host syncs).
struct FirstBadGrad<'a> {
    grads: &'a GradientsParams,
    found: Option<String>,
}

impl ModuleVisitor for FirstBadGrad<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        if self.found.is_some() {
            return;
        }
        let Some(g) = self.grads.get::<D>(param.id) else {
            return;
        };
        let nan = scalar_f32(g.clone().is_nan().any().int().float()) > 0.0;
        let inf = scalar_f32(g.is_inf().any().int().float()) > 0.0;
        if nan || inf {
            self.found = Some(format!(
                "dims={:?} id={:?} (nan={nan}, inf={inf})",
                param.val().dims(),
                param.id
            ));
        }
    }
}

/// The first parameter with a non-finite gradient, if any (see [`FirstBadGrad`]).
pub fn first_bad_grad(model: &MambaVocabNet, grads: &GradientsParams) -> Option<String> {
    let mut v = FirstBadGrad { grads, found: None };
    model.visit(&mut v);
    v.found
}

/// Compact console form of the diagnostics (centered PRs are the primary
/// read-outs).
pub fn format_prs(state_prs: &[StatePr], weight_prs: &WeightPr) -> String {
    let states: Vec<String> = state_prs
        .iter()
        .map(|r| {
            format!(
                "L{}H{} pooled {:.2} (m{:.1e}), final {:.2} (m{:.1e})",
                r.layer, r.head, r.pooled_centered, r.pooled_trace, r.final_centered, r.final_trace
            )
        })
        .collect();
    let blocks: Vec<String> = weight_prs
        .layers
        .iter()
        .map(|l| {
            format!(
                "L{} z {:.1}, x {:.1}, B {:.1}, C {:.1}, out {:.1}, B-alpha {:.1}",
                l.layer, l.z, l.x, l.b, l.c, l.out, l.b_alphabet
            )
        })
        .collect();
    format!(
        "state PR [{}] | weight PR emb {:.2}, head {:.2}, emb-freq {:.2} | block [{}]",
        states.join("; "),
        weight_prs.emb,
        weight_prs.lm_head,
        weight_prs.emb_freq,
        blocks.join("; "),
    )
}

/// Append one metrics row, creating the file with a header on first use.
pub fn append_metrics(
    path: &std::path::Path,
    step: usize,
    lr: f64,
    train_loss: f32,
    train_acc: f64,
    test_acc: f64,
    weight_prs: &WeightPr,
) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the metrics csv");
    if needs_header {
        writeln!(file, "step,lr,train_loss,train_acc,test_acc,emb_pr,head_pr,emb_freq_pr")
            .expect("failed csv header write");
    }
    writeln!(
        file,
        "{step},{lr},{train_loss},{train_acc},{test_acc},{},{},{}",
        weight_prs.emb, weight_prs.lm_head, weight_prs.emb_freq,
    )
    .expect("failed csv write");
}

/// [`append_metrics`] without diagnostics: the weight-PR columns are written
/// as `nan` so the file keeps one schema either way.
pub fn append_metrics_bare(
    path: &std::path::Path,
    step: usize,
    lr: f64,
    train_loss: f32,
    train_acc: f64,
    test_acc: f64,
) {
    let nan = WeightPr {
        emb: f64::NAN,
        lm_head: f64::NAN,
        emb_freq: f64::NAN,
        layers: Vec::new(),
    };
    append_metrics(path, step, lr, train_loss, train_acc, test_acc, &nan);
}

/// Append the per-(layer, head) state-PR rows, creating the file with a
/// header on first use.
pub fn append_pr(path: &std::path::Path, step: usize, state_prs: &[StatePr]) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the pr csv");
    if needs_header {
        writeln!(
            file,
            "step,layer,head,pooled_centered,pooled_uncentered,final_centered,final_uncentered,pooled_trace,final_trace"
        )
        .expect("failed csv header write");
    }
    for r in state_prs {
        writeln!(
            file,
            "{step},{},{},{},{},{},{},{},{}",
            r.layer, r.head, r.pooled_centered, r.pooled_uncentered, r.final_centered, r.final_uncentered, r.pooled_trace, r.final_trace,
        )
        .expect("failed csv write");
    }
}

/// Append the per-layer block-weight PR rows, creating the file with a header
/// on first use.
pub fn append_weight_pr(path: &std::path::Path, step: usize, weight_prs: &WeightPr) {
    use std::io::Write as _;
    let needs_header = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("failed to open the weights csv");
    if needs_header {
        writeln!(file, "step,layer,z,x,b,c,out,b_alphabet").expect("failed csv header write");
    }
    for l in &weight_prs.layers {
        writeln!(
            file,
            "{step},{},{},{},{},{},{},{}",
            l.layer, l.z, l.x, l.b, l.c, l.out, l.b_alphabet,
        )
        .expect("failed csv write");
    }
}
