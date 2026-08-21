//! Detaching a module's parameters from the autodiff graph.
//!
//! Cuts gradient flow through a module. **This does not save memory** — see the
//! warning below; [`Layers::grad_horizon`](crate::modules::Layers::grad_horizon)
//! moves its no-grad prefix to the inner backend instead.
//!
//! # Why detaching the *activations* is not enough
//!
//! Burn tracks gradients **per tensor**, not ambiently: there is no
//! `torch.no_grad()`. An op is untracked only when *every* input node has
//! `Requirement::None`, so `layer.forward(x.detach())` still builds a full graph
//! — the layer's own weights are tracked `Param`s, and one tracked input is
//! enough. Untracked ops, by contrast, are registered with **unit state**, so no
//! activation is retained: that is where the memory saving comes from.
//!
//! Cutting gradient flow through a prefix therefore takes *both* halves:
//! [`detach_params`] on a **clone** of the module (the original keeps its
//! tracked params, which is what lets a weight-shared layer receive gradient
//! from its tracked applications only), and `.detach()` on every tensor entering
//! the prefix from outside.
//!
//! # Warning: this frees no memory
//!
//! An untracked op is still **registered in the graph** — Burn keeps an
//! `UntrackedOpsStep` per op so that a memory-bound op with an untracked parent
//! can still retrieve it — so its output stays retained. Measured on a
//! 64-virtual-layer `Layers<Mamba3>` (flex, `d_model` 128, batch 4, seq 512),
//! peak RSS with a 2-layer gradient horizon was 3144 MB with a detached prefix
//! against 208 MB with an inner-backend one, and only the latter is flat in
//! depth. Detaching buys the gradient semantics and essentially none of the
//! memory.
//!
//! # What this does *not* reach
//!
//! [`Module::map`](burn::module::Module::map) is a **no-op on plain `Tensor` fields** — Burn implements
//! `Module for Tensor` as a constant (`map` returns `self`), so only
//! [`Param`](burn::module::Param)-wrapped tensors go through the mapper. Weights
//! are `Param`s and are covered; bare tensors (notably the caches) are not.

use burn::module::{Module, ModuleMapper, Param};
use burn::prelude::*;

/// Clears `require_grad` and cuts the graph on every parameter it visits.
struct DetachParams;

impl ModuleMapper for DetachParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        // `set_require_grad` fixes the *leaf* requirement (and, via
        // `Param::from_mapped_value`, the param's own `require_grad` flag, which
        // is re-read from the tensor); `detach` additionally re-roots the tensor
        // so nothing upstream of it can be reached. Both are documented no-ops
        // off the autodiff backend, so this is safe on every backend.
        Param::from_mapped_value(id, tensor.set_require_grad(false).detach(), mapper)
    }
}

/// Return `module` with every parameter detached from the autodiff graph.
///
/// Call it on a **clone**: the detached copy is what the no-grad part runs, while
/// the original keeps the tracked parameters the back-propagated part needs.
/// Under weight sharing both refer to the same weights, and the gradient then
/// accumulates from the tracked applications alone.
///
/// A no-op in effect when the module is not on an autodiff backend — and, per the
/// module warning, **not** a way to save memory.
pub fn detach_params<T: Module>(module: T) -> T {
    module.map(&mut DetachParams)
}
