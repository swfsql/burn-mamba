# Mamba-specific code stays in burn-mamba

burn-stack holds the containers around a block (layers, stacks, networks,
caches, optimizer plans). It serves more than one family: burn-mamba and
burn-deltanet use it. So burn-stack *code* is block-agnostic by default. It has
no family name and no family-specific shape assumption. An item that only one
family needs goes into the crate of that family.

burn-stack *docs* can name a family as a citation: the origin of a mechanism, a
reference implementation, or a consumer.

## Consequences

- An exception is rare. A mechanism that more than one family uses can have a
  family-specific setting in burn-stack. The code names it after the
  mechanism, not after the family, and its doc says why it is in burn-stack.
- Every feature of this crate forwards to burn-stack. The `backend-*` cfgs take
  effect where `burn_stack::impl_backend_ext_for_burn_backends!` expands, which
  is in this crate. A backend that only one side adds loses its `BackendExt`
  impls without an error.
