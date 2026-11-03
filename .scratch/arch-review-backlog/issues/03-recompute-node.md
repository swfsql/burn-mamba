# 03: One recompute node

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`. Not compiled.

**Files:** `src/mamba2/ssd/serial_recalculated/backward.rs:29-351`,
`src/mamba3/{double,single}_ssd/ssd/serial_recalculated/backward.rs`,
`../burn-stack/src/utils/combined_grad.rs:24-76`.

**Problem:** The three `backward.rs` hold 1,040 lines and no math. Normalised, 180 of
190 double-SSD lines occur in single-SSD, and 139 in Mamba-2. One input costs about
20 mentions. The author keeps `prepare([...])`, `ops.parents`, `State` and the const
`N` in order by hand. `47c7dd10` (one Burn interface change) edited all three.

**Direction:** One generic node module owns the prepare step, both `OpsKind` arms,
the pair flatten and split, and the gradient consume and register. Each plant keeps
its inputs, constants, inner forward call and `combined_backward` call.
`combined_grad` folds into the module.

**Open:** Which crate owns the module. Only these three files call `combined_grad`,
but burn-stack ADR-0001 names it as plumbing. Split this ticket by repo at triage.

**Cost:** `Backward<B, const N>` needs `Send + Debug + Sized + 'static`. A zero-size
plant type fits. A closure does not, because it is not `Debug`. Static dispatch adds
no run cost. The `UnTracked` arm runs `float_cat` with no gradient to serve, and one
module can remove that copy.

ADR-0007 is not affected, because the node holds no math.
