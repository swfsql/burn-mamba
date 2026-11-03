# 05: Delete the backend traits with no bound site

Status: needs-triage

Blocked by: None

From the architecture review of 2026-11-03. Line numbers are those of commit
`42ca3d0e`.

**Files:** `src/mamba3/mod.rs:94-114`, `src/mamba2/ssd/serial_recalculated/serial_recalculated.rs:396`,
`src/mamba3/double_ssd/ssd/serial_recalculated/serial_recalculated.rs:155`,
`src/mamba3/single_ssd/ssd/serial_recalculated/serial_recalculated.rs:200`,
`src/mamba3/quat_scan/quat_scan.rs:224`.

**Problem:** 9 traits sit at this seam, but only 2 adapters are real: the default
body and `Autodiff`. Every plain impl is empty. Five public traits have no bound site
in `src`, `examples`, `benches` or burn-stack: 4 `…AutodiffBackendExt`, and
`Mamba3BackendExt`.

**Direction:** Delete the five traits. If nothing else calls
`decl_autodiff_backend_ext!`, delete it from burn-stack too. Before that, ask the
user whether a project outside these two repos names it.

**Also worth trying:** `#[extension_type]` for the optional inputs. Today they pass
as a placeholder plus a `bool` (`Restarts::op_inputs`, `seed_bnhpr` with `seeded`).

No memory or speed cost. The burn-stack half of the backend list is in
`../burn-stack/.scratch/arch-review-backlog/issues/01-backend-list.md`.
