A narrow check after the review. It covers rows L40, L134 and L140.

Paths are relative to `mamba_ssm/`.

**Q1: accumulator, same form as SISO.** Additive update: `ops/tilelang/mamba3/mamba3_mimo_fwd.py:407` `T.gemm(k_state_frag, PsiV_shared, states_frag, ...clear_accum=False)`. `k_state_frag` is a copy of `k_shared` (392), which holds K scaled by `trap_scale_shared` (284-285). Scale: :196 `gamma_frag[cs] + shifted_gamma_shared[cs]`. :182 `dt_shifted_frag[cs] * T.sigmoid(-trap_shifted_frag[cs])` reads position t+1. It is 0.0 only at `S-1` (:181), so accumulator equals state at call end.

**Q2: no input state.** Kernel parameters (:70-90) have none. :159 `T.clear(states_frag)` starts from zero. `rg` for `input_`, `INIT` and similar: only :98 `Inputs:`. Module call (`modules/mamba3.py:185-205`) passes only `return_state=ssm_state is not None`. Later calls use `step` (:146-147).

Citation forms for the doc: `TIL/mamba3_mimo_fwd.py:407`, `TIL/mamba3_mimo_fwd.py:159`.
