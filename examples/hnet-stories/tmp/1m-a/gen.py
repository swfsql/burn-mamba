#!/usr/bin/env python3
"""Write cfg/<name>/{model,training}_config.json for an H-Net run.

usage: gen.py <name> k=v …   (the keys and their defaults: DEFAULTS)

A per-stage key takes one value for every stage, or one value per stage
joined by `/` (outermost first): `d=64/96`, `enc=2/1`. `S` is the number of
stages. The main network keys start with `m`.

- `lrm=auto`: the heuristic of the paper, λˢ = √(Dᴹ/Dˢ · Πᵢ₌ₛ Nⁱ), and 1 for
  the main network. `lrm=1/1` (S + 1 values) sets them.
- `smlp`/`mmlp`: the hidden width of a SwiGLU MLP (0: none).
- `mgr`/`mmgr`: Multi-Gate residual streams (0: the standard residual).
- `mV`: the virtual layers of the main network (0: none).
- `init=ref`: N(0, 0.02) with the residual rescale (the reference). `burn`:
  the defaults of Burn.
- The training keys are those of `../../../tiny-stories/tmp/1m-cased/gen.py`.
  `lr`, `warm` and `stages=…` write the LR sequence (`S:LR` a jump,
  `A-B:LR` a cosine anneal).
"""
import json
import math
import os
import sys

from lrseq import HERE, stage

DEFAULTS = dict(
    # stages (per-stage lists with `/`)
    S=1, d="64", enc="2", dec="2", r="32", p="16", e="2", sm="1", smlp="0",
    mgr="0", N="4/2", srot="Complex2D", su="1",
    # main network
    # muntie: the untied lists of mnist-class (Mamba3Config: the block tail and
    # the small per-head tensors; the layer: its pre-norms), held once per
    # virtual application. `auto` (user): on whenever the main network has
    # virtual layers. `0` / `1` force it.
    # su/mu: micro_steps (MambaProduct) of the stages / the main network.
    md=128, mL=3, mV=0, mr=64, mp=16, me=2, mm=1, mu=1, mmlp=0, mmgr=0,
    # msched: the virtual schedule of the main network: Cyclic, Stretched, or
    # the real layer of each application, `0,1,1,1,1,2` (Custom; mV values).
    mrot="Complex2D", muntie="auto", msched="Cyclic",
    # every block (`rot` is per stack: `srot`, `mrot`. Real1D: the
    # Mamba-2-like block, with no rotation)
    rf=1.0, rr=2.0, trap="HorizontalCarryOver", bias=1, onorm=1, mlpmul=32,
    # the network and the recipe
    # cmult: the inner length is a multiple of it (few distinct shapes, so
    # few JIT compilations). Keep it a multiple of the main SSD chunk.
    tie=1, start=4, init="ref", alpha=0.3, lrm="auto", cmult=64, smooth=64,
    # training
    bs=10, seq=1024, lr=4e-3, warm=128, stages="", wd=0.1, muon=1,
    stories=262144, vstories=256, epochs=1000, frontier="always", seed=0,
    workers=2, runlen=1,
    # buckets: the slots of length-bucketed training (`512,768,…`; empty:
    # none), each batch ~bchars scored chars. `seq`/`runlen` then do
    # nothing: the validation takes the slots too (each story whole). The
    # default is the protocol:
    # whole stories, full backprop (the longest story has 4149 chars). One
    # slot for every story up to 1024 chars (88% of them, 11 × 1024 tokens):
    # finer short slots group the stories by length, which cost +0.008
    # (B-c1024). The long slots stay fine, so a batch holds ≤ ~9.2k tokens.
    buckets="1024,1536,2048,3072,4160", bchars=7450,
)


def per_stage(a, key, kind):
    """The values of a per-stage key, one per stage."""
    vals = [kind(v) for v in str(a[key]).split("/")]
    if len(vals) == 1:
        vals = vals * a["S"]
    if len(vals) < a["S"]:
        sys.exit(f"{key}={a[key]}: {len(vals)} values for {a['S']} stages")
    return vals[: a["S"]]


def block(a, d, r, p, e, m, rot, untie=0, u=1):
    return {"Mamba3": {
        "d_model": d, "state_rank": r, "expand": e, "per_head_dim": p,
        "ngroups": 1, "mimo_rank": m, "micro_steps": u, "trapezoid": a["trap"],
        "gain": "Projected", "kalman_kappa_init": 0.01, "tropical": "None",
        "a_floor": 0.0001, "dt_min": 0.001, "dt_max": 0.1,
        "dt_init_floor": 0.0001, "dt_limit": [0.0, 65504.0],
        "has_proj_bias": bool(a["bias"]), "has_learnable_init_state": False,
        "rope_fraction": a["rf"], "has_outproj_norm": bool(a["onorm"]),
        "rotation": rot, "rotation_range": a["rr"],
        "siso_specialization": True, "siso_specialization_decode": True,
        "untied": ["InProjTail", "DtBias", "D", "BNorm", "CNorm", "OutNorm"]
        if untie else [],
    }}


def init(a):
    if a["init"] == "burn":
        return None
    return {"std": 0.02, "zero_bias": True,
            "residual_paths": ["out_proj.weight", "mlp.fc2.weight"],
            "residual_depth": None}


def schedule(sched, virtual, layers):
    """The JSON of a virtual schedule (see `msched`)."""
    if sched in ("Cyclic", "Stretched"):
        return sched
    idx = [int(i) for i in sched.split(",")]
    if len(idx) != virtual or not all(0 <= i < layers for i in idx):
        sys.exit(f"msched={sched}: needs {virtual} real layers in 0..{layers}")
    return {"Custom": idx}


def stack(a, d, layers, mlp, mgr, virtual=0, untie=0, sched="Cyclic"):
    return {
        "n_real_layers": layers,
        "n_virtual_layers": None if virtual in (0, layers) else [virtual, schedule(sched, virtual, layers)],
        "grad_horizon": None, "class_latents": [],
        "ignore_first_residual": False, "ignore_last_residual": False,
        "residuals": "Standard" if mgr == 0 else {"MultiGate": {
            "n_stream": mgr, "init_bias": 0.0, "init_bias_step": 0.0,
            "per_virtual_layer": False}},
        "mlp": None if mlp == 0 else {
            "d_model": d, "d_intermediate": mlp, "multiple_of": a["mlpmul"],
            "has_proj_bias": False},
        "untied": (["Norm"] + (["Norm2"] if mlp else [])) if untie else [],
        "init": init(a),
    }


def model(a):
    ds = per_stage(a, "d", int)
    enc, dec = per_stage(a, "enc", int), per_stage(a, "dec", int)
    rs, ps, es = per_stage(a, "r", int), per_stage(a, "p", int), per_stage(a, "e", int)
    ms, mlps = per_stage(a, "sm", int), per_stage(a, "smlp", int)
    mgrs, targets = per_stage(a, "mgr", int), per_stage(a, "N", float)
    rots, us = per_stage(a, "srot", str), per_stage(a, "su", int)
    stages = [{"encoder": stack(a, ds[s], enc[s], mlps[s], mgrs[s]),
               "decoder": stack(a, ds[s], dec[s], mlps[s], mgrs[s])}
              for s in range(a["S"])]
    muntie = a["mV"] not in (0, a["mL"]) if a["muntie"] == "auto" else bool(int(a["muntie"]))
    main = stack(a, a["md"], a["mL"], a["mmlp"], a["mmgr"], a["mV"], muntie, a["msched"])
    if a["lrm"] == "auto":
        lrm = [math.sqrt(a["md"] / ds[s] * math.prod(targets[s:])) for s in range(a["S"])] + [1.0]
    else:
        lrm = [float(v) for v in str(a["lrm"]).split("/")]
        if len(lrm) != a["S"] + 1:
            sys.exit(f"lrm={a['lrm']}: needs {a['S'] + 1} values")
    head_init = None if a["init"] == "burn" else {
        "std": 0.02, "zero_bias": True, "residual_paths": [], "residual_depth": None}
    return {
        "model": {
            "shape": {
                "hnet": {"stages": stages, "main": main,
                         "chunk_multiple": a["cmult"], "smooth_block": a["smooth"]},
                "vocab_size": 48, "pad_vocab_size_multiple": 1,
                "missing_lm_head": bool(a["tie"]), "head_init": head_init,
                "n_start_tokens": a["start"],
            },
            "stage_blocks": [block(a, ds[s], rs[s], ps[s], es[s], ms[s], rots[s], 0, us[s]) for s in range(a["S"])],
            "main_block": block(a, a["md"], a["mr"], a["mp"], a["me"], a["mm"], a["mrot"], muntie, a["mu"]),
        },
        "ratio_targets": targets,
        "ratio_weight": a["alpha"],
        "lr_multipliers": lrm,
    }


def training(a):
    muon = None if not a["muon"] else {
        "weight_decay": {"penalty": a["wd"]},
        "momentum": {"momentum": 0.95, "dampening": 0.0, "nesterov": True},
        "ns_coefficients": [3.4445, -4.775, 2.0315], "epsilon": 1e-7,
        "ns_steps": 5, "adjust_lr_fn": "MatchRmsAdamW"}
    seq = [{"from": 0, "lr": {"Linear": {"start": 0.0, "end": a["lr"], "steps": a["warm"]}}}]
    for st in (st for st in a["stages"].split(",") if st):
        span, lr = st.split(":")
        start, segment = stage(seq, span, float(lr))
        if start <= seq[-1]["from"]:
            sys.exit(f"the stages must start in order: {a['stages']}")
        seq.append(segment)
    frontier = "Always" if a["frontier"] in ("always", 0) else {
        "Bits": {"max_bits": float(a["frontier"])}}
    return {
        "training": {
            "optimizer": {
                "fallback": {"AdamW": {
                    "grad_clipping": {"Value": 1.0}, "beta_1": 0.9,
                    "beta_2": 0.999, "epsilon": 8.1584695e-8,
                    "weight_decay": a["wd"], "cautious_weight_decay": True,
                    "amsgrad": False}},
                "muon": muon},
            "num_epochs": a["epochs"], "batch_size": a["bs"],
            "num_workers": a["workers"],
            "lr": {"Sequence": seq},
            "seed": a["seed"]},
        "seq_len": a["seq"], "run_len": a["runlen"],
        "frontier": frontier, "train_stories": a["stories"],
        "valid_stories": a["vstories"], "sample_chars": 400,
        "sample_temperature": 0.8, "pack": None,
        "bucket": None if not a["buckets"] else {
            "slots": [int(s) for s in a["buckets"].split(",")],
            "chars": a["bchars"], "probe": 0},
    }


def parse(v):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return None if v == "None" else v


def main():
    name, kv = sys.argv[1], sys.argv[2:]
    a = dict(DEFAULTS)
    for item in kv:
        k, v = item.split("=", 1)
        if k not in a:
            sys.exit(f"unknown key {k}")
        # A string default (a per-stage list, `trap`, `rot`, …) stays a string.
        a[k] = v if isinstance(DEFAULTS[k], str) else parse(v)
    out = os.path.join(HERE, "cfg", name)
    os.makedirs(out, exist_ok=True)
    for fname, obj in (("model_config.json", model(a)),
                       ("training_config.json", training(a))):
        with open(os.path.join(out, fname), "w") as f:
            json.dump(obj, f, indent=2)
    with open(os.path.join(out, "args.txt"), "w") as f:
        f.write(" ".join(kv) + "\n")
    print(out)


if __name__ == "__main__":
    main()
