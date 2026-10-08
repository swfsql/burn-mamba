#!/usr/bin/env python3
"""Write cfg/<name>/{model,training}_config.json from shortcut key=value args.

usage: gen.py <name> [key=value ...]   (see SHORT for keys; unknown keys abort)

The LR schedule is an LR sequence (lrseq.py): a `Linear` warmup from 0 to
`lr` over `warm` steps (the ramp then holds `lr`), and a segment for each
step-down of `stages`. "S:LR" is a jump to LR at S. "A-B:LR" is an anneal: a
cosine from the rate at A to LR at B, which then holds LR. For example
"4630:1e-3,8000-12000:5e-4".
`decay.py` writes a decay branch from a step, and `stage.py` a step-down.
"""
import json, os, sys

from lrseq import stage

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = dict(
    # model
    d=32, L=2, V=8, r=64, e=4, p=32, g=1, m=1, u=1, rot="Complex2D", rf=1.0,
    rr=2.0, bias=1, onorm=1, mgr=4, mlp=0, mlpmul=32, tie=1, cls=4,
    trap="HorizontalCarryOver", gain="Projected", trop="None",
    # untie=1: mnist-class's untied list (block tail + small per-head tensors,
    # the layer pre-norms), held once per virtual application.
    # mgrpv=1: one MultiGate module per virtual layer. sched: the virtual
    # schedule (Cyclic | Stretched).
    untie=0, mgrpv=0, sched="Cyclic",
    # training
    bs=8, seq=256, lr=12e-3, warm=128, stages="", wd=1e-4,
    # epochs: large, so a run never ends on it: runs end by -b / --max-seconds
    # (both stop mid-epoch, resumable with --resume), the LR schedule is explicit.
    muon=1, stories=4096, vstories=256, epochs=1000, frontier=1.6, seed=0,
    workers=2, runlen=18446744073709551615,
)


def model(a):
    block = {
        "d_model": a["d"], "state_rank": a["r"], "expand": a["e"],
        "per_head_dim": a["p"], "ngroups": a["g"], "mimo_rank": a["m"],
        "micro_steps": a["u"], "trapezoid": a["trap"], "gain": a["gain"],
        "kalman_kappa_init": 0.01, "tropical": a["trop"], "a_floor": 0.0001,
        "dt_min": 0.001, "dt_max": 0.1, "dt_init_floor": 0.0001,
        "dt_limit": [0.0, 65504.0], "has_proj_bias": bool(a["bias"]),
        "has_learnable_init_state": False, "rope_fraction": a["rf"],
        "has_outproj_norm": bool(a["onorm"]), "rotation": a["rot"],
        "rotation_range": a["rr"], "siso_specialization": True,
        "siso_specialization_decode": True,
        "untied": ["InProjTail", "DtBias", "D", "BNorm", "CNorm", "OutNorm"]
        if a["untie"] else [],
    }
    residuals = "Standard" if a["mgr"] == 0 else {"MultiGate": {
        "n_stream": a["mgr"], "init_bias": 0.0, "init_bias_step": 0.0,
        "per_virtual_layer": bool(a["mgrpv"])}}
    mlp = None if a["mlp"] == 0 else {
        "d_model": a["d"], "d_intermediate": a["mlp"],
        "multiple_of": a["mlpmul"], "has_proj_bias": False}
    return {"Mamba3": {
        "n_real_layers": a["L"],
        "n_virtual_layers": None if a["V"] in (0, a["L"]) else [a["V"], a["sched"]],
        "grad_horizon": None, "vocab_size": 48, "pad_vocab_size_multiple": 1,
        "mamba_block": block, "missing_lm_head": bool(a["tie"]),
        "class_latents": ["Start"] * a["cls"], "ignore_first_residual": False,
        "ignore_last_residual": False, "residuals": residuals, "mlp": mlp,
        "untied": (["Norm"] + (["Norm2"] if a["mlp"] else [])) if a["untie"] else [],
    }}


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
        "sample_temperature": 0.8,
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
        # A string default (`trap`, `trop`, `rot`, …) keeps `None` a string.
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
