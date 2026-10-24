"""Sweep DNA substrate constants over the trained models. Run: python sweep.py

Writes:
  results/sweep_axes.csv  one-at-a-time sweeps of sigma / leak / n_levels on the main model
  results/sweep_grid.csv  factorial grid over every model, the frontier data
  results/sweep_fuel.csv  fuel depletion: a reused tube over a stream of held-out tokens
  results/summary.json    leak knee, seesaw cost, fuel, headline numbers
Each pipetting seed is an independent physical preparation of the tube.
"""
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dna import DNAConfig, capacity_report, energy_report, reset_counters
from model import CTX, V, CharMLP, one_hot, splits, to_dna, windows
from train import HIDDEN, MAIN

ROOT = Path(__file__).parent
OUT = ROOT / "results"
SEEDS = 5
REALISTIC = DNAConfig()                        # sigma .10, leak 1e-3, 16 levels
IDEAL = DNAConfig().ideal()
AXES = {
    "sigma": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    "leak_rate": [0.0] + [float(x) for x in np.logspace(-5, 0, 21)],
    "n_levels": [None, 256, 64, 32, 16, 8, 6, 4, 3, 2],
}
GRID = {"sigma": [0.0, 0.05, 0.10, 0.15], "leak_rate": [0.0, 1e-3, 3e-3, 1e-2],
        "n_levels": [None, 16, 8]}
GRID_SEEDS = 3
FUEL_BUDGETS = [1e2, 3e2, 1e3, 3e3, 1e4, 3e4, math.inf]  # full-scale releases per neuron
FUEL_TOKENS, FUEL_WINDOW = 60000, 2000


def load(hidden, act):
    ck = torch.load(ROOT / "ckpt" / f"mlp_h{hidden}_{act}.pt")
    m = CharMLP(hidden, act); m.load_state_dict(ck["state"]); m.eval()
    return m


Xva, Yva = windows(splits()[1])
XOH = one_hot(Xva)                              # [N, CTX*V], built once


@torch.no_grad()
def evaluate(model, cfg, seed, act):
    dna = to_dna(model, cfg, seed, act)
    reset_counters(dna)
    nll = sum(F.cross_entropy(dna(XOH[i:i + 32768]), Yva[i:i + 32768], reduction="sum").item()
              for i in range(0, len(Yva), 32768))
    ppl = math.exp(nll / len(Yva))
    cap = capacity_report(dna, CTX * V, cfg)
    rows = []
    for arch in ("bulk", "localised"):
        e = energy_report(dna, replace(cfg, architecture=arch))
        rows.append({"ppl": ppl, "architecture": arch, **e,
                     "nats_per_J_wallplug": (math.log(V) - math.log(ppl)) / e["E_wallplug"],
                     "nats_per_J_amortised": (math.log(V) - math.log(ppl)) / e["E_amortised"],
                     "n_strands": cap["n_strands"], "n_weight_gates": cap["n_weight_gates"],
                     "dual_rail_tax": cap["dual_rail_tax"], "feasible": cap["feasible"],
                     "feasible_1e3": cap["n_strands"] <= 1e3})
    return rows


def run(points, path):
    rows, t0 = [], time.time()
    for p in points:
        cfg = replace(p.pop("base"), sigma=p["sigma"], leak_rate=p["leak_rate"],
                      n_levels=p["n_levels"])
        for r in evaluate(p.pop("model"), cfg, p["seed"], p["act"]):
            rows.append({**p, **r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"  {len(rows)} rows -> {path.name}  ({time.time() - t0:.0f}s)")
    return rows


@torch.no_grad()
def fuel_sweep(model):
    """One tube, fuelled once, serving held-out tokens in order. Rolling ppl over windows."""
    rows, out = [], {}
    for budget in FUEL_BUDGETS:
        curves = []
        for s in range(GRID_SEEDS):
            dna = to_dna(model, replace(REALISTIC, fuel_budget=budget), s, "seesaw")
            reset_counters(dna)
            nll = torch.cat([F.cross_entropy(dna(XOH[i:i + 4096]), Yva[i:i + 4096], reduction="none")
                             for i in range(0, FUEL_TOKENS, 4096)])[:FUEL_TOKENS]
            curves.append(nll.view(-1, FUEL_WINDOW).mean(1).exp().numpy())
        mean = np.mean(curves, axis=0)
        for w, p in enumerate(mean):
            rows.append({"fuel_budget": budget, "tokens_served": (w + 1) * FUEL_WINDOW, "window_ppl": p})
        out[budget] = mean
    ref = out[math.inf]
    summ = {}
    for budget in FUEL_BUDGETS[:-1]:
        bad = np.nonzero(out[budget] > 1.1 * ref)[0]
        summ[str(budget)] = {
            "tokens_until_+10pct_ppl": int(bad[0] * FUEL_WINDOW) if len(bad) else f">{FUEL_TOKENS}",
            "ppl_last_window": float(out[budget][-1]), "ppl_last_window_full_fuel": float(ref[-1])}
    with open(OUT / "sweep_fuel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"  {len(rows)} rows -> sweep_fuel.csv")
    return summ


def knee(xs, ys, y0, frac=0.10):
    """First leak (log-interpolated) where perplexity exceeds y0*(1+frac)."""
    target = y0 * (1 + frac)
    for (x0, a), (x1, b) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if a <= target < b:
            t = (target - a) / (b - a)
            return float(10 ** (math.log10(x0) + t * (math.log10(x1) - math.log10(x0))))
    return None


def stats(rows, **match):
    v = [r["ppl"] for r in rows if r["architecture"] == "bulk"
         and all(r[k] == val for k, val in match.items())]
    return float(np.mean(v)), float(np.std(v)), len(v)


def main():
    torch.manual_seed(0)
    OUT.mkdir(exist_ok=True)
    main_h, main_act = MAIN
    m_relu, m_see = load(main_h, "relu"), load(main_h, "seesaw")
    fp32 = {k: math.exp(F.cross_entropy(m(XOH), Yva).item())
            for k, m in (("relu", m_relu), ("seesaw", m_see))}

    # --- one-at-a-time axes around IDEAL (isolates each effect) and REALISTIC ------------
    print("axes sweep")
    pts = []
    for center_name, center in (("ideal", IDEAL), ("realistic", REALISTIC)):
        for axis, vals in AXES.items():
            for v in vals:
                for trained, act in (("relu", "relu"), ("relu", "seesaw"), ("seesaw", "seesaw")):
                    for s in range(SEEDS):
                        base = center if act == "relu" else (
                            replace(center, y_max=1.0) if center is IDEAL else center)
                        p = {"center": center_name, "axis": axis, "hidden": main_h,
                             "trained": trained, "act": act, "seed": s,
                             "sigma": center.sigma, "leak_rate": center.leak_rate,
                             "n_levels": center.n_levels}
                        p[axis] = v
                        pts.append({**p, "base": base,
                                    "model": m_relu if trained == "relu" else m_see})
    axes = run(pts, OUT / "sweep_axes.csv")

    # --- factorial grid over every model: the frontier ------------------------------------
    print("grid sweep")
    pts = []
    for h in HIDDEN:
        models = {"relu": load(h, "relu"), "seesaw": load(h, "seesaw")}
        for trained, act in (("relu", "relu"), ("relu", "seesaw"), ("seesaw", "seesaw")):
            for sg in GRID["sigma"]:
                for lk in GRID["leak_rate"]:
                    for nl in GRID["n_levels"]:
                        for s in range(GRID_SEEDS):
                            pts.append({"hidden": h, "trained": trained, "act": act, "seed": s,
                                        "sigma": sg, "leak_rate": lk, "n_levels": nl,
                                        "base": REALISTIC, "model": models[trained]})
    run(pts, OUT / "sweep_grid.csv")

    # --- headline analysis ----------------------------------------------------------------
    summ = {"fp32_ppl": fp32, "main_model": {"hidden": main_h, "params": m_relu.n_params()}}
    leak_vals = AXES["leak_rate"][1:]
    summ["leak_knee"] = {}
    for center in ("ideal", "realistic"):
        for trained, act in (("relu", "relu"), ("relu", "seesaw"), ("seesaw", "seesaw")):
            sel = dict(center=center, axis="leak_rate", trained=trained, act=act)
            y0 = stats(axes, leak_rate=0.0, **sel)[0]
            ys = [stats(axes, leak_rate=l, **sel)[0] for l in leak_vals]
            summ["leak_knee"][f"{center}/{trained}-trained/{act}"] = {
                "ppl_at_zero_leak": y0, "knee_leak_rate_+10pct": knee(leak_vals, ys, y0),
                "knee_leak_rate_x2": knee(leak_vals, ys, y0, frac=1.0)}
    summ["seesaw_cost"] = {}
    for center in ("ideal", "realistic"):
        sel = dict(center=center, axis="sigma", sigma=(0.0 if center == "ideal" else 0.10))
        r = stats(axes, trained="relu", act="relu", **sel)
        s = stats(axes, trained="relu", act="seesaw", **sel)
        t = stats(axes, trained="seesaw", act="seesaw", **sel)
        summ["seesaw_cost"][center] = {
            "relu": r, "seesaw_swapped_in": s, "seesaw_trained": t,
            "swap_cost_ppl": s[0] - r[0], "trained_cost_ppl": t[0] - r[0]}
    real = [r for r in axes if r["center"] == "realistic" and r["axis"] == "sigma"
            and r["sigma"] == 0.10 and r["trained"] == "relu" and r["act"] == "seesaw"]
    summ["realistic_main"] = {arch: {k: float(np.mean([r[k] for r in real if r["architecture"] == arch]))
                                     for k in ("ppl", "E_chem", "E_wallplug", "E_amortised",
                                               "E_landauer", "ops_per_token", "s_per_token_latency",
                                               "s_per_token_amortised", "n_strands",
                                               "dual_rail_tax", "nats_per_J_amortised")}
                              for arch in ("bulk", "localised")}
    summ["gpu_same_model_J_per_token"] = 2 * m_relu.n_params() * REALISTIC.gpu_J_per_flop
    # copies_per_inference sensitivity: analog concentrations need many molecules per weight
    # to beat shot noise, which shrinks the batch and multiplies the chemistry.
    summ["copies_sensitivity_E_amortised"] = {
        arch: {f"{c:.0e}": v["E_chem"] * c + (v["E_wallplug"] - v["E_chem"]) * c / REALISTIC.n_parallel
               for c in (1, 1e2, 1e3, 1e4, 1e6)}
        for arch, v in summ["realistic_main"].items()}
    print("fuel sweep")
    summ["fuel_depletion"] = fuel_sweep(m_relu)
    (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
