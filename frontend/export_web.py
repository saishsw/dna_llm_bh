"""Export trained weights, a held-out sample and sweep results for the browser explorer.

Run from BASES_Project/: python frontend/export_web.py  -> frontend/data.js
Also writes reference perplexities (PyTorch) that frontend/check_sim.js must reproduce.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from dna import DNAConfig, reset_counters  # noqa: E402
from model import CTX, VOCAB, CharMLP, encode, load_text, one_hot, splits, to_dna, windows  # noqa: E402
from train import HIDDEN  # noqa: E402

N_SAMPLE = 20000
OUT = ROOT / "frontend"


def load(h, act):
    ck = torch.load(ROOT / "ckpt" / f"mlp_h{h}_{act}.pt")
    m = CharMLP(h, act); m.load_state_dict(ck["state"]); m.eval()
    return m


def flat(lin):
    W = torch.cat([lin.weight, lin.bias[:, None]], 1)
    return [float(f"{v:.6g}") for v in W.flatten().tolist()]


def read(name):
    with open(ROOT / "results" / name) as f:
        return list(csv.DictReader(f))


@torch.no_grad()
def ref_ppl(model, cfg, act, X, Y):
    dna = to_dna(model, cfg, 0, act)
    reset_counters(dna)
    return math.exp(F.cross_entropy(dna(one_hot(X)), Y).item())


def main():
    va = splits()[1]
    Xs, Ys = windows(va[:N_SAMPLE + CTX])
    sample = "".join(VOCAB[i] for i in va[:N_SAMPLE + CTX].tolist())
    models, base = {}, {r["hidden"] + "_" + r["activation"]: r for r in read("baselines.csv")}
    for h in HIDDEN:
        for act in ("relu", "seesaw"):
            m = load(h, act)
            models[f"h{h}_{act}"] = {"hidden": h, "trained": act, "params": m.n_params(),
                                    "fp32_ppl_heldout": float(base[f"{h}_{act}"]["fp32_ppl"]),
                                    "W1": flat(m.fc1), "W2": flat(m.fc2)}

    # reference perplexities on the exported sample, deterministic configs only (sigma = 0)
    I = DNAConfig().ideal()
    refs = []
    for h, tr, act, cfg in [
        (32, "relu", "relu", I),
        (32, "relu", "seesaw", replace(DNAConfig(), sigma=0.0)),
        (32, "seesaw", "seesaw", replace(DNAConfig(), sigma=0.0, n_levels=8, leak_rate=1e-2)),
        (16, "relu", "seesaw", replace(DNAConfig(), sigma=0.0, fuel_budget=1e3)),
        (64, "seesaw", "seesaw", replace(DNAConfig(), sigma=0.0, n_levels=4, leak_rate=3e-3,
                                         threshold=0.05, gain=1.5)),
    ]:
        c = {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in asdict(cfg).items()}
        refs.append({"model": f"h{h}_{tr}", "act": act, "cfg": c,
                     "ppl": ref_ppl(load(h, tr), cfg, act, Xs, Ys)})

    # leak cliff curves: mean / sd over preparations
    curves = defaultdict(lambda: defaultdict(list))
    for r in read("sweep_axes.csv"):
        if r["axis"] == "leak_rate" and r["architecture"] == "bulk" and float(r["leak_rate"]) > 0:
            curves[f'{r["center"]}|{r["trained"]}|{r["act"]}'][float(r["leak_rate"])].append(float(r["ppl"]))
    leak = {k: [[x, float(np.mean(v)), float(np.std(v))] for x, v in sorted(d.items())]
            for k, d in curves.items()}

    # frontier cloud: grid configs averaged over preparations, bulk, seesaw act
    cloud = defaultdict(list)
    for r in read("sweep_grid.csv"):
        if r["architecture"] == "bulk" and r["act"] == "seesaw":
            key = (int(r["hidden"]), r["trained"], r["sigma"], r["leak_rate"], r["n_levels"])
            cloud[key].append((float(r["ppl"]), float(r["E_chem"]), float(r["E_wallplug"]),
                               float(r["E_amortised"]), r["feasible"] == "True"))
    grid = [{"h": k[0], "trained": k[1], "sigma": float(k[2]), "leak": float(k[3]),
             "levels": int(k[4]) if k[4] else None,
             "ppl": float(np.mean([v[0] for v in vs])), "E_chem": vs[0][1], "E_wallplug": vs[0][2],
             "E_amortised": vs[0][3], "feasible": vs[0][4]} for k, vs in cloud.items()]

    fuel = defaultdict(list)
    for r in read("sweep_fuel.csv"):
        fuel[r["fuel_budget"]].append([int(r["tokens_served"]), float(r["window_ppl"])])

    cfg = {k: (None if isinstance(v, float) and math.isinf(v) else v) for k, v in asdict(DNAConfig()).items()}
    data = {"vocab": "".join(VOCAB), "ctx": CTX, "sample": sample, "models": models,
            "config": cfg, "summary": json.loads((ROOT / "results" / "summary.json").read_text()),
            "leak_curves": leak, "grid": grid, "fuel": fuel, "reference": refs}
    js = "window.DNA_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (OUT / "data.js").write_text(js)
    print(f"data.js {len(js) / 1e6:.2f} MB, {len(models)} models, {len(grid)} grid points")
    for r in refs:
        print(f"  ref {r['model']:12s} {r['act']:6s} ppl {r['ppl']:.5f}")


if __name__ == "__main__":
    main()
