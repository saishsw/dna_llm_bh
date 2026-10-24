"""Train the FP32 silicon baselines. Run: python train.py

Trains a small family (hidden width x activation) so the sweep has a real model-size axis.
Writes ckpt/*.pt and results/baselines.csv.
"""
import csv
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import CTX, V, CharMLP, calibrate_to_full_scale, one_hot, perplexity, splits, windows

HIDDEN = [4, 8, 16, 32, 64]
ACTS = ["relu", "seesaw"]
MAIN = (32, "relu")         # the model the headline numbers use
STEPS, BATCH, LR = 8000, 1024, 3e-3
ROOT = Path(__file__).parent


def train_one(hidden, act, Xtr, Ytr, seed=0):
    torch.manual_seed(seed)
    model = CharMLP(hidden, act)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        i = torch.randint(0, len(Ytr), (BATCH,), generator=g)
        loss = F.cross_entropy(model(one_hot(Xtr[i])), Ytr[i])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model


def main():
    torch.manual_seed(0)
    (ckpt := ROOT / "ckpt").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    tr, va = splits()
    Xtr, Ytr = windows(tr)
    Xva, Yva = windows(va)
    print(f"vocab {V}, context {CTX}, train {len(Ytr):,} / held-out {len(Yva):,} chars")
    print(f"uniform ppl = {V}")
    rows, t0 = [], time.time()
    for act in ACTS:
        for h in HIDDEN:
            t = time.time()
            m = train_one(h, act, Xtr, Ytr)
            scale = calibrate_to_full_scale(m, Xtr) if act == "relu" else 1.0
            ppl = perplexity(m, Xva, Yva)
            torch.save({"state": m.state_dict(), "hidden": h, "activation": act},
                       ckpt / f"mlp_h{h}_{act}.pt")
            rows.append({"hidden": h, "activation": act, "params": m.n_params(),
                         "fp32_ppl": round(ppl, 4), "relu_scale": round(scale, 4)})
            print(f"h={h:3d} {act:6s} params={m.n_params():6d}  FP32 held-out ppl={ppl:.3f}"
                  f"  ({time.time() - t:.1f}s)")
    with open(ROOT / "results" / "baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    main = next(r for r in rows if (r["hidden"], r["activation"]) == MAIN)
    print(f"\nMAIN baseline h={MAIN[0]} {MAIN[1]}: FP32 ppl {main['fp32_ppl']} "
          f"({main['params']} params). total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
