"""Figures from the sweep CSVs. Run: python plot.py  -> results/frontier.png, results/leak_cliff.png"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dna import DNAConfig

OUT = Path(__file__).parent / "results"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SERIES = {"E_chem": ("#2a78d6", "o", "E_chem: strand displacement only"),
          "E_wallplug": ("#eb6834", "s", "E_wallplug: + plate reader, batch 1"),
          "E_amortised": ("#1baf7a", "^", "E_amortised: reader shared over 6e13 molecules")}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def read(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def realistic(r):
    return r["sigma"] == "0.1" and r["leak_rate"] == "0.001" and r["n_levels"] == "16"


def frontier():
    rows = [r for r in read("sweep_grid.csv") if r["act"] == "seesaw" and r["architecture"] == "bulk"]
    summ = json.loads((OUT / "summary.json").read_text())
    cfg = DNAConfig()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.grid(True, which="major", color=GRID, lw=0.8); ax.set_axisbelow(True)

    for key, (col, mk, label) in SERIES.items():
        feas = [r for r in rows if r["feasible"] == "True"]
        infe = [r for r in rows if r["feasible"] != "True"]
        ax.scatter([float(r[key]) for r in feas], [float(r["ppl"]) for r in feas], s=14,
                   marker=mk, color=col, alpha=0.25, lw=0)
        ax.scatter([float(r[key]) for r in infe], [float(r["ppl"]) for r in infe], s=14,
                   marker=mk, facecolor="none", edgecolor=col, alpha=0.5, lw=0.7)
        # realistic-constant frontier across model width, seesaw-trained, mean over seeds
        pts = defaultdict(list)
        for r in rows:
            if realistic(r) and r["trained"] == "seesaw":
                pts[int(r["hidden"])].append((float(r[key]), float(r["ppl"]), r["feasible"] == "True"))
        hs = sorted(pts)
        xs = [np.mean([p[0] for p in pts[h]]) for h in hs]
        ys = [np.mean([p[1] for p in pts[h]]) for h in hs]
        ok = [pts[h][0][2] for h in hs]
        n_ok = sum(ok)
        ax.plot(xs[:n_ok], ys[:n_ok], "-", color=col, lw=2, marker=mk, ms=8,
                mec=SURFACE, mew=1.5, label=label, zorder=4)
        ax.plot(xs[n_ok - 1:], ys[n_ok - 1:], "--", color=col, lw=2, zorder=4)
        ax.plot(xs[n_ok:], ys[n_ok:], mk, color=col, ms=8, mfc=SURFACE, mew=1.5, zorder=5)
        if key == "E_chem":
            for h, x, y in zip(hs, xs, ys):
                ax.annotate(f"h={h}", (x, y), xytext=(6, 4), textcoords="offset points",
                            fontsize=8, color=INK2)

    main = summ["realistic_main"]["bulk"]
    land = main["E_landauer"]
    ax.axvline(land, color=INK, lw=1, ls=":")
    ax.text(land * 1.4, 13.6, f"Landauer: {main['ops_per_token']:.0f} ops x kT ln2\n= {land:.1e} J/token",
            fontsize=8, color=INK2, va="top")
    base = summ["fp32_ppl"]["relu"]
    ax.axhline(base, color=INK, lw=1, ls="-.")
    ax.text(2e-17, base * 0.975, f"FP32 silicon baseline (main model h=32, ReLU), ppl {base:.2f}", fontsize=8,
            color=INK2, va="top")
    gpu = summ["gpu_same_model_J_per_token"]
    ax.plot([gpu], [base], marker="*", ms=14, color=INK, zorder=6, ls="none",
            label="GPU, this model, ideal utilisation (est.)")
    lo, hi = cfg.gpu_J_per_token_70B
    ax.axvspan(lo, hi, color=MUTED, alpha=0.15, lw=0)
    ax.text(np.sqrt(lo * hi), 5.9, "GPU, 70B LLM\nJ/token reference\n(different model)",
            fontsize=8, color=INK2, ha="center", va="bottom")
    wp, am = main["E_wallplug"], main["E_amortised"]
    ax.annotate("", xy=(am * 2, 12.5), xytext=(wp / 2, 12.5),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1.2))
    ax.text(np.sqrt(am * wp), 12.8, "batch amortisation: reader energy / 6e13", ha="center",
            fontsize=8, color=INK2)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-17, 1e7); ax.set_ylim(5.8, 14)
    ax.set_yticks([6, 7, 8, 9, 10, 12, 14]); ax.set_yticklabels(["6", "7", "8", "9", "10", "12", "14"])
    ax.minorticks_off()
    ax.set_xlabel("energy per token (J), simulated, literature constants, bulk settling")
    ax.set_ylabel("held-out perplexity (char-level)")
    fig.suptitle("DNA strand-displacement LM: perplexity vs joules per token", x=0.01, y=0.985,
                 ha="left", fontsize=12, color=INK)
    ax.text(0, 1.015, "Lines: seesaw-trained models h=4..64 at realistic constants (sigma .10, leak 1e-3, "
            "16 levels), mean of 3 preparations.\nFaint: all grid configs. Hollow + dashed: exceeds "
            "1e4 orthogonal domains (infeasible, extrapolation).", transform=ax.transAxes,
            fontsize=8, color=INK2, va="bottom")
    ax.legend(loc="center", fontsize=8, frameon=False, bbox_to_anchor=(0.6, 0.47))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "frontier.png", dpi=160)


def leak_cliff():
    rows = [r for r in read("sweep_axes.csv") if r["axis"] == "leak_rate" and r["architecture"] == "bulk"]
    summ = json.loads((OUT / "summary.json").read_text())
    combos = [("relu", "relu", "#2a78d6", "o", "ReLU-trained, ReLU"),
              ("relu", "seesaw", "#eb6834", "s", "ReLU-trained, seesaw swapped in"),
              ("seesaw", "seesaw", "#1baf7a", "^", "seesaw-trained, seesaw")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, center, title in zip(axes, ("ideal", "realistic"),
                                 ("leak only (all other effects off)",
                                  "leak + sigma .10 + 16 levels (5 preparations, +/-1 sd)")):
        ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        ax.axvspan(1e-3, 1e-2, color=MUTED, alpha=0.15, lw=0)
        for trained, act, col, mk, label in combos:
            g = defaultdict(list)
            for r in rows:
                if r["center"] == center and r["trained"] == trained and r["act"] == act \
                        and float(r["leak_rate"]) > 0:
                    g[float(r["leak_rate"])].append(float(r["ppl"]))
            xs = sorted(g)
            m = np.array([np.mean(g[x]) for x in xs]); s = np.array([np.std(g[x]) for x in xs])
            ax.fill_between(xs, np.clip(m - s, 5, 1e5), np.clip(m + s, 5, 1e5), color=col, alpha=0.15, lw=0)
            ax.plot(xs, m, color=col, lw=2, marker=mk, ms=5, mec=SURFACE, label=label)
            k = summ["leak_knee"][f"{center}/{trained}-trained/{act}"]["knee_leak_rate_+10pct"]
            if k:
                ax.axvline(k, color=col, lw=1, ls=":")
        k = summ["leak_knee"][f"{center}/relu-trained/seesaw"]["knee_leak_rate_+10pct"]
        ax.text(k * 0.85, 300, f"knee (+10% ppl)\n~{k:.1e}", fontsize=8, color=INK2, ha="right")
        ax.text(3.2e-3, 25, "literature\nleak range", fontsize=8, color=INK2, ha="center")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(5, 1e5)
        ax.set_xlabel("leak rate (fraction of gate concentration)")
        ax.set_title(title, fontsize=10, loc="left")
    axes[0].set_ylabel("held-out perplexity")
    axes[0].legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0, 0.62))
    fig.suptitle("The leak cliff: flat, then collapse", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "leak_cliff.png", dpi=160)


if __name__ == "__main__":
    frontier()
    leak_cliff()
    print("wrote results/frontier.png, results/leak_cliff.png")
