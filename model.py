"""Char-level MLP language model small enough to be plausible as a DNA circuit.

Input is the one-hot context, which is literally how a DNA circuit sees text: the strand for
"char c at position i" is present or absent. fc1 on a one-hot input is a position-specific
embedding fused with the hidden layer, so the whole model is two weight matrices.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dna import DNAConfig, DNALinear, SeesawActivation

CTX = 6
VOCAB = list("abcdefghijklmnopqrstuvwxyz .,'!?\n")
V = len(VOCAB)
_FOLD = {";": ",", ":": ",", "-": " ", "&": " ", "$": " "}
DATA = Path(__file__).parent / "data" / "tinyshakespeare.txt"

FALLBACK_TEXT = (  # used only if the download is unavailable
    "to be, or not to be, that is the question:\nwhether 'tis nobler in the mind to suffer\n"
    "the slings and arrows of outrageous fortune,\nor to take arms against a sea of troubles\n"
    "and by opposing end them. to die, to sleep;\nno more; and by a sleep to say we end\n"
) * 400


def load_text() -> str:
    raw = DATA.read_text() if DATA.exists() else FALLBACK_TEXT
    out = []
    for ch in raw.lower():
        ch = _FOLD.get(ch, ch)
        out.append(ch if ch in VOCAB else " ")
    return "".join(out)


def encode(text: str) -> torch.Tensor:
    idx = {c: i for i, c in enumerate(VOCAB)}
    return torch.tensor([idx[c] for c in text], dtype=torch.long)


def splits(val_frac=0.1):
    ids = encode(load_text())
    n = int(len(ids) * (1 - val_frac))
    return ids[:n], ids[n:]


def windows(ids: torch.Tensor):
    """(contexts [N, CTX], targets [N])"""
    X = ids.unfold(0, CTX + 1, 1)
    return X[:, :CTX], X[:, CTX]


def one_hot(ctx_ids: torch.Tensor) -> torch.Tensor:
    return F.one_hot(ctx_ids, V).float().flatten(1)  # [N, CTX*V]


class CharMLP(nn.Module):
    def __init__(self, hidden=32, activation="relu", y_max=1.0):
        super().__init__()
        self.hidden, self.activation = hidden, activation
        self.fc1 = nn.Linear(CTX * V, hidden)
        self.fc2 = nn.Linear(hidden, V)
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "seesaw":  # train against DNA's own nonlinearity
            self.act = SeesawActivation(DNAConfig(threshold=0.0, gain=1.0, y_max=y_max))
        else:
            raise ValueError(activation)

    def forward(self, x):  # x: one-hot [N, CTX*V]
        return self.fc2(self.act(self.fc1(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def to_dna(model: CharMLP, cfg: DNAConfig, seed: int = 0, activation: str = "seesaw") -> nn.Module:
    """Copy of a trained model with both layers on the DNA substrate.

    seed selects the physical preparation (pipetting realisation); each layer gets its own.
    """
    dna = nn.Module()
    dna.fc1 = DNALinear(model.fc1, cfg, seed=2 * seed)
    dna.fc2 = DNALinear(model.fc2, cfg, seed=2 * seed + 1)
    dna.act = SeesawActivation(cfg) if activation == "seesaw" else nn.ReLU()
    dna.forward = lambda x: dna.fc2(dna.act(dna.fc1(x)))
    return dna


@torch.no_grad()
def perplexity(model, ctx_ids, targets, batch=16384) -> float:
    nll, n = 0.0, 0
    for i in range(0, len(targets), batch):
        logits = model(one_hot(ctx_ids[i:i + batch]))
        nll += F.cross_entropy(logits, targets[i:i + batch], reduction="sum").item()
        n += len(targets[i:i + batch])
    return math.exp(nll / n)


@torch.no_grad()
def calibrate_to_full_scale(model: CharMLP, ctx_ids, q=0.999):
    """Rescale a ReLU model so hidden activations sit in [0, ~1] full-scale concentration units.

    ReLU is positively homogeneous, so fc1/a and fc2*a is the identical function in FP32. It
    only chooses the units, so that y_max=1 means the same physical ceiling for every model.
    """
    h = model.act(model.fc1(one_hot(ctx_ids[:20000])))
    a = float(torch.quantile(h.flatten(), q))
    model.fc1.weight /= a; model.fc1.bias /= a
    model.fc2.weight *= a
    return a
