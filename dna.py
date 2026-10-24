"""DNA strand-displacement substrate for PyTorch.

Wraps trained nn.Linear layers so they compute the way a seesaw / strand-displacement
network would: non-negative dual-rail concentrations, finite pipetting resolution,
log-normal pipetting error, signal-independent leak, and seesaw threshold-amplify-saturate
nonlinearity. Every physical constant lives in DNAConfig.

All constants are literature-derived defaults (see DNAConfig below and README), not measurements.
"""
from dataclasses import dataclass, field, replace
from typing import Optional
import math

import torch
import torch.nn as nn

K_B = 1.380649e-23  # J/K


@dataclass
class DNAConfig:
    # --- substrate non-idealities (applied in this order) -------------------------------
    n_levels: Optional[int] = 16       # pipetting resolution per rail; None = continuous. 16 ~ 4 bit
    sigma: float = 0.10                # log-normal pipetting error, w*exp(N(0,sigma)). lit 0.05-0.15
    leak_rate: float = 1e-3            # spurious displacement, fraction of gate conc. lit 1e-3-1e-2

    # --- seesaw gate transfer function: clamp(gain*(x - threshold), 0, y_max) ----------
    threshold: float = 0.0
    gain: float = 1.0
    y_max: float = 1.0                 # fuel-limited output, in full-scale concentration units

    # --- fuel depletion (effect 6), for a tube reused across a stream of tokens -----------
    fuel_budget: float = math.inf      # full-scale releases one neuron's fuel supports; inf = off
    fuel_chunk: int = 64               # tokens processed between fuel updates

    # --- thermodynamics ------------------------------------------------------------------
    temperature_K: float = 300.0
    E_op_kT: float = 20.0              # free energy per strand-displacement op, in kT
    ops_per_weight: float = 2.0        # gate displacement + rail annihilation, per active gate
    ops_per_neuron: float = 3.0        # threshold, amplify, reporter/fuel, per seesaw neuron

    # --- timing --------------------------------------------------------------------------
    architecture: str = "bulk"         # "bulk" (diffusion-limited) or "localised" (origami)
    settle_bulk_s: float = 3600.0      # per layer, diffusion limited
    settle_localised_s: float = 60.0   # per layer, Chatterjee et al. 2017

    # --- readout & parallelism -----------------------------------------------------------
    readout_power_W: float = 50.0      # fluorescence plate reader
    readout_time_s: Optional[float] = None  # None = reader runs for the full settling time
    n_parallel: float = 6e13           # molecules in 100 uL at 1 uM
    copies_per_inference: float = 1.0  # molecules of each species one inference consumes

    # --- capacity -------------------------------------------------------------------------
    max_orthogonal_domains: float = 1e4  # orthogonal sequence library, lit 1e3-1e4
    strict_capacity: bool = False        # True: raise on infeasible; False: flag only

    # --- silicon references (plot only) ---------------------------------------------------
    gpu_J_per_token_70B: tuple = (0.1, 1.0)
    gpu_J_per_flop: float = 1e-11      # FP32 at full utilisation, ~H100 class; optimistic for GPU

    @property
    def kT(self) -> float:
        return K_B * self.temperature_K

    @property
    def landauer_J(self) -> float:
        return self.kT * math.log(2)

    @property
    def E_op_J(self) -> float:
        return self.E_op_kT * self.kT

    @property
    def settle_s(self) -> float:
        if self.architecture == "bulk":
            return self.settle_bulk_s
        if self.architecture == "localised":
            return self.settle_localised_s
        raise ValueError(f"unknown architecture {self.architecture!r}")

    def ideal(self) -> "DNAConfig":
        """Same constants, every non-ideality switched off. Must reproduce silicon exactly."""
        return replace(self, n_levels=None, sigma=0.0, leak_rate=0.0, threshold=0.0,
                       gain=1.0, y_max=math.inf, fuel_budget=math.inf)


class InfeasibleConfig(RuntimeError):
    pass


def quantise_rail(w: torch.Tensor, n_levels: Optional[int], full_scale: float) -> torch.Tensor:
    """Snap a non-negative rail to n_levels evenly spaced concentrations in [0, full_scale]."""
    if n_levels is None or full_scale == 0:
        return w
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    step = full_scale / (n_levels - 1)
    return torch.round(w / step) * step


class DNALinear(nn.Module):
    """A trained nn.Linear realised as dual-rail strand-displacement weight gates.

    The bias is treated as a gate driven by a constant unit-concentration source strand, so it
    is split, quantised, noised and leaks exactly like the weights.
    """

    def __init__(self, linear: nn.Linear, cfg: DNAConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.in_features, self.out_features = linear.in_features, linear.out_features
        W = linear.weight.detach().clone()
        b = linear.bias.detach().clone() if linear.bias is not None else torch.zeros(self.out_features)
        self.register_buffer("W_trained", torch.cat([W, b[:, None]], dim=1))  # [out, in+1]
        self.reset_counters()
        self.prepare(seed)

    # ---- the tube: build it once per seed, like one physical preparation ----------------
    def prepare(self, seed: int):
        cfg, W = self.cfg, self.W_trained
        # 1. dual rail
        W_plus, W_minus = W.clamp(min=0), (-W).clamp(min=0)
        # 2. concentration quantisation, shared full scale across both rails
        full_scale = float(torch.maximum(W_plus.max(), W_minus.max()))
        W_plus = quantise_rail(W_plus, cfg.n_levels, full_scale)
        W_minus = quantise_rail(W_minus, cfg.n_levels, full_scale)
        self.n_weight_gates = int((W_plus > 0).sum() + (W_minus > 0).sum())
        # 3. log-normal multiplicative pipetting error, independently per pipetted species
        if cfg.sigma > 0:
            g = torch.Generator().manual_seed(seed)
            W_plus = W_plus * torch.exp(cfg.sigma * torch.randn(W_plus.shape, generator=g))
            W_minus = W_minus * torch.exp(cfg.sigma * torch.randn(W_minus.shape, generator=g))
        self.register_buffer("W_plus", W_plus, persistent=False)
        self.register_buffer("W_minus", W_minus, persistent=False)
        # 4. leak floor: proportional to total gate concentration feeding each output
        self.register_buffer("leak", cfg.leak_rate * (W_plus + W_minus).sum(dim=1), persistent=False)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = torch.cat([x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype)], dim=-1)
        y = x1 @ (self.W_plus - self.W_minus).T
        if self.cfg.leak_rate > 0:
            y = y + self.leak
        # 5. energy and time
        n_tok = x.numel() // x.shape[-1]
        self.tokens += n_tok
        self.ops += n_tok * self.n_weight_gates * self.cfg.ops_per_weight
        self.seconds += self.cfg.settle_s
        return y

    # ---- accounting ----------------------------------------------------------------------
    def reset_counters(self):
        self.tokens, self.ops, self.seconds = 0, 0.0, 0.0

    def strand_count(self) -> dict:
        """Distinct species this layer needs in the tube."""
        rails = 2 * self.out_features        # y+ and y- output strands
        annihilators = self.out_features     # y+ + y- -> waste, the subtraction
        return {"weight_gates": self.n_weight_gates, "rail_strands": rails,
                "annihilators": annihilators,
                "dual_rail_tax": self.out_features + annihilators,  # vs a signed substrate
                "total": self.n_weight_gates + rails + annihilators}


class SeesawActivation(nn.Module):
    """DNA's native nonlinearity: threshold, amplify, saturate at the fuel-limited maximum."""

    def __init__(self, cfg: DNAConfig):
        super().__init__()
        self.cfg = cfg
        self.reset_counters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        y = torch.clamp(c.gain * (x - c.threshold), min=0.0)
        if math.isfinite(c.fuel_budget):
            y = self._deplete(y)
        else:
            y = y.clamp(max=c.y_max)
        n_tok = x.numel() // x.shape[-1]
        self.tokens += n_tok
        self.ops += n_tok * x.shape[-1] * c.ops_per_neuron
        return y

    def _deplete(self, y: torch.Tensor) -> torch.Tensor:
        """Tokens arrive in order along dim 0. Each release burns fuel; the ceiling a neuron can
        reach shrinks with the fuel left: y <= y_max * (1 - consumed / budget)."""
        c, out = self.cfg, []
        if self.consumed is None:
            self.consumed = torch.zeros(y.shape[-1])
        for blk in y.split(c.fuel_chunk, dim=0):
            left = (1 - self.consumed / c.fuel_budget).clamp(min=0)
            blk = torch.minimum(blk, c.y_max * left)
            self.consumed = self.consumed + blk.sum(dim=0)
            out.append(blk)
        return torch.cat(out)

    def refuel(self):
        self.consumed = None

    def reset_counters(self):
        self.tokens, self.ops = 0, 0.0
        self.refuel()

    def strand_count(self, n_neurons: int) -> int:
        return 3 * n_neurons  # threshold gate, amplifier gate, fuel


# ---- model-level helpers -----------------------------------------------------------------

def dna_layers(model):
    return [m for m in model.modules() if isinstance(m, DNALinear)]


def seesaw_layers(model):
    return [m for m in model.modules() if isinstance(m, SeesawActivation)]


def reset_counters(model):
    for m in model.modules():
        if hasattr(m, "reset_counters"):
            m.reset_counters()


def capacity_report(model, n_input_species: int, cfg: DNAConfig) -> dict:
    layers = dna_layers(model)
    per = [l.strand_count() for l in layers]
    n_hidden = sum(l.out_features for l in layers[:-1])
    total = n_input_species + sum(p["total"] for p in per) + 3 * n_hidden
    rep = {"n_strands": total,
           "n_weight_gates": sum(p["weight_gates"] for p in per),
           "dual_rail_tax": sum(p["dual_rail_tax"] for p in per),
           "max_orthogonal_domains": cfg.max_orthogonal_domains,
           "feasible": total <= cfg.max_orthogonal_domains}
    if not rep["feasible"] and cfg.strict_capacity:
        raise InfeasibleConfig(f"needs {total} distinct strands > library of "
                               f"{cfg.max_orthogonal_domains:.0f} orthogonal domains")
    return rep


def energy_report(model, cfg: DNAConfig) -> dict:
    """Per-token energy and time from the counters accumulated since the last reset."""
    layers = dna_layers(model)
    tokens = layers[0].tokens
    if tokens == 0:
        raise RuntimeError("no forward passes recorded")
    ops = (sum(l.ops for l in layers) + sum(s.ops for s in seesaw_layers(model))) / tokens
    ops *= cfg.copies_per_inference
    latency = len(layers) * cfg.settle_s                      # layers settle sequentially
    readout_s = latency if cfg.readout_time_s is None else cfg.readout_time_s
    E_chem = ops * cfg.E_op_J
    E_read = cfg.readout_power_W * readout_s
    batch = cfg.n_parallel / cfg.copies_per_inference
    return {"ops_per_token": ops,
            "E_chem": E_chem,
            "E_wallplug": E_chem + E_read,                    # batch 1: reader serves one token
            "E_amortised": E_chem + E_read / batch,           # chemistry is per inference, never shared
            "E_landauer": ops * cfg.landauer_J,              # same op count at kT ln2 each
            "s_per_token_latency": latency,
            "s_per_token_amortised": latency / batch}
