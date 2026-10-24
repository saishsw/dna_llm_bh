---
name: dna-llm-sim
description: Build and extend the DNA molecular-computing language model simulator for the BASES hackathon. Use whenever work touches the DNA substrate layer, seesaw-gate activations, strand-displacement energy accounting, the perplexity-vs-joules sweep, or any claim about what DNA computing can and cannot do. Covers the physics constants, the modelling rules, the honesty guardrails, and the argument the demo must support.
---

# DNA LLM Simulator

## The thesis this code exists to support

Full argument: `CASE_FOR_DNA.md` in the project root. Read it before making any claim about what DNA computing can or cannot do, and before writing the README's framing.

Among substrates that are genuinely biological, DNA is the only one where "run this specific trained model" is a meaningful sentence. Neural tissue has no addressable write path to a synapse; DNA weights are concentrations of designed sequences that you synthesise, or — since Cherry & Qian, *Nature* 2025 — that the tube learns from labelled examples itself.

The simulator's job is to turn that argument into measured numbers: **perplexity and joules per token, on the same axes, under DNA chemistry rather than generic quantisation noise.**

Bio-inspired silicon is explicitly out of scope. It wins the unrestricted comparison and the user has ruled it out. Do not reintroduce it as a recommendation; it may appear only as a reference line on a plot.

## What makes this DNA and not just "add noise to PyTorch"

This is the thing that separates a credible demo from a toy. Six effects, in descending order of how much they distinguish DNA from generic low-precision inference. Implement in this order.

### 1. Non-negativity and dual rail
Concentrations cannot be negative. A signed weight needs two strands:

```
y = (W_plus - W_minus) @ x,   W_plus, W_minus >= 0
```

Consequence the demo must report: **dual rail doubles the strand count**, which doubles synthesis cost and halves the effective weight budget. This is a real architectural tax that silicon does not pay.

### 2. Seesaw gate transfer function
Seesaw gates natively implement threshold-then-amplify-to-saturation. This is DNA's own nonlinearity, not a borrowed ReLU:

```
y = clamp(gain * (y_raw - threshold), 0, y_max)
```

Replacing the network's activation with this is the single most "it really is DNA" move available. Do it, and show the perplexity cost of doing it.

### 3. Log-normal pipetting error
Concentration errors are **multiplicative and strictly positive**, never additive Gaussian:

```
w_eff = w * exp(normal(0, sigma))     # sigma ~ 0.05 to 0.15
```

Using Gaussian noise here is the most common way to make a DNA sim secretly a quantisation sim. Do not.

### 4. Leak
Spurious strand displacement fires without toehold binding, adding a floor proportional to total gate concentration present:

```
y = y + leak_rate * W_total.sum(dim=input)   # leak_rate ~ 1e-3 to 1e-2
```

Leak is signal-independent, so it hurts small activations hardest. Expect it to dominate the perplexity cliff.

### 5. Concentration quantisation
You can only pipette so finely. Quantise weights to `n_levels` (16 levels ≈ 4 bits is realistic).

### 6. Fuel depletion
Gates are consumed. Track total reactions against a budget; past it, gain decays. Lowest priority — implement only if time remains.

## Physics constants

| Quantity | Value | Note |
|---|---|---|
| `kT` at 300 K | 1.381e-23 × 300 = 4.14e-21 J | |
| Landauer bound, kT·ln2 | 2.87e-21 J | reference line on every energy plot |
| Energy per strand-displacement op | ~20 kT = 8.3e-20 J | ~30× Landauer |
| ATP hydrolysis in vivo | 8.3e-20 J | ~50 kJ/mol |
| Synaptic vesicle release | 1.64e5 ATP ≈ 1.4e-14 J | Attwell & Laughlin 2001 |
| Synaptic event, system level | ~3e-12 J | after brain housekeeping overhead |
| Bulk cascade settling time | ~3600 s per layer | diffusion limited |
| Localised (origami) settling | ~60 s per layer | Chatterjee et al. 2017 |
| Fluorescence readout power | ~50 W | plate reader |
| Molecules in 100 µL at 1 µM | ~6e13 | the parallelism argument |
| Orthogonal domain library | 1e3 to 1e4 | **the binding constraint on scale** |
| GPU reference | ~0.1 to 1 J/token (70B) | |

## The metric

Never report raw "perplexity per joule" — it is degenerate, minimisable by burning energy. Report **joules per token at fixed perplexity**, or the Pareto frontier of the two. If a single scalar is needed:

```
efficiency = (log(ppl_baseline) - log(ppl_model)) / joules_per_token    # nats per joule
```

Report three energy figures separately, never merged:
- `E_chem` — strand displacement only. This is the number that beats silicon.
- `E_wallplug` — including readout instrument. This is the honest number, and it is terrible at batch 1.
- `E_amortised` — wall-plug divided across ~6e13 parallel molecules. This is the batch-inference argument, and it is the demo's punchline.

## Guardrails

- **The identity test is mandatory.** With `sigma=0, leak=0, n_levels=inf, threshold=0, gain=1`, `DNALinear` must reproduce `nn.Linear` to within float tolerance. Write this test first and keep it passing. Everything else is uninterpretable without it.
- **Enforce the capacity ceiling honestly.** If a model needs more distinct sequences than `max_orthogonal_domains` allows, the config is infeasible — say so in the output rather than silently simulating it.
- **Label simulation as simulation.** Constants are hand-chosen from literature, not measured on a benchmark. The defensible claims are the *shape* of the frontier and the *ordering* of effects, not absolute joule figures.
- **Do not claim DNA does interactive generation.** It does batch. The dissertation concedes this; the code should not quietly un-concede it.
- **Cite the real limit.** Demonstrated molecular networks hold 1e2–1e3 weights. Any extrapolation past 1e4 is a projection and must be drawn as a dashed line.

## Code conventions

- Plain PyTorch, no Lightning or Hydra. numpy 1.26, torch 2.14, matplotlib available.
- One substrate config dataclass, `DNAConfig`, with every physical constant as a named field and literature defaults. Everything tunable from one place — the user explicitly wants to "initialize the constants".
- Energy counters live on the layer and accumulate per forward pass; reset between evals.
- Char-level models, small vocabulary, CPU-only. No CUDA assumptions.
- Determinism: seed everything, because the sweep compares noise regimes and unseeded runs make the frontier unreadable.

## Key sources

- [Cherry & Qian 2025, supervised learning in DNA neural networks](https://www.nature.com/articles/s41586-025-09479-w) — in-tube learning, the write-port argument
- [Cherry & Qian 2018, winner-take-all networks](https://www.nature.com/articles/s41586-018-0289-6) — 100-bit patterns, 9 classes, scale ceiling
- [Chatterjee et al. 2017, localised DNA computing](https://www.nature.com/articles/nnano.2017.127/) — hours to minutes, sequence reuse
- [Attwell & Laughlin 2001](https://journals.sagepub.com/doi/10.1097/00004647-200110000-00001) — synaptic energy, the neuron comparison
- `CASE_FOR_DNA.md` — the position paper this build defends
- Deep dive and dissertation: https://claude.ai/artifact/BfUT3kJqE5eoHWM1s3MmTY
