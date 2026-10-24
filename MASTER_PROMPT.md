# Master prompt — DNA LLM simulator, 3 hours

Paste everything below the line into a fresh Claude Code session in this directory.

---

Build a simulator that makes PyTorch behave like a DNA strand-displacement computer, and use it to measure perplexity against joules per token. Read `.claude/skills/dna-llm-sim/SKILL.md` first — it has the physics constants, the six DNA effects to model, and the honesty guardrails. Follow it.

Context: this is a 3-hour hackathon build. The argument being defended is in `CASE_FOR_DNA.md` — read it too. In short: DNA is the best *biological* substrate for a language model because it is the only one you can actually load weights into. Bio-inspired silicon is out of scope by the user's choice. The deliverable is a plot plus a number I can defend to judges, not a framework.

Work in four phases and **show me something runnable at the end of each**. Do not build phases 3 and 4 before phase 1 passes its test.

## Phase 1 (45 min) — `dna.py`, the substrate

A `DNAConfig` dataclass holding every physical constant as a named field with literature defaults, so I can tune all of it from one place.

A `DNALinear(nn.Module)` that wraps a trained `nn.Linear` and applies, in this order:
1. dual-rail split into non-negative `W_plus` / `W_minus`
2. concentration quantisation to `n_levels`
3. log-normal multiplicative pipetting error — `w * exp(N(0, sigma))`, never additive Gaussian
4. leak proportional to total gate concentration
5. energy and time accumulation into counters on the layer

A `SeesawActivation` implementing `clamp(gain * (x - threshold), 0, y_max)` — DNA's native nonlinearity, replacing ReLU.

**Write the identity test first and make it pass before anything else:** with `sigma=0, leak=0, n_levels=None, threshold=0, gain=1`, `DNALinear` must match `nn.Linear` to float tolerance. If that test is not green, nothing downstream means anything.

Also enforce the capacity ceiling: if the model needs more distinct strands than `max_orthogonal_domains`, raise or flag the config as infeasible rather than simulating it silently.

## Phase 2 (45 min) — `model.py`, `train.py`

A char-level language model small enough to be plausible as DNA: embedding + one or two hidden layers, a few thousand parameters, context window of 4–8 characters. Train it on silicon in FP32 on any public-domain English text (tiny Shakespeare is fine — download it, or generate from a built-in string if the network is unavailable).

Report held-out perplexity for the FP32 baseline. That number is the ceiling everything else is measured against. Keep training under about two minutes on CPU.

## Phase 3 (45 min) — `sweep.py`

Swap the trained layers into `DNALinear` and sweep the substrate constants. At minimum sweep `sigma`, `leak_rate`, and `n_levels`, several values each, multiple seeds per point because these are noise regimes.

For every configuration record: perplexity, `E_chem` per token, `E_wallplug` per token including readout, `E_amortised` across ~6e13 parallel molecules, seconds per token, strand count, and whether the config is feasible under the domain ceiling. Write it all to CSV.

Two results I specifically want found and quantified:
- **the leak cliff** — perplexity should be flat as leak rises, then collapse past a threshold. Locate the knee.
- **what the seesaw activation costs** — perplexity with `SeesawActivation` versus with ReLU, all else equal.

## Phase 4 (45 min) — `plot.py`, `README.md`

The main figure: perplexity on y, joules per token on x, log-log, with a vertical line at the Landauer bound, a horizontal line at the FP32 baseline perplexity, and a GPU reference point. Plot all three energy accountings as separate series so the batch-amortisation argument is visible as a leftward jump.

A second figure for the leak cliff.

`README.md`: what was built, how to rerun it, the headline numbers, and an explicit limitations section saying the constants are literature-derived rather than measured, that demonstrated molecular networks hold only 1e2–1e3 weights, and that anything beyond 1e4 is extrapolation.

## Rules

- Stop and tell me if a phase runs long rather than skipping its test.
- Every number that reaches the README must come from a run, not from the prompt.
- Keep it to plain PyTorch, CPU, seeded. No frameworks, no CUDA.
- If you have spare time at the end, add fuel depletion, not more plots.
