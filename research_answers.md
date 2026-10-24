# Research answers: biological LLMs and reversible/thermodynamic computing

Drafted 2026-10-24. Figures marked "(my calc)" are back-of-envelope arithmetic from the cited inputs. Energy-efficiency claims from startups are their own and have not been independently reproduced unless stated.

Reference constants used in both answers:

| Quantity | Value | Source |
|---|---|---|
| Landauer limit, kT ln 2 at 300 K | 2.9 × 10⁻²¹ J (2.9 zJ) | standard physics |
| One ATP hydrolysis in a cell | ~8 × 10⁻²⁰ J (~20 kT) | ~50 kJ/mol (my calc) |
| Charging one CMOS logic node (end of ITRS roadmap) | ~60,000 kT ≈ 0.25 fJ | [Frank, Sandia ABRC preprint](https://www.sandia.gov/app/uploads/sites/210/2022/06/ICRC17-Frank-ABRC-preprint-6.0.pdf) |
| H100, dense FP16 | ~0.7 pJ/FLOP (989 TFLOPS at 700 W) | spec sheet (my calc) |
| HBM access | ~2 to 7 pJ/bit | [Chatterjee et al. 2017](https://research.nvidia.com/sites/default/files/pubs/2017-02_Architecting-an-Energy-Efficient/chatterjee.hpca2017.pdf), [Wing VC](https://www.wing.vc/content/the-memory-triopoly) |
| Llama-3 70B FP8 inference, 8×H100 | ~0.39 J/output token | [FifthRow / John Snow Labs](https://www.johnsnowlabs.com/tokens-per-joule-how-to-quantify-and-reduce-the-energy-footprint-of-clinical-llm-inference/) (secondary) |
| Llama-3.2 1B, H100 TensorRT-LLM | ~0.008 J/token (121 tokens/J) | [ML.ENERGY / co-r-e summary](https://co-r-e.com/method/llm-energy-efficiency-sweet-spots) |

---

## 1. How would you build a biological computer that runs a language model, minimizing perplexity per joule?

### Pin down the metric first

Perplexity is exp(cross-entropy), so it is exponential in model quality and hard to trade against energy linearly. I would optimize a Pareto frontier: wall-plug joules per token at a fixed held-out perplexity (or, equivalently, minimize bits-per-token × joules-per-token). "Wall-plug" is the important word. It has to include the incubator, perfusion pumps, gas control, electrode amplifiers and the silicon that encodes and decodes tokens, because those dwarf the neurons.

A quick ledger shows why. A human neuron runs on roughly 20 W / 86 × 10⁹ ≈ 0.2 nW (my calc). Cortical Labs' CL1 holds about 200,000 neurons, which is ~50 µW of biology, while IEEE Spectrum reports a CL1 rack draws 850 to 1,000 W ([IEEE Spectrum](https://spectrum.ieee.org/biological-computer-for-sale), [Cortical Labs](https://corticallabs.com/cl1)). Life support and electronics outweigh the computing tissue by about seven orders of magnitude. Any design that does not attack overhead cannot win on this metric.

### Where biology's advantage actually is

The headline claims ("a million times less power", [FinalSpark via Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/worlds-first-bioprocessor-uses-16-human-brain-organoids-for-a-million-times-less-power-consumption-than-a-digital-chip)) compare a whole brain to a hypothetical silicon brain simulation. Per elementary operation, biology is not dramatically better:

- One glutamate vesicle release costs about 1.6 × 10⁵ ATP ([Attwell & Laughlin 2001](https://journals.sagepub.com/doi/full/10.1097/00004647-200110000-00001)), roughly 13 fJ (my calc). An 8-bit digital MAC is ~0.05 pJ plus whatever its weight fetch costs.
- Levy and Calvert estimate cortical *computation* uses only ~0.1 W of ATP, and that communication costs 35× more than computation ([PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2008173118)).

So the brain wins through architecture: memory sits in the synapse where computation happens (no HBM round trip), activity is sparse (most neurons are silent most of the time), and signals are low-voltage analog. A biological LM should keep those properties and avoid the things biology is bad at: precise weights, fast clocks, copying parameters, and backpropagation.

### Candidate substrates

| Substrate | State of the art | Fit for an LM |
|---|---|---|
| 2D neuron cultures on MEAs (Cortical Labs) | Pong via closed-loop feedback (DishBrain, 2022); 120-unit CL1 facility planned | Fast enough (ms spikes), trainable by feedback, but only ~10⁵ neurons and ~10³ to 10⁴ electrodes |
| 3D brain organoids (FinalSpark, Brainoware) | Reservoir computing on Japanese vowels, 51% → 78% accuracy ([Cai et al., Nat. Electron. 2023](https://www.nature.com/articles/s41928-023-01069-w)); dopamine-based training; ~100-day lifetimes | More neurons per volume, but necrotic cores without vasculature; roadmap targets ~10⁷ cells via microfluidic perfusion ([Smirnova et al. 2023](https://www.frontiersin.org/journals/science/articles/10.3389/fsci.2023.1017235/full)) |
| DNA strand-displacement networks | Winner-take-all classifier of 10×10 patterns ([Cherry & Qian 2018](https://www.nature.com/articles/s41586-018-0289-6)); in-vitro supervised learning ([Nature 2025](https://www.nature.com/articles/s41586-025-09479-w)) | Each reaction costs only tens of kT, but reactions take minutes to hours and circuits are use-once. Fine for one classification, hopeless for autoregressive generation |
| Engineered gene circuits in cells | Logic gates, toggle switches | Minute-to-hour timescales; same problem as DNA |

Living neurons on electrodes are the only substrate fast enough to emit tokens at a useful rate (tens of ms per step suggests 10 to 100 tokens/s per culture).

### The architecture I would build

**Model shape.** Use a recurrent, fixed-state model such as a state-space or RWKV-style network. A transformer's KV cache grows with context and needs exact recall, which tissue cannot provide. A recurrent state maps naturally onto ongoing network activity.

**Division of labor.** Train a small teacher LM on silicon. Tokens are encoded on silicon into spatiotemporal stimulation patterns delivered through a high-density CMOS MEA. The culture acts as the recurrent core: a high-dimensional, nonlinear, fading-memory dynamical system that does the expensive mixing for almost no energy. A small silicon readout (linear or low-rank) maps recorded spike features to next-token logits and samples.

**Training in three stages.**
1. *Reservoir stage.* Leave the tissue untrained and fit only the readout, as Brainoware did. This works today and gives a baseline bits/token number.
2. *Closed-loop shaping.* Use the DishBrain principle: give the culture predictable, structured stimulation when its readout predicts the next token correctly and unstructured noise when it does not, optionally gated with dopamine uncaging (as FinalSpark has done). The goal is to push the reservoir's dynamics toward the statistics of language so the silicon readout can shrink.
3. *Distillation.* Match the culture-plus-readout output distribution to the silicon teacher with KL loss on the readout and reward-shaped feedback to the tissue. Report perplexity relative to the teacher at matched readout size, so the tissue's contribution is measurable rather than hidden in the readout.

**Cutting overhead (the real lever on joules).**
- Put many organoids in one perfused enclosure so thermal and gas overhead is amortized. Incubator power scales with surface area, tissue count with volume.
- Do spike detection and feature compression at the electrode in low-power CMOS, and send only sparse events off-chip. This copies the brain's own lesson that communication dominates energy.
- Avoid optical I/O (optogenetics plus imaging) for inference. Lasers and cameras cost watts.
- Extend lifetime (perfusion, vascularization) so the energy and cost of growing and characterizing each culture is spread over more tokens, and build per-culture calibration since every culture is unique and weights cannot be copied.

### What I expect it to achieve

A rough target system: 10⁷ neurons (~2 mW of tissue), a shared enclosure amortized to ~1 W per culture, ~0.1 W of MEA and readout electronics, at 10 tokens/s gives about 0.1 J/token (my calc). That is competitive with a 70B model on H100s (~0.39 J/token) on energy but nowhere near on perplexity, and it is ~10× *worse* than a 1B model on an H100 (~0.008 J/token), which will also beat it on perplexity. Reservoir-style language models have historically been weak, and 10⁷ noisy neurons holding maybe 10⁹ to 10¹⁰ imprecise synapses that cannot be set directly is a poor substitute for 10⁹ trained parameters.

My verdict is that the build is feasible as a research system, and the right first deliverable is an honest bits-per-token vs. wall-plug-joules curve on a small corpus such as TinyStories. Biology loses on perplexity per joule for the foreseeable future because the metric rewards precise, trainable, copyable parameters, which are exactly biology's weaknesses. The lessons it does teach (co-located memory, sparse events, cheap communication) transfer better to neuromorphic and in-memory silicon than to tissue. Ethics review is also mandatory for any human-derived neural culture that is trained with feedback.

---

## 2. What would it take for reversible or thermodynamic computing to beat CMOS on energy per operation, and would it matter for AI?

### The gap and why it exists

A CMOS logic node dissipates about ½CV², roughly 60,000 kT, every time it switches, because the charge is dumped to ground through a resistor ([Frank, Sandia](https://www.sandia.gov/app/uploads/sites/210/2022/06/ICRC17-Frank-ABRC-preprint-6.0.pdf)). Landauer's bound is 0.7 kT per *erased* bit, and logically reversible operations erase nothing, so in principle there is 10⁴ to 10⁵× headroom per gate. Two different ideas try to use it.

**Reversible / adiabatic computing** charges nodes slowly from a ramped, resonant power clock and returns the charge to the resonator instead of to ground. Dissipation per switch falls roughly as (RC/T)·CV², so energy is traded for time.

**Thermodynamic computing** (Extropic, Normal Computing) uses thermal noise as the computation, letting coupled analog or probabilistic circuits relax to a distribution and then sampling it. It targets sampling and linear algebra, not general logic.

### Where they stand (late 2026)

| Effort | Measured result | Caveats |
|---|---|---|
| Vaire "Ice River" adiabatic CMOS test chip | Energy recovery factor 1.77 (capacitor array) and 1.41 (adder) vs. square-wave drive, 500 MHz ([EE Times](https://www.eetimes.com/vaire-demos-energy-recovery-with-reversible-computing-test-chip/)) | ~30 to 45% energy saved on deliberately simple logic; roadmap to 4,000× needs MEMS resonators with 99.97% recovery over 10 to 15 years |
| Superconducting AQFP (Yokohama) | 24 kT per junction for an 8-bit adder at 4.2 K ([Sci. Rep. 2019](https://www.nature.com/articles/s41598-019-46595-w)); reversible variants theoretically below kT ln 2 at low frequency | Cryocooling costs hundreds of watts of wall power per watt removed at 4 K |
| Normal Computing SPU | 8-cell RLC board inverted 8×8 matrices, reproduced across 3 boards ([Nat. Commun. 2025](https://www.nature.com/articles/s41467-025-59011-x)) | Advantage over GPUs projected only past ~3,000 dimensions; inductors limit scaling |
| Extropic X0 / Z1 | Room-temperature p-bit prototypes at two fabs; ~10,000× vs. GPU for denoising thermodynamic models in *simulation* ([Extropic](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware)) | The 10,000× is a modeled projection on simple image benchmarks; the Sept 2026 Z1T "140×" claim is self-reported, and an FPGA consumed >95% of system energy in it ([explainx analysis](https://www.explainx.ai/blog/extropic-z1t-thermodynamic-sparse-transformer-hardware-2026)) |

Nobody has yet shown a system-level, wall-plug energy win on a real AI workload against same-node CMOS.

### What it would take to win

1. **Very high-Q energy recovery.** If a resonator returns a fraction *r* of the switching energy, the gain is about 1/(1 − *r*). Vaire's 1.77 factor corresponds to *r* ≈ 0.44. A 10× win needs *r* ≈ 0.9 and 100× needs 0.99, sustained across process variation and at GHz-scale clocks. On-chip inductors cannot get there; MEMS or bulk-acoustic resonators probably can, which makes co-packaging a hard problem.
2. **Leakage-free transistors.** Adiabatic logic runs slower, so static leakage accumulates over a longer cycle and can erase the dynamic savings. Frank argues adiabatic CMOS gives little system-level benefit without much cheaper low-leakage devices ([Frank 2018](https://arxiv.org/pdf/1803.02789)). That means high-Vt or new device types tuned for low off-current rather than peak speed.
3. **Reversible logic design and compilers.** Fully reversible logic needs extra garbage-bit and uncompute paths (Bennett's construction costs roughly 2× time or area). Multi-phase clocks (S2LAL uses 8 phases, [Frank 2020](https://arxiv.org/abs/2009.00448)) need EDA tools that do not exist at production quality.
4. **Paying for area with parallelism.** Slower clocks mean more silicon per unit of throughput. This only pays off for workloads that are massively parallel and tolerant of latency, and AI is the best example of that.
5. **For thermodynamic chips: an algorithm that needs them.** Their advantage is in sampling. They need model families (energy-based models, diffusion, Bayesian inference) where sampling dominates the cost, and they must be measured end to end including digital control and I/O.
6. **Room temperature, or very large per-op gains.** Superconducting logic must beat CMOS by more than the ~10² to 10³× cryocooling penalty before it saves energy at the wall.

### Would it matter for AI?

Yes, but less than the per-gate numbers suggest, and only with co-design. Three reasons:

**Data movement dominates, not logic.** A 70B FP8 model at batch size 1 reads ~70 GB of weights per token. At ~4 pJ/bit that is ~2 J/token in HBM traffic alone (my calc), versus ~0.1 J for its ~140 GFLOP of arithmetic at 0.7 pJ/FLOP. Batching amortizes weight reads, but data movement still costs 3 to 10× the compute energy at scale ([Wing VC](https://www.wing.vc/content/the-memory-triopoly)). Making logic reversible attacks the smaller term, so Amdahl's law caps the gain at maybe 2 to 3× unless wires and memory also go adiabatic. Adiabatic charge recovery on long buses and clock trees is the most promising near-term use for that reason.

**Electricity is a small share of GPU cost, but power is a hard limit.** An H100 at 700 W over five years uses ~30 MWh, about $3k at $0.10/kWh, against a $25k+ purchase price (my calc). A chip that is 10× more efficient but needs 3× the silicon for the same throughput loses on cost per token, unless the site has a fixed power budget. Grid interconnect is increasingly the binding constraint for new data centers, so tokens per second per watt becomes revenue per watt there, and energy efficiency starts to matter commercially.

**Thermodynamic sampling misses most LLM compute.** Transformer inference is deterministic matrix multiplication, and token sampling is a negligible share of it. Thermodynamic hardware matters for AI only if the models move toward energy-based or diffusion-style generation. That is a hardware-lottery bet: a hardware win requires a model shift first.

### Bottom line

To beat CMOS, reversible computing needs >90 to 99% resonant energy recovery, near-zero-leakage devices and reversible EDA tooling. Thermodynamic computing needs an end-to-end benchmark on a workload that is actually sampling-bound. Both are probably 5 to 15 years from mattering at scale. For AI, the biggest near-term energy gains will keep coming from moving less data (in-memory compute, lower precision, sparsity, better batching) rather than from cheaper bit flips. Physics leaves 10⁴× or more of headroom per operation, so the binding limits are architecture and economics. Cheaper operations will also likely raise total AI compute (Jevons' paradox) rather than lower total grid demand.
