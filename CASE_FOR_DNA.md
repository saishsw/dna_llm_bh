# The Case for DNA

### Why molecular computing is the only viable biological substrate for a language model

> Position paper backing the simulator in this directory. This is the argument the demo exists to support — read it before making any claim about what DNA computing can or cannot do. Living version: https://claude.ai/artifact/BfUT3kJqE5eoHWM1s3MmTY

**Thesis.** Restrict the field to substrates that are genuinely biological — molecules, cells, tissue — and the choice collapses to two: living neurons or engineered nucleic acids. Living neurons lose. They lose not on speed or scale, which are the usual complaints, but on a prerequisite so elementary it is easy to walk past: *you cannot tell a neuron what to compute.* DNA you can. Everything else in this argument follows from that one asymmetry.

## I. The test that disqualifies tissue

A trained language model **is** its checkpoint. Perplexity is a property of specific numerical weights, arrived at through a pretraining run that represents essentially all of the model's cost and all of its value. Any substrate proposed as a host must therefore answer one question before any other: *can you load the weights into it?*

Neural tissue cannot. There is no addressable write path to a synapse. Optogenetics addresses cell populations and cell types, not individual synaptic conductances. Chemical LTP induction shifts statistical distributions of strength, not specified values. The synapse is not a register, and in a living culture it is not even individually observable at scale.

The DishBrain result is real, and I do not want to understate it: 800,000 neurons learning Pong in roughly twenty minutes of closed-loop feedback is a genuine scientific achievement. But it is closed-loop *shaping* of an unknown dynamical system toward a reward signal. It is not weight loading, and twenty minutes to Pong is not a step on a path that ends at a function specified in advance.

The standard rescue is reservoir computing: leave the tissue untrained, let it act as a high-dimensional nonlinear mixer, and train only a silicon readout. This rescue concedes the argument. If the readout carries the model, the tissue is not the model — and the test of that is simple. Replace the culture with a random projection matrix and measure how much perplexity you lose. Honest accounting attributes the result to the silicon, and what remains is a conventional language model with an expensive, mortal random-feature generator bolted to its front.

## II. DNA passes the test twice

**First, by construction.** A weight in a strand-displacement network is the concentration of a defined sequence. You compute the weights in silico, synthesise the strands, and mix them at the specified concentrations. The write path is a sequence design tool, a synthesis order and a pipette. This is unglamorous, slow and expensive — and it is a *write path*, which is more than tissue offers at any price.

**Second, and this is new.** In September 2025, Cherry and Qian published [supervised learning in DNA neural networks](https://www.nature.com/articles/s41586-025-09479-w) in *Nature*: an enzyme-free molecular network that learns from labelled examples **inside a single test tube**, trained to classify three distinct sets of 100-bit patterns. Class information is stored as quantitative concentration states and reused to classify novel inputs. Every previous molecular network had its parameters learned in silico and then installed. This one forms its own. Strikingly, the resulting matrix of DNA memory concentrations, read out by fluorescence, visually reconstructs the handwritten digits it was trained on — the weights are legible as the thing they encode.

That is a chemical system forming weights from data: precisely the capability neural cultures are perpetually claimed to possess and have never demonstrated against a specification.

## III. Energy, which is the easy part

Toehold-mediated strand displacement is driven by base-pairing free energy, netting on the order of 20 k<sub>B</sub>T per operation — about 8 × 10⁻²⁰ J, roughly **30× the Landauer bound**. Nothing humans have built, in any substrate, runs closer to the thermodynamic floor.

Against that, a synaptic vesicle release costs 1.64 × 10⁵ ATP ≈ 13 fJ of signalling energy, and once the brain's housekeeping load is included the system-level figure is nearer 3 pJ per synaptic event. DNA is cheaper per operation by **five to seven orders of magnitude**.

The standby comparison is even more lopsided, and it is the one that actually decides deployments. A neuron culture is an obligate consumer: 37 °C thermal control, CO₂ regulation, perfusion, media exchange and amplifier electronics, drawing tens of watts per 10⁵ cells *continuously*, computing or not, until the culture dies in about six months. Dried DNA sits at room temperature, stable for centuries, drawing **nothing at all**. A biological computer that costs zero watts to not use is a different kind of object from one that must be kept alive.

## IV. The four properties, and who has them

| | Neural tissue | DNA |
| --- | --- | --- |
| **Programmable** — weights set to specified values | No | Yes, by synthesis; now also by in-vitro learning |
| **Copyable** — the substrate itself duplicated | No | Yes — PCR duplicates it exponentially, for cents |
| **Simulable** — behaviour predictable before building | No | Yes — nearest-neighbour thermodynamics, NUPACK |
| **Persistent** — survives without maintenance | No, ~6 months | Yes, centuries, at zero power |

The copyability row deserves emphasis because it has no analogue in neuroscience. PCR duplicates the computational substrate itself, exponentially, in a thermocycler. Once a model is synthesised once, every subsequent copy is nearly free. You cannot photocopy a cortex; every culture is a distinct organism that must be grown and shaped from nothing.

Simulability matters almost as much. DNA hybridisation obeys well-parameterised nearest-neighbour thermodynamics, so a circuit can be designed and verified computationally before a single strand is ordered. Neural culture behaviour is not predictable from first principles, which means every experiment is an experiment.

## V. The latency objection, answered

The strongest objection — the one in the background notes for this project, which called DNA "hopeless for autoregressive generation" — is speed. Bulk strand-displacement cascades settle over hours, and autoregression is irreducibly sequential. Two responses, one technical and one about the workload.

**Localisation removes most of the diffusion penalty.** [Chatterjee et al. (2017, *Nature Nanotechnology*)](https://www.nature.com/articles/nnano.2017.127/) tether circuit elements to DNA origami scaffolds, cutting computation time **from hours to minutes** and — critically — allowing the same sequence to be reused at different spatial locations. Subsequent work on surface-fixed origami registers pushes further in the same direction. This is still minutes, not microseconds, and I concede that fully.

**So stop pricing it as a chatbot.** Interactive generation is latency-bound; a datacentre is a throughput machine. One hundred microlitres at 1 µM holds ~6 × 10¹³ molecules, and every one is an independent processor with free fan-out. The correct deployment model is batch inference: 10¹³ queries answered simultaneously over a few hours, at tens of zeptojoules each. For offline corpus work — classification, scoring, embedding, bulk generation — that is the shape of the job regardless of substrate.

DNA will never serve an interactive token stream. It can plausibly serve batch workloads. Tissue serves neither, because tissue cannot be given the model in the first place.

## VI. The real limit, stated honestly

Scale is the genuine problem, and it should not be minimised. Demonstrated molecular networks hold 10²–10³ weights. Anything linguistically interesting needs 10⁶ or more. The binding constraint is sequence orthogonality: every weight requires a domain that does not cross-react with every other domain in the tube, and crosstalk caps a practical library somewhere around 10³–10⁴.

What makes this the *right* problem to have is that it has known lines of attack, and all of them work by converting a global constraint into a local one. Localised circuits reuse sequences at distinct spatial addresses, so orthogonality need only hold within a neighbourhood rather than across the whole tube. Origami tiling multiplexes those neighbourhoods. Compartmentalisation in droplets or vesicles partitions the sequence space again. And sequence design is an established computational discipline with mature tooling, not an open mystery.

Contrast the neuron write-port problem, which has no known line of attack whatsoever. **A hard engineering problem with three plausible routes beats an open scientific problem with none.**

## Conclusion

The case for DNA is not that it is fast, and not that it is large. It is neither, and pretending otherwise would be the weakest version of this argument.

The case is that DNA is the only biological substrate in which the sentence *"run this specific trained model"* has a meaning. It accepts weights, holds them indefinitely at zero power, copies them for cents, and — as of 2025 — forms them from labelled data on its own. It computes within roughly thirty times of the thermodynamic limit of computation, which no other physical system achieves. Every obstacle that remains is an engineering obstacle with a route through it.

Living neurons are faster, and they will stay faster. It does not matter. A substrate you cannot program is not running your model — it is running its own, and nobody can say what that is.

## Sources

- [Cherry & Qian 2025, *Supervised learning in DNA neural networks*, Nature 645:639–647](https://www.nature.com/articles/s41586-025-09479-w)
- [Cherry & Qian 2018, *Scaling up molecular pattern recognition with DNA-based winner-take-all neural networks*, Nature](https://www.nature.com/articles/s41586-018-0289-6)
- [Chatterjee et al. 2017, *A spatially localized architecture for fast and modular DNA computing*, Nature Nanotechnology](https://www.nature.com/articles/nnano.2017.127/)
- [Attwell & Laughlin 2001, *An energy budget for signaling in the grey matter of the brain*](https://journals.sagepub.com/doi/10.1097/00004647-200110000-00001)
- [Kagan et al. 2022, *In vitro neurons learn and exhibit sentience when embodied in a simulated game-world* (DishBrain), Neuron](https://www.cell.com/neuron/fulltext/S0896-6273(22)00806-6)
