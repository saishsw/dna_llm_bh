# Background primer: why anyone asks these two questions

This primer is for readers new to AI hardware, neuroscience, or the physics of computing. It explains the ideas behind the two prompts:

1. How would you build a biological computer capable of running a language model, minimizing perplexity per joule?
2. What would it take for reversible or thermodynamic computing to beat CMOS on energy per operation, and would it matter for AI?

Both questions come from one worry: AI is using a lot of electricity, and today's chips may be close to the limit of how efficient they can get. Each prompt asks about a different way out. The first borrows from living brains. The second changes the physics of how a chip handles energy.

Each source has a quality tag in brackets. The scale is explained in the last section.

---

## Part 1: Shared basics

### What a language model does

A large language model (LLM), such as the models behind ChatGPT or Claude, writes text one token at a time. A token is a word or part of a word. For each token, the model looks at the text so far and outputs a probability for every possible next token. Underneath, this is billions of multiplications and additions between the input and the model's parameters (also called weights), which are the numbers it learned during training. A 70-billion-parameter model does about 140 billion arithmetic operations per token.

### What perplexity measures

Perplexity is the standard score for how well a language model predicts text. It answers this question: on average, how many equally likely options does the model seem to be choosing between at each step? A perplexity of 10 means the model is about as uncertain as if it were picking from 10 words at random. Lower is better. A perfect model would score 1.

Perplexity is exponential: it is e raised to the model's average surprise (its "cross-entropy"). That makes it awkward to combine with energy, which is why the first answer converts it to bits per token before trading it off against joules.

### What a joule is, and why AI energy matters

A joule is a unit of energy. A watt is one joule per second. A 10 W LED bulb uses 10 joules each second. The human brain runs on about 20 W. One Nvidia H100 GPU, a common AI chip, draws up to 700 W, and AI data centers hold tens of thousands of them.

Measured inference costs vary widely. A 70-billion-parameter model on H100s uses roughly 0.4 J per token, and a 1-billion-parameter model about 0.008 J per token [C, secondary blog summaries of benchmark papers: [John Snow Labs](https://www.johnsnowlabs.com/tokens-per-joule-how-to-quantify-and-reduce-the-energy-footprint-of-clinical-llm-inference/), [co-r-e summary](https://co-r-e.com/method/llm-energy-efficiency-sweet-spots)]. The more careful primary sources are the [ML.ENERGY leaderboard](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/) [B, academic benchmark project] and [TokenPowerBench](https://arxiv.org/html/2512.03024v1) [B, preprint]. Energy per token depends heavily on model size, batch size and number precision, so a single number can mislead.

### How a computer chip spends energy

Almost all modern chips are made with CMOS (complementary metal-oxide-semiconductor) technology. A CMOS chip is billions of tiny switches called transistors. Every time a switch flips, a small wire is charged up and then drained to ground, and that energy turns into heat. The energy per flip is about ½CV², where C is the wire's capacitance and V is the voltage.

AI chips spend energy in two places:

- **Arithmetic**: the multiplications themselves. An H100 spends about 0.7 picojoules (0.7 × 10⁻¹² J) per operation, calculated from its published spec of 989 trillion operations per second at 700 W.
- **Moving data**: fetching the model's weights from memory to the arithmetic units. High-bandwidth memory (HBM) costs about 2 to 7 picojoules per bit moved [A, [Nvidia Research, HPCA 2017](https://research.nvidia.com/sites/default/files/pubs/2017-02_Architecting-an-Energy-Efficient/chatterjee.hpca2017.pdf); C, [Wing VC](https://www.wing.vc/content/the-memory-triopoly)]. One number takes 8 to 16 bits, so moving it often costs more than doing math with it.

Engineers call this the "memory wall." It comes up again in both answers.

---

## Part 2: Background for the biological computer prompt

### Why people look to brains

The human brain has about 86 billion neurons and runs on about 20 W. Stanford's Kwabena Boahen has estimated that simulating it on conventional silicon would take around 10 megawatts. That estimate is quoted by [FinalSpark](https://finalspark.com/neuroplatform/) [D, company website citing an academic]. The comparison is loose, because a simulation of a brain is not the same task as doing what a brain does. Even so, the gap is what draws people to "wetware."

### How brains spend energy

Neurons communicate with electrical spikes and chemical signals at junctions called synapses. The cell's energy currency is ATP. Each ATP molecule releases about 8 × 10⁻²⁰ J when used.

- One synaptic signal (the release of one packet of neurotransmitter) costs about 160,000 ATP, or about 13 femtojoules (13 × 10⁻¹⁵ J) [A, [Attwell & Laughlin 2001](https://journals.sagepub.com/doi/full/10.1097/00004647-200110000-00001), a classic and widely cited paper. Later work found it overestimates the cost of spikes somewhat].
- In the cortex, sending signals over long distances costs about 35 times more energy than the computation itself [A, [Levy & Calvert, PNAS 2021](https://www.pnas.org/doi/10.1073/pnas.2008173118). It is peer reviewed, but the result depends on the authors' assumptions, as an accompanying [PNAS commentary](https://www.pnas.org/doi/10.1073/pnas.2107022118) notes].

For a newcomer, the main takeaway is that one synapse is not wildly cheaper than one digital operation. The brain saves energy through its organization. Memory sits at the synapse where computation happens, so nothing has to be fetched from a separate memory chip. Most neurons are silent at any moment, and signals are low voltage.

### The kinds of biological computers being built

**Neurons on electrodes.** Living neurons, usually grown from human stem cells, sit on a multi-electrode array (MEA), a chip with thousands of tiny electrodes that can stimulate the cells and record their spikes.
- Cortical Labs' DishBrain (2022) taught a culture to play Pong. Their CL1 product (2025) holds about 200,000 neurons, costs $35,000, and keeps cells alive for up to 6 months [C, [IEEE Spectrum](https://spectrum.ieee.org/biological-computer-for-sale), a reputable trade magazine; D, [Cortical Labs](https://corticallabs.com/cl1) itself]. Sources disagree on whether its 850 to 1,000 W power figure is per unit or per rack [C, compare IEEE Spectrum with [DCD](https://www.datacenterdynamics.com/en/news/australian-startup-cortical-labs-unveils-worlds-first-commercial-biological-computer/)].

**Brain organoids.** These are small 3D clumps of neural tissue grown from stem cells.
- FinalSpark's Neuroplatform rents remote access to 16 organoids (about 160,000 neurons in total). Its "million times more efficient" claim is the company's own, and the press repeats it with figures that disagree [D, [Tom's Hardware](https://www.tomshardware.com/pc-components/cpus/worlds-first-bioprocessor-uses-16-human-brain-organoids-for-a-million-times-less-power-consumption-than-a-digital-chip) relaying company statements; [New Atlas](https://newatlas.com/computers/finalspark-bio-computers-brain-organoids/) gives "a billion" instead].
- Brainoware used an organoid as a "reservoir" (explained below) to recognize spoken Japanese vowels. Accuracy rose from 51% to 78% with training, still below conventional AI [A, [Cai et al., Nature Electronics 2023](https://www.nature.com/articles/s41928-023-01069-w)].
- The research roadmap for "organoid intelligence" calls for scaling organoids to about 10 million cells. Organoids lack blood vessels, so the interior starves as they grow, and the roadmap proposes artificial perfusion (pumped nutrient flow) as a fix [B, [Smirnova et al., Frontiers in Science 2023](https://www.frontiersin.org/journals/science/articles/10.3389/fsci.2023.1017235/full), a peer-reviewed perspective piece that sets out a program rather than reporting results].

**DNA computers.** Instead of cells, designed DNA strands react in a test tube, and the concentrations of molecules act as numbers.
- Caltech researchers built a DNA "neural network" that sorts 10×10-pixel patterns into nine categories [A, [Cherry & Qian, Nature 2018](https://www.nature.com/articles/s41586-018-0289-6)], and later one that learns from examples [A, [Cherry & Qian, Nature 2025](https://www.nature.com/articles/s41586-025-09479-w)]. These reactions take minutes to hours, and each circuit can be used only once, so they are far too slow to generate text.

### Two terms you will meet

- **Reservoir computing.** You keep a complex dynamic system (here, living tissue) untrained, feed inputs into it, and train only a simple readout that interprets its responses. This works well with biology because nobody knows how to set individual synapse strengths on purpose.
- **Closed-loop training.** The tissue gets feedback that depends on its own output. DishBrain gave neurons predictable stimulation when they played well and random noise when they missed, and the network reorganized itself to get more of the predictable kind.

### Why the prompt is hard

To run a language model you need billions of precisely set, copyable numbers and fast, repeatable computation. Living tissue offers noisy, uncopyable, self-organizing connections, along with a heavy life-support bill (incubators, pumps, sensing electronics) that outweighs the neurons' own energy by orders of magnitude. The question tests whether you can see past the "20-watt brain" headline to what actually costs energy.

---

## Part 3: Background for the reversible and thermodynamic computing prompt

### The physical floor: Landauer's principle

In 1961 Rolf Landauer showed that erasing one bit of information must release at least kT ln 2 of heat. Here k is Boltzmann's constant and T is temperature. At room temperature that is about 3 × 10⁻²¹ J (3 zeptojoules). Experiments have since confirmed the bound. Today's CMOS switching costs about 60,000 times more than that [B, [Frank, Sandia preprint](https://www.sandia.gov/app/uploads/sites/210/2022/06/ICRC17-Frank-ABRC-preprint-6.0.pdf), by a leading researcher in the field, not yet peer reviewed in this form]. So physics leaves a lot of room. The open question is whether engineering can use it.

### Reversible computing

The bound applies only to erasing information. An operation that loses no information, so its inputs can be recovered from its outputs, could in principle cost almost nothing. That is reversible computing.

The practical version is called adiabatic computing. A normal chip dumps each wire's charge to ground, wasting it. An adiabatic chip charges wires slowly using an oscillating power supply (a resonator, which works like a swinging pendulum) and then pulls most of the charge back for reuse. The catch is speed: the slower you charge, the less you waste. Michael Frank at Sandia National Labs has written much of the modern theory [B, [Frank 2018 preprint](https://arxiv.org/pdf/1803.02789); B, [Frank et al. 2020](https://arxiv.org/abs/2009.00448)].

Recent progress:
- UK startup Vaire's first test chip recycled roughly 30 to 45% of its switching energy, the first commercial demonstration of the idea [C, [EE Times](https://www.eetimes.com/vaire-demos-energy-recovery-with-reversible-computing-test-chip/), reputable trade press with independent critique; D, Vaire's own roadmap claiming 4,000× gains in 10 to 15 years, via [IEEE Spectrum](https://spectrum.ieee.org/reversible-computing)]. EE Times points out that recycling energy is not the same as full logical reversibility.
- Superconducting circuits cooled to 4 kelvin (about −269 °C) have reached 24 kT per switch [A, [Takeuchi et al., Scientific Reports 2019](https://www.nature.com/articles/s41598-019-46595-w)]. Cooling to that temperature costs hundreds of watts of electricity for each watt of heat removed, which cancels much of the gain.

### Thermodynamic computing

This is a different idea with a similar name. Chips normally fight electrical noise (random jiggling from heat). Thermodynamic computers use it. A circuit is set up so that, as it jiggles and settles, its states follow a probability distribution, and reading those states gives you random samples or solves certain math problems. Some kinds of AI, such as image-generating diffusion models, spend much of their effort producing random samples, so the hope is to get them almost for free from physics.

- Normal Computing built an 8-cell circuit board that inverted 8×8 matrices this way. It is a small proof of concept, and any advantage over GPUs is only projected for very large problems [A, [Nature Communications 2025](https://www.nature.com/articles/s41467-025-59011-x)].
- Extropic has built prototype chips and claims about 10,000× efficiency over GPUs for one type of generative model. That number comes from a computer simulation of future hardware on simple image tasks, not from a measurement [D, [Extropic blog](https://extropic.ai/writing/tsu-101-an-entirely-new-type-of-computing-hardware); C, critical analyses such as [explainx](https://www.explainx.ai/blog/extropic-z1t-thermodynamic-sparse-transformer-hardware-2026), a blog. A related peer-reviewed paper co-authored with MIT's Isaac Chuang appeared in npj Unconventional Computing in 2026, but I did not read it directly].

### Why "would it matter for AI?" is a fair follow-up

Even a perfect switch would not fix the memory wall from Part 1, because much of an AI chip's energy goes to moving data rather than flipping logic gates. Electricity is also only part of what an AI chip costs over its life. The chip's purchase price is usually larger, so a more efficient chip that is slower or bigger may not save money. Finally, today's language models mostly do deterministic multiplication, not random sampling, so thermodynamic chips help only if AI models change shape to suit them.

---

## Part 4: How to read the sources

| Tag | Meaning | Examples in this primer | How much to trust it |
|---|---|---|---|
| A | Peer-reviewed primary research in a major journal | Nature, Nature Electronics, Nature Communications, PNAS, Scientific Reports, J. Cereb. Blood Flow Metab. | High for what was measured. Check what the paper actually tested, since press summaries often stretch it |
| B | Preprints, perspective pieces, academic benchmarks | arXiv papers by Frank, Sandia preprints, Frontiers roadmap, ML.ENERGY, TokenPowerBench | Moderate to high. Expert work that has not been fully peer reviewed, or that argues for a program rather than reporting results |
| C | Reputable trade press and careful secondary analysis | IEEE Spectrum, EE Times, DCD, Wing VC, technical blogs | Moderate. Good for context and timelines. Numbers are often second-hand and sometimes disagree |
| D | Company claims and press that repeats them | Cortical Labs, FinalSpark, Vaire, Extropic sites; Tom's Hardware and New Atlas restating claims | Low for performance numbers. Treat as marketing until measured independently |

A practical rule for this topic: the biggest efficiency numbers ("a million times," "10,000×," "4,000×") all come from tier D or from simulations. The measured results from tiers A and B are much more modest: 78% vowel accuracy, an 8×8 matrix, 30 to 45% energy recycled. Comparing the two is a good way to see how far each field still has to go.

## Glossary

| Term | Plain meaning |
|---|---|
| ATP | The molecule cells use to store and spend energy |
| Adiabatic logic | Circuits that charge slowly and recycle energy instead of dumping it |
| CMOS | The standard transistor technology in nearly all chips |
| Femto-, pico-, zepto- | 10⁻¹⁵, 10⁻¹², 10⁻²¹ |
| HBM | Fast memory stacked next to AI chips |
| Inference | Running a trained model to get answers (as opposed to training it) |
| kT | The typical energy of thermal jiggling at temperature T, about 4 × 10⁻²¹ J at room temperature |
| MEA | Multi-electrode array, a chip that talks to living neurons |
| Organoid | A small 3D lump of lab-grown tissue that mimics an organ |
| Parameter / weight | A learned number inside a model |
| Perplexity | How uncertain a language model is about the next token (lower is better) |
| Token | A word or word-piece a language model reads and writes |
