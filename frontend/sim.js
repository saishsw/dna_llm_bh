// Browser port of dna.py: same order of effects, same accounting.
// Dual rail -> quantise -> log-normal pipetting -> leak, then seesaw, then energy and strands.
// Pipetting draws use a seeded JS generator, so a "preparation" here is not the same draw as
// a PyTorch seed. With sigma = 0 everything is deterministic and must match PyTorch exactly
// (checked by check_sim.js).
(function (root) {
  const K_B = 1.380649e-23;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    let u = 0;
    while (u === 0) u = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // One layer as a physical preparation. W is row-major [out][in+1], last column = bias.
  function prepareLayer(W, nOut, nIn, cfg, seed) {
    const cols = nIn + 1, n = nOut * cols;
    let fullScale = 0;
    for (let i = 0; i < n; i++) fullScale = Math.max(fullScale, Math.abs(W[i]));
    const levels = cfg.n_levels;
    const step = levels && fullScale > 0 ? fullScale / (levels - 1) : 0;
    const rand = mulberry32(seed * 2654435761 + 12345);
    const net = new Float64Array(n), leak = new Float64Array(nOut);
    let gates = 0;
    for (let i = 0; i < n; i++) {
      let p = Math.max(W[i], 0), m = Math.max(-W[i], 0);          // 1. dual rail
      if (step) { p = Math.round(p / step) * step; m = Math.round(m / step) * step; }  // 2.
      if (p > 0) gates++;
      if (m > 0) gates++;
      if (cfg.sigma > 0) {                                         // 3. log-normal, both rails
        p *= Math.exp(cfg.sigma * gaussian(rand));
        m *= Math.exp(cfg.sigma * gaussian(rand));
      }
      net[i] = p - m;
      leak[(i / cols) | 0] += cfg.leak_rate * (p + m);             // 4. leak floor
    }
    return { net, leak, nOut, nIn, cols, gates };
  }

  function prepare(model, cfg, seed) {
    const V = model.V, ctx = model.ctx, H = model.hidden;
    return {
      H, V, ctx,
      L1: prepareLayer(model.W1, H, ctx * V, cfg, 2 * seed),
      L2: prepareLayer(model.W2, V, H, cfg, 2 * seed + 1),
    };
  }

  function makeActivation(cfg, act, H) {
    const consumed = new Float64Array(H);
    const fuel = act === "seesaw" && cfg.fuel_budget != null && isFinite(cfg.fuel_budget);
    let left = new Float64Array(H).fill(1), inChunk = 0;
    return {
      // h is modified in place for one token; tokens must arrive in stream order
      apply(h) {
        if (act === "relu") { for (let j = 0; j < H; j++) h[j] = Math.max(0, h[j]); return; }
        if (fuel && inChunk === 0) {
          for (let j = 0; j < H; j++) left[j] = Math.max(0, 1 - consumed[j] / cfg.fuel_budget);
        }
        for (let j = 0; j < H; j++) {
          let y = Math.max(0, cfg.gain * (h[j] - cfg.threshold));
          const cap = cfg.y_max == null ? Infinity : cfg.y_max;
          y = Math.min(y, fuel ? cap * left[j] : cap);
          h[j] = y;
          if (fuel) consumed[j] += y;
        }
        if (fuel) inChunk = (inChunk + 1) % (cfg.fuel_chunk || 64);
      },
    };
  }

  function logits(prep, ctxIds, actFn, h, out) {
    const { L1, L2, H, V, ctx } = prep;
    for (let j = 0; j < H; j++) {
      const row = j * L1.cols;
      let s = L1.net[row + L1.nIn] + L1.leak[j];
      for (let p = 0; p < ctx; p++) s += L1.net[row + p * V + ctxIds[p]];
      h[j] = s;
    }
    actFn.apply(h);
    for (let k = 0; k < V; k++) {
      const row = k * L2.cols;
      let s = L2.net[row + L2.nIn] + L2.leak[k];
      for (let j = 0; j < H; j++) s += L2.net[row + j] * h[j];
      out[k] = s;
    }
    return out;
  }

  // Perplexity over a stream of character ids, in order. Returns ppl and windowed ppl.
  function evaluate(prep, cfg, act, ids, windowSize) {
    const actFn = makeActivation(cfg, act, prep.H);
    const h = new Float64Array(prep.H), out = new Float64Array(prep.V);
    const ctxIds = new Int32Array(prep.ctx);
    let nll = 0, wnll = 0;
    const windows = [];
    const n = ids.length - prep.ctx;
    for (let t = 0; t < n; t++) {
      for (let p = 0; p < prep.ctx; p++) ctxIds[p] = ids[t + p];
      logits(prep, ctxIds, actFn, h, out);
      let mx = -Infinity;
      for (let k = 0; k < prep.V; k++) mx = Math.max(mx, out[k]);
      let z = 0;
      for (let k = 0; k < prep.V; k++) z += Math.exp(out[k] - mx);
      const l = Math.log(z) + mx - out[ids[t + prep.ctx]];
      nll += l; wnll += l;
      if (windowSize && (t + 1) % windowSize === 0) { windows.push(Math.exp(wnll / windowSize)); wnll = 0; }
    }
    return { ppl: Math.exp(nll / n), windows };
  }

  function generate(prep, cfg, act, promptIds, nChars, temperature, seed) {
    const actFn = makeActivation({ ...cfg, fuel_budget: null }, act, prep.H);
    const rand = mulberry32(seed * 7919 + 17);
    const h = new Float64Array(prep.H), out = new Float64Array(prep.V);
    const ids = promptIds.slice(-prep.ctx);
    const gen = [];
    for (let i = 0; i < nChars; i++) {
      logits(prep, ids.slice(-prep.ctx), actFn, h, out);
      let mx = -Infinity;
      for (let k = 0; k < prep.V; k++) mx = Math.max(mx, out[k] / temperature);
      const probs = new Float64Array(prep.V);
      let z = 0;
      for (let k = 0; k < prep.V; k++) { probs[k] = Math.exp(out[k] / temperature - mx); z += probs[k]; }
      let r = rand() * z, pick = prep.V - 1;
      for (let k = 0; k < prep.V; k++) { r -= probs[k]; if (r <= 0) { pick = k; break; } }
      ids.push(pick); gen.push(pick);
    }
    return gen;
  }

  function accounting(prep, cfg, act) {
    const kT = K_B * cfg.temperature_K;
    const copies = cfg.copies_per_inference;
    let ops = (prep.L1.gates + prep.L2.gates) * cfg.ops_per_weight;
    if (act === "seesaw") ops += prep.H * cfg.ops_per_neuron;
    ops *= copies;
    const settle = cfg.architecture === "bulk" ? cfg.settle_bulk_s : cfg.settle_localised_s;
    const latency = 2 * settle;
    const readoutS = cfg.readout_time_s == null ? latency : cfg.readout_time_s;
    const E_chem = ops * cfg.E_op_kT * kT;
    const E_read = cfg.readout_power_W * readoutS;
    const batch = cfg.n_parallel / copies;
    const strands = prep.ctx * prep.V + (prep.L1.gates + 3 * prep.H) + (prep.L2.gates + 3 * prep.V) + 3 * prep.H;
    return {
      ops, E_chem, E_wallplug: E_chem + E_read, E_amortised: E_chem + E_read / batch,
      E_landauer: ops * kT * Math.LN2, latency, throughput_s: latency / batch,
      strands, weight_gates: prep.L1.gates + prep.L2.gates,
      dual_rail_tax: 2 * (prep.H + prep.V),
      feasible: strands <= cfg.max_orthogonal_domains,
    };
  }

  function unpackModel(data, key) {
    const m = data.models[key];
    return { ...m, V: data.vocab.length, ctx: data.ctx };
  }

  function encode(data, text) {
    const idx = {};
    for (let i = 0; i < data.vocab.length; i++) idx[data.vocab[i]] = i;
    const space = idx[" "];
    return Array.from(text.toLowerCase(), (c) => (c in idx ? idx[c] : space));
  }

  const api = { prepare, evaluate, generate, accounting, unpackModel, encode };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.DNASim = api;
})(typeof window !== "undefined" ? window : globalThis);
