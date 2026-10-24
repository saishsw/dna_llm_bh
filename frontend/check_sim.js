// node frontend/check_sim.js  -- the JS substrate must reproduce PyTorch on deterministic configs
const fs = require("fs");
const path = require("path");
const sim = require("./sim.js");

const src = fs.readFileSync(path.join(__dirname, "data.js"), "utf8");
const data = JSON.parse(src.slice(src.indexOf("=") + 1).trim().replace(/;$/, ""));
const ids = sim.encode(data, data.sample);
let fail = 0;
for (const r of data.reference) {
  const prep = sim.prepare(sim.unpackModel(data, r.model), r.cfg, 0);
  const t = Date.now();
  const { ppl } = sim.evaluate(prep, r.cfg, r.act, ids);
  const rel = Math.abs(ppl - r.ppl) / r.ppl;
  const ok = rel < 1e-4;
  fail += !ok;
  console.log(`${ok ? "PASS" : "FAIL"}  ${r.model} ${r.act}  torch ${r.ppl.toFixed(5)}  js ${ppl.toFixed(5)}  rel ${rel.toExponential(1)}  ${Date.now() - t}ms`);
}
process.exit(fail ? 1 : 0);
