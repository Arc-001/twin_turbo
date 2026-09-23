// Verifies the dashboard's JS replay engine against the Python results.
// Usage: node scripts/check_dashboard.mjs
import { createRequire } from "module";
import { readFileSync } from "fs";
const require = createRequire(import.meta.url);
globalThis.window = {};
require("../dashboard/demo-data.js");
const R = require("../dashboard/replay.js");
const D = globalThis.window.TWIN_DATA;
const results = JSON.parse(readFileSync(new URL("../artifacts/results/results.json", import.meta.url)));

const keys = ["failures", "mean_wasted_life", "cost_rate_x1000", "twin_sync_rate"];
let worst = 0, mismatchedFailures = 0, checks = 0;
for (const [name, ds] of Object.entries(D.datasets)) {
  const cfg = R.defaultConfig(D.defaults);
  const py = results.datasets[name].decisions.default;
  const rows = [];
  for (const p of [...R.POLICIES, "age_replacement", "run_to_failure"]) {
    const js = R.evaluatePolicy(ds, p, cfg, D.scale).summary;
    const row = { policy: p };
    for (const k of keys) {
      const rel = Math.abs(js[k] - py[p][k]) / Math.max(1e-9, Math.abs(py[p][k]));
      if (k === "failures" && js[k] !== py[p][k]) mismatchedFailures++;
      if (k !== "failures") worst = Math.max(worst, rel);
      row[k] = `${+js[k].toFixed(3)} / ${+py[p][k].toFixed(3)}`;
      checks++;
    }
    rows.push(row);
  }
  console.log(`\n${name}  (js / python)`);
  console.table(rows);
}
console.log(`\n${checks} checks; failure-count mismatches: ${mismatchedFailures}; worst relative diff on continuous metrics: ${(worst * 100).toFixed(2)}%`);
process.exit(mismatchedFailures ? 1 : 0);
