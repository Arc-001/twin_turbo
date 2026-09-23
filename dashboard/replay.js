/* Decision replay engine -- a line-for-line port of src/decision/hybrid.py
 * (EscalationMonitor) and src/decision/replay.py (hybrid_lower_bound,
 * unit_outcome, age_outcome, aggregate). Shared by the dashboard and by
 * scripts/check_dashboard.mjs, which verifies it against the Python results.
 */
(function (root) {
  "use strict";

  const POLICIES = ["hybrid", "edge_conformal", "edge_point", "twin_always", "twin_point",
                    "lstm_mc", "lstm_point", "oracle"];

  function decode(deltas, scale) {
    const out = new Float64Array(deltas.length);
    let acc = 0;
    for (let i = 0; i < deltas.length; i++) { acc += deltas[i]; out[i] = acc / scale; }
    return out;
  }

  /* Decode one exported engine into plain Float64Arrays (cached on the object). */
  function engineSeries(e, scale) {
    if (e._s) return e._s;
    const r = scale.rul;
    const s = {
      edge: decode(e.edge, r), elo: decode(e.elo, r), ehi: decode(e.ehi, r),
      t05: decode(e.t05, r), t50: decode(e.t50, r), t95: decode(e.t95, r),
      lstm: decode(e.lstm, r), llo: decode(e.llo, r),
      hi: decode(e.hi, scale.hi),
      sens: e.sens.map((d) => decode(d, scale.sens)),
      n: e.edge.length,
    };
    s.rulTrue = new Float64Array(s.n);
    for (let i = 0; i < s.n; i++) s.rulTrue[i] = e.life - (i + 1);
    e._s = s;
    return s;
  }

  /* EscalationMonitor.step, stateful across calls. */
  function makeMonitor(esc) {
    let last = null;
    let since = esc.resync;
    return function step(edgeRul) {
      since += 1;
      const reasons = [];
      if (esc.lowRul && edgeRul <= esc.watchRul) reasons.push("low_rul");
      if (esc.sharpDrop && last !== null && last - edgeRul >= esc.deltaDrop) reasons.push("sharp_drop");
      if (esc.periodic && since >= esc.resync) reasons.push("periodic_resync");
      if (last === null && reasons.length === 0) reasons.push("initial_sync");
      if (reasons.length) since = 0;
      last = edgeRul;
      return reasons;
    };
  }

  /* Per-cycle lower bound, twin-synced flags and trigger reasons for one policy. */
  function lowerBound(s, policy, cfg) {
    const n = s.n;
    const synced = new Uint8Array(n);
    let lo;
    let reasons = null;
    switch (policy) {
      case "edge_point": lo = s.edge; break;
      case "edge_conformal": lo = s.elo; break;
      case "lstm_point": lo = s.lstm; break;
      case "lstm_mc": lo = s.llo; break;
      case "twin_point": lo = s.t50; synced.fill(1); break;
      case "twin_always": lo = s.t05; synced.fill(1); break;
      case "oracle": lo = s.rulTrue; break;
      case "hybrid": {
        const step = makeMonitor(cfg);
        lo = Float64Array.from(s.elo);
        reasons = new Array(n);
        for (let i = 0; i < n; i++) {
          const r = step(s.edge[i]);
          reasons[i] = r;
          if (r.length) {
            synced[i] = 1;
            lo[i] = cfg.fusion === "twin" ? s.t05[i] : Math.min(s.t05[i], s.elo[i]);
          }
        }
        break;
      }
      default: throw new Error("unknown policy " + policy);
    }
    return { lo, synced, reasons };
  }

  function zoneOf(lo, cfg) {
    if (lo <= cfg.ground) return "GROUND_NOW";
    if (lo <= cfg.watch) return "SCHEDULE_MAINTENANCE";
    if (lo <= cfg.safe) return "WATCH";
    return "SAFE";
  }

  function unitOutcome(life, lo, synced, ground) {
    const n = lo.length;
    let stop = n - 1;
    for (let i = 0; i < n; i++) if (lo[i] <= ground) { stop = i; break; }
    const rulAtStop = life - (stop + 1);
    const failed = rulAtStop <= 0;
    let syncs = 0;
    for (let i = 0; i <= stop; i++) syncs += synced[i];
    return { life, operated: failed ? life : stop + 1, failed, wasted: failed ? 0 : rulAtStop,
             syncs, cycles: stop + 1, stop };
  }

  function ageOutcome(life, age) {
    const failed = life <= age;
    return { life, operated: failed ? life : age, failed, wasted: failed ? 0 : life - age,
             syncs: 0, cycles: Math.min(life, age), stop: Math.min(life, age) - 1 };
  }

  function aggregate(outcomes, costs) {
    let fail = 0, wasted = 0, operated = 0, life = 0, syncs = 0, cycles = 0;
    for (const o of outcomes) {
      if (o.failed) fail++; else wasted += o.wasted;
      operated += o.operated; life += o.life; syncs += o.syncs; cycles += o.cycles;
    }
    const pm = outcomes.length - fail;
    return {
      n_units: outcomes.length, failures: fail, failure_rate: fail / outcomes.length,
      mean_wasted_life: pm ? wasted / pm : 0, life_utilization: operated / life,
      cost_rate_x1000: 1000 * (costs.failure * fail + costs.preventive * pm) / operated,
      twin_sync_rate: syncs / cycles,
    };
  }

  function evaluatePolicy(ds, policy, cfg, scale) {
    const outcomes = [];
    for (const e of ds.engines) {
      if (policy === "age_replacement") { outcomes.push(ageOutcome(e.life, e.age)); continue; }
      if (policy === "run_to_failure") { outcomes.push(ageOutcome(e.life, 1e9)); continue; }
      const s = engineSeries(e, scale);
      const { lo, synced } = lowerBound(s, policy, cfg);
      outcomes.push(unitOutcome(e.life, lo, synced, cfg.ground));
    }
    return { summary: aggregate(outcomes, { preventive: cfg.costPreventive, failure: cfg.costFailure }), outcomes };
  }

  function defaultConfig(defaults) {
    return {
      safe: defaults.safe, watch: defaults.watch, ground: defaults.ground,
      watchRul: defaults.watchRul, deltaDrop: defaults.deltaDrop, resync: defaults.resync,
      lowRul: true, sharpDrop: true, periodic: true, fusion: "min",
      costPreventive: defaults.costPreventive, costFailure: defaults.costFailure,
    };
  }

  const api = { POLICIES, decode, engineSeries, makeMonitor, lowerBound, zoneOf, unitOutcome,
                ageOutcome, aggregate, evaluatePolicy, defaultConfig };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.TwinReplay = api;
})(typeof window !== "undefined" ? window : globalThis);
