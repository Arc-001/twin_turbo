/* Twin Turbo console. Data: window.TWIN_DATA (scripts/export_dashboard.py).
 * Decision logic: window.TwinReplay (replay.js, verified against Python by
 * scripts/check_dashboard.mjs). Twin lab runs its own particle filter that
 * mirrors src/twin/particle_filter.py. */
(function () {
  "use strict";
  const D = window.TWIN_DATA;
  const R = window.TwinReplay;
  const SC = D.scale;
  const DATASETS = Object.keys(D.datasets);
  const SVGNS = "http://www.w3.org/2000/svg";

  /* ------------------------------------------------------------ helpers */
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  function S(tag, attrs, parent) {
    const n = document.createElementNS(SVGNS, tag);
    if (attrs) for (const k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function T(parent, x, y, text, attrs) {
    const n = S("text", Object.assign({ x, y }, attrs || {}), parent);
    n.textContent = text;
    return n;
  }
  function H(tag, attrs, html) {
    const n = document.createElement(tag);
    if (attrs) for (const k in attrs) {
      if (k === "class") n.className = attrs[k];
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] !== undefined && attrs[k] !== null) n.setAttribute(k, attrs[k]);
    }
    if (html !== undefined) n.innerHTML = html;
    return n;
  }
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
  const f0 = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : Math.round(v).toString();
  const f1 = (v) => (v === null || v === undefined || !isFinite(v)) ? "—" : v.toFixed(1);
  const pct = (v, d = 0) => (100 * v).toFixed(d) + "%";
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function svgFor(container, height) {
    container.innerHTML = "";
    const w = Math.max(280, container.clientWidth || 600);
    const svg = S("svg", { class: "chart", width: w, height, viewBox: `0 0 ${w} ${height}`, role: "img" }, container);
    return { svg, w, h: height };
  }
  function lin(d0, d1, r0, r1) {
    const k = (r1 - r0) / ((d1 - d0) || 1);
    const f = (v) => r0 + (v - d0) * k;
    f.inv = (p) => d0 + (p - r0) / k;
    return f;
  }
  function pathFrom(xs, ys, n) {
    let d = "";
    for (let i = 0; i < n; i++) d += (i ? "L" : "M") + xs(i).toFixed(1) + " " + ys(i).toFixed(1);
    return d;
  }
  function niceTicks(max, target) {
    const raw = max / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = 0; v <= max + 1e-9; v += step) out.push(+v.toFixed(6));
    return out;
  }
  function store(k, v) { try { localStorage.setItem("twinturbo." + k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } }
  function load(k, d) { try { const v = localStorage.getItem("twinturbo." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }

  const ZONE_LABEL = { SAFE: "Safe", WATCH: "Watch", SCHEDULE_MAINTENANCE: "Schedule", GROUND_NOW: "Ground now", GROUNDED: "Removed", FAILED: "Failed" };
  const chip = (zone, text) => `<span class="chip z-${zone}"><i></i>${esc(text || ZONE_LABEL[zone])}</span>`;

  const POLICY = {
    hybrid: { name: "Hybrid (ours)", desc: "edge every cycle, twin on trigger, cautious fusion" },
    edge_conformal: { name: "Edge + conformal", desc: "edge lower 95% bound, no twin" },
    edge_point: { name: "Edge point estimate", desc: "no uncertainty" },
    twin_always: { name: "Twin every cycle", desc: "posterior 5% quantile" },
    twin_point: { name: "Twin median", desc: "no uncertainty margin" },
    lstm_mc: { name: "LSTM + MC-dropout", desc: "mean − 1.645σ" },
    lstm_point: { name: "LSTM point estimate", desc: "deep baseline" },
    age_replacement: { name: "Age replacement", desc: "retire at fixed age (Barlow–Hunter)" },
    oracle: { name: "Oracle", desc: "knows the true RUL" },
    run_to_failure: { name: "Run to failure", desc: "never remove" },
  };
  const REASON = { low_rul: "low RUL", sharp_drop: "sharp drop", periodic_resync: "periodic", initial_sync: "first cycle" };

  const SCENARIOS = [
    { ds: "FD004", u: 50, title: "Point estimates fly it to failure", sub: "FD004 · engine 50 · 6 regimes, 2 fault modes" },
    { ds: "FD003", u: 55, title: "Age replacement wastes most of a life", sub: "FD003 · engine 55 · lives 525 cycles" },
    { ds: "FD004", u: 114, title: "A short-lived engine", sub: "FD004 · engine 114 · fails at cycle 161" },
    { ds: "FD001", u: 64, title: "Textbook degradation", sub: "FD001 · engine 64 · one regime, one fault" },
  ];

  /* ------------------------------------------------------------ state */
  const state = {
    view: "story",
    ds: load("ds", "FD004"),
    cfg: R.defaultConfig(D.defaults),
    engine: 0,
    cycle: 1,
    playing: false,
    speed: 4,
    reveal: false,
    fleetPolicy: "hybrid",
    fleetCycle: 1,
    fleetPlaying: false,
    presenting: false,
    step: 0,
  };
  if (!D.datasets[state.ds]) state.ds = "FD004";
  const ds = () => D.datasets[state.ds];
  const engine = () => ds().engines[state.engine];

  /* ============================================================ top bar */
  function buildTopbar() {
    const seg = $("#ds-seg");
    for (const name of DATASETS) {
      seg.appendChild(H("button", { type: "button", "data-ds": name, onclick: () => setDataset(name) }, name));
    }
    $$(".tab").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
    $("#present-btn").addEventListener("click", () => togglePresent());
    $("#foot-gen").textContent = `Exported ${D.generated}`;
  }
  function syncTopbar() {
    $$("#ds-seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.ds === state.ds));
    $$(".tab").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === state.view));
  }
  function setDataset(name, engineUnit) {
    if (!D.datasets[name]) return;
    const changed = name !== state.ds;
    state.ds = name;
    store("ds", name);
    if (changed || engineUnit !== undefined) {
      const idx = engineUnit !== undefined ? ds().engines.findIndex((e) => e.u === engineUnit) : defaultEngineIndex();
      state.engine = Math.max(0, idx);
      mc.invalidate();
      state.cycle = restingCycle();
      state.playing = false;
      state.fleetCycle = 1;
      state.fleetPlaying = false;
      lab.engine = state.engine;
      lab.reset();
      fleetCache.clear();
    }
    syncTopbar();
    populateEngineSelects();
    renderView();
  }
  function defaultEngineIndex() {
    const sc = SCENARIOS.find((s) => s.ds === state.ds);
    if (sc) return ds().engines.findIndex((e) => e.u === sc.u);
    return 0;
  }
  function setView(v, opts) {
    state.view = v;
    $$(".view").forEach((s) => (s.hidden = s.id !== "view-" + v));
    const slot = $("#settings-slot-" + v);
    if (slot) slot.appendChild(settingsEl);
    syncTopbar();
    if (!(opts && opts.keepHash)) { try { history.replaceState(null, "", "#" + v); } catch (e) { /* sandboxed */ } }
    renderView();
  }
  function renderView() {
    if (state.view === "story") renderStory();
    else if (state.view === "mission") renderMission();
    else if (state.view === "fleet") renderFleet();
    else if (state.view === "twinlab") lab.render();
    else if (state.view === "evidence") renderEvidence();
  }

  /* ============================================================ settings */
  let settingsEl;
  const SETTINGS = [
    ["ground", "s-ground", "o-ground", (v) => v],
    ["watch", "s-watch", "o-watch", (v) => v],
    ["safe", "s-safe", "o-safe", (v) => v],
    ["watchRul", "s-watchrul", "o-watchrul", (v) => v],
    ["deltaDrop", "s-delta", "o-delta", (v) => v],
    ["resync", "s-resync", "o-resync", (v) => v + " cycles"],
    ["costFailure", "s-cost", "o-cost", (v) => v + "×"],
  ];
  function buildSettings() {
    settingsEl = $("#settings-template").content.firstElementChild.cloneNode(true);
    for (const [key, id, out] of SETTINGS) {
      const input = $("#" + id, settingsEl);
      input.addEventListener("input", () => { state.cfg[key] = +input.value; onConfigChange(); });
    }
    const toggles = [["lowRul", "s-lowrul"], ["sharpDrop", "s-sharp"], ["periodic", "s-periodic"]];
    for (const [key, id] of toggles) {
      $("#" + id, settingsEl).addEventListener("change", (e) => { state.cfg[key] = e.target.checked; onConfigChange(); });
    }
    $("#s-fusion", settingsEl).addEventListener("change", (e) => { state.cfg.fusion = e.target.value; onConfigChange(); });
    $("#s-reset", settingsEl).addEventListener("click", () => { state.cfg = R.defaultConfig(D.defaults); onConfigChange(); });
    syncSettings();
  }
  function syncSettings() {
    const c = state.cfg;
    for (const [key, id, out, fmt] of SETTINGS) {
      $("#" + id, settingsEl).value = c[key];
      $("#" + out, settingsEl).textContent = fmt(c[key]);
    }
    $("#s-lowrul", settingsEl).checked = c.lowRul;
    $("#s-sharp", settingsEl).checked = c.sharpDrop;
    $("#s-periodic", settingsEl).checked = c.periodic;
    $("#s-fusion", settingsEl).value = c.fusion;
    const d = R.defaultConfig(D.defaults);
    const changed = Object.keys(d).filter((k) => d[k] !== c[k]).length;
    $("#settings-summary", settingsEl).textContent = changed
      ? `${changed} changed from paper settings · ground ≤ ${c.ground}`
      : `paper settings · ground ≤ ${c.ground}, schedule ≤ ${c.watch}, safe > ${c.safe}`;
  }
  let cfgTimer = null;
  function onConfigChange() {
    if (state.cfg.watch < state.cfg.ground) state.cfg.watch = state.cfg.ground;
    if (state.cfg.safe < state.cfg.watch) state.cfg.safe = state.cfg.watch;
    syncSettings();
    mc.invalidate();
    fleetCache.clear();
    if (state.view === "mission") renderMission();
    if (state.view === "fleet") {
      renderFleetGrid();
      clearTimeout(cfgTimer);
      cfgTimer = setTimeout(renderFleet, 120);
    }
  }

  /* ============================================================ STORY */
  function renderStory() {
    const res = Object.fromEntries(DATASETS.map((n) => [n, D.datasets[n].results]));
    let units = 0, fails = 0, syncW = 0, cycW = 0;
    const savings = [];
    for (const n of DATASETS) {
      const d = res[n].decisions.default;
      units += d.hybrid.n_units; fails += d.hybrid.failures;
      syncW += d.hybrid.twin_sync_rate * d.hybrid.n_units; cycW += d.hybrid.n_units;
      savings.push(1 - d.hybrid.cost_rate_x1000 / d.age_replacement.cost_rate_x1000);
    }
    const minSave = Math.min(...savings), maxSave = Math.max(...savings);
    const fd2 = res.FD002.benchmark.accuracy.twin.rmse;
    const lit2 = Math.min(...D.literature.map((r) => r.FD002));
    $("#hero-stats").innerHTML = `
      <div><span class="label">In-service failures</span><span class="big num">${fails}<span class="muted" style="font-size:.5em"> / ${units}</span></span><p>engines the hybrid policy let fail, across all four datasets.</p></div>
      <div><span class="label">Cost vs age replacement</span><span class="big num">−${Math.round(minSave * 100)}–${Math.round(maxSave * 100)}%</span><p>long-run maintenance cost rate, failures priced at 10× a planned removal.</p></div>
      <div><span class="label">Twin consulted</span><span class="big num">${Math.round((syncW / cycW) * 100)}%</span><p>of cycles; every other cycle runs on the edge model alone.</p></div>
      <div><span class="label">Twin RMSE, FD002</span><span class="big num">${fd2.toFixed(1)}</span><p>on the official test set, vs ${lit2.toFixed(2)} for the best published method we compare against.</p></div>`;

    // one engine, full life
    const d1 = D.datasets.FD001;
    const e = d1.engines.find((x) => x.u === 64) || d1.engines[0];
    const s = R.engineSeries(e, SC);
    $("#story-engine-label").textContent = `FD001 · engine ${e.u} · ${e.life} cycles`;
    const box = $("#story-sensors");
    box.innerHTML = "";
    const grid = H("div", { class: "sensor-row", style: "grid-template-columns:repeat(2,minmax(0,1fr))" });
    box.appendChild(grid);
    d1.strip.forEach((code, k) => grid.appendChild(sparkCard(code, s.sens[k], s.n, s.n, false)));
    grid.appendChild(sparkCard("HI", s.hi, s.n, s.n, true));

    drawArchitecture();
    renderImportance();
  }

  function sensorName(code) {
    if (code === "HI") return ["HI", "Health index (twin input)"];
    return D.sensorInfo[code] || [code, ""];
  }
  function sparkCard(code, arr, n, upto, isHi) {
    const [short, long] = sensorName(code);
    const card = H("div", { class: "sensor" });
    const cur = arr[Math.max(0, upto - 1)];
    card.appendChild(H("div", { class: "sname" }, `<span><b>${esc(short)}</b> ${esc(long)}</span><span class="num">${isHi ? cur.toFixed(2) : cur.toFixed(2)}</span>`));
    const holder = H("div");
    card.appendChild(holder);
    requestAnimationFrame(() => drawSpark(holder, arr, n, upto, isHi));
    return card;
  }
  function drawSpark(holder, arr, n, upto, isHi) {
    const { svg, w, h } = svgFor(holder, 54);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { lo = Math.min(lo, arr[i]); hi = Math.max(hi, arr[i]); }
    const pad = (hi - lo) * 0.08 || 1;
    const x = lin(0, Math.max(1, n - 1), 2, w - 2), y = lin(lo - pad, hi + pad, h - 3, 3);
    S("line", { x1: 0, x2: w, y1: y(isHi ? 0 : 0), y2: y(isHi ? 0 : 0), class: "gridline" }, svg);
    S("path", { d: pathFrom((i) => x(i), (i) => y(arr[i]), Math.min(upto, n)), class: isHi ? "spark-hi" : "spark" }, svg);
    if (upto > 0) S("circle", { cx: x(upto - 1), cy: y(arr[upto - 1]), r: 2.6, class: isHi ? "twin-mid" : "hi-dot" }, svg);
  }

  function drawArchitecture() {
    const svg = $("#arch-diagram");
    if (svg.childNodes.length) return;
    const box = (x, y, w, h, cls, title, sub, titleCls) => {
      S("rect", { x, y, width: w, height: h, rx: 4, class: "box " + cls }, svg);
      T(svg, x + 12, y + 27, title, { class: "t-title " + (titleCls || "") });
      T(svg, x + 12, y + 48, sub);
    };
    const arrow = (x, y, dir, cls) => {
      const p = { r: `M${x} ${y} l-7 -4 v8 z`, l: `M${x} ${y} l7 -4 v8 z`, u: `M${x} ${y} l-4 7 h8 z`, d: `M${x} ${y} l-4 -7 h8 z` }[dir];
      S("path", { d: p, class: "arrow " + (cls || "") }, svg);
    };
    box(16, 110, 150, 70, "", "Sensors", "30-cycle window");
    box(206, 110, 170, 70, "edge", "Edge model", "every cycle · conformal", "t-edge");
    box(426, 110, 150, 70, "policy", "Policy", "min lower bound");
    box(616, 110, 128, 70, "", "Action", "4 zones");
    box(326, 222, 190, 62, "twin", "Digital twin", "particle filter · 1/engine", "t-twin");
    box(566, 222, 178, 62, "", "Fleet prior", "run-to-failure fleet");
    S("path", { d: "M166 145 H199", class: "wire" }, svg); arrow(206, 145, "r");
    S("path", { d: "M376 145 H419", class: "wire" }, svg); arrow(426, 145, "r");
    S("path", { d: "M576 145 H609", class: "wire" }, svg); arrow(616, 145, "r");
    S("path", { d: "M291 180 V253 H319", class: "wire dash" }, svg); arrow(326, 253, "r", "twin");
    T(svg, 222, 206, "trigger:");
    T(svg, 222, 222, "sync HI buffer");
    S("path", { d: "M501 222 V187", class: "wire dash" }, svg); arrow(501, 180, "u", "twin");
    T(svg, 510, 206, "RUL posterior");
    S("path", { d: "M566 253 H523", class: "wire" }, svg); arrow(516, 253, "l");
    T(svg, 16, 40, "on board, every cycle", { class: "t-edge" });
    T(svg, 16, 58, "off board, on trigger", { class: "t-twin" });
    S("line", { x1: 16, x2: 180, y1: 70, y2: 70, class: "wire" }, svg);
    T(svg, 16, 90, "Sensor window → edge → policy runs locally.");
  }

  function renderImportance() {
    const d = ds();
    $("#imp-ds").textContent = `${d.name} · permutation importance, test set`;
    const entries = Object.entries(d.edgeImportance).filter(([k]) => k.startsWith("s")).sort((a, b) => b[1] - a[1]).slice(0, 7);
    const box = $("#edge-importance");
    const rowH = 26;
    const { svg, w } = svgFor(box, entries.length * rowH + 22);
    const labelW = Math.min(230, w * 0.46);
    const max = Math.max(...entries.map((e) => e[1]), 0.1);
    const x = lin(0, max, labelW, w - 48);
    entries.forEach(([code, v], i) => {
      const [short, long] = sensorName(code);
      const y = i * rowH + 4;
      T(svg, 0, y + 14, `${short}`, { class: "strong" });
      T(svg, 64, y + 14, long.length > 24 && labelW < 200 ? long.slice(0, 22) + "…" : long);
      S("rect", { x: labelW, y: y + 5, width: Math.max(1, x(Math.max(0, v)) - labelW), height: 11, class: "bar-edge" }, svg);
      T(svg, x(Math.max(0, v)) + 6, y + 14, `+${v.toFixed(1)}`, { class: "soft-text" });
    });
    T(svg, labelW, entries.length * rowH + 18, "RMSE increase when the sensor is shuffled (cycles)");
  }

  /* ============================================================ MISSION CONTROL */
  const mc = {
    key: null, s: null, lb: null, outcome: null, others: null,
    invalidate() { this.key = null; },
    ensure() {
      const e = engine();
      const key = state.ds + ":" + e.u + ":" + JSON.stringify(state.cfg);
      if (key === this.key) return;
      this.key = key;
      this.s = R.engineSeries(e, SC);
      this.lb = R.lowerBound(this.s, "hybrid", state.cfg);
      this.outcome = R.unitOutcome(e.life, this.lb.lo, this.lb.synced, state.cfg.ground);
      this.others = {};
      for (const p of ["hybrid", "edge_conformal", "edge_point", "twin_point", "lstm_mc", "age_replacement"]) {
        if (p === "age_replacement") { this.others[p] = R.ageOutcome(e.life, e.age); continue; }
        const l = p === "hybrid" ? this.lb : R.lowerBound(this.s, p, state.cfg);
        this.others[p] = R.unitOutcome(e.life, l.lo, l.synced, state.cfg.ground);
      }
    },
  };

  function populateEngineSelects() {
    const d = ds();
    for (const sel of [$("#engine-select"), $("#lab-engine")]) {
      sel.innerHTML = "";
      d.engines.forEach((e, i) => sel.appendChild(H("option", { value: i }, `#${e.u} · ${e.life} cycles`)));
    }
    $("#engine-select").value = state.engine;
    $("#lab-engine").value = lab.engine;
  }

  function buildMission() {
    const sc = $("#scenarios");
    SCENARIOS.forEach((s, i) => {
      sc.appendChild(H("button", { type: "button", class: "scenario", "data-i": i, onclick: () => loadScenario(i) },
        `<b>${esc(s.title)}</b><span>${esc(s.sub)}</span>`));
    });
    $("#engine-select").addEventListener("change", (e) => { state.engine = +e.target.value; state.playing = false; state.cycle = restingCycle(); renderMission(); });
    $("#reveal-truth").addEventListener("change", (e) => { state.reveal = e.target.checked; renderMission(); });
    $("#play-btn").addEventListener("click", () => togglePlay());
    $("#restart-btn").addEventListener("click", () => { state.cycle = 1; renderMission(); });
    $("#scrub").addEventListener("input", (e) => { state.cycle = +e.target.value; state.playing = false; renderMission(); });
    const speeds = [1, 4, 16];
    const seg = $("#speed-seg");
    speeds.forEach((v) => seg.appendChild(H("button", { type: "button", "data-speed": v, onclick: () => { state.speed = v; renderTransport(); } }, v + "×")));
  }
  function restingCycle() {
    mc.ensure();
    return Math.max(1, Math.round(mc.outcome.stop * 0.72));
  }
  function loadScenario(i, opts) {
    const s = SCENARIOS[i];
    setDataset(s.ds, s.u);
    state.cycle = opts && opts.play ? 1 : restingCycle();
    state.reveal = !!(opts && opts.reveal);
    if (state.view !== "mission") setView("mission");
    else renderMission();
    if (opts && opts.play) startPlay();
  }
  function togglePlay() { if (state.playing) state.playing = false; else startPlay(); renderTransport(); }
  function startPlay() {
    mc.ensure();
    const e = engine();
    if (state.cycle >= e.life) state.cycle = 1;
    state.playing = true;
    renderTransport();
  }
  let lastTick = 0;
  function tick(ts) {
    if (state.playing && state.view === "mission") {
      const interval = 1000 / (8 * state.speed);
      if (ts - lastTick >= interval) {
        lastTick = ts;
        mc.ensure();
        const e = engine();
        const stopCycle = mc.outcome.stop + 1;
        state.cycle += 1;
        if (state.cycle === stopCycle || state.cycle >= e.life) { state.playing = false; state.cycle = Math.min(state.cycle, e.life); }
        renderMission(true);
      }
    }
    if (state.fleetPlaying && state.view === "fleet") {
      if (ts - fleetTick >= 1000 / 30) {
        fleetTick = ts;
        state.fleetCycle += 2;
        const maxLife = Math.max(...ds().engines.map((e) => e.life));
        if (state.fleetCycle >= maxLife) { state.fleetCycle = maxLife; state.fleetPlaying = false; }
        renderFleetGrid();
      }
    }
    if (lab.playing && state.view === "twinlab") {
      if (ts - lab.tick >= 1000 / 24) { lab.tick = ts; lab.advance(); }
    }
    requestAnimationFrame(tick);
  }
  let fleetTick = 0;

  function renderTransport() {
    const e = engine();
    const scrub = $("#scrub");
    scrub.max = e.life;
    scrub.value = state.cycle;
    $("#cycle-readout").textContent = `cycle ${state.cycle} / ${state.reveal ? e.life : "?"}`;
    const pb = $("#play-btn");
    pb.innerHTML = (state.playing ? "Pause" : "Play") + ' <kbd>Space</kbd>';
    pb.setAttribute("aria-label", state.playing ? "Pause" : "Play");
    $$("#speed-seg button").forEach((b) => b.setAttribute("aria-pressed", +b.dataset.speed === state.speed));
  }

  function renderMission(fromTick) {
    mc.ensure();
    const e = engine();
    state.cycle = clamp(state.cycle, 1, e.life);
    $("#engine-select").value = state.engine;
    $("#reveal-truth").checked = state.reveal;
    $("#legend-truth").hidden = !state.reveal;
    $$(".scenario").forEach((b) => {
      const s = SCENARIOS[+b.dataset.i];
      b.setAttribute("aria-pressed", s.ds === state.ds && s.u === e.u);
    });
    renderTransport();
    drawMissionChart();
    renderStatePanel();
    if (!fromTick || state.cycle % 4 === 0 || !state.playing) {
      renderSensorRow();
      renderCompare();
    }
  }

  function drawMissionChart() {
    const e = engine(), s = mc.s, { lo, synced, reasons } = mc.lb, cfg = state.cfg;
    const box = $("#mc-chart");
    const H0 = box.clientWidth < 600 ? 300 : 380;
    const { svg, w, h } = svgFor(box, H0);
    const m = { l: 40, r: 14, t: 10, b: 58 };
    const plotB = h - m.b;
    const YMAX = 160;
    const x = lin(1, Math.max(2, e.life), m.l, w - m.r);
    const y = lin(0, YMAX, plotB, m.t);
    const idx = state.cycle - 1;
    const stop = mc.outcome.stop;

    // zone bands
    const bands = [["GROUND_NOW", 0, cfg.ground], ["SCHEDULE_MAINTENANCE", cfg.ground, cfg.watch], ["WATCH", cfg.watch, cfg.safe], ["SAFE", cfg.safe, YMAX]];
    for (const [z, a, b] of bands) {
      if (b <= a) continue;
      S("rect", { x: m.l, y: y(b), width: w - m.r - m.l, height: y(a) - y(b), class: "zone-" + z }, svg);
      T(svg, w - m.r - 6, y(b) + 12, ZONE_LABEL[z].toUpperCase(), { class: "zone-label", "text-anchor": "end" });
    }
    for (const v of [0, 30, 60, 90, 120, 150]) {
      S("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v), class: "gridline" }, svg);
      T(svg, m.l - 6, y(v) + 4, v, { "text-anchor": "end" });
    }
    const xt = niceTicks(e.life, w < 600 ? 4 : 8);
    for (const v of xt) {
      if (v < 1) continue;
      S("line", { x1: x(v), x2: x(v), y1: plotB, y2: plotB + 4, class: "axis-line" }, svg);
      T(svg, x(v), plotB + 16, v, { "text-anchor": "middle" });
    }
    S("line", { x1: m.l, x2: w - m.r, y1: plotB, y2: plotB, class: "axis-line" }, svg);

    const n = idx + 1;
    const cy = (v) => y(clamp(v, 0, YMAX));
    // edge band + line
    let band = "";
    for (let i = 0; i < n; i++) band += (i ? "L" : "M") + x(i + 1).toFixed(1) + " " + cy(s.ehi[i]).toFixed(1);
    for (let i = n - 1; i >= 0; i--) band += "L" + x(i + 1).toFixed(1) + " " + cy(s.elo[i]).toFixed(1);
    S("path", { d: band + "Z", class: "edge-band" }, svg);
    S("path", { d: pathFrom((i) => x(i + 1), (i) => cy(s.edge[i]), n), class: "edge-line" }, svg);
    // twin syncs
    const g = S("g", {}, svg);
    for (let i = 0; i < n; i++) {
      if (!synced[i]) continue;
      S("line", { x1: x(i + 1), x2: x(i + 1), y1: cy(s.t05[i]), y2: cy(s.t95[i]), class: "twin-bar" }, g);
      S("circle", { cx: x(i + 1), cy: cy(s.t50[i]), r: 2.2, class: "twin-mid" }, g);
    }
    // truth
    if (state.reveal) S("path", { d: pathFrom((i) => x(i + 1), (i) => cy(s.rulTrue[i]), e.life), class: "truth" }, svg);
    // decision bound
    S("path", { d: pathFrom((i) => x(i + 1), (i) => cy(lo[i]), Math.min(n, stop + 1)), class: "lb-line" }, svg);
    // cursor
    S("line", { x1: x(state.cycle), x2: x(state.cycle), y1: m.t, y2: plotB, class: "cursor" }, svg);
    // event
    if (idx >= stop) {
      if (mc.outcome.failed) {
        S("rect", { x: x(e.life) - 5, y: y(0) - 5, width: 10, height: 10, class: "event-fail" }, svg);
        T(svg, x(e.life) - 8, y(0) - 10, "FAILED IN SERVICE", { class: "strong", "text-anchor": "end" });
      } else {
        S("line", { x1: x(stop + 1), x2: x(stop + 1), y1: m.t, y2: plotB, class: "event-ground" }, svg);
        const tx = x(stop + 1) > w * 0.7 ? x(stop + 1) - 8 : x(stop + 1) + 8;
        T(svg, tx, m.t + 30, `GROUNDED AT CYCLE ${stop + 1}`, { class: "strong", "text-anchor": x(stop + 1) > w * 0.7 ? "end" : "start" });
        if (state.reveal) T(svg, tx, m.t + 46, `${e.life - (stop + 1)} cycles of life left`, { class: "soft-text", "text-anchor": x(stop + 1) > w * 0.7 ? "end" : "start" });
      }
    }
    // sync lane
    const laneY = plotB + 36;
    T(svg, m.l - 6, laneY + 4, "twin", { "text-anchor": "end" });
    S("line", { x1: m.l, x2: w - m.r, y1: laneY, y2: laneY, class: "gridline" }, svg);
    for (let i = 0; i < n; i++) {
      if (!reasons[i] || !reasons[i].length) continue;
      const r = reasons[i][0], cx = x(i + 1);
      if (r === "low_rul") S("rect", { x: cx - 2.5, y: laneY - 6, width: 5, height: 12, class: "lane-mark low_rul" }, svg);
      else if (r === "sharp_drop") S("path", { d: `M${cx} ${laneY - 6} l5 11 h-10 z`, class: "lane-mark sharp_drop" }, svg);
      else S("circle", { cx, cy: laneY, r: 3, class: "lane-mark " + r }, svg);
    }
    T(svg, m.l, h - 4, "■ low RUL   ▲ sharp drop   ● periodic / first", { class: "soft-text" });
    T(svg, w - m.r, h - 4, "flight cycle →", { "text-anchor": "end" });
  }

  function renderStatePanel() {
    const e = engine(), s = mc.s, { lo, synced, reasons } = mc.lb, idx = state.cycle - 1;
    const out = mc.outcome;
    let zone = R.zoneOf(lo[idx], state.cfg);
    let removed = false;
    if (idx > out.stop) { zone = out.failed ? "FAILED" : "GROUNDED"; removed = true; }
    if (out.failed && idx >= e.life - 1) zone = "FAILED";
    let lastSync = -1;
    for (let i = idx; i >= 0; i--) if (synced[i]) { lastSync = i; break; }
    let syncs = 0;
    for (let i = 0; i <= idx; i++) syncs += synced[i];
    const now = reasons[idx] && reasons[idx].length ? reasons[idx].map((r) => REASON[r]).join(", ") : "none";
    const truth = state.reveal ? `${f0(s.rulTrue[idx])} <small>cycles</small>` : `<small>hidden</small>`;
    let verdict = "";
    if (idx >= out.stop) {
      verdict = out.failed
        ? `<div class="verdict"><b>Failed in service</b>The lower bound never reached ${state.cfg.ground} before the engine ran out of life.</div>`
        : `<div class="verdict"><b>Grounded at cycle ${out.stop + 1}</b>${state.reveal ? `${out.wasted} cycles of life were left, well clear of failure.` : "Tick “Show true RUL” to see how close it was."}</div>`;
    }
    $("#state-panel").innerHTML = `
      <div class="state-zone z-${zone}">
        <span class="label">Action at cycle ${state.cycle}</span>
        <div class="zone-name"><i></i>${ZONE_LABEL[zone]}</div>
      </div>
      <dl class="kv">
        <dt>Decision lower bound</dt><dd>${removed ? "—" : f1(lo[idx])} <small>cycles</small></dd>
        <dt class="swatch-edge">Edge estimate</dt><dd>${f1(s.edge[idx])} <small>[${f0(s.elo[idx])}, ${f0(s.ehi[idx])}]</small></dd>
        <dt class="swatch-twin">Twin median</dt><dd>${lastSync >= 0 ? f1(s.t50[lastSync]) + ` <small>[${f0(s.t05[lastSync])}, ${f0(s.t95[lastSync])}]</small>` : "—"}</dd>
        <dt>Twin last synced</dt><dd>${lastSync >= 0 ? (lastSync === idx ? "this cycle" : `${idx - lastSync} cycles ago`) : "—"}</dd>
        <dt>Triggers this cycle</dt><dd>${esc(now)}</dd>
        <dt>Twin syncs so far</dt><dd>${syncs} <small>of ${idx + 1} (${pct(syncs / (idx + 1))})</small></dd>
        <dt>True RUL</dt><dd>${truth}</dd>
      </dl>${verdict}`;
  }

  function renderCompare() {
    const e = engine();
    const box = $("#compare");
    box.innerHTML = "";
    for (const [p, o] of Object.entries(mc.others)) {
      const res = o.failed ? chip("FAILED", "Failed in service") : chip("GROUNDED", `Removed at ${o.stop + 1}`);
      const detail = o.failed ? `ran to cycle ${e.life}` : `${o.wasted} cycles of life unused`;
      box.appendChild(H("div", { class: "cell" + (p === "hybrid" ? " me" : "") },
        `<span class="pname">${esc(POLICY[p].name)}</span><span class="pdesc">${esc(POLICY[p].desc)}</span>${res}<span class="pdesc">${state.reveal || p === "age_replacement" ? detail : o.failed ? "" : "tick Show true RUL"}</span>`));
    }
  }

  function renderSensorRow() {
    const d = ds(), s = mc.s, box = $("#sensor-row");
    box.innerHTML = "";
    d.strip.forEach((code, k) => box.appendChild(sparkCard(code, s.sens[k], s.n, state.cycle, false)));
    box.appendChild(sparkCard("HI", s.hi, s.n, state.cycle, true));
  }

  /* ============================================================ FLEET */
  const fleetCache = new Map();
  function fleetPolicyData(policy) {
    const key = state.ds + ":" + policy + ":" + JSON.stringify(state.cfg);
    if (fleetCache.has(key)) return fleetCache.get(key);
    const rows = ds().engines.map((e) => {
      if (policy === "age_replacement" || policy === "run_to_failure") {
        const o = R.ageOutcome(e.life, policy === "age_replacement" ? e.age : 1e9);
        return { e, lo: null, o };
      }
      const s = R.engineSeries(e, SC);
      const l = R.lowerBound(s, policy, state.cfg);
      return { e, lo: l.lo, o: R.unitOutcome(e.life, l.lo, l.synced, state.cfg.ground) };
    });
    fleetCache.set(key, rows);
    return rows;
  }
  function buildFleet() {
    const sel = $("#fleet-policy");
    for (const p of ["hybrid", "edge_conformal", "edge_point", "twin_always", "twin_point", "lstm_mc", "lstm_point", "age_replacement", "oracle"]) {
      sel.appendChild(H("option", { value: p }, POLICY[p].name));
    }
    sel.addEventListener("change", (e) => { state.fleetPolicy = e.target.value; renderFleetGrid(); });
    $("#fleet-play").addEventListener("click", () => {
      const maxLife = Math.max(...ds().engines.map((e) => e.life));
      if (!state.fleetPlaying && state.fleetCycle >= maxLife) state.fleetCycle = 1;
      state.fleetPlaying = !state.fleetPlaying;
      renderFleetGrid();
    });
    $("#fleet-scrub").addEventListener("input", (e) => { state.fleetCycle = +e.target.value; state.fleetPlaying = false; renderFleetGrid(); });
  }
  function renderFleet() {
    renderFleetGrid();
    renderScorecard();
    renderFrontier();
  }
  function renderFleetGrid() {
    const rows = fleetPolicyData(state.fleetPolicy);
    const t = state.fleetCycle;
    const maxLife = Math.max(...rows.map((r) => r.e.life));
    $("#fleet-scrub").max = maxLife;
    $("#fleet-scrub").value = t;
    $("#fleet-cycle").textContent = t;
    $("#fleet-policy").value = state.fleetPolicy;
    $("#fleet-play").textContent = state.fleetPlaying ? "Pause" : "Play";
    const cells = $("#fleet-cells");
    if (cells.childElementCount !== rows.length || cells.dataset.ds !== state.ds) {
      cells.innerHTML = "";
      cells.dataset.ds = state.ds;
      rows.forEach((r, i) => cells.appendChild(H("button", { type: "button", class: "cell-e", title: "", "aria-label": `Engine ${r.e.u}`, onclick: () => { state.engine = i; state.cycle = 1; setView("mission"); } }, "")));
    }
    let inService = 0, removed = 0, failed = 0, syncs = 0, wasted = 0;
    rows.forEach((r, i) => {
      const { e, lo, o } = r;
      let z;
      if (o.failed && t >= e.life) { z = "FAILED"; failed++; }
      else if (!o.failed && t > o.stop + 1) { z = "GROUNDED"; removed++; wasted += o.wasted; }
      else {
        inService++;
        z = lo ? R.zoneOf(lo[Math.min(t, e.life) - 1], state.cfg) : "SAFE";
        if (!o.failed && t === o.stop + 1) z = "GROUND_NOW";
      }
      const c = cells.children[i];
      c.className = "cell-e z-" + z;
      c.title = `Engine ${e.u}: ${ZONE_LABEL[z]}${z === "GROUNDED" ? `, ${o.wasted} cycles unused` : ""}`;
    });
    const cost = state.cfg.costFailure * failed + state.cfg.costPreventive * removed;
    $("#fleet-stats").innerHTML = `
      <div><span class="label">In service</span><span class="big num">${inService}</span></div>
      <div><span class="label">Removed in time</span><span class="big num">${removed}</span></div>
      <div><span class="label">Failed</span><span class="big num">${failed}</span></div>
      <div><span class="label">Cost so far</span><span class="big num">${cost}</span></div>`;
  }
  function renderScorecard() {
    const policies = ["hybrid", "edge_conformal", "edge_point", "twin_always", "twin_point", "lstm_mc", "lstm_point", "age_replacement", "oracle", "run_to_failure"];
    const rows = policies.map((p) => {
      const outs = fleetPolicyData(p).map((r) => r.o);
      return { p, a: R.aggregate(outs, { preventive: state.cfg.costPreventive, failure: state.cfg.costFailure }) };
    });
    const safe = rows.filter((r) => r.a.failures === 0 && r.p !== "oracle");
    const best = safe.length ? safe.reduce((a, b) => (a.a.cost_rate_x1000 <= b.a.cost_rate_x1000 ? a : b)).p : null;
    const maxCost = Math.max(...rows.filter((r) => r.p !== "run_to_failure").map((r) => r.a.cost_rate_x1000));
    $("#score-n").textContent = `${ds().name} · ${ds().engines.length} engines`;
    let html = `<thead><tr><th>Policy</th><th class="r">Failed</th><th class="r">Unused life</th><th class="r">Life used</th><th class="r">Cost rate</th><th class="r">Twin</th></tr></thead><tbody>`;
    for (const { p, a } of rows) {
      const w = Math.min(90, (a.cost_rate_x1000 / maxCost) * 90);
      html += `<tr class="${p === "hybrid" ? "hl" : ""}"><td>${esc(POLICY[p].name)}${p === best ? '<span class="badge">cheapest safe</span>' : ""}<span class="sub">${esc(POLICY[p].desc)}</span></td>
        <td class="r">${a.failures ? `<span class="chip z-FAILED" style="padding:2px 7px 2px 5px"><i></i>${a.failures}</span>` : "0"}</td>
        <td class="r">${a.failures === a.n_units ? "—" : f1(a.mean_wasted_life)}</td>
        <td class="r">${pct(a.life_utilization, 1)}</td>
        <td class="r"><div class="bar-cell"><span>${a.cost_rate_x1000.toFixed(2)}</span><span class="bar ${a.failures ? "fail" : ""}" style="width:${p === "run_to_failure" ? 90 : w}px"></span></div></td>
        <td class="r">${a.twin_sync_rate ? pct(a.twin_sync_rate) : "—"}</td></tr>`;
    }
    $("#scorecard").innerHTML = html + "</tbody>";
  }
  function renderFrontier() {
    const policies = [["hybrid", "var(--ink)", 2.6, ""], ["edge_conformal", "var(--accent)", 1.8, ""], ["edge_point", "var(--accent)", 1.6, "5 4"],
                      ["twin_always", "var(--twin)", 1.8, ""], ["lstm_mc", "var(--ink-mute)", 1.6, "2 3"]];
    const grounds = [];
    for (let g = 0; g <= 30; g += 2) grounds.push(g);
    const series = policies.map(([p]) => {
      const lows = ds().engines.map((e) => { const s = R.engineSeries(e, SC); const l = R.lowerBound(s, p, state.cfg); return { e, l }; });
      return grounds.map((g) => lows.reduce((acc, { e, l }) => acc + (R.unitOutcome(e.life, l.lo, l.synced, g).failed ? 1 : 0), 0));
    });
    const box = $("#frontier");
    const { svg, w, h } = svgFor(box, 250);
    const m = { l: 40, r: 120, t: 12, b: 34 };
    const ymax = Math.max(4, ...series.flat());
    const x = lin(0, 30, m.l, w - m.r), y = lin(0, ymax, h - m.b, m.t);
    for (const v of niceTicks(ymax, 4)) {
      S("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v), class: "gridline" }, svg);
      T(svg, m.l - 6, y(v) + 4, v, { "text-anchor": "end" });
    }
    for (const g of [0, 5, 10, 15, 20, 25, 30]) T(svg, x(g), h - m.b + 16, g, { "text-anchor": "middle" });
    T(svg, (m.l + w - m.r) / 2, h - 4, "ground threshold on the lower bound (cycles)", { "text-anchor": "middle" });
    S("line", { x1: x(state.cfg.ground), x2: x(state.cfg.ground), y1: m.t, y2: h - m.b, class: "cursor" }, svg);
    const ends = [];
    series.forEach((vals, k) => {
      const [p, color, width, dash] = policies[k];
      S("path", { d: pathFrom((i) => x(grounds[i]), (i) => y(vals[i]), vals.length), fill: "none", stroke: color, "stroke-width": width, "stroke-dasharray": dash || null }, svg);
      ends.push({ p, color, yv: y(vals[0]), v: vals[0] });
    });
    // labels at x=0 end, placed on the right with collision avoidance
    ends.sort((a, b) => a.yv - b.yv);
    let lastY = -Infinity;
    for (const en of ends) {
      const ly = Math.max(en.yv, lastY + 14);
      lastY = ly;
      T(svg, w - m.r + 8, ly + 4, `${POLICY[en.p].name.replace(" (ours)", "")}: ${en.v}`, { fill: en.color, style: `fill:${en.color}` });
    }
    T(svg, w - m.r + 8, m.t - 0, "failures at 0 →", { class: "soft-text" });
  }

  /* ============================================================ TWIN LAB (particle filter) */
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function cholesky(A) {
    const n = A.length, L = A.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let sum = A[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      L[i][j] = i === j ? Math.sqrt(Math.max(sum, 1e-12)) : sum / L[j][j];
    }
    return L;
  }
  class ParticleTwin {
    constructor(prior, n, seed) {
      this.N = n; this.rng = mulberry32(seed); this.prior = prior;
      this.sigma = prior.obs_std; this.shrink = 0.98; this.resamples = 0; this.tSeen = 0;
      this.z = new Float64Array(n * 4); this.logw = new Float64Array(n);
      const L = cholesky(prior.cov);
      for (let i = 0; i < n; i++) {
        const g = [this.gauss(), this.gauss(), this.gauss()];
        for (let a = 0; a < 3; a++) {
          let v = prior.mean[a];
          for (let b = 0; b <= a; b++) v += L[a][b] * g[b];
          this.z[i * 4 + a] = v;
        }
        this.z[i * 4 + 3] = prior.h_fail_mean + prior.h_fail_std * this.gauss();
      }
    }
    gauss() {
      let u = 0, v = 0;
      while (u === 0) u = this.rng();
      while (v === 0) v = this.rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    weights() {
      let mx = -Infinity;
      for (let i = 0; i < this.N; i++) mx = Math.max(mx, this.logw[i]);
      const w = new Float64Array(this.N);
      let s = 0;
      for (let i = 0; i < this.N; i++) { w[i] = Math.exp(this.logw[i] - mx); s += w[i]; }
      for (let i = 0; i < this.N; i++) w[i] /= s;
      return w;
    }
    ess() { const w = this.weights(); let s = 0; for (let i = 0; i < this.N; i++) s += w[i] * w[i]; return 1 / s; }
    assimilate(t, hi) {
      const z = this.z;
      for (let i = 0; i < this.N; i++) {
        const pred = z[i * 4] - Math.exp(z[i * 4 + 1]) * Math.exp(Math.exp(z[i * 4 + 2]) * t);
        const r = (hi - pred) / this.sigma;
        this.logw[i] += -0.5 * r * r;
      }
      if (this.ess() < this.N / 2) this.resample();
      this.tSeen = t;
    }
    resample() {
      const N = this.N, w = this.weights(), z = this.z, nz = new Float64Array(N * 4);
      const u0 = this.rng();
      let c = w[0], j = 0;
      for (let i = 0; i < N; i++) {
        const pos = (u0 + i) / N;
        while (pos > c && j < N - 1) { j++; c += w[j]; }
        for (let a = 0; a < 4; a++) nz[i * 4 + a] = z[j * 4 + a];
      }
      const mean = [0, 0, 0, 0];
      for (let i = 0; i < N; i++) for (let a = 0; a < 4; a++) mean[a] += nz[i * 4 + a] / N;
      const cov = [0, 1, 2, 3].map(() => [0, 0, 0, 0]);
      for (let i = 0; i < N; i++) for (let a = 0; a < 4; a++) for (let b = 0; b <= a; b++)
        cov[a][b] += (nz[i * 4 + a] - mean[a]) * (nz[i * 4 + b] - mean[b]) / (N - 1);
      for (let a = 0; a < 4; a++) { cov[a][a] += 1e-9; for (let b = 0; b < a; b++) cov[b][a] = cov[a][b]; }
      const aS = this.shrink, h2 = 1 - aS * aS;
      const L = cholesky(cov.map((r) => r.map((v) => v * h2)));
      for (let i = 0; i < N; i++) {
        const g = [this.gauss(), this.gauss(), this.gauss(), this.gauss()];
        for (let a = 0; a < 4; a++) {
          let jit = 0;
          for (let b = 0; b <= a; b++) jit += L[a][b] * g[b];
          z[i * 4 + a] = aS * nz[i * 4 + a] + (1 - aS) * mean[a] + jit;
        }
      }
      this.logw.fill(0);
      this.resamples++;
    }
    rul(tNow) {
      const out = new Float64Array(this.N), z = this.z;
      for (let i = 0; i < this.N; i++) {
        const theta = Math.exp(z[i * 4 + 1]), beta = Math.exp(z[i * 4 + 2]);
        const ratio = Math.max((z[i * 4] - z[i * 4 + 3]) / theta, 1);
        out[i] = Math.max(Math.log(ratio) / beta - tNow, 0);
      }
      return out;
    }
  }
  function wQuantiles(vals, w, qs) {
    const idx = Array.from(vals.keys()).sort((a, b) => vals[a] - vals[b]);
    const out = [];
    let c = 0, k = 0;
    for (const i of idx) {
      c += w[i];
      while (k < qs.length && c >= qs[k]) { out.push(vals[i]); k++; }
      if (k === qs.length) break;
    }
    while (out.length < qs.length) out.push(vals[idx[idx.length - 1]]);
    return out;
  }

  const lab = {
    engine: 0, cycle: 40, pf: null, fed: 0, playing: false, tick: 0,
    fault: null, outlier: null,
    obs(i) {
      const e = ds().engines[this.engine];
      const s = R.engineSeries(e, SC);
      const t = i + 1;
      let v = s.hi[i];
      if (this.fault !== null && t >= this.fault) v -= 0.02 * (Math.exp(0.045 * (t - this.fault)) - 1);
      if (this.outlier !== null && t === this.outlier) v -= 0.45;
      return v;
    },
    reset() { this.pf = null; this.fed = 0; this.fault = null; this.outlier = null; },
    ensurePf() {
      const e = ds().engines[this.engine];
      if (!this.pf || this.fed > this.cycle) {
        this.pf = new ParticleTwin(ds().prior, 2000, e.u);
        this.fed = 0;
      }
      while (this.fed < this.cycle) { this.pf.assimilate(this.fed + 1, this.obs(this.fed)); this.fed++; }
    },
    advance() {
      const e = ds().engines[this.engine];
      if (this.cycle >= e.life) { this.playing = false; this.render(); return; }
      this.cycle++;
      this.render();
    },
    render() {
      if (state.view !== "twinlab") return;
      const e = ds().engines[this.engine];
      this.cycle = clamp(this.cycle, 1, e.life);
      this.ensurePf();
      $("#lab-engine").value = this.engine;
      $("#lab-scrub").max = e.life;
      $("#lab-scrub").value = this.cycle;
      $("#lab-readout").textContent = `observed ${this.cycle} cycles`;
      $("#lab-play").textContent = this.playing ? "Pause" : "Play";
      const notes = [];
      if (this.fault !== null) notes.push(`accelerated wear from cycle ${this.fault}`);
      if (this.outlier !== null) notes.push(`bad reading at cycle ${this.outlier}`);
      $("#lab-inject-note").textContent = notes.length ? "Injected: " + notes.join(", ") : "";
      const w = this.pf.weights();
      const rul = this.pf.rul(this.cycle);
      const [q05, q50, q95] = wQuantiles(rul, w, [0.05, 0.5, 0.95]);
      this.drawHi(e, w, q50);
      this.drawRul(e, rul, w, q05, q50, q95);
      this.drawCloud(w);
      const reveal = $("#lab-reveal").checked;
      const trueRul = e.life - this.cycle;
      const zone = R.zoneOf(q05, state.cfg);
      const injected = this.fault !== null;
      $("#lab-state").innerHTML = `
        <div class="state-zone z-${zone}"><span class="label">Twin verdict at cycle ${this.cycle}</span><div class="zone-name"><i></i>${ZONE_LABEL[zone]}</div></div>
        <dl class="kv">
          <dt>RUL median</dt><dd>${f0(q50)} <small>cycles</small></dd>
          <dt>90% interval</dt><dd>${f0(q05)} – ${f0(q95)}</dd>
          <dt>Interval width</dt><dd>${f0(q95 - q05)} <small>cycles</small></dd>
          <dt>True RUL</dt><dd>${reveal ? (injected ? `<small>original engine:</small> ${trueRul}` : trueRul) : "<small>hidden</small>"}</dd>
          <dt>Particles</dt><dd>2,000</dd>
          <dt>Effective sample size</dt><dd>${f0(this.pf.ess())}</dd>
          <dt>Resampling events</dt><dd>${this.pf.resamples}</dd>
        </dl>
        <div class="verdict soft">Each particle is one guess at φ, θ, β and the failure threshold for <b style="display:inline;font:inherit;text-transform:none;letter-spacing:0">this</b> engine. Observations that disagree with a guess shrink its weight; when too few guesses carry the weight, the cloud is resampled around the survivors.</div>`;
    },
    drawHi(e, w, q50) {
      const box = $("#lab-hi");
      const { svg, w: W, h } = svgFor(box, 300);
      const m = { l: 44, r: 14, t: 12, b: 30 };
      const xmax = Math.min(700, Math.max(e.life + 10, this.cycle + q50 * 1.3 + 20));
      const x = lin(1, xmax, m.l, W - m.r), y = lin(-0.35, 1.25, h - m.b, m.t);
      for (const v of [0, 0.5, 1]) {
        S("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: "gridline" }, svg);
        T(svg, m.l - 6, y(v) + 4, v.toFixed(1), { "text-anchor": "end" });
      }
      for (const v of niceTicks(xmax, 6)) if (v >= 1) T(svg, x(v), h - m.b + 16, v, { "text-anchor": "middle" });
      const pr = ds().prior;
      S("rect", { x: m.l, y: y(pr.h_fail_mean + pr.h_fail_std), width: W - m.r - m.l, height: y(pr.h_fail_mean - pr.h_fail_std) - y(pr.h_fail_mean + pr.h_fail_std), class: "thresh" }, svg);
      S("line", { x1: m.l, x2: W - m.r, y1: y(pr.h_fail_mean), y2: y(pr.h_fail_mean), class: "thresh-line" }, svg);
      T(svg, W - m.r - 4, y(pr.h_fail_mean) - 6, "failure threshold", { "text-anchor": "end" });
      // forecast fan from weighted resample of 300 particles
      const K = 300, z = this.pf.z, idxs = [];
      const rng = mulberry32(7);
      const cdf = new Float64Array(w.length);
      let c = 0;
      for (let i = 0; i < w.length; i++) { c += w[i]; cdf[i] = c; }
      for (let k = 0; k < K; k++) {
        const u = rng();
        let lo = 0, hi = w.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
        idxs.push(lo);
      }
      const steps = 60, t0 = this.cycle, ts = [];
      for (let k = 0; k <= steps; k++) ts.push(t0 + (xmax - t0) * (k / steps));
      const qs = [[], [], [], [], []];
      for (const t of ts) {
        const vals = idxs.map((i) => z[i * 4] - Math.exp(z[i * 4 + 1]) * Math.exp(Math.exp(z[i * 4 + 2]) * t)).sort((a, b) => a - b);
        [0.05, 0.25, 0.5, 0.75, 0.95].forEach((q, j) => qs[j].push(clamp(vals[Math.floor(q * (K - 1))], -0.35, 1.25)));
      }
      const area = (a, b) => {
        let d = "";
        ts.forEach((t, k) => (d += (k ? "L" : "M") + x(t).toFixed(1) + " " + y(qs[a][k]).toFixed(1)));
        for (let k = ts.length - 1; k >= 0; k--) d += "L" + x(ts[k]).toFixed(1) + " " + y(qs[b][k]).toFixed(1);
        return d + "Z";
      };
      S("path", { d: area(0, 4), class: "fan-outer" }, svg);
      S("path", { d: area(1, 3), class: "fan-inner" }, svg);
      S("path", { d: pathFrom((k) => x(ts[k]), (k) => y(qs[2][k]), ts.length), class: "fan-mid" }, svg);
      const g = S("g", {}, svg);
      for (let i = 0; i < this.cycle; i++) S("circle", { cx: x(i + 1), cy: y(clamp(this.obs(i), -0.35, 1.25)), r: 1.9, class: "hi-dot" }, g);
      S("line", { x1: x(this.cycle), x2: x(this.cycle), y1: m.t, y2: h - m.b, class: "cursor" }, svg);
      if ($("#lab-reveal").checked) {
        S("line", { x1: x(e.life), x2: x(e.life), y1: m.t, y2: h - m.b, class: "event-ground" }, svg);
        T(svg, x(e.life) - 6, m.t + 12, this.fault !== null ? "original failure" : "actual failure", { class: "strong", "text-anchor": "end" });
      }
    },
    drawRul(e, rul, w, q05, q50, q95) {
      const box = $("#lab-rul");
      const { svg, w: W, h } = svgFor(box, 170);
      const m = { l: 44, r: 14, t: 14, b: 30 };
      const xmax = Math.max(60, Math.min(400, q95 * 1.25 + 10, 400));
      const bins = 60, bw = xmax / bins, hist = new Float64Array(bins);
      for (let i = 0; i < rul.length; i++) hist[Math.min(bins - 1, Math.floor(rul[i] / bw))] += w[i];
      const hmax = Math.max(...hist) || 1;
      const x = lin(0, xmax, m.l, W - m.r), y = lin(0, hmax, h - m.b, m.t);
      for (let b = 0; b < bins; b++) {
        if (hist[b] <= 0) continue;
        S("rect", { x: x(b * bw) + 0.5, y: y(hist[b]), width: Math.max(1, x(bw) - x(0) - 1), height: y(0) - y(hist[b]), class: "hist" }, svg);
      }
      S("line", { x1: m.l, x2: W - m.r, y1: y(0), y2: y(0), class: "axis-line" }, svg);
      for (const v of niceTicks(xmax, 6)) T(svg, x(v), h - m.b + 16, v, { "text-anchor": "middle" });
      for (const [v, lab] of [[q05, "5%"], [q50, "median"], [q95, "95%"]]) {
        if (v > xmax) continue;
        S("line", { x1: x(v), x2: x(v), y1: m.t, y2: h - m.b, class: "cursor" }, svg);
        T(svg, x(v) + 4, m.t + 8, lab, { class: "soft-text" });
      }
      if ($("#lab-reveal").checked && this.fault === null) {
        const tr = e.life - this.cycle;
        if (tr <= xmax) { S("line", { x1: x(tr), x2: x(tr), y1: m.t - 6, y2: h - m.b, class: "truth-mark" }, svg); T(svg, x(tr) + 4, h - m.b - 6, "truth", { class: "strong" }); }
      }
      T(svg, W - m.r, h - 2, "remaining cycles →", { "text-anchor": "end" });
    },
    drawCloud(w) {
      const box = $("#lab-cloud");
      const { svg, w: W, h } = svgFor(box, 240);
      const m = { l: 40, r: 10, t: 10, b: 30 };
      const z = this.pf.z, N = this.pf.N;
      const pr = ds().prior;
      const sd1 = Math.sqrt(pr.cov[1][1]), sd2 = Math.sqrt(pr.cov[2][2]);
      const x = lin(pr.mean[2] - 3.2 * sd2, pr.mean[2] + 3.2 * sd2, m.l, W - m.r);
      const y = lin(pr.mean[1] - 3.2 * sd1, pr.mean[1] + 3.2 * sd1, h - m.b, m.t);
      S("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: h - m.t - m.b, fill: "none", class: "axis-line", stroke: "var(--line)" }, svg);
      const g = S("g", {}, svg);
      let wmax = 0;
      for (let i = 0; i < N; i++) wmax = Math.max(wmax, w[i]);
      for (let i = 0; i < N; i += 2) {
        const px = x(z[i * 4 + 2]), py = y(z[i * 4 + 1]);
        if (px < m.l || px > W - m.r || py < m.t || py > h - m.b) continue;
        S("circle", { cx: px.toFixed(1), cy: py.toFixed(1), r: 1.6, class: "particle", "fill-opacity": (0.08 + 0.9 * (w[i] / wmax)).toFixed(2) }, g);
      }
      T(svg, (m.l + W - m.r) / 2, h - 8, "log β  (how fast damage grows)", { "text-anchor": "middle" });
      T(svg, 12, (m.t + h - m.b) / 2, "log θ", { "text-anchor": "middle", transform: `rotate(-90 12 ${(m.t + h - m.b) / 2})` });
    },
  };
  function buildLab() {
    $("#lab-engine").addEventListener("change", (e) => { lab.engine = +e.target.value; lab.cycle = 40; lab.reset(); lab.render(); });
    $("#lab-scrub").addEventListener("input", (e) => { lab.cycle = +e.target.value; lab.playing = false; lab.render(); });
    $("#lab-play").addEventListener("click", () => {
      const e = ds().engines[lab.engine];
      if (!lab.playing && lab.cycle >= e.life) { lab.cycle = 1; lab.pf = null; }
      lab.playing = !lab.playing; lab.render();
    });
    $("#lab-reset").addEventListener("click", () => { lab.reset(); lab.cycle = 1; lab.render(); });
    $("#lab-fault").addEventListener("click", () => { lab.fault = lab.cycle; lab.pf = null; lab.render(); });
    $("#lab-outlier").addEventListener("click", () => { lab.outlier = lab.cycle + 1; lab.pf = null; lab.cycle = Math.min(lab.cycle + 1, ds().engines[lab.engine].life); lab.render(); });
    $("#lab-reveal").addEventListener("change", () => lab.render());
  }

  /* ============================================================ EVIDENCE */
  function renderEvidence() {
    drawBenchmark();
    drawCoverage();
    drawHorizon();
    drawCost();
    renderTriggerTable();
    renderOps();
  }
  function markShape(svg, kind, cx, cy) {
    if (kind === "edge") return S("rect", { x: cx - 5, y: cy - 5, width: 10, height: 10, class: "ours-edge" }, svg);
    if (kind === "twin") return S("path", { d: `M${cx} ${cy - 7} l7 7 l-7 7 l-7 -7 z`, class: "ours-twin" }, svg);
    return S("circle", { cx, cy, r: 5, class: "ours-lstm" }, svg);
  }
  function drawBenchmark() {
    const box = $("#bench-chart");
    const rowH = 54;
    const narrow = box.clientWidth < 700;
    const { svg, w, h } = svgFor(box, DATASETS.length * rowH + 30);
    const m = { l: 58, r: narrow ? 14 : 300, t: 6 };
    const x = lin(8, 32, m.l, w - m.r);
    for (const v of [10, 15, 20, 25, 30]) {
      S("line", { x1: x(v), x2: x(v), y1: m.t, y2: h - 24, class: "gridline" }, svg);
      T(svg, x(v), h - 8, v, { "text-anchor": "middle" });
    }
    DATASETS.forEach((name, k) => {
      const cy = m.t + k * rowH + rowH / 2;
      T(svg, 0, cy + 4, name, { class: "strong" });
      S("line", { x1: m.l, x2: w - m.r, y1: cy, y2: cy, class: "axis-line" }, svg);
      const lits = D.literature.map((r) => ({ m: r.method, v: r[name] }));
      for (const l of lits) S("circle", { cx: x(l.v), cy, r: 4, class: "lit-dot", "fill-opacity": 0.75 }, svg).appendChild(Object.assign(document.createElementNS(SVGNS, "title"), { textContent: `${l.m}: ${l.v}` }));
      const acc = D.datasets[name].results.benchmark.accuracy;
      markShape(svg, "lstm", x(acc.lstm.rmse), cy);
      markShape(svg, "edge", x(acc.edge.rmse), cy);
      markShape(svg, "twin", x(acc.twin.rmse), cy);
      if (!narrow) {
        const best = lits.reduce((a, b) => (a.v <= b.v ? a : b));
        T(svg, w - m.r + 16, cy - 4, `edge ${acc.edge.rmse.toFixed(2)} · twin ${acc.twin.rmse.toFixed(2)} · LSTM ${acc.lstm.rmse.toFixed(2)}`, { class: "ink-text" });
        T(svg, w - m.r + 16, cy + 12, `best published ${best.v} (${best.m.replace(/ \(.*\)/, "")})`, { class: "soft-text" });
      }
    });
    T(svg, m.l, h - 8, "RMSE", { class: "soft-text", "text-anchor": "end", dx: -8 });
  }
  function drawCoverage() {
    const box = $("#coverage-chart");
    const rowH = 38;
    const { svg, w, h } = svgFor(box, DATASETS.length * rowH + 36);
    const m = { l: 58, r: 16, t: 8 };
    const x = lin(0.3, 1.0, m.l, w - m.r);
    for (const v of [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) {
      S("line", { x1: x(v), x2: x(v), y1: m.t, y2: h - 26, class: "gridline" }, svg);
      T(svg, x(v), h - 10, Math.round(v * 100) + "%", { "text-anchor": "middle" });
    }
    S("line", { x1: x(0.9), x2: x(0.9), y1: m.t, y2: h - 26, class: "nominal" }, svg);
    T(svg, x(0.9) + 4, m.t + 8, "nominal 90%", { class: "strong" });
    DATASETS.forEach((name, k) => {
      const cy = m.t + 14 + k * rowH + rowH / 2 - 6;
      T(svg, 0, cy + 4, name, { class: "strong" });
      S("line", { x1: m.l, x2: w - m.r, y1: cy, y2: cy, class: "axis-line" }, svg);
      const cov = D.datasets[name].results.benchmark.coverage;
      markShape(svg, "lstm", x(cov.lstm_mc_dropout.picp), cy);
      markShape(svg, "edge", x(cov.edge_conformal.picp), cy);
      markShape(svg, "twin", x(cov.twin_posterior.picp), cy);
    });
  }
  function drawHorizon() {
    const d = ds();
    $("#horizon-ds").textContent = `${d.name} · run-to-failure CV, all cycles`;
    const rows = d.results.cvAccuracy.by_horizon;
    const box = $("#horizon-chart");
    const { svg, w, h } = svgFor(box, 230);
    const m = { l: 40, r: 60, t: 12, b: 34 };
    const ymax = Math.ceil(Math.max(...rows.flatMap((r) => [r.edge, r.twin, r.lstm])) / 5) * 5;
    const x = lin(0, rows.length - 1, m.l + 20, w - m.r - 10), y = lin(0, ymax, h - m.b, m.t);
    for (const v of niceTicks(ymax, 4)) { S("line", { x1: m.l, x2: w - m.r, y1: y(v), y2: y(v), class: "gridline" }, svg); T(svg, m.l - 6, y(v) + 4, v, { "text-anchor": "end" }); }
    rows.forEach((r, i) => T(svg, x(i), h - m.b + 16, `${r.lo}–${r.hi === 126 ? 125 : r.hi}`, { "text-anchor": "middle" }));
    T(svg, (m.l + w - m.r) / 2, h - 2, "true cycles to failure", { "text-anchor": "middle" });
    const lines = [["lstm", "var(--ink-mute)", "3 3"], ["edge", "var(--accent)", ""], ["twin", "var(--twin)", ""]];
    for (const [k, color, dash] of lines) {
      S("path", { d: pathFrom((i) => x(i), (i) => y(rows[i][k]), rows.length), fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash || null }, svg);
      rows.forEach((r, i) => markShape(svg, k, x(i), y(r[k])));
      T(svg, x(rows.length - 1) + 12, y(rows[rows.length - 1][k]) + 4, { lstm: "LSTM", edge: "edge", twin: "twin" }[k], { fill: color, style: `fill:${color}` });
    }
    T(svg, m.l, m.t - 2, "RMSE", { class: "soft-text" });
  }
  function drawCost() {
    const box = $("#cost-chart");
    const pols = ["age_replacement", "edge_point", "lstm_mc", "edge_conformal", "twin_always", "hybrid", "oracle"];
    const narrow = box.clientWidth < 760;
    const cols = narrow ? 1 : 4;
    const colW = box.clientWidth / cols;
    box.innerHTML = "";
    const wrap = H("div", { style: `display:grid;grid-template-columns:repeat(${cols},minmax(0,1fr));gap:18px` });
    box.appendChild(wrap);
    DATASETS.forEach((name) => {
      const cell = H("div");
      wrap.appendChild(cell);
      const dd = D.datasets[name].results.decisions.default;
      const rowH = 24;
      const { svg, w, h } = svgFor(cell, pols.length * rowH + 40);
      const m = { l: 116, r: 44, t: 22 };
      const xmax = 10;
      const x = lin(0, xmax, m.l, w - m.r);
      T(svg, 0, 13, name, { class: "strong" });
      pols.forEach((p, i) => {
        const a = dd[p];
        const y = m.t + i * rowH;
        T(svg, m.l - 8, y + 13, POLICY[p].name.replace(" (ours)", "").replace("Edge point estimate", "Edge point").replace("LSTM + MC-dropout", "LSTM MC").replace("Twin every cycle", "Twin always"), { "text-anchor": "end", class: p === "hybrid" ? "strong" : "soft-text" });
        const cls = p === "hybrid" ? "bar-hybrid" : p === "oracle" ? "bar-muted" : p === "age_replacement" ? "bar-lstm" : "bar-muted";
        S("rect", { x: m.l, y: y + 4, width: Math.max(1, x(Math.min(a.cost_rate_x1000, xmax)) - m.l), height: 12, class: cls }, svg);
        T(svg, x(Math.min(a.cost_rate_x1000, xmax)) + 5, y + 14, a.cost_rate_x1000.toFixed(2), { class: "ink-text" });
        if (a.failures) {
          S("rect", { x: m.l + 3, y: y + 7, width: 6, height: 6, fill: "var(--page)" }, svg);
          T(svg, m.l + 12, y + 14, `${a.failures} failed`, { fill: "var(--page)", style: "fill:var(--page);font-size:10px" });
        }
      });
      void colW;
    });
  }
  function renderTriggerTable() {
    const d = ds();
    $("#trig-ds").textContent = `${d.name} · ground ≤ 2, the aggressive setting`;
    const rows = d.results.decisions.triggers.filter((r) => r.ground === 2);
    const name = (v) => v === "all" ? "All three triggers" : v.replace("without_", "Without ").replace("only_", "Only ").replace("low_rul", "low RUL").replace("sharp_drop", "sharp drop").replace("periodic_resync", "periodic");
    let html = `<thead><tr><th>Variant</th><th class="r">Failed</th><th class="r">Twin syncs</th><th class="r">Cost rate</th></tr></thead><tbody>`;
    for (const r of rows) html += `<tr class="${r.variant === "all" ? "hl" : ""}"><td>${esc(name(r.variant))}</td><td class="r">${r.failures ? `<span class="chip z-FAILED" style="padding:2px 7px 2px 5px"><i></i>${r.failures}</span>` : "0"}</td><td class="r">${pct(r.twin_sync_rate, 1)}</td><td class="r">${r.cost_rate_x1000.toFixed(2)}</td></tr>`;
    $("#trigger-table").innerHTML = html + "</tbody>";
  }
  function renderOps() {
    const d = ds(), L = d.results.latency, o = L.ops;
    $("#ops-ds").textContent = d.name;
    const k = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(n);
    $("#ops-table").innerHTML = `<thead><tr><th>Component</th><th class="r">Work per call</th><th class="r">State</th><th class="r">Laptop time</th></tr></thead><tbody>
      <tr><td>Edge model<span class="sub">150 trees, depth ≤ 6</span></td><td class="r">${k(o.edge_comparisons_per_inference)} compares<span class="sub">+ ${k(o.edge_feature_flops)} feature flops</span></td><td class="r">none</td><td class="r">${L.edge_predict.mean_ms.toFixed(2)} ms</td></tr>
      <tr><td>LSTM baseline<span class="sub">${k(L.lstm_params)} params</span></td><td class="r">${k(o.lstm_macs_per_pass)} MACs<span class="sub">×20 for MC-dropout</span></td><td class="r">none</td><td class="r">${L.lstm_cpu_predict.mean_ms.toFixed(2)} ms<span class="sub">${L.lstm_cpu_mc20.mean_ms.toFixed(1)} ms ×20</span></td></tr>
      <tr><td>Digital twin<span class="sub">2,000 particles</span></td><td class="r">~${k(o.twin_flops_per_observation)} flops / obs</td><td class="r">${k(o.twin_state_floats_per_engine)} floats<span class="sub">per engine</span></td><td class="r">${L.twin_sync_20obs.mean_ms.toFixed(2)} ms<span class="sub">sync of 20 obs</span></td></tr></tbody>`;
  }

  /* ============================================================ PRESENTER */
  const STEPS = [
    { title: "The problem", note: "Too late is a failure in service; too early wastes life you paid for. The baseline everyone uses is a fixed retirement age.", go: () => storyAt("ch-problem") },
    { title: "The data", note: "NASA C-MAPSS: 709 simulated engines run to failure, 21 sensors, up to 6 flight regimes and 2 fault modes. Note how noisy the raw sensors are next to the health index.", go: () => storyAt("ch-data") },
    { title: "Architecture", note: "A cheap stateless model on board every cycle; a stateful twin per engine off board, consulted on trigger; a policy that only acts on calibrated lower bounds.", go: () => storyAt("ch-arch") },
    { title: "Edge model", note: "Gradient-boosted trees on 30-cycle window statistics, wrapped in Mondrian conformal prediction for a guaranteed 90% interval.", go: () => storyAt("ch-edge") },
    { title: "Digital twin", note: "Health index + exponential degradation + particle filter: Bayesian updating of a fleet prior, one engine at a time.", go: () => storyAt("ch-twin") },
    { title: "Decision layer", note: "Three triggers, cautious fusion of two calibrated bounds, four actions.", go: () => storyAt("ch-hybrid") },
    { title: "Live: one engine", note: "FD004 engine 50, never seen in training. Blue is the edge model, violet bars are twin syncs, black is what the policy acts on. Watch where it grounds.", go: () => loadScenario(0, { play: true }) },
    { title: "What would have happened", note: "Reveal the truth. The edge point estimate and even the edge conformal bound fly this engine to failure; the hybrid grounds it with margin.", go: () => { loadScenario(0); state.cycle = engine().life; state.reveal = true; renderMission(); } },
    { title: "Twin lab", note: "2,000 hypotheses about one engine. Each observation re-weights them; the forecast fan tightens as the twin learns this engine.", go: () => { setView("twinlab"); lab.cycle = 20; lab.reset(); lab.playing = true; lab.render(); } },
    { title: "Break the engine", note: "Inject accelerated wear. The twin had no idea this was coming; within a few cycles the posterior moves.", go: () => { setView("twinlab"); lab.playing = false; lab.cycle = Math.max(lab.cycle, 90); lab.fault = lab.cycle; lab.pf = null; lab.playing = true; lab.render(); } },
    { title: "The fleet", note: "Every held-out FD004 engine at once, under the hybrid policy. Zero failures; engines retired close to the end of their lives.", go: () => { setDataset("FD004"); state.cfg = R.defaultConfig(D.defaults); syncSettings(); fleetCache.clear(); setView("fleet"); state.fleetCycle = 1; state.fleetPlaying = true; state.fleetPolicy = "hybrid"; renderFleet(); } },
    { title: "Stress test", note: "Remove the safety margin (ground at 0) and trust the twin alone at syncs. Failures appear. Switch fusion back and they disappear.", go: () => { setDataset("FD004"); setView("fleet"); state.cfg = Object.assign(R.defaultConfig(D.defaults), { ground: 0, fusion: "twin" }); syncSettings(); fleetCache.clear(); settingsEl.open = true; state.fleetCycle = 600; state.fleetPlaying = false; renderFleet(); } },
    { title: "Accuracy vs the literature", note: "Official test sets. Caveat on screen: the twin uses the full history, windowed models only 30 cycles.", go: () => { state.cfg = R.defaultConfig(D.defaults); syncSettings(); fleetCache.clear(); settingsEl.open = false; evidenceAt("ev-benchmark"); } },
    { title: "Calibration", note: "A 90% interval should contain the truth 90% of the time. MC-dropout does not; conformal and particle-filter intervals do.", go: () => evidenceAt("ev-coverage") },
    { title: "Cost", note: "Renewal-reward cost rate. Predictive policies cut cost 34–47% vs age replacement; the hybrid is the only one that needs no safety buffer on any dataset.", go: () => evidenceAt("ev-cost") },
  ];
  function storyAt(id) { setView("story"); requestAnimationFrame(() => { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: "smooth", block: "start" }); }); }
  function evidenceAt(id) { setView("evidence"); requestAnimationFrame(() => { const el = document.getElementById(id); if (el) el.scrollIntoView({ behavior: "smooth", block: "center" }); }); }
  function togglePresent(on) {
    state.presenting = on === undefined ? !state.presenting : on;
    document.body.classList.toggle("presenting", state.presenting);
    $("#presenter").hidden = !state.presenting;
    if (state.presenting) goStep(state.step);
    else { state.playing = false; lab.playing = false; state.fleetPlaying = false; }
    requestAnimationFrame(renderView);
  }
  function goStep(i) {
    state.step = clamp(i, 0, STEPS.length - 1);
    state.playing = false; lab.playing = false; state.fleetPlaying = false;
    const st = STEPS[state.step];
    $("#p-step").textContent = `${state.step + 1} / ${STEPS.length}`;
    $("#p-title").textContent = st.title;
    $("#p-note").textContent = st.note;
    $("#p-ticks").innerHTML = STEPS.map((_, k) => `<i class="${k <= state.step ? "on" : ""}"></i>`).join("");
    st.go();
  }

  /* ============================================================ keyboard */
  document.addEventListener("keydown", (ev) => {
    const tag = (ev.target && ev.target.tagName) || "";
    const typing = tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA";
    if (ev.key === "p" || ev.key === "P") { if (!typing) { togglePresent(); ev.preventDefault(); } return; }
    if (state.presenting) {
      if (["ArrowRight", "PageDown", "Enter"].includes(ev.key) && !typing) { goStep(state.step + 1); ev.preventDefault(); return; }
      if (["ArrowLeft", "PageUp"].includes(ev.key) && !typing) { goStep(state.step - 1); ev.preventDefault(); return; }
      if (ev.key === "Escape") { togglePresent(false); return; }
    }
    if (ev.key === " " && !typing) {
      ev.preventDefault();
      if (state.view === "mission") togglePlay();
      else if (state.view === "fleet") $("#fleet-play").click();
      else if (state.view === "twinlab") $("#lab-play").click();
      return;
    }
    if (!state.presenting && state.view === "mission" && !typing && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) {
      state.playing = false;
      state.cycle += ev.key === "ArrowRight" ? 1 : -1;
      renderMission();
      ev.preventDefault();
    }
  });
  $("#p-prev").addEventListener("click", () => goStep(state.step - 1));
  $("#p-next").addEventListener("click", () => goStep(state.step + 1));
  $("#p-exit").addEventListener("click", () => togglePresent(false));

  /* ============================================================ boot */
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderView, 120); });

  buildTopbar();
  buildSettings();
  buildMission();
  buildFleet();
  buildLab();
  state.engine = Math.max(0, defaultEngineIndex());
  lab.engine = state.engine;
  populateEngineSelects();
  state.cycle = restingCycle();
  lab.cycle = Math.round(engine().life * 0.45);
  $("#loading").hidden = true;
  const hash = (location.hash || "").slice(1);
  setView(["story", "mission", "fleet", "twinlab", "evidence"].includes(hash) ? hash : "story", { keepHash: true });
  requestAnimationFrame(tick);
})();
