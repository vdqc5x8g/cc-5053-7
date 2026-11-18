"use strict";

/* Modal synthesis percussion UI.
 * Every strike POSTs to /api/hit, which runs the modal solver server-side
 * and returns freshly synthesized audio plus the actual per-mode solver
 * inputs (delta_v) that the chart and table display. No pre-rendered
 * audio, no animation standing in for the sound.
 */

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const state = {
  models: [],
  spec: null,
  audioCtx: null,
  lastHit: null,        // {u, v}
  lastResponse: null,
  selectedMode: null,   // mode index or null
  modeShape: null,      // sampled shape of selected mode
  history: [],          // {label, buffer}
};

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

async function fetchJSON(url, options) {
  const resp = await fetch(url, options);
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || resp.statusText);
  return data;
}

function sliderLog(id, min, max) {
  const t = $(id).value / 1000;
  return min * Math.pow(max / min, t);
}

const velocity = () => sliderLog("velocity", 0.05, 10.0);       // m/s
const contactTime = () => sliderLog("contact", 1e-4, 0.02);     // s
const dampingScale = () => sliderLog("damping", 0.05, 100.0);
const durationS = () => sliderLog("duration", 0.5, 5.0);        // s

function fmt(x, digits = 2) {
  if (x === 0) return "0";
  const ax = Math.abs(x);
  if (ax >= 1000 || ax < 0.01) return x.toExponential(1);
  return x.toFixed(digits);
}

function updateSliderLabels() {
  $("velocityVal").textContent = fmt(velocity()) + " m/s";
  $("contactVal").textContent = fmt(contactTime() * 1000, 1) + " ms";
  $("dampingVal").textContent = fmt(dampingScale()) + "x";
  $("durationVal").textContent = fmt(durationS(), 1) + " s";
}

// ---------------------------------------------------------------------------
// surface drawing + hit mapping
// ---------------------------------------------------------------------------

function surfaceLayout() {
  const canvas = $("surface");
  const geo = state.spec.geometry;
  const margin = 40;
  const aw = geo.type === "plate" ? geo.a_m : geo.length_m;
  const ah = geo.type === "plate" ? geo.b_m : geo.width_m;
  const scale = Math.min(
    (canvas.width - 2 * margin) / aw,
    (canvas.height - 2 * margin) / ah
  );
  const w = aw * scale;
  const h = ah * scale;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h, geo };
}

function drawSurface() {
  const canvas = $("surface");
  const ctx = canvas.getContext("2d");
  const L = surfaceLayout();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // mode shape overlay (if a mode is selected)
  if (state.modeShape) drawShapeOverlay(ctx, L);

  // surface outline
  ctx.strokeStyle = "#4a5568";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(L.x, L.y, L.w, L.h);

  // quarter grid lines
  ctx.strokeStyle = "rgba(74,85,104,0.35)";
  ctx.lineWidth = 0.5;
  for (const f of [0.25, 0.5, 0.75]) {
    ctx.beginPath();
    ctx.moveTo(L.x + f * L.w, L.y);
    ctx.lineTo(L.x + f * L.w, L.y + L.h);
    ctx.moveTo(L.x, L.y + f * L.h);
    ctx.lineTo(L.x + L.w, L.y + f * L.h);
    ctx.stroke();
  }

  // last hit marker
  if (state.lastHit) {
    const px = L.x + state.lastHit.u * L.w;
    const py = L.y + state.lastHit.v * L.h;
    ctx.strokeStyle = "#ffb74d";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, 2 * Math.PI);
    ctx.moveTo(px - 12, py); ctx.lineTo(px + 12, py);
    ctx.moveTo(px, py - 12); ctx.lineTo(px, py + 12);
    ctx.stroke();
  }

  // axis labels with units
  ctx.fillStyle = "#8b96a5";
  ctx.font = "11px sans-serif";
  const aw = L.geo.type === "plate" ? L.geo.a_m : L.geo.length_m;
  const ah = L.geo.type === "plate" ? L.geo.b_m : L.geo.width_m;
  ctx.fillText(`0`, L.x - 12, L.y + L.h + 4);
  ctx.fillText(`${aw.toFixed(2)} m`, L.x + L.w - 34, L.y + L.h + 16);
  ctx.fillText(`${ah.toFixed(2)} m`, L.x - 38, L.y + 6);
}

function drawShapeOverlay(ctx, L) {
  const shape = state.modeShape;
  if (shape.type === "plate") {
    const n = shape.grid.length;
    const img = ctx.createImageData(n, n);
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const v = shape.grid[j][i]; // [-1, 1]
        const k = (j * n + i) * 4;
        img.data[k] = v > 0 ? 255 * v : 30;          // R
        img.data[k + 1] = 40;                        // G
        img.data[k + 2] = v < 0 ? 255 * -v : 30;     // B
        img.data[k + 3] = 200;
      }
    }
    const off = document.createElement("canvas");
    off.width = n; off.height = n;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, L.x, L.y, L.w, L.h);
  } else {
    // bar: draw phi(x) as a filled curve along the length
    const xs = shape.values;
    const n = xs.length;
    ctx.beginPath();
    ctx.moveTo(L.x, L.y + L.h / 2);
    for (let i = 0; i < n; i++) {
      const px = L.x + (i / (n - 1)) * L.w;
      const py = L.y + L.h / 2 - xs[i] * (L.h * 1.6);
      ctx.lineTo(px, py);
    }
    ctx.strokeStyle = "#4fc3f7";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "rgba(79,195,247,0.12)";
    ctx.fillRect(L.x, L.y, L.w, L.h);
  }
}

function canvasToHit(evt) {
  const canvas = $("surface");
  const rect = canvas.getBoundingClientRect();
  const L = surfaceLayout();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const cx = (evt.clientX - rect.left) * sx;
  const cy = (evt.clientY - rect.top) * sy;
  const u = (cx - L.x) / L.w;
  const v = (cy - L.y) / L.h;
  return { u: Math.min(Math.max(u, 0), 1), v: Math.min(Math.max(v, 0), 1) };
}

// ---------------------------------------------------------------------------
// mode chart (frequencies vs actual solver excitation |delta_v|)
// ---------------------------------------------------------------------------

const chart = { bars: [] }; // {x0, x1, modeIndex}

function drawModeChart(modes) {
  const canvas = $("modeChart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  chart.bars = [];
  if (!state.spec) return;

  const fMin = 20, fMax = state.spec.f_max_hz;
  const xOf = (f) => ((Math.log(f / fMin)) / Math.log(fMax / fMin)) * (W - 50) + 40;

  // axes
  ctx.strokeStyle = "#3a4350";
  ctx.beginPath();
  ctx.moveTo(40, H - 24); ctx.lineTo(W - 10, H - 24);
  ctx.moveTo(40, H - 24); ctx.lineTo(40, 8);
  ctx.stroke();
  ctx.fillStyle = "#8b96a5";
  ctx.font = "10px sans-serif";
  for (const f of [50, 100, 200, 500, 1000, 2000, 5000, 10000]) {
    if (f > fMax) break;
    ctx.fillText(f >= 1000 ? `${f / 1000}k` : `${f}`, xOf(f) - 8, H - 10);
    ctx.strokeStyle = "rgba(58,67,80,0.4)";
    ctx.beginPath(); ctx.moveTo(xOf(f), H - 24); ctx.lineTo(xOf(f), 8); ctx.stroke();
  }
  ctx.fillText("Hz", W - 24, H - 10);

  const activ = modes || state.spec.modes.map(() => null);
  const dvs = activ.map((m) => (m ? Math.abs(m.delta_v_m_s) : 0));
  const dvMax = Math.max(...dvs, 1e-12);
  const yOf = (dv) => {
    if (dv <= 0) return H - 24;
    const decades = Math.log10(dv / dvMax); // 0 .. -inf
    return H - 24 - Math.max(0, 1 + decades / 4) * (H - 40);
  };

  activ.forEach((m, i) => {
    const f = m ? m.frequency_hz : state.spec.modes[i].frequency_hz;
    const x = xOf(f);
    const y = yOf(dvs[i]);
    const selected = state.selectedMode === i;
    ctx.fillStyle = selected ? "#ffb74d" : dvs[i] > 0 ? "#4fc3f7" : "#2c3542";
    ctx.fillRect(x - 2, y, 4, H - 24 - y);
    chart.bars.push({ x0: x - 4, x1: x + 4, modeIndex: i });
  });
}

function chartModeAt(evt) {
  const canvas = $("modeChart");
  const rect = canvas.getBoundingClientRect();
  const x = ((evt.clientX - rect.left) / rect.width) * canvas.width;
  let best = null, bestD = 8;
  for (const b of chart.bars) {
    const c = (b.x0 + b.x1) / 2;
    const d = Math.abs(x - c);
    if (d < bestD) { bestD = d; best = b.modeIndex; }
  }
  return best;
}

// ---------------------------------------------------------------------------
// mode table
// ---------------------------------------------------------------------------

function renderModeList(activation) {
  const tbody = $("modeList").querySelector("tbody");
  tbody.innerHTML = "";
  if (!state.spec) return;
  state.spec.modes.forEach((m, i) => {
    const act = activation ? activation[i] : null;
    const tr = document.createElement("tr");
    if (state.selectedMode === i) tr.classList.add("selected");
    const phi = act ? act.shape_at_hit : null;
    const dv = act ? Math.abs(act.delta_v_m_s) : null;
    tr.innerHTML =
      `<td>${i}</td><td>${m.label}</td>` +
      `<td>${m.frequency_hz.toFixed(1)}</td><td>${m.tau_s.toFixed(3)}</td>` +
      `<td class="${phi !== null && Math.abs(phi) < 1e-3 ? "zero" : ""}">` +
      `${phi !== null ? phi.toFixed(3) : "–"}</td>` +
      `<td class="${dv !== null && dv < 1e-6 ? "zero" : ""}">` +
      `${dv !== null ? dv.toExponential(2) : "–"}</td>`;
    const td = document.createElement("td");
    const solo = document.createElement("button");
    solo.textContent = "solo";
    solo.title = "render this mode alone at the last strike point";
    solo.addEventListener("click", (e) => {
      e.stopPropagation();
      if (state.lastHit) strike(state.lastHit.u, state.lastHit.v, i);
    });
    td.appendChild(solo);
    tr.appendChild(td);
    tr.addEventListener("click", () => selectMode(i));
    tbody.appendChild(tr);
  });
}

async function selectMode(i) {
  state.selectedMode = i;
  const m = state.spec.modes[i];
  state.modeShape = await fetchJSON(
    `/api/mode_shape?id=${state.spec.id}&mode=${i}&n=41&damping_scale=${dampingScale()}`
  );
  const act = state.lastResponse ? state.lastResponse.modes[i] : null;
  $("selectedModeInfo").textContent =
    `mode ${m.label}: f = ${m.frequency_hz.toFixed(1)} Hz, ` +
    `tau = ${m.tau_s.toFixed(3)} s, modal mass = ${m.modal_mass_kg.toFixed(4)} kg` +
    (act ? `, phi(hit) = ${act.shape_at_hit.toFixed(3)}, |delta_v| = ${Math.abs(act.delta_v_m_s).toExponential(2)} m/s` : "");
  drawSurface();
  drawModeChart(state.lastResponse ? state.lastResponse.modes : null);
  renderModeList(state.lastResponse ? state.lastResponse.modes : null);
}

// ---------------------------------------------------------------------------
// audio
// ---------------------------------------------------------------------------

function ensureAudioCtx() {
  if (!state.audioCtx) {
    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  }
  return state.audioCtx;
}

function decodeB64Audio(b64) {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  return ensureAudioCtx().decodeAudioData(bytes.buffer.slice(0));
}

function playBuffer(buffer) {
  const ctx = ensureAudioCtx();
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start();
}

// ---------------------------------------------------------------------------
// strike
// ---------------------------------------------------------------------------

async function strike(u, v, soloMode = null) {
  state.lastHit = { u, v };
  const body = {
    model: state.spec.id,
    u, v,
    velocity_m_s: velocity(),
    contact_time_s: contactTime(),
    damping_scale: dampingScale(),
    duration_s: durationS(),
    solo_mode: soloMode,
  };
  let resp;
  try {
    resp = await fetchJSON("/api/hit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    $("status").textContent = "error: " + err.message;
    return;
  }
  state.lastResponse = resp;
  const buffer = await decodeB64Audio(resp.audio_b64);
  playBuffer(buffer);
  pushHistory(resp, buffer);
  drawSurface();
  drawModeChart(resp.modes);
  renderModeList(resp.modes);
  showStatus(resp);
}

function showStatus(resp) {
  const s = resp.solver;
  let text =
    `solver: ${s.type} | max |pole| = ${s.max_pole_magnitude.toFixed(8)} (< 1: stable)\n` +
    `modes: ${s.modes_used} used, ${s.excluded_mode_count} below f_max=${s.f_max_hz} Hz excluded (reported, not hidden)\n` +
    `peak ${resp.peak_linear.toExponential(2)} -> normalized x${resp.gain_applied.toExponential(2)} (no clipping) | ` +
    `impulse ${resp.impulse_n_s.toFixed(3)} N s, contact ${(resp.contact_time_s * 1000).toFixed(2)} ms`;
  if (resp.solo_mode !== null && resp.solo_mode !== undefined) {
    text += `\nsolo: mode ${resp.solo_mode} only`;
  }
  $("status").innerHTML = resp.warnings.length
    ? text + `\n<span class="warn">warning: ${resp.warnings.join("; ")}</span>`
    : text;
}

function pushHistory(resp, buffer) {
  const label =
    `${resp.solo_mode !== null && resp.solo_mode !== undefined ? "solo " + resp.solo_mode + " " : ""}` +
    `@(${resp.x_m.toFixed(3)}, ${resp.y_m.toFixed(3)}) m, ` +
    `${resp.velocity_m_s.toFixed(2)} m/s`;
  state.history.unshift({ label, buffer });
  if (state.history.length > 8) state.history.pop();
  const div = $("history");
  div.innerHTML = "";
  state.history.forEach((h) => {
    const el = document.createElement("span");
    el.className = "entry";
    el.innerHTML = `<span class="pos">${h.label}</span>`;
    el.addEventListener("click", () => playBuffer(h.buffer));
    div.appendChild(el);
  });
}

// ---------------------------------------------------------------------------
// model info panel
// ---------------------------------------------------------------------------

function renderModelInfo() {
  const s = state.spec;
  const g = s.geometry;
  const geoText =
    g.type === "plate"
      ? `plate ${g.a_m} x ${g.b_m} m, thickness ${s.thickness_m * 1000} mm`
      : `bar ${g.length_m} m long, ${g.width_m * 1000} mm wide, thickness ${s.thickness_m * 1000} mm`;
  const mat = s.material;
  $("modelInfo").innerHTML =
    `<h3>${s.name}</h3>` +
    `<div class="desc">${s.description}</div>` +
    `<table><tr><th>geometry</th><td>${geoText}</td></tr>` +
    `<tr><th>material</th><td>${mat.name}: E = ${mat.youngs_modulus_pa.toExponential(2)} Pa, ` +
    `rho = ${mat.density_kg_m3} kg/m^3, nu = ${mat.poisson_ratio}, loss factor eta = ${mat.loss_factor}</td></tr>` +
    `<tr><th>solver</th><td>exact matrix-exponential integrator per mode, ` +
    `${s.modes.length} modes below ${s.f_max_hz} Hz ` +
    `(${s.excluded_mode_count} more excluded by the mode cap - reported, not hidden)</td></tr></table>` +
    `<ul>${s.notes.map((n) => `<li>${n}</li>`).join("")}</ul>` +
    `<div class="warn">All parameters are declared analytical/procedural models - nothing here is a measurement of a real object.</div>`;
}

// ---------------------------------------------------------------------------
// model loading + init
// ---------------------------------------------------------------------------

let dampingReloadTimer = null;

async function loadModel() {
  const id = $("modelSelect").value;
  state.spec = await fetchJSON(`/api/model?id=${id}&damping_scale=${dampingScale()}`);
  state.selectedMode = null;
  state.modeShape = null;
  state.lastResponse = null;
  drawSurface();
  drawModeChart(null);
  renderModeList(null);
  renderModelInfo();
  $("selectedModeInfo").textContent =
    "no mode selected - click a bar in the chart or a row in the table";
  $("status").textContent =
    `${state.spec.modes.length} modes loaded, ${state.spec.excluded_mode_count} excluded by cap`;
}

async function init() {
  const data = await fetchJSON("/api/models");
  state.models = data.models;
  const sel = $("modelSelect");
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.name;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", loadModel);

  for (const id of ["velocity", "contact", "duration"]) {
    $(id).addEventListener("input", updateSliderLabels);
  }
  $("damping").addEventListener("input", () => {
    updateSliderLabels();
    clearTimeout(dampingReloadTimer);
    dampingReloadTimer = setTimeout(loadModel, 300);
  });
  updateSliderLabels();

  $("modelInfoBtn").addEventListener("click", () =>
    $("modelInfo").classList.toggle("hidden")
  );

  $("surface").addEventListener("click", (evt) => {
    const { u, v } = canvasToHit(evt);
    strike(u, v, null);
  });

  $("modeChart").addEventListener("click", (evt) => {
    const i = chartModeAt(evt);
    if (i !== null) selectMode(i);
  });

  await loadModel();
}

init();
