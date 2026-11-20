"use strict";

/* Modal synthesis percussion UI - sustained engine edition.
 *
 * The page owns a *session*: an ordered list of hit events, each with a
 * position, a velocity and a sample-accurate time relative to the session
 * start. A strike appends an event and re-renders the whole session through
 * /api/hit, which runs one persistent modal engine (the server-side
 * equivalent of "the body that is still ringing"). Nothing about the
 * previous clicks is reset by a new one.
 *
 * Audition modes:
 *   mix     - the whole session as one sustained signal
 *   event N - event N alone, on an otherwise silent body (position and
 *             time preserved, so its slot is audible)
 *   iso N   - event N's independent response, struck at t=0 (peak-normalised
 *             by the server), for A/B against the mix
 * Playback is driven by the Web Audio clock, not by redraws, so changing
 * the drawing cadence cannot affect the sound.
 */

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------

const BLOCK_SIZE = 512;            // render block; adjustable via the UI
const MAX_EVENTS = 8;

const state = {
  models: [],
  spec: null,
  audioCtx: null,
  session: [],                     // [{u, v, time_s, velocity_m_s, contact_time_s, direction, label}]
  lastResponse: null,
  viewEvent: null,                 // event index whose excitation is shown, or null
  selectedMode: null,
  modeShape: null,
  audition: "mix",                 // "mix" | "event" | "iso"
  blockSize: BLOCK_SIZE,
  replaySizes: null,                // set while a block-replay audit is in flight
  playing: null,                   // {source, startedAt, duration}
  buffers: {},                     // {mix, events:[], isolated:[]}
  history: [],                     // [{label, response}] - last full responses
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
const intervalMs = () => sliderLog("interval", 30, 900);        // ms between auto hits

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
  $("intervalVal").textContent = fmt(intervalMs(), 0) + " ms";
  $("blockVal").textContent = `${state.blockSize} smp`;
}

// ---------------------------------------------------------------------------
// surface drawing + hit mapping (unchanged visual contract)
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

  if (state.modeShape) drawShapeOverlay(ctx, L);

  ctx.strokeStyle = "#4a5568";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(L.x, L.y, L.w, L.h);

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

  // session marks: numbered, colour by direction, ring by view selection
  state.session.forEach((ev, i) => {
    const px = L.x + ev.u * L.w;
    const py = L.y + ev.v * L.h;
    const viewed = state.viewEvent === i;
    ctx.strokeStyle = ev.direction > 0 ? "#ffb74d" : "#4fc3f7";
    ctx.lineWidth = viewed ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.arc(px, py, viewed ? 10 : 7, 0, 2 * Math.PI);
    ctx.moveTo(px - 12, py); ctx.lineTo(px + 12, py);
    ctx.moveTo(px, py - 12); ctx.lineTo(px, py + 12);
    ctx.stroke();
    ctx.fillStyle = ev.direction > 0 ? "#ffb74d" : "#4fc3f7";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${i + 1}${ev.direction < 0 ? "′" : ""}`, px + 10, py - 9);
  });

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
        const v = shape.grid[j][i];
        const k = (j * n + i) * 4;
        img.data[k] = v > 0 ? 255 * v : 30;
        img.data[k + 1] = 40;
        img.data[k + 2] = v < 0 ? 255 * -v : 30;
        img.data[k + 3] = 200;
      }
    }
    const off = document.createElement("canvas");
    off.width = n; off.height = n;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, L.x, L.y, L.w, L.h);
  } else {
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
// excitation chart for one event
// ---------------------------------------------------------------------------

const chart = { bars: [] };

function activeModes() {
  const r = state.lastResponse;
  if (!r) return null;
  const ev = state.viewEvent === null ? r.events[0] : r.events[state.viewEvent];
  return ev ? ev.modes : null;
}

function drawModeChart(modes) {
  const canvas = $("modeChart");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  chart.bars = [];
  if (!state.spec) return;

  const fMin = 20, fMax = state.spec.f_max_hz;
  const xOf = (f) => ((Math.log(f / fMin)) / Math.log(fMax / fMin)) * (W - 50) + 40;

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
    const decades = Math.log10(dv / dvMax);
    return H - 24 - Math.max(0, 1 + decades / 4) * (H - 40);
  };

  const cats = state.lastResponse ? state.lastResponse.cancellation : null;
  activ.forEach((m, i) => {
    const f = m ? m.frequency_hz : state.spec.modes[i].frequency_hz;
    const x = xOf(f);
    const y = yOf(dvs[i]);
    const selected = state.selectedMode === i;
    let color = dvs[i] > 0 ? "#4fc3f7" : "#2c3542";
    if (selected) color = "#ffb74d";
    else if (cats && cats.cancelled.includes(i)) color = "#ef9a9a";   // cancelled by pair
    else if (cats && cats.reinforced.includes(i)) color = "#9ccc65";  // reinforced by pair
    ctx.fillStyle = color;
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
// mode table for one event, with per-row solo
// ---------------------------------------------------------------------------

function renderModeList(modes) {
  const tbody = $("modeList").querySelector("tbody");
  tbody.innerHTML = "";
  if (!state.spec) return;
  const cats = state.lastResponse ? state.lastResponse.cancellation : null;
  state.spec.modes.forEach((m, i) => {
    const act = modes ? modes[i] : null;
    const tr = document.createElement("tr");
    if (state.selectedMode === i) tr.classList.add("selected");
    const phi = act ? act.shape_at_hit : null;
    const dv = act ? act.delta_v_m_s : null;
    let status = "";
    if (cats && cats.cancelled.includes(i)) status = `<span class="tag cancel">cancel</span>`;
    else if (cats && cats.reinforced.includes(i)) status = `<span class="tag add">add</span>`;
    tr.innerHTML =
      `<td>${i}</td><td>${m.label}</td>` +
      `<td>${m.frequency_hz.toFixed(1)}</td><td>${m.tau_s.toFixed(3)}</td>` +
      `<td class="${phi !== null && Math.abs(phi) < 1e-3 ? "zero" : ""}">` +
      `${phi !== null ? phi.toFixed(3) : "–"}</td>` +
      `<td class="${dv !== null && Math.abs(dv) < 1e-9 ? "zero" : ""}">` +
      `${dv !== null ? dv.toExponential(2) : "–"}</td>` +
      `<td>${status}</td>`;
    const td = document.createElement("td");
    const solo = document.createElement("button");
    solo.textContent = "solo";
    solo.title = "render this session with only this mode, then audition";
    solo.addEventListener("click", (e) => {
      e.stopPropagation();
      rerender({ solo_mode: i }).then(() => play("mix"));
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
  const act = activeModes() ? activeModes()[i] : null;
  const rb = state.lastResponse ? state.lastResponse.body_state.modes[i] : null;
  $("selectedModeInfo").textContent =
    `mode ${m.label}: f = ${m.frequency_hz.toFixed(1)} Hz, ` +
    `tau = ${m.tau_s.toFixed(3)} s` +
    (act ? `, phi(hit) = ${act.shape_at_hit.toFixed(3)}, |Δv| = ${Math.abs(act.delta_v_m_s).toExponential(2)} m/s` : "") +
    (rb
      ? ` | live: amplitude ${rb.amplitude.toExponential(2)}, ` +
        `phase ${rb.phase_deg.toFixed(1)}°, tail ${rb.tail_s.toFixed(3)} s`
      : "");
  drawSurface();
  drawModeChart(activeModes());
  renderModeList(activeModes());
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

async function play(kind) {
  state.audition = kind;
  if (kind === "mix") {
    await ensureBuffers();
    startBuffer(state.buffers.mix);
  } else if (kind.startsWith("iso:")) {
    await playEventIsolated(Number(kind.split(":")[1]));
  }
  updateAuditionLabel();
}

function stopPlayback() {
  if (state.playing) {
    try { state.playing.source.stop(); } catch (_) { /* already stopped */ }
    state.playing = null;
  }
}

async function ensureBuffers() {
  const r = state.lastResponse;
  if (!r || state.buffers.responseRef === r) return;
  state.buffers = { mix: await decodeB64Audio(r.mix_audio_b64), responseRef: r };
}

function startBuffer(buffer) {
  if (!buffer) return;
  stopPlayback();
  const ctx = ensureAudioCtx();
  const src = ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(ctx.destination);
  src.start();
  state.playing = { source: src, startedAt: ctx.currentTime, duration: buffer.duration };
  src.onended = () => { state.playing = null; };
}

async function playEventAlone(i) {
  // Re-render with only event i submitted (its time slot preserved), then
  // audition that buffer. The full session is re-fetched on the next action.
  const resp = await fetchJSON("/api/hit", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(sessionBody({ solo_event: i, include_isolated: false })),
  });
  startBuffer(await decodeB64Audio(resp.audio_b64));
  state.audition = `event:${i}`;
  updateAuditionLabel();
}

async function playEventIsolated(i) {
  // Already-rendered independent stem (struck at t=0, peak-normalised).
  const b64 = state.lastResponse && state.lastResponse.isolated_audio_b64[i];
  if (!b64) return;
  startBuffer(await decodeB64Audio(b64));
  state.audition = `iso:${i}`;
  updateAuditionLabel();
}

function updateAuditionLabel() {
  const el = $("auditionLabel");
  if (!el) return;
  const a = state.audition;
  el.textContent = a === "mix" ? "playing: full mix" :
    a.startsWith("event:") ? `playing: event ${Number(a.split(":")[1]) + 1} alone (in its slot)` :
    a.startsWith("iso:") ? `playing: event ${Number(a.split(":")[1]) + 1} isolated (struck at t=0)` : "";
}

// ---------------------------------------------------------------------------
// session -> server
// ---------------------------------------------------------------------------

function sessionBody(extra = {}) {
  const body = {
    model: state.spec.id,
    duration_s: durationS(),
    damping_scale: dampingScale(),
    block_size: state.blockSize,
    include_isolated: true,
    events: state.session.map((ev, i) => ({
      u: ev.u,
      v: ev.v,
      time_s: ev.time_s,
      velocity_m_s: ev.velocity_m_s,
      contact_time_s: ev.contact_time_s,
      direction: ev.direction,
      label: ev.label || `hit ${i + 1}`,
    })),
  };
  if (state.session.length > 1 && state.replaySizes) {
    // Only when the user explicitly asks for the block-invariance audit:
    // it re-renders the whole session once per block size.
    body.replay_block_sizes = state.replaySizes;
  }
  return Object.assign(body, extra);
}

let renderToken = 0;

async function rerender(extra = {}) {
  if (!state.session.length) return;
  const token = ++renderToken;
  let resp;
  try {
    resp = await fetchJSON("/api/hit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(sessionBody(extra)),
    });
  } catch (err) {
    $("status").textContent = "error: " + err.message;
    throw err;
  }
  if (token !== renderToken) return resp;  // a newer render superseded this one
  state.lastResponse = resp;
  state.buffers = {};
  if (state.viewEvent === null || state.viewEvent >= resp.events.length) {
    state.viewEvent = resp.events.length - 1;
  }
  await ensureBuffers();
  drawSurface();
  drawModeChart(activeModes());
  renderModeList(activeModes());
  renderSessionList();
  renderSoloControls();
  showStatus(resp);
  return resp;
}

async function strike(u, v, direction = 1) {
  if (state.session.length >= MAX_EVENTS) {
    state.session.shift();  // keep the most recent window; clock keeps counting
  }
  const now = state.session.length
    ? state.session[state.session.length - 1].time_s + intervalMs() / 1000
    : 0;
  state.session.push({
    u, v,
    time_s: now,
    velocity_m_s: velocity(),
    contact_time_s: contactTime(),
    direction,
    label: `hit ${state.session.length + 1}${direction < 0 ? "′" : ""}`,
  });
  state.viewEvent = state.session.length - 1;
  const resp = await rerender();
  if (!resp) return;
  await play("mix");
  pushHistory(resp);
}

function pushHistory(resp) {
  const label = resp.events
    .map((e) => `${e.label}@(${e.x_m.toFixed(3)},${e.y_m.toFixed(3)})`)
    .join(" + ");
  state.history.unshift({ label, response: resp });
  if (state.history.length > 6) state.history.pop();
  renderHistory();
}

function renderHistory() {
  const div = $("history");
  div.innerHTML = "";
  state.history.forEach((h) => {
    const el = document.createElement("span");
    el.className = "entry";
    el.innerHTML = `<span class="pos">${h.label}</span>`;
    el.addEventListener("click", () => {
      state.lastResponse = h.response;
      state.buffers = {};
      renderSoloControls();
      showStatus(h.response);
    });
    div.appendChild(el);
  });
}

async function renderSessionList() {
  const div = $("sessionEvents");
  div.innerHTML = "";
  state.session.forEach((ev, i) => {
    const el = document.createElement("div");
    el.className = "evrow" + (state.viewEvent === i ? " viewed" : "");
    el.innerHTML =
      `<span class="idx">${i + 1}${ev.direction < 0 ? "′" : ""}</span>` +
      `<span class="pos">(${ev.u.toFixed(3)}, ${ev.v.toFixed(3)})</span>` +
      `<span class="t">t=${ev.time_s.toFixed(3)} s</span>`;
    const solo = document.createElement("button");
    solo.textContent = "alone";
    solo.title = "submit only this event to a silent body (keeps its time slot)";
    solo.addEventListener("click", (e) => { e.stopPropagation(); playEventAlone(i); });
    const iso = document.createElement("button");
    iso.textContent = "iso";
    iso.title = "play this event's independent response (struck at t=0)";
    iso.addEventListener("click", (e) => { e.stopPropagation(); playEventIsolated(i); });
    const flip = document.createElement("button");
    flip.textContent = "flip ⇄";
    flip.title = "reverse this event's excitation direction and re-render";
    flip.addEventListener("click", async (e) => {
      e.stopPropagation();
      ev.direction = -ev.direction;
      state.viewEvent = i;
      await rerender();
      await play("mix");
    });
    const rm = document.createElement("button");
    rm.textContent = "×";
    rm.title = "remove this event and re-render the session without it";
    rm.addEventListener("click", async (e) => {
      e.stopPropagation();
      state.session.splice(i, 1);
      state.viewEvent = state.session.length ? Math.min(i, state.session.length - 1) : null;
      if (state.session.length) { await rerender(); await play("mix"); }
      else { clearSession(); }
    });
    el.appendChild(solo);
    el.appendChild(iso);
    el.appendChild(flip);
    el.appendChild(rm);
    el.addEventListener("click", () => { state.viewEvent = i; drawSurface(); drawModeChart(activeModes()); renderModeList(activeModes()); renderSessionList(); });
    div.appendChild(el);
  });
}

function renderSoloControls() {
  const cats = state.lastResponse ? state.lastResponse.cancellation : null;
  const rc = $("readback");
  if (!state.lastResponse) { rc.innerHTML = ""; return; }
  // Show every mode that is still ringing, plus cancelled modes so their
  // near-zero amplitude and zero tail are visible, not filtered out.
  const cancelled = new Set(cats ? cats.cancelled : []);
  const rows = state.lastResponse.body_state.modes
    .filter((m) => m.amplitude > 0 || cancelled.has(m.index))
    .slice(0, 14)
    .map((m) => {
      const tag = cancelled.has(m.index) ? ' <span class="tag cancel">cancel</span>' : "";
      return (
        `<tr><td>${m.index}</td><td>${m.label}${tag}</td>` +
        `<td>${m.amplitude.toExponential(2)}</td>` +
        `<td>${m.phase_deg.toFixed(1)}°</td>` +
        `<td>${m.tail_s.toFixed(3)} s</td></tr>`
      );
    })
    .join("");
  rc.innerHTML =
    `<table><thead><tr><th>#</th><th>mode</th><th>amp</th><th>phase</th>` +
    `<th>tail (to 1 % of loudest)</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>` +
    (cats
      ? `<div class="catnote">cancelled by the pair: ${cats.cancelled.slice(0, 8).join(", ") || "none"}` +
        ` | reinforced: ${cats.reinforced.slice(0, 8).join(", ") || "none"}</div>`
      : "");
}

async function placeOppositePair() {
  // Two points mirrored across the plate/bar centre: |phi| at the two is
  // equal for every mode (so they sound alike hit alone), while the signed
  // coupling flips for every mode whose index along that axis is even.
  // Flipping one event's direction then lets a listener compare the two
  // couplings directly.
  const t0 = state.session.length
    ? state.session[state.session.length - 1].time_s + intervalMs() / 1000
    : 0;
  const dt = intervalMs() / 1000;
  state.session.push({
    u: 0.35, v: 0.40, time_s: t0, velocity_m_s: velocity(),
    contact_time_s: contactTime(), direction: 1, label: `pair L`,
  });
  state.session.push({
    u: 0.65, v: 0.40, time_s: t0 + dt, velocity_m_s: velocity(),
    contact_time_s: contactTime(), direction: 1, label: `pair R`,
  });
  state.viewEvent = state.session.length - 1;
  const resp = await rerender();
  if (!resp) return;
  await play("mix");
  pushHistory(resp);
}

function clearSession() {
  state.session = [];
  state.lastResponse = null;
  state.viewEvent = null;
  state.buffers = {};
  stopPlayback();
  drawSurface();
  drawModeChart(null);
  renderModeList(null);
  renderSessionList();
  renderSoloControls();
  $("status").textContent = "session cleared";
}

function showStatus(resp) {
  const s = resp.solver;
  const evInfo = resp.events
    .map((e) => `${e.label}: t=${e.applied_time_s.toFixed(4)} s (Δ${(e.time_quantization_s * 1e6).toFixed(0)} µs)`)
    .join(" | ");
  let text =
    `solver: ${s.type} (sustained) | max |pole| = ${s.max_pole_magnitude.toFixed(8)} (< 1: stable)\n` +
    `events: ${evInfo}\n` +
    `modes: ${s.modes_used} used, ${s.excluded_mode_count} above the cap or f_max=${s.f_max_hz} Hz (reported, not hidden)\n` +
    `block size ${resp.block_size} smp | mix peak ${resp.mix_peak_linear.toExponential(2)} -> x${resp.mix_gain_applied.toExponential(2)} (no clipping)`;
  if (resp.replay) {
    const worst = Object.entries(resp.replay.comparisons)
      .map(([k, v]) => `${k}: ${v.max_rel_diff.toExponential(1)}`)
      .join(", ");
    text += `\nblock replay agreement (rel): ${worst}`;
  }
  if (resp.solo_event !== null && resp.solo_event !== undefined) {
    text += `\nauditioning only event ${resp.solo_event + 1}`;
  }
  if (resp.solo_mode !== null && resp.solo_mode !== undefined) {
    text += `\nauditioning only mode ${resp.solo_mode}`;
  }
  $("status").innerHTML = resp.warnings.length
    ? text + `\n<span class="warn">warning: ${resp.warnings.join("; ")}</span>`
    : text;
}

// ---------------------------------------------------------------------------
// model info + loading
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
    `(${s.excluded_mode_count} more excluded by the mode cap - reported, not hidden)</td></tr>` +
    `<tr><th>body swap</th><td>changing the model installs a new body: the old tail is discarded, ` +
    `never mixed into the new modes; the session clock is preserved</td></tr></table>` +
    `<ul>${s.notes.map((n) => `<li>${n}</li>`).join("")}</ul>` +
    `<div class="warn">All parameters are declared analytical/procedural models - nothing here is a measurement of a real object.</div>`;
}

let dampingReloadTimer = null;

async function loadModel() {
  const id = $("modelSelect").value;
  state.spec = await fetchJSON(`/api/model?id=${id}&damping_scale=${dampingScale()}`);
  state.selectedMode = null;
  state.modeShape = null;
  // A body change discards the ringing session: the server-side engine
  // never mixes one body's tail into another's.
  if (state.session.length) {
    const dropped = state.session.length;
    state.session = [];
    state.lastResponse = null;
    state.viewEvent = null;
    state.buffers = {};
    stopPlayback();
    $("status").textContent =
      `model changed: new body installed, ${dropped} ringing event(s) discarded (clock kept)`;
  }
  drawSurface();
  drawModeChart(null);
  renderModeList(null);
  renderSessionList();
  renderModelInfo();
  $("selectedModeInfo").textContent =
    "no mode selected - click a bar in the chart or a row in the table";
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
  sel.addEventListener("change", async () => {
    await loadModel();
  });

  for (const id of ["velocity", "contact", "duration", "interval"]) {
    $(id).addEventListener("input", () => { updateSliderLabels(); renderSoloControls(); });
  }
  $("damping").addEventListener("input", () => {
    updateSliderLabels();
    clearTimeout(dampingReloadTimer);
    dampingReloadTimer = setTimeout(loadModel, 300);
  });

  $("blockSize").addEventListener("input", () => {
    state.blockSize = 2 ** Number($("blockSize").value);
    updateSliderLabels();
    if (state.session.length) rerender().then(() => play("mix")).catch(() => {});
  });

  $("modelInfoBtn").addEventListener("click", () =>
    $("modelInfo").classList.toggle("hidden")
  );

  $("surface").addEventListener("click", (evt) => {
    const { u, v } = canvasToHit(evt);
    strike(u, v, evt.shiftKey ? -1 : 1).catch(() => {});
  });

  $("modeChart").addEventListener("click", (evt) => {
    const i = chartModeAt(evt);
    if (i !== null) selectMode(i);
  });

  $("clearSession").addEventListener("click", clearSession);
  $("placePair").addEventListener("click", () => placeOppositePair().catch(() => {}));
  $("runReplay").addEventListener("click", async () => {
    if (state.session.length < 2) {
      $("status").textContent = "block-replay audit needs at least two events";
      return;
    }
    state.replaySizes = [64, 256, state.blockSize];
    try { await rerender(); } finally { state.replaySizes = null; }
  });

  updateSliderLabels();
  await loadModel();
}

init();
