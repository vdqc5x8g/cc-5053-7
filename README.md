# Modal Synthesis Percussion

A struck-object synthesizer: virtual metal plates and wooden/aluminum bars
are represented by a declared set of vibrational modes. Clicking the
surface submits a **hit event** — a position and a sample-accurate time —
and the sustained modal engine adds that event's excitation to a body that
may still be ringing from the previous click. The decaying sound is
synthesized sample by sample by an exactly stable discrete solver. No
pre-recorded audio is used anywhere; every click runs the solver.

## Run

```bash
python3 server.py --port 8000
# open http://127.0.0.1:8000
```

Dependencies: `numpy` (solver + tests), `pytest` (tests only).
The server itself is Python standard library only.

## Test

```bash
python3 -m pytest tests/ -q
```

Covers: analytical mode formulas and boundary conditions; solver
correctness (rendered frequency and decay vs. declared parameters);
solver stability over the full supported parameter range (including
modes at 0.45·fs and extreme damping); hit mapping (center vs. edge,
bar nodal points); velocity linearity and mallet spectral effects;
high-damping settings; spectral peaks and decay times of rendered
audio; WAV normalization without clipping; the HTTP API end to end.

The refactor-specific suite (`tests/test_sustained.py`) additionally
covers: a hit event carrying only position + time; a second click landing
on a still-ringing body; two clicks straddling a nodal line cancelling
that mode while reinforcing its neighbours; a direction flip negating one
click's excitation; isolated per-hit responses summing exactly to the
accumulated response (superposition); block-size invariance and no mode
being zeroed at a block edge; body swaps discarding old tails; and the
live readback of a mode's phase and remaining tail.

## Physics: model sources

All quantities are SI. Each mode is declared as
`(frequency [Hz], tau [s], modal mass [kg], mode shape φ (dimensionless, max |φ| = 1))`.

| Model | Source | Formula |
|---|---|---|
| Rectangular thin plate, simply supported | Navier's closed-form solution (Leissa, *Vibration of Plates*, 1969) | `f_mn = (π/2)·√(D/ρh)·(m²/a² + n²/b²)`, `D = Eh³/(12(1−ν²))`, `φ = sin(mπx/a)sin(nπy/b)` |
| Free-free bar | Euler–Bernoulli beam theory (Blevins, *Formulas for Natural Frequency and Mode Shape*) | `f_n = (β_nL)²/(2πL²)·h·√(E/12ρ)`, `cos(βL)cosh(βL)=1` solved by Newton iteration |
| Perturbed plate | procedural | plate modes with seeded random jitter (±4 % f, ±30 % τ) — a statistical model of manufacturing variability |

Damping is a constant loss factor η per model: `τ_n = 1/(π f_n η)`
(steel 3e-4, aluminum 1e-3, wood 8e-3).

## Declared approximations (not measurements)

* Simply supported plate edges are exactly solvable; real free/clamped
  edges are not. This is a deliberate simplification.
* Thin-plate/thin-beam theory: no shear deformation or rotary inertia.
* Constant loss factor; real damping is frequency dependent.
* Rigid-body bar modes (0 Hz) excluded; bar torsion/width modes neglected.
* Radiation model: output = sum of modal velocities with unit weight per
  mode (radiation efficiency and directivity ignored). Output is in
  arbitrary linear units, normalized to peak 0.89 on WAV export —
  **never clipped**.
* Mallet: impulse `J = m·v` fully transferred; velocity-dependent
  low-pass `W(f) = 1/(1+(π f t_c)²)` with Hertzian contact-time scaling
  `t_c ∝ v^(−1/5)`. This is why harder strikes sound brighter, not just
  louder.
* Nothing is a measurement of a physical instrument. The UI's
  "model & approximations" panel shows this same declaration.

## The solver and its stability

Each mode integrates `q̈ + 2αq̇ + ω₀²q = 0` by the **exact matrix
exponential** of its state transition over one sample:

```
[q; v]_{k+1} = expm(A·T) [q; v]_k      A = [[0, 1], [−ω₀², −2α]]
```

The discrete poles are `exp((−α ± jω_d)T)` with magnitude `exp(−αT) ≤ 1`
for **every** α ≥ 0, sample rate, and mode frequency — including modes
near fs/2 and arbitrarily light or heavy damping (the overdamped branch
is evaluated as decaying exponentials so nothing can overflow). The
recursion cannot diverge, so no output clipping exists to hide
divergence; tests assert boundedness against a theoretical envelope and
verify that a 0.45·fs mode rings for seconds without growth.

`render()` evaluates the same recursion in closed form (the matrix power
applied analytically); a test verifies it matches per-sample `step()`
calls to floating-point accuracy.

## Sustained vibration: events, body state and block rendering

The interaction layer submits only a **hit event** (`HitEvent`): a
surface position, a strike time in seconds, and the mallet/velocity data
that produced it. It carries no oscillators, durations or engine handles.
`ModalInstrument` maps a physical strike to a vector of per-mode velocity
kicks; `ModalEngine` owns one persistent `ModalVoice` for the body and
schedules each event's kicks at its sample time.

* **Accumulation, not reset.** A second click is *added* to the modal
  state that is still decaying from the first — the physical
  superposition. Nothing is cleared on a hit, on a redraw, or between
  render calls.
* **Sample-accurate timing.** Event times are quantized to the next
  integer sample; the response reports the applied sample and the
  quantization (≤ 1 sample ≈ 23 µs at 44.1 kHz). A late event is
  rejected rather than silently moved to the present.
* **Block rendering.** `render_block(n)` advances every mode by exactly n
  samples and splits only at impulse times. Because the state is carried
  across calls and `M^a·M^b = M^(a+b)`, rendering the same session with
  different block lengths gives the same signal up to float rounding, and
  no mode is zeroed at a block edge. Tests replay one session at 1, 17,
  64, 512, 4096 and full-length blocks and trace per-mode stems across
  edges to prove it.
* **Body replacement.** Swapping the model installs a new body and
  *discards* the old tail; the new body's state is zero at the current
  clock time, so old tails cannot restart on, or leak into, the new mode
  table. This is the inheritance rule: strike-time parameters (mallet,
  geometry, material) are re-derived for the new body's event; nothing of
  the old oscillation is inherited. The session clock is preserved.
* **Direction.** A hit event carries `direction ∈ {+1, −1}`, which negates
  that event's excitation (equivalent to striking from the other face).
  Reversing one of a mirrored pair flips which modes the pair cancels
  versus reinforces.

### Isolated responses vs. one accumulated state

Because the body is linear, "render each hit as its own response and sum
them" and "add every hit to one growing modal state" must agree. The UI
exposes both: each event can be auditioned **alone** (submitted to an
otherwise silent body, keeping its time slot) or **isolated** (an
independent response struck at t=0). The audit helper
`isolated_vs_accumulated` renders both and reports the residual, which the
tests require to be below 1e-10 relative.

### Nodal-line cancellation

Two clicks placed on opposite sides of a mode's nodal line receive
opposite-signed `φ_n(p)`, so their kicks cancel for that mode while adding
for modes whose antinode lies between the points. The UI's *place mirror
pair* button puts a mirrored pair like this in the session; the mode table
tags each mode `cancelled` or `add`, and the live readback shows the
surviving modes' phase and remaining tail.

## Hit mapping (position → excitation)

A strike at surface point `p` with velocity `v` gives mode `n` the modal
velocity kick

```
Δv_n = J · W(f_n, v) · φ_n(p) / M_n        [m/s]
```

Position enters through the mode shape `φ_n(p)`: hitting a nodal line
silences that mode at any force. The API returns every mode's
`φ_n(p)`, mallet weight and `Δv_n`; the UI chart and table display
exactly these solver inputs.

## API

* `GET /api/models` — list models
* `GET /api/model?id=...` — mode table, geometry, material, notes
* `GET /api/mode_shape?id=...&mode=...&n=...` — sampled mode shape
* `POST /api/hit` — submit hit events, get one sustained render

  ```json
  {
    "model": "steel_plate",
    "damping_scale": 1.0,
    "duration_s": 2.0,
    "block_size": 512,
    "events": [
      {"u": 0.42, "v": 0.37, "time_s": 0.0, "velocity_m_s": 2.0,
       "contact_time_s": 0.001, "direction": 1, "label": "A"},
      {"u": 0.58, "v": 0.37, "time_s": 0.12, "velocity_m_s": 2.0,
       "contact_time_s": 0.001, "direction": -1, "label": "B"}
    ],
    "solo_event": 1,
    "solo_mode": 2,
    "include_isolated": true,
    "replay_block_sizes": [64, 256, 2048]
  }
  ```

  Returns base64 WAV audio (the auditioned channel `audio_b64`, plus the
  always-full `mix_audio_b64` and per-event `isolated_audio_b64`),
  per-event activation with their applied sample times, the pair
  `cancellation` report, the live `body_state` readback (phase / amplitude
  / tail), optional `replay` block-invariance diagnostics, and solver
  diagnostics (max pole magnitude, excluded modes). A legacy single-hit
  body (`u`, `v`, ...) is accepted and treated as one event at t=0.

## Layout

```
modal_synth/
  physics.py     analytical mode models (plate, bar), damping law
  resonator.py   ModalOscillator (numerical core) + ModalVoice (sustained, block-rendered)
  synth.py       hit event mapping, mallet model, ModalInstrument, ModalEngine
  audit.py       superposition / block-replay / cancellation / tail readback tools
  models.py      declared presets (incl. procedural perturbed plate)
  audio.py       WAV export (normalize only, never clip)
server.py        stdlib HTTP server: static UI + JSON API
ui/              web interface (vanilla JS, canvas)
tests/           pytest suite (physics, solver, instrument, sustained, API)
```
