# Modal Synthesis Percussion

A struck-object synthesizer: virtual metal plates and wooden/aluminum bars
are represented by a declared set of vibrational modes. Clicking the
surface excites the modes according to **where** and **how hard** you hit,
and the decaying sound is synthesized sample by sample by an exactly
stable discrete solver. No pre-recorded audio is used anywhere; every
click runs the solver.

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
* `POST /api/hit` — `{model, u, v, velocity_m_s, contact_time_s,
  damping_scale, duration_s, solo_mode}` → base64 WAV + per-mode
  activation + solver diagnostics (max pole magnitude, excluded modes)

## Layout

```
modal_synth/
  physics.py     analytical mode models (plate, bar), damping law
  resonator.py   the discrete modal oscillator (signal solver)
  synth.py       hit mapping, mallet model, instrument, rendering
  models.py      declared presets (incl. procedural perturbed plate)
  audio.py       WAV export (normalize only, never clip)
server.py        stdlib HTTP server: static UI + JSON API
ui/              web interface (vanilla JS, canvas)
tests/           pytest suite (physics, solver, instrument, API)
```
