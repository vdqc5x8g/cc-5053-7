#!/usr/bin/env python3
"""HTTP server for the modal synthesis percussion instrument.

Serves the web UI and a JSON API. The interaction layer submits *hit
events* - a position, a velocity and a sample-accurate time - and the
server runs one sustained modal engine over the whole event list, so a
second click lands on a body that is still ringing from the first.

API
---
GET  /api/models                      list of available instrument models
GET  /api/model?id=...                full mode table + geometry + notes
GET  /api/mode_shape?id=...&mode=...&n=...   sampled mode shape grid
POST /api/hit                         sustained render of submitted hit events

The hit request body:

    {
      "model": "steel_plate",
      "damping_scale": 1.0,                  # optional, default 1
      "duration_s": 2.0,                     # render length
      "block_size": 512,                     # optional render block length
      "events": [
        {"u": 0.42, "v": 0.37, "time_s": 0.0,
         "velocity_m_s": 2.0, "contact_time_s": 0.001,
         "direction": 1, "label": "A"},
        ...
      ],
      "solo_event": 1,                       # optional: audition one hit
      "solo_mode": 2,                        # optional: audition one mode
      "include_isolated": true,              # per-event stems for A/B
      "replay_block_sizes": [64, 256, 2048]  # optional block-invariance audit
    }

A legacy single-hit body without ``events`` (``u``, ``v``, ...) is
accepted and treated as one event at t=0.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from modal_synth import audit, models
from modal_synth.audio import to_wav_bytes
from modal_synth.synth import HitEvent, Mallet, ModalEngine, ModalInstrument

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"

SAMPLE_RATE = 44100
MAX_DURATION_S = 5.0
MAX_EVENTS = 8
VELOCITY_RANGE = (0.05, 10.0)  # m/s
CONTACT_TIME_RANGE = (0.0, 0.02)  # s
DAMPING_SCALE_RANGE = (0.05, 100.0)
BLOCK_SIZE_RANGE = (1, 65536)

_spec_cache: dict[tuple[str, float], models.InstrumentSpec] = {}


def get_spec(model_id: str, damping_scale: float = 1.0) -> models.InstrumentSpec:
    key = (model_id, round(damping_scale, 6))
    if key not in _spec_cache:
        _spec_cache[key] = models.build(model_id, damping_scale)
    return _spec_cache[key]


def spec_to_json(spec: models.InstrumentSpec) -> dict:
    kind = spec.geometry[0]
    geometry = (
        {"type": "plate", "a_m": spec.geometry[1], "b_m": spec.geometry[2]}
        if kind == "plate"
        else {"type": "bar", "length_m": spec.geometry[1], "width_m": spec.geometry[2]}
    )
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "geometry": geometry,
        "thickness_m": spec.thickness_m,
        "material": {
            "name": spec.material.name,
            "youngs_modulus_pa": spec.material.youngs_modulus_pa,
            "density_kg_m3": spec.material.density_kg_m3,
            "poisson_ratio": spec.material.poisson_ratio,
            "loss_factor": spec.material.loss_factor,
        },
        "f_max_hz": spec.f_max_hz,
        "excluded_mode_count": spec.excluded_mode_count,
        "notes": spec.notes,
        "modes": [
            {
                "index": i,
                "label": m.label,
                "frequency_hz": m.frequency_hz,
                "tau_s": m.tau_s,
                "modal_mass_kg": m.modal_mass_kg,
            }
            for i, m in enumerate(spec.modes)
        ],
    }


def make_instrument(spec: models.InstrumentSpec, contact_time_s: float) -> ModalInstrument:
    return ModalInstrument(
        spec.modes,
        spec.geometry,
        sample_rate_hz=SAMPLE_RATE,
        mallet=Mallet(mass_kg=0.02, contact_time_s=contact_time_s),
        thickness_m=spec.thickness_m,
    )


def _parse_events(body: dict, instrument: ModalInstrument) -> list[HitEvent]:
    default_velocity = float(body.get("velocity_m_s", 2.0))
    default_contact = float(body.get("contact_time_s", 0.001))
    raw = body.get("events")
    if raw is None:
        raw = [
            {
                "u": body.get("u", 0.5),
                "v": body.get("v", 0.5),
                "time_s": 0.0,
                "velocity_m_s": default_velocity,
                "contact_time_s": default_contact,
                "direction": 1,
            }
        ]
    if not isinstance(raw, list) or not raw:
        raise ValueError("events must be a non-empty list")
    if len(raw) > MAX_EVENTS:
        raise ValueError(f"at most {MAX_EVENTS} events per request")
    events = []
    for i, ev in enumerate(raw):
        u = float(ev.get("u", 0.5))
        v = float(ev.get("v", 0.5))
        time_s = float(ev.get("time_s", 0.0))
        velocity = float(ev.get("velocity_m_s", default_velocity))
        contact = float(ev.get("contact_time_s", default_contact))
        direction = int(ev.get("direction", 1))
        if time_s < 0.0:
            raise ValueError(f"event {i}: time_s must be non-negative")
        if not (VELOCITY_RANGE[0] <= velocity <= VELOCITY_RANGE[1]):
            raise ValueError(
                f"event {i}: velocity_m_s must be in {VELOCITY_RANGE}"
            )
        if not (CONTACT_TIME_RANGE[0] <= contact <= CONTACT_TIME_RANGE[1]):
            raise ValueError(
                f"event {i}: contact_time_s must be in {CONTACT_TIME_RANGE}"
            )
        if direction not in (1, -1):
            raise ValueError(f"event {i}: direction must be +1 or -1")
        x_m, y_m = instrument.map_hit(u, v)
        events.append(
            HitEvent(
                position_m=(x_m, y_m),
                time_s=time_s,
                velocity_m_s=velocity,
                contact_time_s=contact,
                direction=direction,
                label=str(ev.get("label", f"hit {i}")),
            )
        )
    return events


def _event_json(index: int, body_ev: dict, report) -> dict:
    ev = report.event
    return {
        "index": index,
        "label": ev.label,
        "u": float(body_ev.get("u", 0.5)),
        "v": float(body_ev.get("v", 0.5)),
        "x_m": ev.position_m[0],
        "y_m": ev.position_m[1],
        "time_s": ev.time_s,
        "applied_sample": report.applied_sample,
        "applied_time_s": report.applied_time_s,
        "time_quantization_s": report.time_quantization_s,
        "velocity_m_s": ev.velocity_m_s,
        "contact_time_s": ev.contact_time_s,
        "direction": ev.direction,
        "impulse_n_s": report.impulse_n_s,
        "warnings": report.warnings,
        "modes": [
            {
                "index": em.index,
                "label": em.label,
                "frequency_hz": em.frequency_hz,
                "tau_s": em.tau_s,
                "shape_at_hit": em.shape_at_hit,
                "mallet_weight": em.mallet_weight,
                "delta_v_m_s": em.delta_v_m_s,
                "peak_displacement_m": em.peak_displacement_m,
            }
            for em in report.modes
        ],
    }


def _render_engine(
    instrument: ModalInstrument,
    events: list[HitEvent],
    duration_s: float,
    block_size: int,
    solo_event: int | None = None,
    solo_mode: int | None = None,
):
    """Build an engine, submit events, render one sustained mix.

    Returns (engine, mix, chosen) where ``chosen`` is the sample array the
    caller should audition (mix, solo-event stem, or solo-mode rendering).
    """
    engine = ModalEngine(instrument)
    for i, ev in enumerate(events):
        if solo_event is not None and i != solo_event:
            continue
        engine.submit(ev, only_mode=solo_mode)
    mix = engine.render_blocks(
        int(round(duration_s * SAMPLE_RATE)), block_size
    )
    return engine, mix


def handle_hit(body: dict) -> dict:
    model_id = str(body.get("model", ""))
    if model_id not in models.BUILDERS:
        raise ValueError(f"unknown model '{model_id}'")
    damping_scale = float(body.get("damping_scale", 1.0))
    duration = float(body.get("duration_s", 2.0))
    block_size = int(body.get("block_size", 512))
    if not (DAMPING_SCALE_RANGE[0] <= damping_scale <= DAMPING_SCALE_RANGE[1]):
        raise ValueError(f"damping_scale must be in {DAMPING_SCALE_RANGE}")
    if not (0.05 <= duration <= MAX_DURATION_S):
        raise ValueError(f"duration_s must be in (0.05, {MAX_DURATION_S}]")
    if not (BLOCK_SIZE_RANGE[0] <= block_size <= BLOCK_SIZE_RANGE[1]):
        raise ValueError(f"block_size must be in {BLOCK_SIZE_RANGE}")

    spec = get_spec(model_id, damping_scale)
    contact_time = float(body.get("contact_time_s", 0.001))
    if not (CONTACT_TIME_RANGE[0] <= contact_time <= CONTACT_TIME_RANGE[1]):
        raise ValueError(f"contact_time_s must be in {CONTACT_TIME_RANGE}")
    instrument = make_instrument(spec, contact_time)
    events = _parse_events(body, instrument)

    solo_event = body.get("solo_event", None)
    if solo_event is not None:
        solo_event = int(solo_event)
        if not (0 <= solo_event < len(events)):
            raise ValueError("solo_event out of range")
    solo_mode = body.get("solo_mode", None)
    if solo_mode is not None:
        solo_mode = int(solo_mode)
        if not (0 <= solo_mode < len(spec.modes)):
            raise ValueError("solo_mode out of range")

    # One sustained mix of every event, plus the possibly soloed audition.
    full_engine, full_mix = _render_engine(instrument, events, duration, block_size)
    if solo_event is None and solo_mode is None:
        chosen = full_mix
    else:
        _, chosen = _render_engine(
            instrument,
            events,
            duration,
            block_size,
            solo_event=solo_event,
            solo_mode=solo_mode,
        )

    wav, peak, gain = to_wav_bytes(chosen, SAMPLE_RATE)
    full_wav, full_peak, full_gain = to_wav_bytes(full_mix, SAMPLE_RATE)

    raw_events = body.get("events")
    raw_events = raw_events if raw_events is not None else [
        {"u": body.get("u", 0.5), "v": body.get("v", 0.5)}
    ]
    event_json = [
        _event_json(i, raw_events[i], rep)
        for i, rep in enumerate(full_engine.reports)
    ]

    readback = audit.mode_tail_readback(
        full_engine, n_tail=1.0, floor_ratio=0.01
    )

    # Per-event isolated stems (each struck at t=0) for A/B listening.
    isolated_b64 = []
    if body.get("include_isolated", True):
        for ev in events:
            eng = ModalEngine(instrument)
            eng.submit(HitEvent(
                position_m=ev.position_m,
                time_s=0.0,
                velocity_m_s=ev.velocity_m_s,
                contact_time_s=ev.contact_time_s,
                mallet_mass_kg=ev.mallet_mass_kg,
                direction=ev.direction,
                label=ev.label,
            ), only_mode=solo_mode)
            iso = eng.render_blocks(int(round(duration * SAMPLE_RATE)), block_size)
            iso_wav, _, _ = to_wav_bytes(iso, SAMPLE_RATE)
            isolated_b64.append(base64.b64encode(iso_wav).decode("ascii"))

    replay = None
    sizes = body.get("replay_block_sizes")
    if sizes:
        sizes = [int(s) for s in sizes][:6]
        if any(s < 1 for s in sizes):
            raise ValueError("replay_block_sizes must be positive")
        replay = audit.replay_blocks(instrument, events, duration, sizes)
        replay.pop("outputs", None)

    cancellation = audit.overlap_cancellation(full_engine.reports, len(events))

    return {
        "audio_b64": base64.b64encode(wav).decode("ascii"),
        "mix_audio_b64": base64.b64encode(full_wav).decode("ascii"),
        "isolated_audio_b64": isolated_b64,
        "sample_rate_hz": SAMPLE_RATE,
        "duration_s": duration,
        "peak_linear": peak,
        "gain_applied": gain,
        "mix_peak_linear": full_peak,
        "mix_gain_applied": full_gain,
        "block_size": block_size,
        "events": event_json,
        "solo_event": solo_event,
        "solo_mode": solo_mode,
        "cancellation": cancellation,
        "body_state": readback,
        "replay": replay,
        "warnings": sorted({w for e in event_json for w in e["warnings"]}),
        "solver": {
            "type": "exact matrix-exponential state integrator (per mode)",
            "max_pole_magnitude": full_engine.voice.max_pole_magnitude(),
            "f_max_hz": spec.f_max_hz,
            "modes_used": len(spec.modes),
            "excluded_mode_count": spec.excluded_mode_count,
        },
    }


def handle_mode_shape(query: dict) -> dict:
    model_id = query.get("id", [""])[0]
    mode_index = int(query.get("mode", ["0"])[0])
    n = min(int(query.get("n", ["41"])[0]), 81)
    damping_scale = float(query.get("damping_scale", ["1.0"])[0])
    spec = get_spec(model_id, damping_scale)
    if not (0 <= mode_index < len(spec.modes)):
        raise ValueError("mode index out of range")
    mode = spec.modes[mode_index]
    kind = spec.geometry[0]
    if kind == "plate":
        a, b = spec.geometry[1], spec.geometry[2]
        xs = [a * i / (n - 1) for i in range(n)]
        ys = [b * j / (n - 1) for j in range(n)]
        grid = [
            [round(float(mode.shape(x, y)), 4) for x in xs] for y in ys
        ]
        return {"type": "plate", "xs_m": xs, "ys_m": ys, "grid": grid}
    length = spec.geometry[1]
    xs = [length * i / (n - 1) for i in range(n)]
    values = [round(float(mode.shape(x)), 4) for x in xs]
    return {"type": "bar", "xs_m": xs, "values": values}


class Handler(BaseHTTPRequestHandler):
    server_version = "ModalSynth/2.0"

    def _send_json(self, obj: dict, status: int = 200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: Path, content_type: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str):
        self._send_json({"error": message}, status)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        try:
            if path == "/":
                self._send_file(UI_DIR / "index.html", "text/html; charset=utf-8")
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif path == "/app.js":
                self._send_file(UI_DIR / "app.js", "text/javascript; charset=utf-8")
            elif path == "/style.css":
                self._send_file(UI_DIR / "style.css", "text/css; charset=utf-8")
            elif path == "/api/models":
                self._send_json(
                    {
                        "models": [
                            {
                                "id": mid,
                                "name": get_spec(mid).name,
                                "description": get_spec(mid).description,
                            }
                            for mid in models.BUILDERS
                        ]
                    }
                )
            elif path == "/api/model":
                query = urllib.parse.parse_qs(parsed.query)
                model_id = query.get("id", [""])[0]
                damping_scale = float(query.get("damping_scale", ["1.0"])[0])
                if model_id not in models.BUILDERS:
                    self._error(404, f"unknown model '{model_id}'")
                    return
                self._send_json(spec_to_json(get_spec(model_id, damping_scale)))
            elif path == "/api/mode_shape":
                query = urllib.parse.parse_qs(parsed.query)
                self._send_json(handle_mode_shape(query))
            else:
                self._error(404, "not found")
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/hit":
            self._error(404, "not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(handle_hit(body))
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # noqa: BLE001
            self._error(500, f"{type(exc).__name__}: {exc}")

    def log_message(self, fmt, *args):  # quieter logs
        pass


def main():
    parser = argparse.ArgumentParser(description="Modal synthesis percussion server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"modal synth server on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
