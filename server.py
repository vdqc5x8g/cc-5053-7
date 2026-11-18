#!/usr/bin/env python3
"""HTTP server for the modal synthesis percussion instrument.

Serves the web UI and a JSON API. Every hit runs the modal solver
server-side; no pre-rendered audio is stored anywhere.

API
---
GET  /api/models                      list of available instrument models
GET  /api/model?id=...                full mode table + geometry + notes
GET  /api/mode_shape?id=...&mode=...&n=...   sampled mode shape grid
POST /api/hit                         strike: runs the solver, returns WAV
                                      audio (base64) + per-mode activation
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from modal_synth import models
from modal_synth.audio import to_wav_bytes
from modal_synth.synth import Mallet, ModalInstrument

ROOT = Path(__file__).resolve().parent
UI_DIR = ROOT / "ui"

SAMPLE_RATE = 44100
MAX_DURATION_S = 5.0
VELOCITY_RANGE = (0.05, 10.0)  # m/s
CONTACT_TIME_RANGE = (0.0, 0.02)  # s
DAMPING_SCALE_RANGE = (0.05, 100.0)

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


def handle_hit(body: dict) -> dict:
    model_id = str(body.get("model", ""))
    if model_id not in models.BUILDERS:
        raise ValueError(f"unknown model '{model_id}'")
    u = float(body.get("u", 0.5))
    v = float(body.get("v", 0.5))
    velocity = float(body.get("velocity_m_s", 2.0))
    contact_time = float(body.get("contact_time_s", 0.001))
    damping_scale = float(body.get("damping_scale", 1.0))
    duration = float(body.get("duration_s", 2.0))
    solo_mode = body.get("solo_mode", None)
    if not (VELOCITY_RANGE[0] <= velocity <= VELOCITY_RANGE[1]):
        raise ValueError(f"velocity_m_s must be in {VELOCITY_RANGE}")
    if not (CONTACT_TIME_RANGE[0] <= contact_time <= CONTACT_TIME_RANGE[1]):
        raise ValueError(f"contact_time_s must be in {CONTACT_TIME_RANGE}")
    if not (DAMPING_SCALE_RANGE[0] <= damping_scale <= DAMPING_SCALE_RANGE[1]):
        raise ValueError(f"damping_scale must be in {DAMPING_SCALE_RANGE}")
    if not (0.05 <= duration <= MAX_DURATION_S):
        raise ValueError(f"duration_s must be in (0.05, {MAX_DURATION_S}]")

    spec = get_spec(model_id, damping_scale)
    if solo_mode is not None:
        solo_mode = int(solo_mode)
        if not (0 <= solo_mode < len(spec.modes)):
            raise ValueError("solo_mode out of range")

    inst = make_instrument(spec, contact_time)
    x_m, y_m = inst.map_hit(u, v)
    report = inst.strike(x_m, y_m, velocity, solo_mode=solo_mode)
    out = inst.render(duration)
    wav, peak, gain = to_wav_bytes(out, SAMPLE_RATE)
    pole_max = max(
        max(osc.pole_magnitudes()) for osc in inst.oscillators
    )
    return {
        "audio_b64": base64.b64encode(wav).decode("ascii"),
        "sample_rate_hz": SAMPLE_RATE,
        "duration_s": duration,
        "peak_linear": peak,
        "gain_applied": gain,
        "x_m": x_m,
        "y_m": y_m,
        "velocity_m_s": velocity,
        "impulse_n_s": report.impulse_n_s,
        "contact_time_s": report.contact_time_s,
        "solo_mode": solo_mode,
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
        "warnings": report.warnings,
        "solver": {
            "type": "exact matrix-exponential state integrator (per mode)",
            "max_pole_magnitude": pole_max,
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
    server_version = "ModalSynth/1.0"

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
