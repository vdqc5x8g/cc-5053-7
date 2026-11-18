"""End-to-end API tests against a live server instance."""

import base64
import io
import json
import threading
import urllib.request
import urllib.error
import wave
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

import server


@pytest.fixture(scope="module")
def api():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()


def get(base, path):
    with urllib.request.urlopen(base + path) as resp:
        return json.loads(resp.read())


def post(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def decode_wav(b64):
    data = base64.b64decode(b64)
    with wave.open(io.BytesIO(data)) as wf:
        frames = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32767.0
        return pcm, wf.getframerate()


def test_models_endpoint(api):
    models = get(api, "/api/models")["models"]
    ids = {m["id"] for m in models}
    assert {"steel_plate", "aluminum_bar", "wood_bar", "perturbed_plate"} <= ids


def test_model_endpoint_reports_modes_and_notes(api):
    spec = get(api, "/api/model?id=steel_plate")
    assert len(spec["modes"]) > 10
    assert spec["excluded_mode_count"] > 0  # honestly reported truncation
    assert any("not a measurement" in note for note in spec["notes"])
    m0 = spec["modes"][0]
    assert m0["frequency_hz"] > 0 and m0["tau_s"] > 0


def test_hit_returns_audio_and_activation(api):
    resp = post(
        api,
        "/api/hit",
        {"model": "steel_plate", "u": 0.5, "v": 0.5, "velocity_m_s": 2.0,
         "duration_s": 1.0},
    )
    pcm, fs = decode_wav(resp["audio_b64"])
    assert fs == 44100
    assert pcm.size == 44100
    assert np.max(np.abs(pcm)) <= 0.90  # normalized, never clipped
    assert len(resp["modes"]) == 64
    # Center hit: fundamental dominates; |delta_v| of mode 0 is the largest.
    mags = [abs(m["delta_v_m_s"]) for m in resp["modes"]]
    assert mags[0] == max(mags)
    assert resp["solver"]["max_pole_magnitude"] < 1.0


def test_hit_position_changes_activation(api):
    center = post(api, "/api/hit", {"model": "steel_plate", "u": 0.5, "v": 0.5,
                                    "velocity_m_s": 2.0, "duration_s": 0.5})
    corner = post(api, "/api/hit", {"model": "steel_plate", "u": 0.05, "v": 0.05,
                                    "velocity_m_s": 2.0, "duration_s": 0.5})
    c0 = abs(center["modes"][0]["delta_v_m_s"])
    k0 = abs(corner["modes"][0]["delta_v_m_s"])
    assert c0 > 20 * k0


def test_hit_validation_errors(api):
    for body, status in [
        ({"model": "nope"}, 400),
        ({"model": "steel_plate", "velocity_m_s": 100.0}, 400),
        ({"model": "steel_plate", "duration_s": 99.0}, 400),
        ({"model": "steel_plate", "solo_mode": 999}, 400),
    ]:
        req = urllib.request.Request(
            api + "/api/hit",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(req)
        assert excinfo.value.code == status


def test_solo_mode_frequency(api):
    spec = get(api, "/api/model?id=aluminum_bar")
    f1 = spec["modes"][1]["frequency_hz"]
    resp = post(api, "/api/hit", {"model": "aluminum_bar", "u": 0.3, "v": 0.5,
                                  "velocity_m_s": 2.0, "duration_s": 1.0,
                                  "solo_mode": 1})
    pcm, fs = decode_wav(resp["audio_b64"])
    spec_mag = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(pcm.size, 1.0 / fs)
    peak_f = freqs[int(np.argmax(spec_mag))]
    assert abs(peak_f - f1) < 0.01 * f1


def test_mode_shape_endpoint(api):
    shape = get(api, "/api/mode_shape?id=steel_plate&mode=0&n=21")
    assert shape["type"] == "plate"
    grid = np.array(shape["grid"])
    assert grid.shape == (21, 21)
    # (1,1) mode: zero on the boundary, max at the center.
    assert np.allclose(grid[0, :], 0.0) and np.allclose(grid[:, 0], 0.0)
    assert abs(grid[10, 10]) == pytest.approx(1.0, abs=0.01)
    bar = get(api, "/api/mode_shape?id=wood_bar&mode=0&n=41")
    assert bar["type"] == "bar" and len(bar["values"]) == 41
