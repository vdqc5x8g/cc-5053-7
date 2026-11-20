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


def post(base, body):
    req = urllib.request.Request(
        base + "/api/hit",
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
        {"model": "steel_plate", "u": 0.5, "v": 0.5, "velocity_m_s": 2.0,
         "duration_s": 1.0},
    )
    pcm, fs = decode_wav(resp["audio_b64"])
    assert fs == 44100
    assert pcm.size == 44100
    assert np.max(np.abs(pcm)) <= 0.90  # normalized, never clipped
    assert len(resp["events"]) == 1
    assert len(resp["events"][0]["modes"]) == 64
    # Center hit: fundamental dominates; |delta_v| of mode 0 is the largest.
    mags = [abs(m["delta_v_m_s"]) for m in resp["events"][0]["modes"]]
    assert mags[0] == max(mags)
    assert resp["solver"]["max_pole_magnitude"] < 1.0


def test_hit_position_changes_activation(api):
    center = post(api, {"model": "steel_plate", "u": 0.5, "v": 0.5,
                                    "velocity_m_s": 2.0, "duration_s": 0.5})
    corner = post(api, {"model": "steel_plate", "u": 0.05, "v": 0.05,
                                    "velocity_m_s": 2.0, "duration_s": 0.5})
    c0 = abs(center["events"][0]["modes"][0]["delta_v_m_s"])
    k0 = abs(corner["events"][0]["modes"][0]["delta_v_m_s"])
    assert c0 > 20 * k0


def test_hit_validation_errors(api):
    for body, status in [
        ({"model": "nope"}, 400),
        ({"model": "steel_plate", "velocity_m_s": 100.0}, 400),
        ({"model": "steel_plate", "duration_s": 99.0}, 400),
        ({"model": "steel_plate", "solo_mode": 999}, 400),
        ({"model": "steel_plate", "events": []}, 400),
        ({"model": "steel_plate",
          "events": [{"u": 0.5, "v": 0.5, "time_s": -1.0}]}, 400),
        ({"model": "steel_plate",
          "events": [{"u": 0.5, "v": 0.5, "direction": 0}]}, 400),
        ({"model": "steel_plate", "solo_event": 5,
          "events": [{"u": 0.5, "v": 0.5}]}, 400),
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
    resp = post(api, {"model": "aluminum_bar", "u": 0.3, "v": 0.5,
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


# ---------------------------------------------------------------------------
# Sustained multi-event refactor
# ---------------------------------------------------------------------------


def two_event_body(**overrides):
    body = {
        "model": "steel_plate",
        "duration_s": 0.8,
        "events": [
            {"u": 0.42, "v": 0.37, "time_s": 0.0, "velocity_m_s": 2.0, "label": "A"},
            {"u": 0.58, "v": 0.37, "time_s": 0.12, "velocity_m_s": 2.0, "label": "B"},
        ],
    }
    body.update(overrides)
    return body


def test_two_events_are_sample_timed(api):
    resp = post(api, two_event_body())
    assert [e["label"] for e in resp["events"]] == ["A", "B"]
    assert resp["events"][0]["applied_sample"] == 0
    assert resp["events"][1]["applied_sample"] == round(0.12 * 44100)
    assert abs(resp["events"][1]["time_quantization_s"]) < 1 / 44100


def test_second_hit_lands_on_still_ringing_body(api):
    """The second click's audio contains the first click's tail plus itself."""
    resp = post(api, two_event_body(include_isolated=True))
    mix, fs = decode_wav(resp["mix_audio_b64"])
    assert len(resp["isolated_audio_b64"]) == 2
    iso_a, _ = decode_wav(resp["isolated_audio_b64"][0])
    iso_b, _ = decode_wav(resp["isolated_audio_b64"][1])
    start = round(0.12 * fs)
    # Before B arrives, the mix is just A's tail (same normalization scale
    # aside: compare shapes after scaling each to its own peak).
    mix_pre = mix[:start]
    a_pre = iso_a[:start]
    assert np.corrcoef(mix_pre, a_pre)[0, 1] > 0.999
    # After B the mix is not simply B alone.
    mix_post = mix[start:]
    b_post = iso_b[: len(mix_post)]
    assert np.corrcoef(mix_post, b_post)[0, 1] < 0.995


def test_mix_and_audition_channels(api):
    """Without solo, the auditioned channel is the full mix; with solo it is not."""
    plain = post(api, two_event_body())
    assert plain["audio_b64"] == plain["mix_audio_b64"]
    solo = post(api, two_event_body(solo_event=0))
    assert solo["audio_b64"] != solo["mix_audio_b64"]


def test_replay_block_sizes_agree(api):
    resp = post(
        api,
        two_event_body(replay_block_sizes=[64, 256, 4096]),
    )
    comps = resp["replay"]["comparisons"]
    assert set(comps) == {"64", "256", "4096"}
    for size, c in comps.items():
        # Adjacent blocks are exact continuations: differences are rounding.
        assert c["max_rel_diff"] < 1e-9, size


def test_solo_event_auditions_one_hit(api):
    resp = post(api, two_event_body(solo_event=1))
    assert resp["solo_event"] == 1
    assert [e["label"] for e in resp["events"]] == ["A", "B"]
    # The auditioned audio is B alone, so it starts at B's scheduled time.
    pcm, fs = decode_wav(resp["audio_b64"])
    start = round(0.12 * fs)
    assert np.max(np.abs(pcm[: start - 1])) < 1e-3 * np.max(np.abs(pcm))


def test_direction_flip_negates_excitation(api):
    a = post(api, two_event_body())
    flipped = two_event_body(
        events=[
            {"u": 0.42, "v": 0.37, "time_s": 0.0, "velocity_m_s": 2.0,
             "direction": -1},
        ]
    )
    b = post(api, flipped)
    dv_a = [m["delta_v_m_s"] for m in a["events"][0]["modes"]]
    dv_b = [m["delta_v_m_s"] for m in b["events"][0]["modes"]]
    assert all(x == pytest.approx(-y, rel=1e-12) for x, y in zip(dv_a, dv_b))


def test_cancellation_report_flags_opposite_nodal_sides(api):
    resp = post(api, two_event_body())
    cancelled = resp["cancellation"]["cancelled"]
    reinforced = resp["cancellation"]["reinforced"]
    assert cancelled and reinforced
    assert not (set(cancelled) & set(reinforced))
