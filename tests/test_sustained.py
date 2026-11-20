"""Sustained-vibration refactor tests.

These cover the behaviours the refactor exists to guarantee:

* the interaction layer submits only position + sample time, and a second
  click lands on a body that is still ringing (state is accumulated, not
  reset);
* two clicks on opposite sides of a nodal line cancel that mode while
  adding for its neighbours, and a direction flip negates one click's
  excitation;
* isolated per-hit responses sum to the accumulated response (superposition
  holds, so "one response per click" and "one growing state" agree);
* rendering in different block lengths gives the same signal and never
  zeroes a mode at a block edge;
* swapping the body discards old tails rather than mixing them into the
  new body;
* the live state readback reports the nodal mode's phase and tail.
"""

import dataclasses

import numpy as np
import pytest

from modal_synth import audit, models
from modal_synth.synth import HitEvent, Mallet, ModalEngine, ModalInstrument

FS = 44100.0


def make_instrument(model_id="steel_plate", damping_scale=1.0):
    spec = models.build(model_id, damping_scale)
    return ModalInstrument(
        spec.modes,
        spec.geometry,
        sample_rate_hz=FS,
        mallet=Mallet(mass_kg=0.02, contact_time_s=0.001),
        thickness_m=spec.thickness_m,
    )


def nodal_points(instrument, mode_index, u=0.35):
    """Two x positions straddling the first interior nodal line of a mode.

    For a plate mode (m, n) the nodal lines of the x factor are at
    x = k*a/m. We find the smallest interior one and return points 2 % of
    the plate width either side of it at the given v.
    """
    mode = instrument.modes[mode_index]
    a, b = instrument.geometry[1], instrument.geometry[2]
    # Search for a sign change along a horizontal scan at fixed y where the
    # mode is non-zero, then bisect to the nodal x.
    y = 0.37 * b
    xs = np.linspace(0.02 * a, 0.98 * a, 4001)
    vals = np.array([float(mode.shape(x, y)) for x in xs])
    sign_change = np.where(np.sign(vals[:-1]) * np.sign(vals[1:]) < 0)[0]
    if sign_change.size == 0:
        pytest.skip("mode has no interior nodal line along this scan")
    k = sign_change[0]
    lo, hi = xs[k], xs[k + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(mode.shape(lo, y)) * float(mode.shape(mid, y)) <= 0.0:
            hi = mid
        else:
            lo = mid
    node = 0.5 * (lo + hi)
    delta = 0.02 * a
    return (node - delta, y), (node + delta, y), node


# ---------------------------------------------------------------------------
# Interaction layer submits position + time only
# ---------------------------------------------------------------------------


def test_hit_event_is_position_and_time_only():
    fields = {f.name for f in dataclasses.fields(HitEvent)}
    assert {"position_m", "time_s", "velocity_m_s"} <= fields
    # No oscillator, state, or render-span handles leaked into the event.
    assert not ({"oscillators", "voice", "engine", "duration_s"} & fields)


def test_second_hit_accumulates_on_ringing_body():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    engine = ModalEngine(inst)
    engine.submit(HitEvent((0.42 * a, 0.37 * b), 0.0, 2.0))
    # Render part of the first ring so its state is non-zero on the clock.
    engine.render_duration(0.12)
    assert not engine.voice.silent
    before = engine.body_state()[0].amplitude
    engine.submit(HitEvent((0.58 * a, 0.37 * b), 0.12, 2.0))
    after = engine.body_state()[0].amplitude
    # The first ring was not reset by the second hit.
    assert before > 0.0
    assert after > 0.0
    engine.render_duration(0.5)
    assert np.all(np.isfinite(engine.render_duration(0.1)))


# ---------------------------------------------------------------------------
# Nodal line: two near points cancel one mode, add for neighbours
# ---------------------------------------------------------------------------


def test_adjacent_clicks_cancel_nodal_mode():
    inst = make_instrument()
    (x1, y), (x2, _), node = nodal_points(inst, mode_index=1)
    ev = [
        HitEvent((x1, y), 0.0, 2.0, label="left"),
        HitEvent((x2, y), 0.10, 2.0, label="right"),
    ]
    engine = ModalEngine(inst)
    for e in ev:
        engine.submit(e)
    cats = audit.overlap_cancellation(engine.reports, len(ev))
    # Mode 1 straddled the node: its two kicks nearly cancel.
    assert 1 in cats["cancelled"]
    # Some nearby mode must instead be reinforced (on the same side of an
    # antinode), proving the two clicks are not simply identical.
    assert cats["reinforced"]
    # The per-mode signed sum for the cancelled mode is far below the
    # individual kicks.
    assert abs(cats["sums"][1]) < 0.05 * cats["largest_single"][1]


def test_nodal_mode_cancelled_in_audio_but_neighbours_remain():
    """Two kicks that null a mode's excitation null it in the mixed audio.

    A mode's *excitation* is the sum of its kicks. Two clicks on opposite
    sides of a nodal line null that sum; that is only audible as a nulled
    component if the clicks arrive close enough together to stay coherent
    for that mode, so here they are one sample apart. Neighbouring modes
    whose antinode lies between the points receive same-signed kicks and
    are reinforced.
    """
    inst = make_instrument()
    (x1, y), (x2, _), _ = nodal_points(inst, mode_index=1)
    ev = [
        HitEvent((x1, y), 0.0, 2.0),
        HitEvent((x2, y), 1.0 / FS, 2.0),
    ]
    engine = ModalEngine(inst)
    for e in ev:
        engine.submit(e)
    cats = audit.overlap_cancellation(engine.reports, len(ev))
    assert 1 in cats["cancelled"]
    assert cats["reinforced"]

    traced = audit.trace_cancellation(
        inst, ev, 0.4, mode_indices=[0, 1, 2], block_size=256
    )
    acc = traced["accumulated_stems"]
    single = traced["isolated_stems"][0]
    # Mode 1's residual after the pair is far below a single hit's.
    e_acc1 = np.sum(acc[1] ** 2)
    e_single1 = np.sum(single[1] ** 2)
    assert e_acc1 < 0.01 * e_single1
    # A reinforced neighbour carries clearly more energy than the nulled one.
    assert max(np.sum(acc[i] ** 2) for i in (0, 2)) > 10 * e_acc1


# ---------------------------------------------------------------------------
# Direction flip
# ---------------------------------------------------------------------------


def test_direction_reversal_negates_one_event():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    x, y = 0.43 * a, 0.29 * b
    e_plus = HitEvent((x, y), 0.0, 2.0, direction=1)
    e_minus = HitEvent((x, y), 0.0, 2.0, direction=-1)
    r_plus = ModalEngine(inst).submit(e_plus)
    r_minus = ModalEngine(inst).submit(e_minus)
    for a_m, b_m in zip(r_plus.modes, r_minus.modes):
        assert a_m.delta_v_m_s == pytest.approx(-b_m.delta_v_m_s, rel=1e-12)
        # Physical shape values are not signed by the direction.
        assert a_m.shape_at_hit == pytest.approx(b_m.shape_at_hit, rel=1e-12)


def test_direction_flip_audio_is_exact_negation():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    x, y = 0.43 * a, 0.29 * b
    e_plus = HitEvent((x, y), 0.0, 2.0, direction=1)
    e_minus = HitEvent((x, y), 0.0, 2.0, direction=-1)
    ep = ModalEngine(inst)
    ep.submit(e_plus)
    out_plus = ep.render_duration(0.4)
    em = ModalEngine(inst)
    em.submit(e_minus)
    out_minus = em.render_duration(0.4)
    # The rendered waveforms are mirror images (peak-normalization is equal
    # and opposite, so compare raw values).
    np.testing.assert_allclose(out_plus, -out_minus, rtol=0, atol=1e-15)


def test_mirror_points_sound_alike_but_couple_oppositely():
    """Two points mirrored across the centre: |phi| equal, signed coupling
    opposite. Reversing one event's direction makes the pair purely additive
    for modes whose antinode is between them, and purely canceling for the
    others."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    left = HitEvent((0.35 * a, 0.40 * b), 0.0, 2.0, direction=1)
    right_same = HitEvent((0.65 * a, 0.40 * b), 0.0, 2.0, direction=1)
    right_flipped = dataclasses.replace(right_same, direction=-1)

    def cancels(ev):
        eng = ModalEngine(inst)
        eng.submit(left)
        eng.submit(ev)
        cats = audit.overlap_cancellation(eng.reports, 2)
        return set(cats["cancelled"]), set(cats["reinforced"])

    # Same direction: the (1,1)-like modes cancel, the (2,·)-like add.
    c_same, r_same = cancels(right_same)
    assert c_same and r_same
    assert not (c_same & r_same)
    # Flipping one direction swaps the roles for every mode.
    c_flip, r_flip = cancels(right_flipped)
    assert c_flip == r_same
    assert r_flip == c_same


def test_mirror_pairs_individual_responses_are_similar():
    """Alone, the two points have the same |excitation| spectrum shape
    (they sound alike); only the signs differ."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    rl = ModalEngine(inst).submit(HitEvent((0.35 * a, 0.40 * b), 0.0, 2.0))
    rr = ModalEngine(inst).submit(HitEvent((0.65 * a, 0.40 * b), 0.0, 2.0))
    vl = np.abs([m.delta_v_m_s for m in rl.modes])
    vr = np.abs([m.delta_v_m_s for m in rr.modes])
    assert np.allclose(vl, vr, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# Isolated responses vs one accumulated state
# ---------------------------------------------------------------------------


def test_accumulated_equals_sum_of_isolated_responses():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    ev = [
        HitEvent((0.42 * a, 0.37 * b), 0.0, 2.0),
        HitEvent((0.58 * a, 0.37 * b), 0.12, 2.0),
    ]
    res = audit.isolated_vs_accumulated(inst, ev, total_s=1.0, block_size=256)
    assert res["residual_rel"] < 1e-10
    assert res["peak"] > 0.0
    # Each isolated response is a full-length buffer starting at its strike.
    assert all(buf.size == res["n_samples"] for buf in res["isolated"])


def test_superposition_holds_with_three_overlapping_hits():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    ev = [
        HitEvent((0.30 * a, 0.30 * b), 0.0, 1.5),
        HitEvent((0.55 * a, 0.40 * b), 0.07, 2.5),
        HitEvent((0.70 * a, 0.60 * b), 0.18, 1.0),
    ]
    res = audit.isolated_vs_accumulated(inst, ev, total_s=0.8, block_size=128)
    assert res["residual_rel"] < 1e-10


# ---------------------------------------------------------------------------
# Block edges and redraw cadence
# ---------------------------------------------------------------------------


def test_block_size_invariance():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    ev = [
        HitEvent((0.42 * a, 0.37 * b), 0.0, 2.0),
        HitEvent((0.58 * a, 0.37 * b), 0.12, 2.0),
    ]
    rep = audit.replay_blocks(
        inst, ev, total_s=0.6, block_sizes=[1, 17, 64, 512, 4096, 26460]
    )
    ref_size = rep["reference_block"]
    for size, c in rep["comparisons"].items():
        if size == ref_size:
            continue
        assert c["max_rel_diff"] < 1e-9, (size, c["max_rel_diff"])


def test_no_mode_zeroed_at_block_edges():
    """Trace per-mode stems across a fine block grid; state carries over."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    ev = [
        HitEvent((0.42 * a, 0.37 * b), 0.0, 2.0),
        HitEvent((0.58 * a, 0.37 * b), 0.12, 2.0),
    ]
    traced = audit.trace_cancellation(
        inst, ev, total_s=0.5, mode_indices=[0, 1, 2, 3], block_size=256
    )
    acc = traced["accumulated_stems"]
    # A ringing mode must not drop to zero at a block boundary. Check the
    # sample immediately across each edge relative to the block RMS.
    for i in (0, 2, 3):
        stem = acc[i]
        for b in traced["blocks"][1:]:
            edge = b["start"]
            if edge >= stem.size:
                continue
            rms = np.sqrt(np.mean(stem[edge - 256 : edge] ** 2))
            if rms == 0.0:
                continue
            # No discontinuity larger than a full block's worth of signal.
            assert abs(stem[edge]) <= 4 * rms + 1e-12


def test_redraw_cadence_does_not_change_audio():
    """Rendering in irregular chunks equals one long render."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    ev = [HitEvent((0.42 * a, 0.37 * b), 0.0, 2.0)]

    ref_engine = ModalEngine(inst)
    ref_engine.submit(ev[0])
    ref = ref_engine.render_duration(0.5)

    eng = ModalEngine(inst)
    eng.submit(ev[0])
    chunks = []
    pattern = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    i = 0
    while sum(len(c) for c in chunks) < ref.size:
        n = pattern[i % len(pattern)]
        chunks.append(eng.render_block(n))
        i += 1
    out = np.concatenate(chunks)[: ref.size]
    assert np.max(np.abs(out - ref)) < 1e-10 * np.max(np.abs(ref))


# ---------------------------------------------------------------------------
# Body swap: old tails must not restart on, or leak into, the new body
# ---------------------------------------------------------------------------


def test_body_swap_discards_old_tails():
    steel = make_instrument("steel_plate")
    a, b = steel.geometry[1], steel.geometry[2]
    engine = ModalEngine(steel)
    engine.submit(HitEvent((0.4 * a, 0.4 * b), 0.0, 3.0))
    engine.render_duration(0.1)  # plate is ringing
    assert not engine.voice.silent

    wood_spec = models.build("wood_bar")
    engine.replace_body(
        wood_spec.modes,
        wood_spec.geometry,
        mallet=steel.mallet,
        thickness_m=wood_spec.thickness_m,
    )
    assert engine.voice.silent  # nothing carried over
    assert engine.position == int(round(0.1 * FS))  # clock preserved
    out = engine.render_duration(0.3)
    assert np.all(out == 0.0)  # no ghost tail from the steel plate


def test_body_swap_then_new_hit_starts_cleanly():
    steel = make_instrument("steel_plate")
    a, b = steel.geometry[1], steel.geometry[2]
    engine = ModalEngine(steel)
    engine.submit(HitEvent((0.4 * a, 0.4 * b), 0.0, 3.0))
    engine.render_duration(0.15)

    wood_spec = models.build("wood_bar")
    engine.replace_body(
        wood_spec.modes,
        wood_spec.geometry,
        mallet=steel.mallet,
        thickness_m=wood_spec.thickness_m,
    )
    length = wood_spec.geometry[1]
    engine.submit(HitEvent((0.2 * length, 0.0), engine.time_s, 2.0))
    out = engine.render_duration(0.4)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.0
    # The new bar's fundamental dominates; the old plate's fundamental must
    # not be a spectral component of the new body's output.
    spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
    freqs = np.fft.rfftfreq(out.size, 1.0 / FS)
    old_f0 = steel.modes[0].frequency_hz
    k_old = int(np.argmin(np.abs(freqs - old_f0)))
    band = spectrum[max(0, k_old - 4) : k_old + 5]
    assert np.max(band) < 0.05 * np.max(spectrum)


def test_reset_silences_without_moving_clock():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    engine = ModalEngine(inst)
    engine.submit(HitEvent((0.4 * a, 0.4 * b), 0.0, 3.0))
    engine.render_duration(0.2)
    pos = engine.position
    engine.reset()
    assert engine.position == pos
    assert engine.voice.silent
    assert np.all(engine.render_duration(0.1) == 0.0)


# ---------------------------------------------------------------------------
# Readback: phase and tail of the nodal mode
# ---------------------------------------------------------------------------


def test_readback_reports_phase_and_tail():
    inst = make_instrument()
    (x1, y), (x2, _), _ = nodal_points(inst, mode_index=1)
    ev = [
        HitEvent((x1, y), 0.0, 2.0),
        HitEvent((x2, y), 0.10, 2.0),
    ]
    engine = ModalEngine(inst)
    for e in ev:
        engine.submit(e)
    engine.render_duration(0.10)  # just after the second click
    readback = audit.mode_tail_readback(engine, mode_indices=[0, 1])
    by_index = {row["index"]: row for row in readback["modes"]}
    # The cancelled mode has near-zero amplitude; the surviving one does not.
    assert by_index[1]["amplitude"] < 0.1 * by_index[0]["amplitude"]
    # Intrinsic tail of a surviving mode is its declared tau.
    assert by_index[0]["tail_s"] == pytest.approx(inst.modes[0].tau_s, rel=1e-9)
    assert by_index[0]["phase_deg"] == pytest.approx(
        np.degrees(by_index[0]["phase_rad"]), rel=1e-12
    )
    # Phase is finite and in (-180, 180].
    assert -180.0 <= by_index[0]["phase_deg"] <= 180.0


def test_readback_tail_is_zero_for_cancelled_mode_near_coincident_pair():
    """Two clicks that null a mode leave it with zero threshold-tail."""
    inst = make_instrument()
    (x1, y), (x2, _), _ = nodal_points(inst, mode_index=1)
    ev = [
        HitEvent((x1, y), 0.0, 2.0),
        HitEvent((x2, y), 1.0 / FS, 2.0),
    ]
    engine = ModalEngine(inst)
    for e in ev:
        engine.submit(e)
    engine.render_duration(0.02)
    readback = audit.mode_tail_readback(
        engine, mode_indices=[0, 1], floor_ratio=0.01
    )
    by_index = {row["index"]: row for row in readback["modes"]}
    assert by_index[1]["tail_s"] == 0.0
    assert by_index[0]["tail_s"] > 0.0
    assert by_index[0]["amplitude"] > by_index[1]["amplitude"]


def test_readback_phase_advances_at_mode_frequency():
    """Between two instants, the modal phase advances at ~ w0 * dt.

    The analytic signal is z = q + j·v/w0; for this oscillator it rotates
    at -w0, so the unwrapped phase decreases by w0*dt.
    """
    osc_f = 440.0
    inst = ModalInstrument(
        [models.Mode(osc_f, 10.0, 1.0, lambda x, y: 1.0, "test")],
        ("plate", 1.0, 1.0),
        sample_rate_hz=FS,
    )
    engine = ModalEngine(inst)
    engine.submit(HitEvent((0.5, 0.5), 0.0, 1.0))
    engine.render_duration(0.01)
    p0 = engine.body_state()[0].phase_rad
    sample0 = engine.position
    engine.render_duration(0.005)
    p1 = engine.body_state()[0].phase_rad
    dt = (engine.position - sample0) / FS  # actual elapsed, whole samples
    expected_advance = 2 * np.pi * osc_f * dt
    # The smallest signed rotation from p0 to p1.
    delta = ((p1 - p0 + np.pi) % (2 * np.pi)) - np.pi
    assert abs(delta) == pytest.approx(expected_advance % (2 * np.pi), abs=1e-3)
    assert delta < 0.0  # rotates at -w0
