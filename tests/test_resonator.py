"""Solver tests: correctness of the discrete modal oscillator and its
stability over the whole supported parameter range.

The stability requirement: no parameter combination inside the supported
range (0 <= f < fs/2, tau > 0) may produce a diverging state. The exact
matrix-exponential integrator has poles at exp(-alpha T) <= 1 by
construction; these tests verify that property numerically, including
high-frequency modes near fs/2 and extreme damping.
"""

import math

import numpy as np
import pytest

from modal_synth.resonator import ModalOscillator
from spectral import fft_peaks, measure_decay_tau

FS = 44100.0


# ---------------------------------------------------------------------------
# Correctness: rendered signal matches the declared f and tau
# ---------------------------------------------------------------------------


def test_rendered_frequency_and_decay_match_parameters():
    f0, tau = 440.0, 0.5
    osc = ModalOscillator(f0, tau, FS)
    osc.impulse(1.0)
    out = osc.render(int(1.5 * FS))
    peaks = fft_peaks(out, FS, n_peaks=1)
    assert peaks[0][0] == pytest.approx(f0, rel=2e-3)
    measured_tau = measure_decay_tau(out, FS, f0, 0.1, 1.1, 0.2)
    assert measured_tau == pytest.approx(tau, rel=0.02)


def test_step_matches_render():
    """The per-sample recursion and the closed-form render must agree."""
    for f, tau in [(0.5, 1e-3), (440.0, 0.5), (5000.0, 0.01), (19000.0, 0.1)]:
        a = ModalOscillator(f, tau, FS)
        b = ModalOscillator(f, tau, FS)
        a.impulse(0.7)
        b.impulse(0.7)
        n = 4096
        stepped = [b.v] + [b.step() for _ in range(n - 1)]
        rendered = a.render(n)
        scale = max(1.0, float(np.max(np.abs(rendered))))
        assert np.max(np.abs(rendered - np.array(stepped))) < 1e-10 * scale


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


def test_pole_magnitudes_inside_unit_circle():
    for f in [0.0, 0.05, 1.0, 440.0, 5000.0, 19800.0]:
        for tau in [1e-4, 1e-3, 0.1, 100.0]:
            osc = ModalOscillator(f, tau, FS)
            r1, r2 = osc.pole_magnitudes()
            assert r1 <= 1.0 + 1e-15
            assert r2 <= 1.0 + 1e-15
            if osc.underdamped:
                expected = math.exp(-osc.alpha / FS)
                assert r1 == pytest.approx(expected, rel=1e-12)
                assert r2 == pytest.approx(expected, rel=1e-12)


def velocity_bound(osc: ModalOscillator, v0: float) -> float:
    """Theoretical |v(t)| envelope bound after a velocity impulse v0."""
    if osc.underdamped:
        return abs(v0) * (1.0 + osc.alpha / osc.wd)
    return abs(v0)


@pytest.mark.parametrize("f", [0.05, 1.0, 100.0, 1000.0, 5000.0, 15000.0, 19800.0])
@pytest.mark.parametrize("tau", [1e-4, 1e-3, 0.01, 0.1, 1.0, 100.0])
def test_stability_sweep_step_recursion(f, tau):
    """Run the actual recursion; state must stay finite and bounded."""
    osc = ModalOscillator(f, tau, FS)
    osc.impulse(1.0)
    bound = velocity_bound(osc, 1.0) * (1.0 + 1e-12)
    n = 20000
    for k in range(n):
        v = osc.step()
        assert math.isfinite(v), f"diverged at sample {k}: f={f}, tau={tau}"
        assert abs(v) <= bound, f"bound exceeded at sample {k}: f={f}, tau={tau}"
    assert math.isfinite(osc.q)


@pytest.mark.parametrize("f", [0.05, 440.0, 5000.0, 19800.0])
@pytest.mark.parametrize("tau", [1e-4, 0.01, 1.0, 100.0])
def test_stability_sweep_render(f, tau):
    osc = ModalOscillator(f, tau, FS)
    osc.impulse(1.0)
    out = osc.render(int(2.0 * FS))
    assert np.all(np.isfinite(out))
    bound = velocity_bound(osc, 1.0) * (1.0 + 1e-9)
    assert np.max(np.abs(out)) <= bound


def test_high_frequency_mode_does_not_diverge_long_run():
    """Mode at ~0.45 fs, light damping, long render: the classic divergence
    trap for naive discrete oscillators. Must stay bounded without any
    output clipping."""
    f = 0.45 * FS  # 19845 Hz
    osc = ModalOscillator(f, 50.0, FS)  # tau = 50 s: nearly undamped
    osc.impulse(1.0)
    out = osc.render(int(5.0 * FS))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-9
    # Nearly undamped: amplitude after 5 s still close to 1.
    tail = np.max(np.abs(out[-4410:]))
    assert tail > 0.8


def test_overdamped_mode_decays_without_oscillation():
    """tau so small that alpha > w0: overdamped branch, monotone decay."""
    osc = ModalOscillator(0.5, 1e-4, FS)  # alpha = 1e4 >> w0 = 3.14
    assert not osc.underdamped
    osc.impulse(1.0)
    out = osc.render(int(0.5 * FS))
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0 + 1e-12
    # Fast pole (tau = 1e-4 s) dies immediately; the slow pole's creep-back
    # residue is ~v0 * w0^2/(2 alpha^2) ~ 2.5e-8 here - correct physics.
    assert abs(out[-1]) < 1e-6


def test_state_decays_to_zero_no_limit_cycle():
    osc = ModalOscillator(1000.0, 0.05, FS)
    osc.impulse(1.0)
    osc.render(int(2.0 * FS))  # 40 tau
    assert abs(osc.v) < 1e-12
    assert abs(osc.q) < 1e-12


def test_silent_oscillator_stays_exactly_silent():
    osc = ModalOscillator(440.0, 1.0, FS)
    out = osc.render(10000)
    assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# ModalVoice: sustained bank, block rendering, sample-timed impulses
# ---------------------------------------------------------------------------


class _Mode:
    def __init__(self, f, tau):
        self.frequency_hz = f
        self.tau_s = tau


def test_voice_block_render_matches_single_oscillator():
    """The vectorized voice must equal the per-mode oscillator recursion."""
    from modal_synth.resonator import ModalVoice

    f, tau = 837.0, 0.4
    voice = ModalVoice([_Mode(f, tau)], FS)
    voice.schedule_impulse(0, 0, 1.0)
    out = voice.render_block(5000)

    osc = ModalOscillator(f, tau, FS)
    osc.impulse(1.0)
    ref = osc.render(5001)[:5000]
    assert np.max(np.abs(out - ref)) < 1e-12 * max(1.0, np.max(np.abs(ref)))


def test_voice_impulse_applied_at_requested_sample():
    from modal_synth.resonator import ModalVoice

    voice = ModalVoice([_Mode(1000.0, 1.0)], FS)
    applied = voice.schedule_impulse(100.0, 0, 1.0)
    assert applied == 100
    out = voice.render_block(200)
    assert np.all(out[:100] == 0.0)  # silent before the impulse
    assert np.any(out[100:] != 0.0)


def test_voice_block_size_invariance():
    from modal_synth.resonator import ModalVoice

    def run(block):
        voice = ModalVoice([_Mode(440.0, 1.0), _Mode(1770.0, 0.3)], FS)
        voice.schedule_impulse(0, 0, 1.0)
        voice.schedule_impulse(500.5, 1, -0.7)
        chunks = []
        left = 4000
        while left > 0:
            n = min(block, left)
            chunks.append(voice.render_block(n))
            left -= n
        return np.concatenate(chunks)

    ref = run(4000)
    for block in (1, 7, 64, 999):
        assert np.max(np.abs(run(block) - ref)) < 1e-11 * max(1.0, np.max(np.abs(ref)))


def test_voice_stems_match_total():
    from modal_synth.resonator import ModalVoice

    voice = ModalVoice([_Mode(300.0, 1.0), _Mode(900.0, 0.5)], FS)
    voice.schedule_impulse(0, 0, 1.0)
    voice.schedule_impulse(0, 1, 0.5)
    total, stems = voice.render_block(2000, stem_modes=[0, 1])
    assert np.max(np.abs((stems[0] + stems[1]) - total)) < 1e-12
