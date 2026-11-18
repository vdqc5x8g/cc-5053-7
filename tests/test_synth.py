"""Instrument tests: hit mapping, velocity behavior, damping, spectra.

These verify the actual requirements: strike position changes *which*
modes are excited (not just loudness), strike velocity changes the
spectrum through the mallet model, high damping stays stable and matches
its tau, and the solver output needs no clipping.
"""

import numpy as np
import pytest

from modal_synth import models
from modal_synth.audio import to_wav_bytes
from modal_synth.synth import Mallet, ModalInstrument
from spectral import fft_peaks, measure_decay_tau, spectral_centroid, band_energy

FS = 44100.0


def make_instrument(model_id="steel_plate", damping_scale=1.0, mallet=None):
    spec = models.build(model_id, damping_scale)
    return ModalInstrument(
        spec.modes,
        spec.geometry,
        sample_rate_hz=FS,
        mallet=mallet or Mallet(mass_kg=0.02, contact_time_s=0.001),
        thickness_m=spec.thickness_m,
    )


# ---------------------------------------------------------------------------
# Hit mapping: position selects modes, not just volume
# ---------------------------------------------------------------------------


def test_center_vs_edge_fundamental():
    """Mode (1,1) peaks at the plate center and nearly vanishes at a corner."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    center = inst.strike(0.5 * a, 0.5 * b, 2.0)
    inst.reset()
    corner = inst.strike(0.05 * a, 0.05 * b, 2.0)
    dv_center = abs(center.modes[0].delta_v_m_s)
    dv_corner = abs(corner.modes[0].delta_v_m_s)
    # phi(1,1) at corner = sin(0.05 pi)^2 ~ 0.0246 of center.
    assert dv_center / dv_corner == pytest.approx(
        1.0 / (np.sin(0.05 * np.pi) ** 2), rel=1e-6
    )
    assert dv_center > 20 * dv_corner


def test_edge_hit_excites_more_high_modes():
    """Spectral centroid of the rendered sound must rise for an edge hit."""
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.5 * a, 0.5 * b, 2.0)
    out_center = inst.render(1.0)
    inst.reset()
    inst.strike(0.06 * a, 0.07 * b, 2.0)
    out_edge = inst.render(1.0)
    c_center = spectral_centroid(out_center, FS)
    c_edge = spectral_centroid(out_edge, FS)
    assert c_edge > 1.5 * c_center


def test_hit_mapping_matches_mode_shape_values():
    """delta_v in the report must equal J * W * phi(p) / M exactly."""
    mallet = Mallet(mass_kg=0.02, contact_time_s=0.001)
    inst = make_instrument(mallet=mallet)
    a, b = inst.geometry[1], inst.geometry[2]
    x, y, v = 0.37 * a, 0.41 * b, 2.5
    report = inst.strike(x, y, v)
    j = mallet.impulse_n_s(v)
    for em, mode in zip(report.modes, inst.modes):
        phi = float(mode.shape(x, y))
        w = mallet.weight(mode.frequency_hz, v)
        assert em.shape_at_hit == pytest.approx(phi, rel=1e-12)
        assert em.delta_v_m_s == pytest.approx(j * w * phi / mode.modal_mass_kg, rel=1e-12)


def test_bar_hit_position_selects_modes():
    """Free-free bar mode 2 has a node at mid-length: silent for center hits."""
    inst = make_instrument("wood_bar")
    length = inst.geometry[1]
    center = inst.strike(0.5 * length, 0.0, 2.0)
    inst.reset()
    near_end = inst.strike(0.05 * length, 0.0, 2.0)
    dv2_center = abs(center.modes[1].delta_v_m_s)
    dv2_end = abs(near_end.modes[1].delta_v_m_s)
    assert dv2_center < 0.02 * dv2_end
    # Free-free fundamental: antinodes at the bar ends, smaller at center.
    assert abs(near_end.modes[0].delta_v_m_s) > abs(center.modes[0].delta_v_m_s)


# ---------------------------------------------------------------------------
# Velocity: linear scaling and spectral effect of the mallet
# ---------------------------------------------------------------------------


def test_velocity_linear_scaling_with_rigid_mallet():
    """contact_time_s = 0 disables the mallet filter: exact linearity."""
    inst = make_instrument(mallet=Mallet(mass_kg=0.02, contact_time_s=0.0))
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.4 * a, 0.35 * b, 1.0)
    out1 = inst.render(0.5)
    inst.reset()
    inst.strike(0.4 * a, 0.35 * b, 2.0)
    out2 = inst.render(0.5)
    ratio = np.max(np.abs(out2)) / np.max(np.abs(out1))
    assert ratio == pytest.approx(2.0, rel=1e-9)


def test_harder_strike_shifts_spectrum_up():
    """Hertz contact: harder hit -> shorter contact -> relatively more
    high-mode energy in the rendered sound.

    The observable is the high-band / fundamental-band energy ratio: the
    overall centroid barely moves because the fundamental always dominates
    total energy, but the *relative* band energies must shift clearly.
    """
    inst = make_instrument(mallet=Mallet(mass_kg=0.02, contact_time_s=0.002))
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.4 * a, 0.35 * b, 0.2)
    soft = inst.render(1.0)
    inst.reset()
    inst.strike(0.4 * a, 0.35 * b, 8.0)
    hard = inst.render(1.0)
    f0 = inst.modes[0].frequency_hz

    def hi_lo_ratio(out):
        lo = band_energy(out, FS, 0.9 * f0, 1.1 * f0)
        hi = band_energy(out, FS, 700.0, 4000.0)
        return hi / lo

    assert hi_lo_ratio(hard) > 2.0 * hi_lo_ratio(soft)
    assert spectral_centroid(hard, FS) > spectral_centroid(soft, FS)


def test_mallet_model_hertz_scaling():
    """Contact time scales as v^(-1/5); weight is a low-pass in f that
    opens up with velocity."""
    mallet = Mallet(mass_kg=0.02, contact_time_s=0.002)
    assert mallet.contact_time_at(8.0) == pytest.approx(
        0.002 * 8.0 ** (-0.2), rel=1e-12
    )
    for v in (0.5, 4.0):
        assert mallet.weight(100.0, v) > mallet.weight(5000.0, v)
    assert mallet.weight(5000.0, 8.0) > mallet.weight(5000.0, 0.2)
    assert mallet.impulse_n_s(3.0) == pytest.approx(0.06, rel=1e-12)


# ---------------------------------------------------------------------------
# Spectral peaks and decay against declared parameters
# ---------------------------------------------------------------------------


def test_rendered_spectrum_matches_declared_mode_frequencies():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.4 * a, 0.35 * b, 2.0)  # generic point: no low-mode nodes
    out = inst.render(2.0)
    peaks = fft_peaks(out, FS, n_peaks=12)
    peak_freqs = [p[0] for p in peaks]
    for mode in inst.modes[:6]:
        assert any(
            abs(pf - mode.frequency_hz) < max(0.005 * mode.frequency_hz, 0.75)
            for pf in peak_freqs
        ), f"mode {mode.label} at {mode.frequency_hz:.1f} Hz missing from spectrum"


def test_decay_time_matches_tau():
    """Aluminum bar, solo fundamental: measured envelope decay equals tau."""
    inst = make_instrument("aluminum_bar")
    length = inst.geometry[1]
    inst.strike(0.3 * length, 0.0, 2.0, solo_mode=0)
    out = inst.render(2.0)
    tau_expected = inst.modes[0].tau_s
    measured = measure_decay_tau(out, FS, inst.modes[0].frequency_hz, 0.2, 1.4, 0.3)
    assert measured == pytest.approx(tau_expected, rel=0.03)


def test_high_damping_stable_and_correct():
    """25x damping: tau/25, still stable, no divergence, correct decay."""
    inst = make_instrument("aluminum_bar", damping_scale=25.0)
    length = inst.geometry[1]
    inst.strike(0.3 * length, 0.0, 3.0, solo_mode=0)
    out = inst.render(1.0)
    assert np.all(np.isfinite(out))
    tau_expected = inst.modes[0].tau_s
    assert tau_expected < 0.1  # genuinely fast decay being measured
    measured = measure_decay_tau(out, FS, inst.modes[0].frequency_hz, 0.03, 0.25, 0.05)
    assert measured == pytest.approx(tau_expected, rel=0.05)
    # After 20 tau essentially nothing remains (no residue, no limit cycle).
    tail = np.max(np.abs(out[int(0.8 * FS) :]))
    assert tail < 1e-4 * np.max(np.abs(out))


# ---------------------------------------------------------------------------
# Output integrity: bounded solver output, normalization without clipping
# ---------------------------------------------------------------------------


def test_output_bounded_by_theoretical_envelope():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    report = inst.strike(0.5 * a, 0.5 * b, 8.0)  # hardest supported hit
    out = inst.render(2.0)
    bound = sum(
        abs(em.delta_v_m_s) * (1.0 + osc.alpha / osc.wd)
        for em, osc in zip(report.modes, inst.oscillators)
        if osc.underdamped
    )
    assert np.max(np.abs(out)) <= bound * (1.0 + 1e-9)


def test_wav_normalization_never_clips():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.5 * a, 0.5 * b, 8.0)
    out = inst.render(1.0)
    peak_raw = float(np.max(np.abs(out)))
    wav, peak_before, gain = to_wav_bytes(out, int(FS))
    assert peak_before == peak_raw
    pcm = np.frombuffer(wav[44:], dtype=np.int16).astype(np.float64) / 32767.0
    assert np.max(np.abs(pcm)) <= 0.90
    assert np.max(np.abs(pcm)) == pytest.approx(0.89, abs=1e-3)


def test_solo_mode_isolates_single_mode():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    inst.strike(0.4 * a, 0.35 * b, 2.0, solo_mode=2)
    out = inst.render(1.5)
    f2 = inst.modes[2].frequency_hz
    f0 = inst.modes[0].frequency_hz
    e_solo = band_energy(out, FS, 0.9 * f2, 1.1 * f2)
    e_other = band_energy(out, FS, 0.9 * f0, 1.1 * f0)
    assert e_solo > 1e4 * e_other


def test_warning_when_linear_range_exceeded():
    inst = make_instrument()
    a, b = inst.geometry[1], inst.geometry[2]
    hard = inst.strike(0.5 * a, 0.5 * b, 8.0)
    # 2 mm plate, 8 m/s hardest hit: the honesty warning must fire.
    assert any("nonlinear" in w for w in hard.warnings)
    inst.reset()
    soft = inst.strike(0.5 * a, 0.5 * b, 0.5)
    assert not soft.warnings
