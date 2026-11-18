"""Spectral measurement helpers shared by tests (numpy only)."""

from __future__ import annotations

import numpy as np


def fft_peaks(samples, fs, f_min=20.0, n_peaks=8, min_prominence_ratio=0.02):
    """Return [(freq_hz, magnitude), ...] of the strongest spectral peaks.

    Peak frequencies are refined by parabolic interpolation on the
    magnitude spectrum, giving sub-bin accuracy.
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    lo = np.searchsorted(freqs, f_min)
    mag = spec.copy()
    mag[:lo] = 0.0
    if mag.max() == 0.0:
        return []
    thresh = min_prominence_ratio * mag.max()
    peaks = []
    used = np.zeros_like(mag, dtype=bool)
    for _ in range(n_peaks):
        idx = int(np.argmax(np.where(used, 0.0, mag)))
        if mag[idx] < thresh or idx == 0 or idx >= mag.size - 1:
            break
        # Parabolic interpolation around idx.
        a, b, c = mag[idx - 1], mag[idx], mag[idx + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
        f = (idx + delta) * (freqs[1] - freqs[0])
        peaks.append((f, float(b)))
        # Exclude a +-2% frequency neighborhood so the next search finds
        # another mode, not a sidelobe of this one.
        half_width = max(3, int(0.02 * idx))
        used[max(0, idx - half_width) : idx + half_width + 1] = True
    return peaks


def measure_decay_tau(samples, fs, freq_hz, t1, t2, window_s):
    """Decay time constant [s] from windowed FFT magnitudes at t1 and t2.

    A(t) ~ exp(-t/tau)  =>  tau = (t2 - t1) / ln(A1 / A2).
    Window centers define the measurement times.
    """
    x = np.asarray(samples, dtype=np.float64)
    w = int(window_s * fs)

    def mag_at(t):
        i0 = int(t * fs) - w // 2
        seg = x[i0 : i0 + w]
        if seg.size < w:
            raise ValueError("window outside signal")
        spec = np.abs(np.fft.rfft(seg * np.hanning(w)))
        freqs = np.fft.rfftfreq(w, 1.0 / fs)
        k = int(np.argmin(np.abs(freqs - freq_hz)))
        lo, hi = max(0, k - 3), min(spec.size, k + 4)
        return float(np.max(spec[lo:hi]))

    a1 = mag_at(t1)
    a2 = mag_at(t2)
    if a1 <= 0.0 or a2 <= 0.0 or a2 >= a1:
        raise AssertionError(f"no measurable decay: A1={a1}, A2={a2}")
    return (t2 - t1) / np.log(a1 / a2)


def spectral_centroid(samples, fs, f_min=20.0):
    """Magnitude-weighted mean frequency [Hz]."""
    x = np.asarray(samples, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    mask = freqs >= f_min
    num = float(np.sum(freqs[mask] * spec[mask]))
    den = float(np.sum(spec[mask]))
    return num / den if den > 0 else 0.0


def band_energy(samples, fs, f_lo, f_hi):
    """Energy of the spectrum within [f_lo, f_hi] Hz."""
    x = np.asarray(samples, dtype=np.float64)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sum(spec[mask]))
