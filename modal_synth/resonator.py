"""Discrete-time modal oscillator: the signal solver.

Two classes:

* ``ModalOscillator`` - one damped mode as an exactly integrated 2-state
  system (the numerical core).
* ``ModalVoice`` - a bank of oscillators with a sample-timed impulse
  schedule. This is the sustained-vibration state: it advances in blocks,
  injects kicks at absolute sample indices, and never resets at block
  boundaries. Rendering the same session in different block sizes yields
  the same output (up to floating-point rounding) because the state is
  carried across every call and only impulses change it.

Each mode obeys  q'' + 2 alpha q' + w0^2 q = 0  with
  w0    = 2 pi f_n        [rad/s]
  alpha = 1 / tau_n       [1/s]

The state [q, v] is advanced by the exact matrix exponential of the
continuous system over one sample period T:

    [q; v]_{k+1} = M [q; v]_k,   M = expm(A T),   A = [[0, 1], [-w0^2, -2 alpha]]

Underdamped (alpha < w0), wd = sqrt(w0^2 - alpha^2), theta = wd T:

    M = e^{-alpha T} [[cos t + (alpha/wd) sin t,   sin t / wd        ],
                      [-(w0^2/wd) sin t,            cos t - (alpha/wd) sin t]]

Overdamped (alpha > w0), gamma = sqrt(alpha^2 - w0^2): same with
cos -> cosh, sin -> sinh, wd -> gamma.

Stability
---------
The eigenvalues of M are exp((-alpha +/- j wd) T) (underdamped) or
exp((-alpha +/- gamma) T) (overdamped). Their magnitudes are
exp(-alpha T) <= 1 for every alpha >= 0, at every sample rate and every
mode frequency. The recursion is therefore stable by construction over
the whole supported parameter range; there is no regime in which a
high-frequency mode can diverge and no output clipping is needed or used.
This is an exact integrator, not a forward-Euler or impulse-invariant
approximation, so the discrete pole frequencies and decay rates equal the
continuous ones to machine precision.

`render()` evaluates the same recursion in closed form (the matrix power
M^k applied analytically), which is mathematically identical to calling
`step()` k times; a test verifies they agree to floating-point accuracy.
"""

from __future__ import annotations

import math

import numpy as np


def _sin_over(w: float, t: np.ndarray) -> np.ndarray:
    """sin(w t) / w, series-safe as w -> 0."""
    if abs(w) < 1e-9:
        wt = w * t
        return t * (1.0 - wt * wt / 6.0)
    return np.sin(w * t) / w


def _sin_over_arr(w: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Elementwise sin(w_i t_j)/w_i for column vector w, column vector t."""
    small = np.abs(w) < 1e-9
    safe_w = np.where(small, 1.0, w)
    out = np.sin(safe_w * t) / safe_w
    if np.any(small):
        wt = w[small] * t
        out[:, small] = t * (1.0 - wt * wt / 6.0)
    return out


def _overdamped_terms(alpha: float, gamma: float, t: np.ndarray):
    """cosh/sinh terms for the overdamped solution as decaying exponentials.

    Direct evaluation of exp(-alpha t) cosh(gamma t) overflows for large
    gamma t (0 * inf -> NaN). Factored as
        exp(-alpha t) cosh(gamma t) = (ep + em) / 2
        exp(-alpha t) sinh(gamma t) = (ep - em) / 2
    with ep = exp((gamma - alpha) t), em = exp(-(gamma + alpha) t), every
    exponent is negative, so nothing can overflow.

    Returns (cosh_term, sinh_over_gamma) where sinh_over_gamma is
    exp(-alpha t) sinh(gamma t) / gamma, series-safe as gamma -> 0.
    """
    ep = np.exp((gamma - alpha) * t)
    em = np.exp(-(gamma + alpha) * t)
    cosh_term = 0.5 * (ep + em)
    if abs(gamma) < 1e-9:
        gt = gamma * t
        sinh_over = np.exp(-alpha * t) * t * (1.0 + gt * gt / 6.0)
    else:
        sinh_over = 0.5 * (ep - em) / gamma
    return cosh_term, sinh_over


class ModalOscillator:
    """One damped mode as an exactly integrated 2-state system."""

    def __init__(self, frequency_hz: float, tau_s: float, sample_rate_hz: float):
        if tau_s <= 0.0:
            raise ValueError("tau_s must be positive")
        if frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative")
        self.frequency_hz = float(frequency_hz)
        self.tau_s = float(tau_s)
        self.sample_rate_hz = float(sample_rate_hz)
        self.w0 = 2.0 * math.pi * self.frequency_hz
        self.alpha = 1.0 / self.tau_s
        self.q = 0.0  # modal displacement [m]
        self.v = 0.0  # modal velocity [m/s]

        t = 1.0 / self.sample_rate_hz
        self.underdamped = self.alpha < self.w0
        if self.underdamped:
            self.wd = math.sqrt(self.w0 * self.w0 - self.alpha * self.alpha)
        else:
            self.wd = math.sqrt(self.alpha * self.alpha - self.w0 * self.w0)
        self._m = self._state_matrix(t)

    def _state_matrix(self, t: float) -> tuple[float, float, float, float]:
        """Exact expm(A t) for the oscillator, as a flat 2x2 tuple."""
        ts = np.asarray(t)
        if self.underdamped:
            env = math.exp(-self.alpha * t)
            c = math.cos(self.wd * t)
            so = _sin_over(self.wd, ts).item()
            m11 = env * (c + self.alpha * so)
            m12 = env * so
            m21 = -env * self.w0 * self.w0 * so
            m22 = env * (c - self.alpha * so)
        else:
            cosh_term, sinh_over = _overdamped_terms(self.alpha, self.wd, ts)
            m11 = cosh_term + self.alpha * sinh_over
            m12 = sinh_over
            m21 = -self.w0 * self.w0 * sinh_over
            m22 = cosh_term - self.alpha * sinh_over
        return (float(m11), float(m12), float(m21), float(m22))

    def impulse(self, delta_v_m_s: float) -> None:
        """Apply a momentum kick: v += delta_v (impulse / modal mass)."""
        self.v += float(delta_v_m_s)

    def step(self) -> float:
        """Advance one sample; returns the current modal velocity [m/s]."""
        m11, m12, m21, m22 = self._m
        q, v = self.q, self.v
        self.q = m11 * q + m12 * v
        self.v = m21 * q + m22 * v
        return self.v

    def render(self, n_samples: int) -> np.ndarray:
        """Exact modal velocity for n samples, then advance the state.

        Uses the closed-form matrix power M^k on the current state; identical
        to n successive step() calls up to floating-point rounding.
        """
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float64)
        k = np.arange(n_samples, dtype=np.float64)
        t = k / self.sample_rate_hz
        q0, v0 = self.q, self.v
        if self.underdamped:
            env = np.exp(-self.alpha * t)
            c = np.cos(self.wd * t)
            so = _sin_over(self.wd, t)
            q = env * (q0 * (c + self.alpha * so) + v0 * so)
            v = env * (v0 * (c - self.alpha * so) - q0 * self.w0 * self.w0 * so)
        else:
            cosh_term, sinh_over = _overdamped_terms(self.alpha, self.wd, t)
            q = q0 * (cosh_term + self.alpha * sinh_over) + v0 * sinh_over
            v = (
                v0 * (cosh_term - self.alpha * sinh_over)
                - q0 * self.w0 * self.w0 * sinh_over
            )
        self.q = float(q[-1])
        self.v = float(v[-1])
        return v

    def pole_magnitudes(self) -> tuple[float, float]:
        """|poles| of the discrete recursion; must be <= 1 for stability."""
        t = 1.0 / self.sample_rate_hz
        if self.underdamped:
            r = math.exp(-self.alpha * t)
            return (r, r)
        return (
            math.exp((-self.alpha + self.wd) * t),
            math.exp((-self.alpha - self.wd) * t),
        )


class ModalVoice:
    """A bank of modal oscillators with a sample-timed impulse schedule.

    A ``ModalVoice`` is the sustained vibration state of one body. It knows
    nothing about where the surface was touched; the interaction layer
    submits kicks already resolved to per-mode velocity increments at a
    (fractional) sample time, and this class applies them to the running
    state.

    Time is measured in samples since the voice was created. Rendering is
    blockwise: ``render_block(n)`` advances every mode by exactly n samples
    and returns the summed modal velocity. Because the oscillator state is
    never reset between calls, adjacent blocks are sample-accurate
    continuations - a block edge is not a state boundary. Two renderings
    of the same voice with different block lengths therefore agree up to
    floating-point rounding (the closed-form M^k is evaluated per block).

    Impulses scheduled with a fractional sample position are applied at the
    next integer sample; the requested time is reported back so callers can
    quantify that quantization (<= 1 sample, i.e. ~23 us at 44.1 kHz).
    """

    def __init__(self, modes, sample_rate_hz: float):
        self.modes = list(modes)
        self.sample_rate_hz = float(sample_rate_hz)
        self.oscillators = [
            ModalOscillator(m.frequency_hz, m.tau_s, self.sample_rate_hz)
            for m in self.modes
        ]
        self.position = 0  # samples elapsed since creation
        self._pending: list[tuple[int, int, float]] = []  # (sample, mode, dv)
        self._params = self._vector_params()

    def _vector_params(self) -> dict:
        """Per-mode constants as arrays for vectorized block rendering."""
        n = len(self.oscillators)
        alpha = np.array([o.alpha for o in self.oscillators], dtype=np.float64)
        w0 = np.array([o.w0 for o in self.oscillators], dtype=np.float64)
        under = np.array([o.underdamped for o in self.oscillators], dtype=bool)
        wd = np.array([o.wd if o.underdamped else 0.0 for o in self.oscillators])
        gamma = np.array([0.0 if o.underdamped else o.wd for o in self.oscillators])
        return {
            "alpha": alpha, "w0": w0, "under": under, "wd": wd, "gamma": gamma,
            "under_idx": np.flatnonzero(under),
            "over_idx": np.flatnonzero(~under),
            "n": n,
        }

    # -- scheduling -----------------------------------------------------

    def schedule_impulse(self, at_sample: float, mode_index: int, delta_v: float) -> int:
        """Queue a modal velocity kick for the given absolute sample.

        Returns the integer sample at which it will actually be applied.
        A time already in the past is clamped to the current position: the
        scheduling layer (not the voice) decides how to handle late
        events, and the voice reports the applied sample so it stays
        auditable.
        """
        sample = max(self.position, int(math.ceil(at_sample - 1e-12)))
        self._pending.append((sample, int(mode_index), float(delta_v)))
        return sample

    def _take_through(self, end_sample: int) -> list[tuple[int, float]]:
        """Remove and return (mode, dv) impulses with sample index < end_sample."""
        if not self._pending:
            return []
        due = [p for p in self._pending if p[0] < end_sample]
        if not due:
            return []
        self._pending = [p for p in self._pending if p[0] >= end_sample]
        due.sort(key=lambda p: p[0])
        return [(mode_index, dv) for _, mode_index, dv in due]

    # -- rendering ------------------------------------------------------

    _MAX_CHUNK = 8192

    def _advance(self, length: int, q0: np.ndarray, v0: np.ndarray):
        """Advance all modes by `length` samples from (q0, v0).

        Returns (velocity_rows, q_end, v_end) where velocity_rows has shape
        (length, n_modes): row j is the velocity at sample offset j within
        the span (t = j/fs), and (q_end, v_end) is the state at t =
        length/fs (one sample beyond the last emitted row), so consecutive
        spans join without a one-sample gap.
        """
        p = self._params
        alpha, w0 = p["alpha"], p["w0"]
        t = np.arange(length + 1, dtype=np.float64)[:, None] / self.sample_rate_hz
        a_term = np.empty((length + 1, p["n"]), dtype=np.float64)
        b_term = np.empty((length + 1, p["n"]), dtype=np.float64)
        ui, oi = p["under_idx"], p["over_idx"]
        if ui.size:
            wd = p["wd"][ui]
            al = alpha[ui]
            env = np.exp(-al * t)
            a_term[:, ui] = env * np.cos(wd * t)
            b_term[:, ui] = env * _sin_over_arr(wd, t)
        if oi.size:
            gamma = p["gamma"][oi]
            al = alpha[oi]
            ep = np.exp((gamma - al) * t)
            em = np.exp(-(gamma + al) * t)
            a_term[:, oi] = 0.5 * (ep + em)
            # sinh(gamma t)/gamma, series-safe as gamma -> 0 (alpha == w0).
            tiny = gamma < 1e-9
            safe_g = np.where(tiny, 1.0, gamma)
            b = 0.5 * (ep - em) / safe_g
            if np.any(tiny):
                gt = gamma[tiny] * t
                b[:, tiny] = t * (1.0 + gt * gt / 6.0)
            b_term[:, oi] = b
        q = q0 * (a_term + alpha * b_term) + v0 * b_term
        v = v0 * (a_term - alpha * b_term) - q0 * (w0 * w0) * b_term
        return v[:length], q[-1], v[-1]

    def render_block(self, n_samples: int, stem_modes=None):
        """Advance n samples, applying scheduled impulses at their times.

        The returned array is the sum of modal velocities for the samples
        [position, position + n_samples). State is carried exactly across
        calls; nothing is zeroed at the boundary.

        If ``stem_modes`` is given, returns (sum, stems) where stems maps
        each requested mode index to that mode's velocity over the block.
        Modes not requested are not materialized.
        """
        n = int(n_samples)
        if n <= 0:
            empty = np.zeros(0, dtype=np.float64)
            if stem_modes is None:
                return empty
            return empty, {i: empty.copy() for i in stem_modes}
        start = self.position
        end = start + n
        out = np.zeros(n, dtype=np.float64)
        stems = (
            {i: np.zeros(n, dtype=np.float64) for i in stem_modes}
            if stem_modes is not None
            else None
        )
        q = np.array([o.q for o in self.oscillators], dtype=np.float64)
        v = np.array([o.v for o in self.oscillators], dtype=np.float64)

        # Split the block at impulse times so each segment starts from the
        # exact state at that sample. Splitting a block is numerically
        # identical to rendering it whole (M^a M^b = M^(a+b)).
        cuts = sorted({s for s, _, _ in self._pending if start < s < end})
        segment_start = start
        for cut in cuts + [end]:
            for mode_index, dv in self._take_through(cut):
                v[mode_index] += dv
            length = cut - segment_start
            offset = segment_start - start
            while length > 0:
                step = min(self._MAX_CHUNK, length)
                vel, q, v = self._advance(step, q, v)
                out[offset : offset + step] += vel.sum(axis=1)
                if stems is not None:
                    for i in stems:
                        stems[i][offset : offset + step] += vel[:, i]
                offset += step
                length -= step
            segment_start = cut
        self.position = end
        for i, osc in enumerate(self.oscillators):
            osc.q = float(q[i])
            osc.v = float(v[i])
        return (out, stems) if stems is not None else out

    def render_block_stems(self, n_samples: int, stem_modes) -> tuple[np.ndarray, dict]:
        """render_block with per-mode stems (see render_block)."""
        return self.render_block(n_samples, stem_modes=stem_modes)

    # -- inspection -----------------------------------------------------

    @property
    def silent(self) -> bool:
        return not self._pending and all(
            osc.q == 0.0 and osc.v == 0.0 for osc in self.oscillators
        )

    def max_pole_magnitude(self) -> float:
        """Largest |pole| over all modes; <= 1 means the voice is stable."""
        return max((max(osc.pole_magnitudes()) for osc in self.oscillators), default=0.0)

