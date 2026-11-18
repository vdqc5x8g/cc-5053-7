"""Discrete-time modal oscillator: the signal solver.

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
