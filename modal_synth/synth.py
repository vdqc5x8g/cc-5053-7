"""Modal instrument: surface hit mapping + sound rendering.

Hit mapping (position and velocity -> per-mode excitation)
----------------------------------------------------------
A strike at surface point p with mallet velocity v applies an impulse
J = m_mallet * v [N s] (fully transferred momentum; contact time much
shorter than every modal period - declared simplification). Mode n with
mode shape phi_n and modal mass M_n receives the modal velocity kick

    delta_v_n = J * W(f_n, v) * phi_n(p) / M_n      [m/s]

so the hit position changes *which* modes are excited (through phi_n(p)),
not just the overall loudness. A strike on a nodal line of mode n leaves
that mode silent regardless of force.

Mallet model (declared approximation)
-------------------------------------
Real mallets act as a low-pass whose cutoff rises with strike velocity
(Hertzian contact: contact time scales as v^(-1/5)). We model the
velocity-dependent impulse spectrum as

    t_c(v) = t_c0 * (v / 1 m/s)^(-1/5)
    W(f, v) = 1 / (1 + (pi f t_c(v))^2)

so harder strikes excite relatively more high-frequency modes - velocity
changes the spectrum, not only the amplitude.

Radiation model (declared approximation)
----------------------------------------
The output signal is the equally weighted sum of modal velocities,
s(t) = sum_n v_n(t). Radiation efficiency, directivity and air loading
are ignored (unit radiation coefficient per mode). The output is in
arbitrary linear units proportional to surface velocity; WAV export
normalizes to a declared peak level. No clipping is applied anywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .physics import Mode
from .resonator import ModalOscillator


@dataclass(frozen=True)
class Mallet:
    """Mallet parameters (SI).

    mass_kg         : mallet mass [kg]; impulse J = mass * velocity [N s]
    contact_time_s  : contact time at 1 m/s [s]; scales as v^(-1/5) (Hertz)
    """

    mass_kg: float = 0.02
    contact_time_s: float = 0.001

    def impulse_n_s(self, velocity_m_s: float) -> float:
        return self.mass_kg * velocity_m_s

    def contact_time_at(self, velocity_m_s: float) -> float:
        return self.contact_time_s * velocity_m_s ** (-0.2)

    def weight(self, frequency_hz: float, velocity_m_s: float) -> float:
        tc = self.contact_time_at(velocity_m_s)
        return 1.0 / (1.0 + (math.pi * frequency_hz * tc) ** 2)


@dataclass
class ExcitedMode:
    """Report of one mode's activation after a strike (all SI)."""

    index: int
    label: str
    frequency_hz: float
    tau_s: float
    modal_mass_kg: float
    shape_at_hit: float  # phi_n(p), dimensionless, signed
    mallet_weight: float  # W(f_n, v), dimensionless
    delta_v_m_s: float  # modal velocity kick, signed
    peak_displacement_m: float  # |delta_v| / wd (light damping)


@dataclass
class StrikeReport:
    x_m: float
    y_m: float
    velocity_m_s: float
    impulse_n_s: float
    contact_time_s: float
    modes: list[ExcitedMode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ModalInstrument:
    """A set of modes on a surface, struck and rendered.

    geometry: ("plate", a_m, b_m) or ("bar", length_m, width_m). For bars
    the mode shapes are 1-D along the length; the across-width coordinate
    is accepted but ignored (torsional/width modes neglected - declared).
    """

    def __init__(
        self,
        modes: list[Mode],
        geometry: tuple,
        sample_rate_hz: float = 44100.0,
        mallet: Mallet | None = None,
        thickness_m: float = 0.0,
    ):
        self.modes = modes
        self.geometry = geometry
        self.sample_rate_hz = float(sample_rate_hz)
        self.mallet = mallet or Mallet()
        self.thickness_m = thickness_m
        self.oscillators = [
            ModalOscillator(m.frequency_hz, m.tau_s, self.sample_rate_hz)
            for m in modes
        ]
        self.report: StrikeReport | None = None

    def map_hit(self, u: float, v: float) -> tuple[float, float]:
        """Normalized surface coords (0..1) -> physical coords [m]."""
        kind = self.geometry[0]
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        if kind == "plate":
            return u * self.geometry[1], v * self.geometry[2]
        return u * self.geometry[1], v * self.geometry[2]

    def strike(
        self,
        x_m: float,
        y_m: float,
        velocity_m_s: float,
        solo_mode: int | None = None,
    ) -> StrikeReport:
        """Strike at (x_m, y_m) with mallet velocity; excites all modes."""
        if velocity_m_s <= 0.0:
            raise ValueError("velocity_m_s must be positive")
        j = self.mallet.impulse_n_s(velocity_m_s)
        report = StrikeReport(
            x_m=x_m,
            y_m=y_m,
            velocity_m_s=velocity_m_s,
            impulse_n_s=j,
            contact_time_s=self.mallet.contact_time_at(velocity_m_s),
        )
        kind = self.geometry[0]
        total_disp = 0.0
        for i, (mode, osc) in enumerate(zip(self.modes, self.oscillators)):
            phi = float(mode.shape(x_m, y_m)) if kind == "plate" else float(mode.shape(x_m))
            w = self.mallet.weight(mode.frequency_hz, velocity_m_s)
            dv = j * w * phi / mode.modal_mass_kg
            if solo_mode is not None and i != solo_mode:
                dv = 0.0
            osc.impulse(dv)
            wd = osc.wd if osc.underdamped else osc.w0
            peak_disp = abs(dv) / wd if wd > 0.0 else 0.0
            total_disp += peak_disp
            report.modes.append(
                ExcitedMode(
                    index=i,
                    label=mode.label,
                    frequency_hz=mode.frequency_hz,
                    tau_s=mode.tau_s,
                    modal_mass_kg=mode.modal_mass_kg,
                    shape_at_hit=phi,
                    mallet_weight=w,
                    delta_v_m_s=dv if solo_mode is None or i == solo_mode else 0.0,
                    peak_displacement_m=peak_disp,
                )
            )
        if self.thickness_m > 0.0 and total_disp > 0.3 * self.thickness_m:
            report.warnings.append(
                f"estimated peak displacement {total_disp * 1e3:.2f} mm exceeds "
                f"0.3x thickness ({0.3 * self.thickness_m * 1e3:.2f} mm); real "
                "plates/bars approach the nonlinear regime there - linear "
                "model extrapolated"
            )
        self.report = report
        return report

    def render(self, duration_s: float) -> np.ndarray:
        """Sum of modal velocities [arb. units ~ m/s], float64, no clipping."""
        n = int(round(duration_s * self.sample_rate_hz))
        out = np.zeros(n, dtype=np.float64)
        for osc in self.oscillators:
            # Silent oscillators (q == v == 0, no input) stay silent exactly.
            if osc.q != 0.0 or osc.v != 0.0:
                out += osc.render(n)
        return out

    def reset(self) -> None:
        for osc in self.oscillators:
            osc.q = 0.0
            osc.v = 0.0
