"""Modal instrument: surface hit mapping + sustained sound rendering.

Layering
--------
The interaction layer submits only *hit events*: a position on the
surface plus a sample-accurate strike time (and the mallet/velocity
information that produced it). It does not own oscillators, does not
render, and does not care whether another strike is still ringing. The
mapping from a hit event to per-mode excitation lives here, and the body
state lives in :class:`ModalEngine`.

``ModalEngine`` owns one persistent ``ModalVoice`` for the current body
and accepts hit events at absolute sample times. A second strike while the
first is still decaying is added (superposed) to the same modal state, so
the sound is the physical sum of both excitations. Nothing is reset on a
hit, at a render-block edge, or when the drawing/refresh cadence changes.

Hit mapping (position and velocity -> per-mode excitation)
----------------------------------------------------------
A strike at surface point p with mallet velocity v applies an impulse
J = m_mallet * v [N s] (fully transferred momentum; contact time much
shorter than every modal period - declared simplification). Mode n with
mode shape phi_n and modal mass M_n receives the modal velocity kick

    delta_v_n = J * W(f_n, v) * phi_n(p) / M_n      [m/s]

so the hit position changes *which* modes are excited (through phi_n(p)),
not just the overall loudness. A strike on a nodal line of mode n leaves
that mode silent regardless of force. Two nearby points on opposite sides
of a nodal line receive opposite-signed phi_n, so their kicks cancel for
that mode while adding for modes whose antinode lies between them.

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

Excitation direction
--------------------
A hit event carries an optional ``direction`` (+1 or -1). Reversing it
negates every mode's kick, which is the physical equivalent of striking
the same point from the other side of the plate (or reversing the mallet
polarity). Because the body is linear, this negates the whole contribution
of that event and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .physics import Mode
from .resonator import ModalVoice


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


@dataclass(frozen=True)
class HitEvent:
    """What the interaction layer submits. This is the whole contract.

    position_m      : strike point on the surface [m]; for bars (x, 0)
    time_s          : strike time in seconds, measured on the engine clock
    velocity_m_s    : mallet speed at contact [m/s]
    contact_time_s  : mallet contact time at 1 m/s [s]
    mallet_mass_kg  : mallet mass [kg]
    direction       : +1 or -1; flips the sign of this event's excitation
    label           : optional identifier used by the audit tools

    The event carries no oscillator references and no durations: how the
    body responds is entirely the engine's concern.
    """

    position_m: tuple[float, float]
    time_s: float
    velocity_m_s: float
    contact_time_s: float = 0.001
    mallet_mass_kg: float = 0.02
    direction: int = 1
    label: str = ""

    @property
    def mallet(self) -> Mallet:
        return Mallet(mass_kg=self.mallet_mass_kg, contact_time_s=self.contact_time_s)


@dataclass
class ExcitedMode:
    """Report of one mode's activation for a single hit event (all SI)."""

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
class HitReport:
    """Per-event excitation report, produced by the mapping layer."""

    event: HitEvent
    impulse_n_s: float
    applied_sample: int
    applied_time_s: float
    time_quantization_s: float
    modes: list[ExcitedMode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def excitation_vector(self) -> np.ndarray:
        return np.array([m.delta_v_m_s for m in self.modes], dtype=np.float64)


@dataclass
class BodyState:
    """A settled snapshot of one mode's sustained state at an instant.

    phase_rad : arg(q + j v / w0) of the analytic modal signal - the phase
                a listener/analyser would read at that instant
    amplitude : sqrt(q^2 + (v/w0)^2), the instantaneous modal amplitude
    tail_s    : time until the envelope falls by 1/e^n_tail from here
    """

    index: int
    displacement: float
    velocity: float
    amplitude: float
    phase_rad: float
    tail_s: float


class ModalInstrument:
    """Mapping from surface coordinates to per-mode excitation.

    This object is stateless with respect to the running vibration: it
    owns the geometry and the mode table and turns a hit at (x, y) into a
    vector of modal velocity kicks. It is deliberately separate from
    ``ModalEngine`` so a body (modes + geometry) can be replaced without
    touching the interaction layer, and so a single hit can be compared
    as a standalone response versus as an addition to a ringing body.
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

    def map_hit(self, u: float, v: float) -> tuple[float, float]:
        """Normalized surface coords (0..1) -> physical coords [m]."""
        kind = self.geometry[0]
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        if kind == "plate":
            return u * self.geometry[1], v * self.geometry[2]
        return u * self.geometry[1], v * self.geometry[2]

    def shape_at(self, mode_index: int, x_m: float, y_m: float) -> float:
        mode = self.modes[mode_index]
        if self.geometry[0] == "plate":
            return float(mode.shape(x_m, y_m))
        return float(mode.shape(x_m))

    def excitation(
        self,
        x_m: float,
        y_m: float,
        velocity_m_s: float,
        direction: int = 1,
        mallet: Mallet | None = None,
    ) -> tuple[float, list[ExcitedMode], list[str]]:
        """Resolve a physical strike into per-mode velocity kicks.

        Returns (impulse_n_s, modes, warnings). ``direction`` of -1 negates
        every kick; the per-mode shape values and mallet weights reported
        are physical and unsigned by the direction, so the audit table
        always shows which mode is on which side of its nodal line.
        ``mallet`` overrides the instrument's default mallet.
        """
        if velocity_m_s <= 0.0:
            raise ValueError("velocity_m_s must be positive")
        if direction not in (1, -1):
            raise ValueError("direction must be +1 or -1")
        mallet = mallet or self.mallet
        j = mallet.impulse_n_s(velocity_m_s)
        excited: list[ExcitedMode] = []
        total_disp = 0.0
        for i, mode in enumerate(self.modes):
            phi = self.shape_at(i, x_m, y_m)
            w = mallet.weight(mode.frequency_hz, velocity_m_s)
            dv = direction * j * w * phi / mode.modal_mass_kg
            w0 = 2.0 * math.pi * mode.frequency_hz
            alpha = 1.0 / mode.tau_s
            wd = math.sqrt(w0 * w0 - alpha * alpha) if alpha < w0 else w0
            peak_disp = abs(dv) / wd if wd > 0.0 else 0.0
            total_disp += peak_disp
            excited.append(
                ExcitedMode(
                    index=i,
                    label=mode.label,
                    frequency_hz=mode.frequency_hz,
                    tau_s=mode.tau_s,
                    modal_mass_kg=mode.modal_mass_kg,
                    shape_at_hit=phi,
                    mallet_weight=w,
                    delta_v_m_s=dv,
                    peak_displacement_m=peak_disp,
                )
            )
        warnings: list[str] = []
        if self.thickness_m > 0.0 and total_disp > 0.3 * self.thickness_m:
            warnings.append(
                f"estimated peak displacement {total_disp * 1e3:.2f} mm exceeds "
                f"0.3x thickness ({0.3 * self.thickness_m * 1e3:.2f} mm); real "
                "plates/bars approach the nonlinear regime there - linear "
                "model extrapolated"
            )
        return j, excited, warnings


class ModalEngine:
    """Persistent sustained vibration for one body, driven by hit events.

    Each call to :meth:`submit` resolves one hit event against the current
    body and schedules its modal kicks at the event's sample time. The
    oscillator state is never cleared by rendering, block resizing or a
    redraw; the only operations that change it are scheduled hits and
    :meth:`reset` / :meth:`replace_body`.

    Block rendering
    ---------------
    ``render_block(n)`` advances the clock by n samples and returns that
    block. ``render_blocks(total, block_size)`` covers the same span with
    explicit block edges. Both apply the state continuously, so the output
    does not depend on where the block boundaries fall (up to float
    rounding); a test traces per-mode cancellation through block edges to
    prove no mode is zeroed there.

    Body replacement
    ----------------
    ``replace_body(modes, geometry, ...)`` swaps the mode table and/or
    geometry. The old voice is discarded, never mixed into the new body:
    the new voice starts at the current clock time with zero state, so
    tails from the previous body cannot leak across. ``keep_tails=False``
    is therefore the only supported policy - "restart with new parameters"
    is explicit, not silent.
    """

    def __init__(
        self,
        instrument: ModalInstrument,
        sample_rate_hz: float | None = None,
    ):
        self.instrument = instrument
        self.sample_rate_hz = float(
            sample_rate_hz or instrument.sample_rate_hz
        )
        self.mallet = instrument.mallet
        self._new_voice()
        self.reports: list[HitReport] = []

    def _new_voice(self) -> None:
        self.voice = ModalVoice(self.instrument.modes, self.sample_rate_hz)

    # -- clock ----------------------------------------------------------

    @property
    def position(self) -> int:
        return self.voice.position

    @property
    def time_s(self) -> float:
        return self.voice.position / self.sample_rate_hz

    # -- hits -----------------------------------------------------------

    def submit(self, event: HitEvent, only_mode: int | None = None) -> HitReport:
        """Resolve and schedule one hit event; returns its per-mode report.

        Late events (time in the past relative to the engine clock) are
        rejected rather than silently folded into the present, so the
        caller can resubmit at the correct time. The report states the
        exact sample at which the impulses were applied.

        ``only_mode`` restricts the applied kicks to a single mode; the
        report still lists every mode's excitation so the audit table stays
        complete.
        """
        x_m, y_m = event.position_m
        impulse, modes, warnings = self.instrument.excitation(
            x_m,
            y_m,
            event.velocity_m_s,
            direction=event.direction,
            mallet=event.mallet,
        )
        if only_mode is not None and not (0 <= only_mode < len(modes)):
            raise ValueError("only_mode out of range")
        requested_sample = event.time_s * self.sample_rate_hz
        if requested_sample < self.voice.position - 1e-9:
            raise ValueError(
                f"hit at {event.time_s:.6f} s is in the past "
                f"(engine clock {self.time_s:.6f} s)"
            )
        applied = self.voice.position
        for em in modes:
            if only_mode is not None and em.index != only_mode:
                continue
            applied = self.voice.schedule_impulse(
                requested_sample, em.index, em.delta_v_m_s
            )
        applied_time = applied / self.sample_rate_hz
        report = HitReport(
            event=event,
            impulse_n_s=impulse,
            applied_sample=applied,
            applied_time_s=applied_time,
            time_quantization_s=applied_time - event.time_s,
            modes=modes,
            warnings=list(warnings),
        )
        self.reports.append(report)
        return report

    # -- rendering ------------------------------------------------------

    def render_block(self, n_samples: int) -> np.ndarray:
        return self.voice.render_block(n_samples)

    def render_blocks(self, total_samples: int, block_size: int) -> np.ndarray:
        """Render total_samples in fixed-size blocks (last block shorter)."""
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        out = np.zeros(int(total_samples), dtype=np.float64)
        cursor = 0
        while cursor < total_samples:
            n = min(block_size, total_samples - cursor)
            out[cursor : cursor + n] = self.voice.render_block(n)
            cursor += n
        return out

    def render_duration(self, duration_s: float) -> np.ndarray:
        n = int(round(duration_s * self.sample_rate_hz))
        return self.voice.render_block(n)

    # -- body swapping --------------------------------------------------

    def replace_body(
        self,
        modes: list[Mode],
        geometry: tuple,
        mallet: Mallet | None = None,
        thickness_m: float = 0.0,
        sample_rate_hz: float | None = None,
    ) -> None:
        """Install a new body and explicitly drop the old tails.

        The engine clock is preserved; the new body's state is zero at the
        current time. This is the documented inheritance rule: parameters
        that describe the *new* strike (geometry, material, mallet) are
        re-derived for the new body; nothing of the old oscillation is
        inherited.
        """
        self.instrument = ModalInstrument(
            modes,
            geometry,
            sample_rate_hz=sample_rate_hz or self.sample_rate_hz,
            mallet=mallet or self.mallet,
            thickness_m=thickness_m,
        )
        self.sample_rate_hz = self.instrument.sample_rate_hz
        self.mallet = self.instrument.mallet
        current = self.voice.position
        self._new_voice()
        self.voice.position = current
        self.reports.clear()

    def reset(self) -> None:
        """Silence the body immediately (keeps the clock position)."""
        current = self.voice.position
        self._new_voice()
        self.voice.position = current
        self.reports.clear()

    # -- inspection -----------------------------------------------------

    def body_state(
        self, n_tail: float = 1.0, floor_ratio: float | None = None
    ) -> list[BodyState]:
        """Read back each mode's instantaneous phase, amplitude and tail.

        By default ``tail_s`` is the time for this mode's envelope to fall
        by a factor e^n_tail from its present amplitude (an intrinsic
        property, independent of amplitude). If ``floor_ratio`` is given,
        the tail instead measures how long this mode remains within
        ``floor_ratio`` of the loudest mode in the body right now: a mode
        that has been cancelled reads a tail of 0 s, while a ringing mode
        reads a positive time until it drops below the floor.
        """
        states: list[BodyState] = []
        raw = []
        for i, (mode, osc) in enumerate(
            zip(self.instrument.modes, self.voice.oscillators)
        ):
            w0 = osc.w0
            analytic = complex(osc.q, osc.v / w0) if w0 > 0.0 else complex(osc.q, 0.0)
            raw.append((mode, osc, analytic))
        ref = max((abs(z) for _, _, z in raw), default=0.0)
        for i, (mode, osc, analytic) in enumerate(raw):
            amp = abs(analytic)
            phase = math.atan2(analytic.imag, analytic.real) if amp > 0.0 else 0.0
            if floor_ratio is None:
                tail = n_tail * mode.tau_s if amp > 0.0 else 0.0
            else:
                floor = floor_ratio * ref
                tail = (
                    max(0.0, mode.tau_s * math.log(amp / floor))
                    if amp > floor > 0.0
                    else 0.0
                )
            states.append(
                BodyState(
                    index=i,
                    displacement=osc.q,
                    velocity=osc.v,
                    amplitude=amp,
                    phase_rad=phase,
                    tail_s=tail,
                )
            )
        return states
