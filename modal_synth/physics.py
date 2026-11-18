"""Analytical modal models for simplified, exactly solvable shapes.

All quantities are SI: frequencies in Hz, decay times in seconds, positions in
meters, masses in kg. Mode shapes are dimensionless and normalized to
max |phi| = 1.

Model sources
-------------
* Simply supported rectangular thin plate: Navier's closed-form solution
  (Leissa, "Vibration of Plates", 1969, ch. 1). Mode shapes are products of
  sines; frequencies follow f_mn = (pi/2) sqrt(D/(rho h)) (m^2/a^2 + n^2/b^2)
  with flexural rigidity D = E h^3 / (12 (1 - nu^2)).
  Real plates are usually free-edged (cymbals) or clamped; those boundary
  conditions have no closed form. The simply supported plate is used here
  because it is exactly solvable - it is a declared simplification, not a
  measurement of a real object.
* Free-free Euler-Bernoulli bar: classical thin-beam theory
  (Blevins, "Formulas for Natural Frequency and Mode Shape", 1979, tab. 8-1).
  Wavenumbers are roots of cos(beta L) cosh(beta L) = 1, solved numerically
  by Newton iteration from the asymptotic guess (2n+1) pi / 2.

Declared approximations
-----------------------
* Thin-plate / thin-beam theory (no shear deformation, no rotary inertia).
* Constant loss factor eta for all modes of a model, so tau_n = 1/(pi f_n eta).
  Real damping is frequency dependent (thermoelastic, air loading,
  viscoelastic supports); a single eta is a declared simplification.
* Rigid-body modes of the free-free bar (0 Hz translation/rotation) are
  excluded: they do not produce audible radiation.
* Nothing here is a measurement of a physical instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Material:
    """Isotropic material properties (SI)."""

    name: str
    youngs_modulus_pa: float
    density_kg_m3: float
    poisson_ratio: float
    loss_factor: float  # eta, dimensionless: tau_n = 1 / (pi * f_n * eta)


@dataclass
class Mode:
    """One vibration mode.

    frequency_hz : modal frequency f_n [Hz]
    tau_s        : amplitude decay time constant [s]; envelope ~ exp(-t/tau)
    modal_mass_kg: generalized mass rho * integral(phi^2 dV)-style [kg]
    shape        : phi(x) for bars or phi(x, y) for plates, dimensionless,
                   normalized to max |phi| = 1
    label        : human-readable mode index, e.g. "(2,1)" or "n=3"
    """

    frequency_hz: float
    tau_s: float
    modal_mass_kg: float
    shape: Callable
    label: str


def tau_from_loss_factor(frequency_hz: float, loss_factor: float) -> float:
    """Decay time constant [s] from a constant loss factor.

    Amplitude decays as exp(-pi f eta t), i.e. tau = 1 / (pi f eta).
    """
    return 1.0 / (math.pi * frequency_hz * loss_factor)


# ---------------------------------------------------------------------------
# Simply supported rectangular thin plate (Navier solution)
# ---------------------------------------------------------------------------


def plate_flexural_rigidity(material: Material, thickness_m: float) -> float:
    """D = E h^3 / (12 (1 - nu^2))  [N m]."""
    return (
        material.youngs_modulus_pa
        * thickness_m**3
        / (12.0 * (1.0 - material.poisson_ratio**2))
    )


def plate_mode_frequency_hz(
    material: Material, a_m: float, b_m: float, thickness_m: float, m: int, n: int
) -> float:
    """f_mn = (pi/2) sqrt(D/(rho h)) (m^2/a^2 + n^2/b^2)  [Hz]."""
    d = plate_flexural_rigidity(material, thickness_m)
    return (
        (math.pi / 2.0)
        * math.sqrt(d / (material.density_kg_m3 * thickness_m))
        * (m * m / (a_m * a_m) + n * n / (b_m * b_m))
    )


def plate_modes(
    material: Material,
    a_m: float,
    b_m: float,
    thickness_m: float,
    f_max_hz: float,
    max_modes: int,
    damping_scale: float = 1.0,
) -> tuple[list[Mode], int]:
    """All (m, n) plate modes up to f_max_hz, sorted by frequency.

    Returns (modes, excluded_count) where excluded_count is how many modes
    below f_max_hz were dropped by the max_modes cap (reported, not hidden).
    """
    modal_mass = material.density_kg_m3 * thickness_m * a_m * b_m / 4.0
    entries = []
    m = 1
    # f increases with m and n; stop when the (m,1) mode alone exceeds f_max.
    while plate_mode_frequency_hz(material, a_m, b_m, thickness_m, m, 1) <= f_max_hz:
        n = 1
        while True:
            f = plate_mode_frequency_hz(material, a_m, b_m, thickness_m, m, n)
            if f > f_max_hz:
                break
            entries.append((f, m, n))
            n += 1
        m += 1
    entries.sort()
    excluded = max(0, len(entries) - max_modes)
    entries = entries[:max_modes]

    modes = []
    for f, m, n in entries:
        mm, nn = m, n  # bind for closure

        def shape(x, y, mm=mm, nn=nn):
            return np.sin(mm * math.pi * x / a_m) * np.sin(nn * math.pi * y / b_m)

        modes.append(
            Mode(
                frequency_hz=f,
                tau_s=tau_from_loss_factor(f, material.loss_factor * damping_scale),
                modal_mass_kg=modal_mass,
                shape=shape,
                label=f"({mm},{nn})",
            )
        )
    return modes, excluded


# ---------------------------------------------------------------------------
# Free-free Euler-Bernoulli bar
# ---------------------------------------------------------------------------


def free_free_bar_wavenumber_root(n: int) -> float:
    """n-th positive root (beta_n L) of cos(z) cosh(z) = 1, n = 1, 2, ...

    Solved as cos(z) - sech(z) = 0 by Newton iteration from the asymptotic
    guess (2n + 1) pi / 2. First roots: 4.7300408, 7.8532046, 10.9956078.
    """
    z = (2 * n + 1) * math.pi / 2.0
    for _ in range(50):
        cosh_z = math.cosh(z)
        f = math.cos(z) - 1.0 / cosh_z
        fp = -math.sin(z) + math.tanh(z) / cosh_z
        dz = f / fp
        z -= dz
        if abs(dz) < 1e-14:
            break
    return z


def free_free_bar_mode_shape_fn(beta_l: float, length_m: float) -> Callable:
    """phi(x) = cosh(bx) + cos(bx) - k (sinh(bx) + sin(bx)), max |phi| = 1.

    k = (cosh(bL) - cos(bL)) / (sinh(bL) - sin(bL)). Satisfies the free
    boundary conditions phi''(0) = phi''(L) = phi'''(0) = phi'''(L) = 0.
    """
    beta = beta_l / length_m
    k = (math.cosh(beta_l) - math.cos(beta_l)) / (math.sinh(beta_l) - math.sin(beta_l))

    def raw(x):
        bx = beta * np.asarray(x, dtype=float)
        return np.cosh(bx) + np.cos(bx) - k * (np.sinh(bx) + np.sin(bx))

    # Normalize to max |phi| = 1 (sampled; extrema are at the ends/antinodes).
    grid = np.linspace(0.0, length_m, 2001)
    scale = float(np.max(np.abs(raw(grid))))
    norm = 1.0 / scale

    def shape(x):
        return raw(x) * norm

    return shape


def free_free_bar_modes(
    material: Material,
    length_m: float,
    width_m: float,
    thickness_m: float,
    f_max_hz: float,
    max_modes: int,
    damping_scale: float = 1.0,
) -> tuple[list[Mode], int]:
    """Flexural modes of a free-free bar, bending across its thickness.

    f_n = (beta_n L)^2 / (2 pi L^2) sqrt(E I / (rho A)) with I = w h^3 / 12,
    A = w h, so sqrt(EI/(rho A)) = h sqrt(E / (12 rho))  [m^2/s].
    """
    stiffness_factor = thickness_m * math.sqrt(
        material.youngs_modulus_pa / (12.0 * material.density_kg_m3)
    )
    area = width_m * thickness_m
    modes: list[Mode] = []
    n = 1
    while len(modes) < max_modes:
        beta_l = free_free_bar_wavenumber_root(n)
        f = beta_l**2 / (2.0 * math.pi * length_m**2) * stiffness_factor
        if f > f_max_hz:
            break
        shape = free_free_bar_mode_shape_fn(beta_l, length_m)
        grid = np.linspace(0.0, length_m, 4001)
        integral_phi2 = float(np.trapezoid(shape(grid) ** 2, grid))
        modal_mass = material.density_kg_m3 * area * integral_phi2
        modes.append(
            Mode(
                frequency_hz=f,
                tau_s=tau_from_loss_factor(f, material.loss_factor * damping_scale),
                modal_mass_kg=modal_mass,
                shape=shape,
                label=f"n={n}",
            )
        )
        n += 1
    # All bar modes below f_max fit or the cap was hit; count the remainder.
    excluded = 0
    while True:
        beta_l = free_free_bar_wavenumber_root(n)
        f = beta_l**2 / (2.0 * math.pi * length_m**2) * stiffness_factor
        if f > f_max_hz:
            break
        excluded += 1
        n += 1
    return modes, excluded
