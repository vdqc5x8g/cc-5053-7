"""Declared instrument presets.

Every preset states its geometry, material and the model it comes from.
None of these are measurements of real objects; they are textbook
analytical models (see physics.py) or, for "perturbed_plate", a seeded
procedural perturbation of the analytical plate, clearly labeled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import physics
from .physics import Material, Mode


@dataclass
class InstrumentSpec:
    id: str
    name: str
    description: str
    geometry: tuple  # ("plate", a_m, b_m) or ("bar", length_m, width_m)
    thickness_m: float
    material: Material
    modes: list[Mode]
    excluded_mode_count: int
    f_max_hz: float
    notes: list[str] = field(default_factory=list)


STEEL = Material("steel", 200e9, 7850.0, 0.30, 3.0e-4)
ALUMINUM = Material("aluminum", 69e9, 2700.0, 0.33, 1.0e-3)
WOOD = Material("spruce-like wood (along grain)", 12e9, 600.0, 0.35, 8.0e-3)

F_MAX_HZ = 18000.0
MAX_MODES = 64


def build_steel_plate(damping_scale: float = 1.0) -> InstrumentSpec:
    a, b, h = 0.30, 0.22, 0.002
    modes, excluded = physics.plate_modes(
        STEEL, a, b, h, F_MAX_HZ, MAX_MODES, damping_scale
    )
    return InstrumentSpec(
        id="steel_plate",
        name="Steel plate 30 x 22 cm, 2 mm (simply supported)",
        description=(
            "Navier's analytical solution for a simply supported rectangular "
            "thin plate. Simplification: real free/clamped edges have no "
            "closed-form modes."
        ),
        geometry=("plate", a, b),
        thickness_m=h,
        material=STEEL,
        modes=modes,
        excluded_mode_count=excluded,
        f_max_hz=F_MAX_HZ,
        notes=[
            "f_mn = (pi/2) sqrt(D/(rho h)) (m^2/a^2 + n^2/b^2), D = E h^3/(12(1-nu^2))",
            "mode shapes sin(m pi x/a) sin(n pi y/b); modal mass rho h a b / 4",
            "constant loss factor eta=3e-4 (steel), tau_n = 1/(pi f_n eta)",
            "not a measurement of a real plate",
        ],
    )


def build_aluminum_bar(damping_scale: float = 1.0) -> InstrumentSpec:
    length, width, h = 0.50, 0.04, 0.010
    modes, excluded = physics.free_free_bar_modes(
        ALUMINUM, length, width, h, F_MAX_HZ, MAX_MODES, damping_scale
    )
    return InstrumentSpec(
        id="aluminum_bar",
        name="Aluminum bar 50 cm, 10 mm thick (free-free)",
        description=(
            "Euler-Bernoulli free-free bar (vibraphone-like, without "
            "resonators). Wavenumbers from cos(beta L) cosh(beta L) = 1."
        ),
        geometry=("bar", length, width),
        thickness_m=h,
        material=ALUMINUM,
        modes=modes,
        excluded_mode_count=excluded,
        f_max_hz=F_MAX_HZ,
        notes=[
            "f_n = (beta_n L)^2/(2 pi L^2) * h sqrt(E/(12 rho))",
            "mode shapes cosh+cos-k(sinh+sin), free-free boundary conditions",
            "rigid-body modes at 0 Hz excluded; width/torsion modes neglected",
            "constant loss factor eta=1e-3 (aluminum)",
            "not a measurement of a real bar",
        ],
    )


def build_wood_bar(damping_scale: float = 1.0) -> InstrumentSpec:
    length, width, h = 0.40, 0.05, 0.020
    modes, excluded = physics.free_free_bar_modes(
        WOOD, length, width, h, F_MAX_HZ, MAX_MODES, damping_scale
    )
    return InstrumentSpec(
        id="wood_bar",
        name="Wooden bar 40 cm, 20 mm thick (free-free)",
        description=(
            "Euler-Bernoulli free-free bar in a spruce-like wood "
            "(marimba-like, without resonators). Isotropic approximation: "
            "real wood is strongly orthotropic."
        ),
        geometry=("bar", length, width),
        thickness_m=h,
        material=WOOD,
        modes=modes,
        excluded_mode_count=excluded,
        f_max_hz=F_MAX_HZ,
        notes=[
            "same Euler-Bernoulli model as the aluminum bar",
            "isotropic E=12 GPa approximates along-grain behavior only",
            "constant loss factor eta=8e-3 (wood, much higher damping than metal)",
            "not a measurement of a real bar",
        ],
    )


def build_perturbed_plate(damping_scale: float = 1.0, seed: int = 7) -> InstrumentSpec:
    """Procedurally generated parameters: analytical steel plate with seeded
    random perturbations (+-4% frequency, +-30% decay time).

    This is the 'procedurally generated modal parameters' option: a
    statistical model standing in for manufacturing variability. It is not
    derived from any measurement.
    """
    spec = build_steel_plate(damping_scale)
    rng = np.random.default_rng(seed)
    perturbed = []
    for m in spec.modes:
        f = m.frequency_hz * float(1.0 + rng.uniform(-0.04, 0.04))
        tau = m.tau_s * float(1.0 + rng.uniform(-0.3, 0.3))
        perturbed.append(
            Mode(
                frequency_hz=f,
                tau_s=tau,
                modal_mass_kg=m.modal_mass_kg,
                shape=m.shape,
                label=m.label,
            )
        )
    spec.id = "perturbed_plate"
    spec.name = "Perturbed steel plate (procedural, seed=7)"
    spec.description = (
        "Steel plate modes with seeded random jitter (+-4% frequency, "
        "+-30% decay time) as a statistical model of manufacturing "
        "variability. Procedurally generated, not measured."
    )
    spec.modes = perturbed
    spec.notes = spec.notes + [
        "frequencies/decays jittered with numpy default_rng(7): "
        "U(-4%,+4%) on f, U(-30%,+30%) on tau",
        "procedural statistical model - explicitly not a measurement",
    ]
    return spec


BUILDERS = {
    "steel_plate": build_steel_plate,
    "aluminum_bar": build_aluminum_bar,
    "wood_bar": build_wood_bar,
    "perturbed_plate": build_perturbed_plate,
}


def build(model_id: str, damping_scale: float = 1.0) -> InstrumentSpec:
    if model_id not in BUILDERS:
        raise KeyError(f"unknown model '{model_id}'")
    return BUILDERS[model_id](damping_scale)
