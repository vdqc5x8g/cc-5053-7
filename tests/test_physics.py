"""Physics model tests: formulas, mode shapes, boundary conditions."""

import math

import numpy as np
import pytest

from modal_synth import physics
from modal_synth.models import STEEL, ALUMINUM, WOOD


# ---------------------------------------------------------------------------
# Plate (Navier solution)
# ---------------------------------------------------------------------------


def test_plate_frequency_ratios_square():
    """Square plate: f_mn proportional to m^2 + n^2 (exact ratios)."""
    a = b = 0.3
    h = 0.002
    f11 = physics.plate_mode_frequency_hz(STEEL, a, b, h, 1, 1)
    f12 = physics.plate_mode_frequency_hz(STEEL, a, b, h, 1, 2)
    f22 = physics.plate_mode_frequency_hz(STEEL, a, b, h, 2, 2)
    f13 = physics.plate_mode_frequency_hz(STEEL, a, b, h, 1, 3)
    assert f12 / f11 == pytest.approx(2.5, rel=1e-12)
    assert f22 / f11 == pytest.approx(4.0, rel=1e-12)
    assert f13 / f11 == pytest.approx(5.0, rel=1e-12)


def test_plate_frequency_value():
    """Absolute check against the closed form, computed independently."""
    a, b, h = 0.30, 0.22, 0.002
    d = STEEL.youngs_modulus_pa * h**3 / (12 * (1 - STEEL.poisson_ratio**2))
    expected = (math.pi / 2) * math.sqrt(d / (STEEL.density_kg_m3 * h)) * (
        1 / a**2 + 1 / b**2
    )
    f = physics.plate_mode_frequency_hz(STEEL, a, b, h, 1, 1)
    assert f == pytest.approx(expected, rel=1e-12)
    assert 100.0 < f < 250.0  # sanity: 2 mm steel plate of this size


def test_plate_modal_mass():
    modes, _ = physics.plate_modes(STEEL, 0.3, 0.22, 0.002, 2000.0, 8)
    expected = STEEL.density_kg_m3 * 0.002 * 0.3 * 0.22 / 4.0
    for m in modes:
        assert m.modal_mass_kg == pytest.approx(expected, rel=1e-12)


def test_plate_mode_shape_boundary_and_normalization():
    modes, _ = physics.plate_modes(STEEL, 0.3, 0.22, 0.002, 2000.0, 1)
    shape = modes[0].shape  # (1,1) mode
    y = np.linspace(0, 0.22, 50)
    x = np.linspace(0, 0.3, 50)
    assert np.allclose(shape(np.zeros_like(y), y), 0.0)
    assert np.allclose(shape(np.full_like(y, 0.3), y), 0.0)
    assert np.allclose(shape(x, np.zeros_like(x)), 0.0)
    assert np.allclose(shape(x, np.full_like(x, 0.22)), 0.0)
    # (1,1) peaks at the plate center with normalized amplitude 1.
    assert shape(0.15, 0.11) == pytest.approx(1.0, rel=1e-12)


def test_plate_mode_cap_reports_excluded():
    modes_all, excluded_all = physics.plate_modes(STEEL, 0.3, 0.22, 0.002, 5000.0, 1000)
    modes_few, excluded_few = physics.plate_modes(STEEL, 0.3, 0.22, 0.002, 5000.0, 5)
    assert len(modes_few) == 5
    assert excluded_few == len(modes_all) - 5
    assert excluded_all == 0


# ---------------------------------------------------------------------------
# Free-free bar (Euler-Bernoulli)
# ---------------------------------------------------------------------------


def test_bar_wavenumber_roots_match_tabulated():
    """Blevins tab. 8-1: 4.7300408, 7.8532046, 10.9956078, 14.1371655."""
    known = [4.7300408, 7.8532046, 10.9956078, 14.1371655]
    for n, z in enumerate(known, start=1):
        assert physics.free_free_bar_wavenumber_root(n) == pytest.approx(z, abs=1e-7)


def test_bar_frequency_ratio():
    modes, _ = physics.free_free_bar_modes(ALUMINUM, 0.5, 0.04, 0.01, 20000.0, 4)
    ratio = modes[1].frequency_hz / modes[0].frequency_hz
    assert ratio == pytest.approx((7.8532046 / 4.7300408) ** 2, rel=1e-6)


def test_bar_mode_shape_free_boundary_conditions():
    """Free ends: phi'' = phi''' = 0 at x = 0 and x = L.

    phi'' is checked by finite differences. phi'''(L) = 0 is equivalent,
    given phi''(L) = 0 and the mode-shape form, to the characteristic
    equation cos(beta L) cosh(beta L) = 1; a direct finite difference of
    phi''' would drown in cosh-cancellation roundoff, so the residual of
    the characteristic equation is checked instead.
    """
    length = 0.5
    beta_l = physics.free_free_bar_wavenumber_root(1)
    shape = physics.free_free_bar_mode_shape_fn(beta_l, length)
    h = 1e-3 * length

    def d2(x):
        return (shape(x + h) - 2 * shape(x) + shape(x - h)) / h**2

    interior_curvature = abs(d2(0.3 * length))
    assert interior_curvature > 10.0
    for x in (0.0, length):
        assert abs(d2(x)) < 1e-3 * interior_curvature
    residual = math.cos(beta_l) * math.cosh(beta_l) - 1.0
    assert abs(residual) < 1e-10


def test_bar_mode_shapes_orthogonal():
    modes, _ = physics.free_free_bar_modes(ALUMINUM, 0.5, 0.04, 0.01, 20000.0, 3)
    x = np.linspace(0, 0.5, 20001)
    phi = [m.shape(x) for m in modes]
    norm0 = np.trapezoid(phi[0] ** 2, x)
    for i in range(3):
        for j in range(i + 1, 3):
            cross = np.trapezoid(phi[i] * phi[j], x)
            assert abs(cross / norm0) < 1e-6


def test_bar_mode_shape_normalized():
    modes, _ = physics.free_free_bar_modes(ALUMINUM, 0.5, 0.04, 0.01, 20000.0, 2)
    x = np.linspace(0, 0.5, 20001)
    for m in modes:
        assert np.max(np.abs(m.shape(x))) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Damping law
# ---------------------------------------------------------------------------


def test_tau_from_loss_factor():
    assert physics.tau_from_loss_factor(100.0, 1e-3) == pytest.approx(
        1.0 / (math.pi * 100.0 * 1e-3), rel=1e-12
    )


def test_wood_more_heavily_damped_than_steel():
    """Wood bar modes must decay much faster than steel plate modes."""
    from modal_synth import models

    steel = models.build("steel_plate")
    wood = models.build("wood_bar")
    assert wood.modes[0].tau_s < 0.2 * steel.modes[0].tau_s
