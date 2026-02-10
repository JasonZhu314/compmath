import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import interpolation as interp


def test_horner_simple_product():
    x = np.array([2.0, 3.0])
    c = np.array([1.0, 4.0, 5.0])
    assert interp.horner(c, x) == 39.0


def test_newton_coeff_quadratic():
    xs = np.array([0.0, 1.0, 2.0])
    ys = xs ** 2
    coeffs = interp.newton_coeff(xs, ys)
    expected = np.array([0.0, 1.0, 1.0])
    assert np.allclose(coeffs, expected)


def test_newton_interpolates_nodes():
    xs = np.array([-1.0, 0.0, 2.0])
    ys = xs ** 2 - 2 * xs + 1
    values = interp.newton(xs, ys, xs)
    assert np.allclose(values, ys)


def test_lagrange_matches_newton():
    xs = np.array([-2.0, -0.5, 1.0, 3.0])
    ys = 0.5 * xs ** 3 - xs + 2
    x_eval = np.linspace(-2.0, 3.0, 11)
    newton_vals = interp.newton(xs, ys, x_eval)
    lagrange_vals = interp.lagrange(xs, ys, x_eval)
    assert np.allclose(lagrange_vals, newton_vals)


def test_piecewise_linear_interpolates_and_clamps():
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 2.0, 0.0])
    x_eval = np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    expected = np.array([0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
    values = interp.piecewise_linear(xs, ys, x_eval)
    assert np.allclose(values, expected)


def test_spline_second_derivative_linear_zero():
    xs = np.array([0.0, 1.0, 2.0, 4.0])
    ys = 3.0 * xs - 5.0
    z = interp.spline_second_derivative(xs, ys)
    assert np.allclose(z, np.zeros_like(xs))


def test_spline_linear_exact():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = 2.0 * xs + 1.0
    x_eval = np.array([0.5, 1.5, 2.5])
    expected = 2.0 * x_eval + 1.0
    values = interp.spline(xs, ys, x_eval)
    assert np.allclose(values, expected)
