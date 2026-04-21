"""
Test.py

This script tests the functions in:
- Differentiate.py
- integrate.py
- Linear_System_Solver.py
- RegressionInterpolation.py
- Root.py
"""

import math
import numpy as np

from Calculus.Differentiate import diff
from Calculus.Integrate import integ
from Linear_Algebra.Linear_System_Solver import linear_system_solve
from Regression_and_Interpolation.polynomial_fit import PolynomialFit, linear_regression, polynomial_regression
from Root_Finding.Root import find_root


def test_differentiate():
    print("----- Testing Differentiate.py -----")

    f = lambda x: x**2

    print("Forward difference of x^2 at x=2:", diff(f, 2, h=0.001, mode=0))
    print("Backward difference of x^2 at x=2:", diff(f, 2, h=0.001, mode=1))
    print("Central difference of x^2 at x=2:", diff(f, 2, h=0.001, mode=2))
    print("Five-point difference of x^2 at x=2:", diff(f, 2, h=0.001, mode=3))

    data = [(0, 0), (1, 1), (2, 4), (3, 9)]
    print("Derivative from list data:", diff(data))

    print()


def test_integrate():
    print("----- Testing integrate.py -----")

    f = lambda x: x**2

    print("Left Riemann integral of x^2 on [0,2]:", integ(f, 0, 2, n=100, mode=0))
    print("Right Riemann integral of x^2 on [0,2]:", integ(f, 0, 2, n=100, mode=1))
    print("Midpoint integral of x^2 on [0,2]:", integ(f, 0, 2, n=100, mode=2))
    print("Trapezoidal integral of x^2 on [0,2]:", integ(f, 0, 2, n=100, mode=3))
    print("Simpson integral of x^2 on [0,2]:", integ(f, 0, 2, n=100, mode=4))

    data = [(0, 0), (1, 1), (2, 4)]
    print("Integral from list data:", integ(data))

    print()


def test_linear_system_solver():
    print("----- Testing Linear_System_Solver.py -----")

    A1 = [[2, 1], [1, 3]]
    b1 = [5, 6]
    print("Square system solution:", linear_system_solve(A1, b1))

    A2 = [[1, 1], [1, 2], [1, 3]]
    b2 = [1, 2, 2]
    print("Least squares solution:", linear_system_solve(A2, b2))

    print()


def test_regression_interpolation():
    print("----- Testing RegressionInterpolation.py -----")

    data = [(0, 1), (1, 3), (2, 5), (3, 7)]
    model = PolynomialFit(data)

    coeffs_linear = model.linear_regression()
    print("Linear regression coefficients:", coeffs_linear)
    print("Linear regression polynomial:", model.polynomial_string())
    print("Evaluate at x=4:", model.evaluate(4))

    data2 = [(0, 1), (1, 2), (2, 5)]
    model2 = PolynomialFit(data2)
    coeffs_poly = model2.polynomial_regression(2)
    print("Polynomial regression coefficients:", coeffs_poly)
    print("Polynomial regression polynomial:", model2.polynomial_string())
    print("Evaluate at x=3:", model2.evaluate(3))

    model3 = PolynomialFit(data2)
    coeffs_interp = model3.polynomial_interpolation()
    print("Interpolation coefficients:", coeffs_interp)
    print("Interpolation polynomial:", model3.polynomial_string())

    x = [0, 1, 2, 3]
    y = [1, 3, 5, 7]
    print("Standalone linear_regression function:", linear_regression(x, y))
    print("Standalone polynomial_regression function:", polynomial_regression(x, y, 2))

    print()


def test_root_finding():
    print("----- Testing Root.py -----")

    f = lambda x: x**2 - 4
    df = lambda x: 2 * x

    print("Bisection root of x^2 - 4 on [0,3]:", find_root(f, 0, 3))
    print("Secant root of x^2 - 4 starting at x0=3:", find_root(f, 3))
    print("Newton root of x^2 - 4 starting at x0=3:", find_root(f, df, 3))

    g = lambda x: math.cos(x) - x
    dg = lambda x: -math.sin(x) - 1

    print("Bisection root of cos(x) - x on [0,1]:", find_root(g, 0, 1))
    print("Newton root of cos(x) - x starting at x0=1:", find_root(g, dg, 1))

    print()

    input()


def run_all_tests():
    test_differentiate()
    test_integrate()
    test_linear_system_solver()
    test_regression_interpolation()
    test_root_finding()
    print("All tests completed.")


if __name__ == "__main__":
    run_all_tests()
