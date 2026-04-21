
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



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
import matplotlib.pyplot as plt


from Calculus import diff, integ
from Linear_Algebra import linear_system_solve
from Regression_and_Interpolation import PolynomialFit, linear_regression, polynomial_regression
from Root_Finding import find_root



def test_differentiate():
    print("----- Testing Differentiate.py -----")

    f = lambda x: -x**3 + 6*x**2 - 9*x + 7

    print("Exact derivative at x=2: 3")
    print("Forward difference at x=2:", diff(f, 2, h=0.001, mode=0))
    print("Backward difference at x=2:", diff(f, 2, h=0.001, mode=1))
    print("Central difference at x=2:", diff(f, 2, h=0.001, mode=2))
    print("Five-point difference at x=2:", diff(f, 2, h=0.001, mode=3))
    
    # Try calling with an invalid mode option
    try:
        print("Mode 4 at x=2:", diff(f, 2, h=0.001, mode=4))
    except ValueError:
        print("mode=4 is an invalid option.")

    data = [(0.5+0.1*i, f(0.5+0.1*i)) for i in np.linspace(0.5, 3.5, 101)]
    ddata = diff(data)
    
    # Plot test data and its derivative
    x = [data[i][0] for i in range(len(data))]
    y = [data[i][1] for i in range(len(data))]
    dy = [ddata[i][1] for i in range(len(data))]
    plt.plot(x, y, "b-", label="f(x)")
    plt.plot(x, dy, "r-", label="f'(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid()
    plt.show()

    print()


def test_integrate(): #step size, low mult num
    print("----- Testing integrate.py -----")

    f = lambda x: -3*x**2 + 12*x - 9

    print("Exact integral on [0,3]: 18")
    print("Left Riemann integral on [0,3]:", integ(f, 0, 3, n=100, mode=0))
    print("Right Riemann integral on [0,3]:", integ(f, 0, 3, n=100, mode=1))
    print("Midpoint integral on [0,3]:", integ(f, 0, 3, n=100, mode=2))
    print("Trapezoidal integral on [0,3]:", integ(f, 0, 3, n=100, mode=3))
    print("Simpson integral on [0,3]:", integ(f, 0, 3, n=100, mode=4))
    
    # Try calling with an invalid mode option
    try:
        print("Mode 5 on [0,3]:", integ(f, 0, 3, n=100, mode=5))
    except ValueError:
        print("mode=5 is an invalid option.")
    
    # Try reversing the order
    print("Left Riemann integral on [3,0]:", integ(f, 3, 0, n=100, mode=0))

    data = [(x, f(x)) for x in np.linspace(0, 3, 101)]
    print("Integral from data on [0,3]:", integ(data))

    print()


def test_linear_system_solver():
    print("----- Testing Linear_System_Solver.py -----")

    A1 = [[2, 1], [1, 3]]
    b1 = [5, 6]
    # Expected solution: [1.8, 1.4]
    print("Square system solution:", linear_system_solve(A1, b1))

    A2 = [[1, 1], [1, 2], [1, 3]]
    b2 = [1, 2, 2]
    # Expected solution: [0.667, 0.5]
    print("Least squares solution:", linear_system_solve(A2, b2))

    print()


def test_regression_interpolation():
    print("----- Testing RegressionInterpolation.py -----")

    data = [(0.9,3.5), (1.4,2.4), (2.4,2.3), (3.1,1.4), (4.6,2.7), (6.3,4.5)]
    model = PolynomialFit(data)

    coeffs_linear = model.linear_regression()
    print("Linear regression coefficients:", coeffs_linear)
    print("Linear regression polynomial:", model.polynomial_string())
    print("Evaluate at x=4:", model.evaluate(4))
    model.plot()

    model2 = PolynomialFit(data)
    coeffs_poly = model2.polynomial_regression(2)
    print("Polynomial regression coefficients:", coeffs_poly)
    print("Polynomial regression polynomial:", model2.polynomial_string())
    print("Evaluate at x=3:", model2.evaluate(3))
    model2.plot()

    data2 = [(0, 1), (1, 2), (2, 5)]
    model3 = PolynomialFit(data2)
    coeffs_interp = model3.polynomial_interpolation()
    print("Interpolation coefficients:", coeffs_interp)
    print("Interpolation polynomial:", model3.polynomial_string())
    model3.plot()

    x = [0, 1, 2, 3]
    y = [1, 3, 5, 7]
    print("Standalone linear_regression function:", linear_regression(x, y))
    print("Standalone polynomial_regression function:", polynomial_regression(x, y, 2))

    print()


def test_root_finding(): # secant -> approx close to 0 -> throw zero division error
    print("----- Testing Root.py -----")

    # Define a test polynomial and its derivative
    f = lambda x: -x**3 + 6*x**2 - 9*x + 7
    df = lambda x: -3*x**2 + 12*x - 9

    print("Bisection root on [4,5]:", find_root(f, 4, 5, tol=1e-10))
    print("Bisection root on [5,4]:", find_root(f, 5, 4, tol=1e-10))
    try:
        print("Bisection root on [2,4]:", find_root(f, 2, 4, tol=1e-10))
    except ValueError:
        print("No root on [2,4].")
    print("Secant root starting at x0=4:", find_root(f, 4, tol=1e-10))
    print("Newton root starting at x0=4:", find_root(f, df, 4, tol=1e-10))
    try:
        print("Newton root starting at x0=3:", find_root(f, df, 3, tol=1e-10))
    except ZeroDivisionError:
        print("Derivative is zero at x=3.")
    print("Secant root starting at x0=3:", find_root(f, 3, tol=1e-10))

    print()


def run_all_tests():
    test_differentiate()
    test_integrate()
    test_linear_system_solver()
    test_regression_interpolation()
    test_root_finding()
    print("All tests completed.")


if __name__ == "__main__":
    run_all_tests()
    #input()
