"""
This file contains numerical integration functions.
"""

import numpy as np


def _left(f, a, b, n=100):
    h = (b - a) / n
    s = 0.0

    for i in range(n):
        xi = a + i * h
        s += f(xi)

    return h * s


def _right(f, a, b, n=100):
    h = (b - a) / n
    s = 0.0

    for i in range(1, n + 1):
        xi = a + i * h
        s += f(xi)

    return h * s


def _midpoint(f, a, b, n=100):
    h = (b - a) / n
    s = 0.0

    for i in range(n):
        xi = a + (i + 0.5) * h
        s += f(xi)

    return h * s


def _trapezoidal(f, a, b, n=100):
    h = (b - a) / n
    s = 0.0

    for i in range(1, n):
        xi = a + i * h
        s += f(xi)

    return h * (0.5 * f(a) + s + 0.5 * f(b))


def _simpson(f, a, b, n=100):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")

    h = (b - a) / n
    s_odd = 0.0
    s_even = 0.0

    for i in range(1, n):
        xi = a + i * h
        if i % 2 == 1:
            s_odd += f(xi)
        else:
            s_even += f(xi)

    return (h / 3) * (f(a) + 4 * s_odd + 2 * s_even + f(b))


def _integ_list(data):
    """
    Integrate a function represented by a list of ordered pairs.

    data = [(x0, y0), (x1, y1), ..., (xn, yn)]

    Uses the trapezoidal rule between consecutive data points.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of ordered pairs")

    if len(data) < 2:
        raise ValueError("data must contain at least two points")

    area = 0.0

    for i in range(len(data) - 1):
        x0, y0 = data[i]
        x1, y1 = data[i + 1]

        if x1 == x0:
            raise ValueError("x-values must be distinct")

        area += (x1 - x0) * (y0 + y1) / 2

    return area


def integ(f, a=None, b=None, n=100, mode=3):
    """
    Numerical integration.

    If f is a function:
        integ(f, a, b, n=100, mode=3)

    If f is a list of ordered pairs:
        integ(data)
    """
    if isinstance(f, list):
        return _integ_list(f)

    if not callable(f):
        raise TypeError("f must be either a callable function or a list of ordered pairs")

    if a is None or b is None:
        raise ValueError("a and b must be provided when f is a function")

    n = int(n)
    if n <= 0:
        raise ValueError("n must be a positive integer")

    if mode == 0:
        return _left(f, a, b, n)
    elif mode == 1:
        return _right(f, a, b, n)
    elif mode == 2:
        return _midpoint(f, a, b, n)
    elif mode == 3:
        return _trapezoidal(f, a, b, n)
    elif mode == 4:
        return _simpson(f, a, b, n)
    else:
        raise ValueError("mode must be 0, 1, 2, 3, or 4")
