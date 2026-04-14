'''
This file contains numerical integration functions.
'''

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


def integ(f, a, b, n=100, mode=3):
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
        return "Please enter a number 0-4."