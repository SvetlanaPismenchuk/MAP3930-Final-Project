"""
Differentate.py

This file contains numerical differentiation functions.

Modes:
0 - forward difference
1 - backward difference
2 - central difference
3 - five-point difference
"""


def _forward_diff(f, x0, h=0.01):
    return (f(x0 + h) - f(x0)) / h


def _backward_diff(f, x0, h=0.01):
    return (f(x0) - f(x0 - h)) / h


def _central_diff(f, x0, h=0.01):
    return (f(x0 + h) - f(x0 - h)) / (2 * h)


def _five_point_diff(f, x0, h=0.01):
    return (
        f(x0 - 2 * h)
        - 8 * f(x0 - h)
        + 8 * f(x0 + h)
        - f(x0 + 2 * h)
    ) / (12 * h)


def _diff_list(data):
    """
    Differentiate a list of ordered pairs.

    data = [(x0, y0), (x1, y1), ..., (xn, yn)]

    Returns a list of ordered pairs:
    [(x0, y0'), (x1, y1'), ..., (xn, yn')]
    """
    result = []
    n = len(data)

    for i in range(n):
        x, y = data[i]

        if i == 0:
            x1, y1 = data[i + 1]
            dydx = (y1 - y) / (x1 - x)

        elif i == n - 1:
            x0, y0 = data[i - 1]
            dydx = (y - y0) / (x - x0)

        else:
            x0, y0 = data[i - 1]
            x1, y1 = data[i + 1]
            dydx = (y1 - y0) / (x1 - x0)

        result.append((x, dydx))

    return result


def diff(f, x0=None, h=0.01, mode=0):
    """
    If f is a function:
        diff(f, x0, h=0.01, mode=0)

    If f is a list of ordered pairs:
        diff(data)
    """

    if isinstance(f, list):
        return _diff_list(f)

    if mode == 0:
        return _forward_diff(f, x0, h)

    elif mode == 1:
        return _backward_diff(f, x0, h)

    elif mode == 2:
        return _central_diff(f, x0, h)

    elif mode == 3:
        return _five_point_diff(f, x0, h)

    else:
        return "Please enter a number 0-3."