"""
Root.py

This module provides numerical root-finding methods:
- Bisection method
- Newton's method
- Secant method

The public interface is find_root(), which selects the
appropriate method based on the user's arguments.
"""


def _bisection(f, a, b, tol=1e-6, max_iter=100):
    """
    Approximate a root of a function using the bisection method.
    """
    if a > b:
        a, b = b, a

    fa = f(a)
    fb = f(b)

    if abs(fa) < tol:
        return a
    if abs(fb) < tol:
        return b

    if fa * fb > 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs.")

    for _ in range(max_iter):
        c = (a + b) / 2
        fc = f(c)

        if abs(fc) < tol or abs(b - a) / 2 < tol:
            return c

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return (a + b) / 2


def _newton(f, df, x0, tol=1e-6, max_iter=100):
    """
    Approximate a root of a function using Newton's method.
    """
    x = x0

    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(fx) < tol:
            return x

        if abs(dfx) < 1e-14:
            raise ZeroDivisionError("Newton's method failed because derivative is zero or too close to zero.")

        x_new = x - fx / dfx

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return x


def _secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Approximate a root of a function using the secant method.
    """
    for _ in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)

        if abs(f1) < tol:
            return x1

        if abs(f1 - f0) < 1e-14:
            raise ZeroDivisionError("Secant method failed because denominator became zero or too close to zero.")

        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

        if abs(x2 - x1) < tol:
            return x2

        x0, x1 = x1, x2

    return x1


def find_root(f, *args, tol=1e-6, max_iter=100):
    """
    Find a root of a function using bisection, secant, or Newton's method.

    Usage
    -----
    1. Bisection:
       find_root(f, a, b, tol=..., max_iter=...)

    2. Secant with one initial guess:
       find_root(f, x0, tol=..., max_iter=...)

    3. Secant with two initial guesses:
       find_root(f, x0, x1, "secant", tol=..., max_iter=...)

    4. Newton:
       find_root(f, df, x0, tol=..., max_iter=...)
    """
    if not callable(f):
        raise TypeError("f must be callable")

    if tol <= 0:
        raise ValueError("tol must be positive")

    if max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")

    if len(args) == 2 and callable(args[0]):
        # Newton's method: find_root(f, df, x0)
        df, x0 = args
        return _newton(f, df, x0, tol=tol, max_iter=max_iter)

    elif len(args) == 3 and args[2] == "secant":
        x0, x1, method = args
        return _secant(f, x0, x1, tol=tol, max_iter=max_iter)

    elif len(args) == 2 and not callable(args[0]):
        # Bisection method: find_root(f, a, b)
        a, b = args
        return _bisection(f, a, b, tol=tol, max_iter=max_iter)

    elif len(args) == 1:
        # Secant method with one initial guess: find_root(f, x0)
        x0 = args[0]
        x1 = x0 + 1e-4
        return _secant(f, x0, x1, tol=tol, max_iter=max_iter)

    else:
        raise ValueError(
            "Invalid arguments.\n"
            "Use one of the following:\n"
            "  find_root(f, a, b, tol=..., max_iter=...)         # bisection\n"
            "  find_root(f, x0, tol=..., max_iter=...)           # secant\n"
            "  find_root(f, df, x0, tol=..., max_iter=...)       # Newton"
        )
