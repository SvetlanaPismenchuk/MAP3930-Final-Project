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

    Parameters
    ----------
    f : callable
        Function whose root is being approximated.
    a : float
        Left endpoint of the interval.
    b : float
        Right endpoint of the interval.
    tol : float, optional
        Error tolerance used as a stopping condition.
    max_iter : int, optional
        Maximum number of iterations allowed.

    Returns
    -------
    float
        Approximate root of the function.

    Raises
    ------
    ValueError
        If f(a) and f(b) do not have opposite signs.
    """
    fa = f(a)
    fb = f(b)

    if fa == 0:
        return a
    if fb == 0:
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

    Parameters
    ----------
    f : callable
        Function whose root is being approximated.
    df : callable
        Derivative of the function f.
    x0 : float
        Initial guess for the root.
    tol : float, optional
        Error tolerance used as a stopping condition.
    max_iter : int, optional
        Maximum number of iterations allowed.

    Returns
    -------
    float
        Approximate root of the function.

    Raises
    ------
    ZeroDivisionError
        If the derivative is zero during an iteration.
    """
    x = x0

    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(fx) < tol:
            return x

        if dfx == 0:
            raise ZeroDivisionError("Newton's method failed because derivative is zero.")

        x_new = x - fx / dfx

        if abs(x_new - x) < tol:
            return x_new

        x = x_new

    return x


def _secant(f, x0, x1, tol=1e-6, max_iter=100):
    """
    Approximate a root of a function using the secant method.

    Parameters
    ----------
    f : callable
        Function whose root is being approximated.
    x0 : float
        First initial guess.
    x1 : float
        Second initial guess.
    tol : float, optional
        Error tolerance used as a stopping condition.
    max_iter : int, optional
        Maximum number of iterations allowed.

    Returns
    -------
    float
        Approximate root of the function.

    Raises
    ------
    ZeroDivisionError
        If f(x1) - f(x0) is zero during an iteration.
    """
    for _ in range(max_iter):
        f0 = f(x0)
        f1 = f(x1)

        if abs(f1) < tol:
            return x1

        if f1 - f0 == 0:
            raise ZeroDivisionError("Secant method failed because denominator became zero.")

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

    2. Secant:
       find_root(f, x0, tol=..., max_iter=...)

       Uses x0 and x0 + 1e-4 as two starting guesses.

    3. Newton:
       find_root(f, df, x0, tol=..., max_iter=...)

    Parameters
    ----------
    f : callable
        Function whose root is being approximated.
    *args
        Extra arguments that determine which method is used.
    tol : float, optional
        Error tolerance.
    max_iter : int, optional
        Maximum number of iterations.

    Returns
    -------
    float
        Approximate root of the function.

    Raises
    ------
    ValueError
        If the arguments do not match a supported usage.
    """
    if len(args) == 2 and callable(args[0]):
        # Newton's method: find_root(f, df, x0)
        df, x0 = args
        return _newton(f, df, x0, tol=tol, max_iter=max_iter)

    elif len(args) == 2:
        # Bisection method: find_root(f, a, b)
        a, b = args
        return _bisection(f, a, b, tol=tol, max_iter=max_iter)

    elif len(args) == 1:
        # Secant method: find_root(f, x0)
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