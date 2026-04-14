"""
Linear_System_Solver.py

This module contains functions for solving linear systems
using common numerical linear algebra methods.

Functionality:
- Solve square systems using Gaussian elimination
  with scaled partial pivoting.
- Solve overdetermined systems using least squares regression.
- Provide one public interface that selects the appropriate method
  based on the shape of the system.
"""

import numpy as np


def _gaussian_elimination_scaled(A, b):
    """
    Solve a square linear system Ax = b using Gaussian elimination
    with scaled partial pivoting.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    n = len(b)

    scale = np.max(np.abs(A), axis=1)

    for k in range(n - 1):
        ratios = np.abs(A[k:n, k]) / scale[k:n]
        max_row = np.argmax(ratios) + k

        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        scale[[k, max_row]] = scale[[max_row, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:n] = A[i, k:n] - factor * A[k, k:n]
            b[i] = b[i] - factor * b[k]

    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:n] @ x[i + 1:n]) / A[i, i]

    return x


def _least_squares_regression(A, b):
    """
    Solve an overdetermined linear system Ax = b using least squares.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    ATA = A.T @ A
    ATb = A.T @ b

    return _gaussian_elimination_scaled(ATA, ATb)


def linear_system_solve(A, b):
    """
    Solve a linear system using an appropriate numerical method.

    - If A is square, use Gaussian elimination with scaled partial pivoting.
    - If A is overdetermined, use least squares regression.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    rows, cols = A.shape

    if rows == cols:
        return _gaussian_elimination_scaled(A, b)
    else:
        return _least_squares_regression(A, b)