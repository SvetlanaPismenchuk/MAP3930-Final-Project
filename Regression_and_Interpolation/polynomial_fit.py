"""
RegressionInterpolation.py

This module contains tools for polynomial regression, linear regression,
and polynomial interpolation.
"""

import numpy as np
import matplotlib.pyplot as plt


class PolynomialFit:
    """
    Represent a polynomial regression or interpolation model built from data.
    """

    def __init__(self, data):
        """
        Initialize a PolynomialFit object with raw data.

        Parameters
        ----------
        data : list of tuple
            List of ordered pairs [(x1, y1), (x2, y2), ..., (xn, yn)].
        """
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("data must be a nonempty list of ordered pairs")

        self.raw_data = data
        self.coefficients = None

    def polynomial_regression(self, degree):
        """
        Fit a polynomial of a specified degree to the data using least squares.
        """
        if degree < 0:
            raise ValueError("degree must be nonnegative")

        x = np.array([point[0] for point in self.raw_data], dtype=float)
        y = np.array([point[1] for point in self.raw_data], dtype=float)

        self.coefficients = np.polyfit(x, y, degree)
        return self.coefficients

    def linear_regression(self):
        """
        Fit a line to the data as a special case of polynomial regression.
        """
        return self.polynomial_regression(1)

    def polynomial_interpolation(self):
        """
        Construct an interpolating polynomial that passes through all data points.
        """
        x = np.array([point[0] for point in self.raw_data], dtype=float)
        y = np.array([point[1] for point in self.raw_data], dtype=float)

        if len(set(x)) != len(x):
            raise ValueError("x-values must be distinct for interpolation")

        degree = len(x) - 1
        self.coefficients = np.polyfit(x, y, degree)
        return self.coefficients

    def get_coefficients(self):
        """
        Return the fitted polynomial coefficients.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial has been fit yet")

        return self.coefficients

    def polynomial_string(self):
        """
        Return the fitted polynomial as a readable string.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial has been fit yet")

        terms = []
        degree = len(self.coefficients) - 1

        for i, coeff in enumerate(self.coefficients):
            power = degree - i

            if np.isclose(coeff, 0):
                continue

            coeff_str = f"{coeff:.6g}"

            if power == 0:
                terms.append(f"{coeff_str}")
            elif power == 1:
                terms.append(f"{coeff_str}x")
            else:
                terms.append(f"{coeff_str}x^{power}")

        if not terms:
            return "0"

        poly = " + ".join(terms)
        poly = poly.replace("+ -", "- ")
        return poly

    def evaluate(self, x):
        """
        Evaluate the fitted polynomial at a given input value.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial has been fit yet")

        return np.polyval(self.coefficients, x)

    def plot(self, **kwargs):
        """
        Plot the raw data together with the fitted regression or interpolation curve.
        """
        if self.coefficients is None:
            raise ValueError("No polynomial has been fit yet")

        x_data = np.array([point[0] for point in self.raw_data], dtype=float)
        y_data = np.array([point[1] for point in self.raw_data], dtype=float)

        num_points = kwargs.get("num_points", 200)
        title = kwargs.get("title", "Polynomial Fit")
        xlabel = kwargs.get("xlabel", "x")
        ylabel = kwargs.get("ylabel", "y")
        show_data = kwargs.get("show_data", True)
        data_label = kwargs.get("data_label", "Raw Data")
        curve_label = kwargs.get("curve_label", "Fitted Curve")

        x_curve = np.linspace(np.min(x_data), np.max(x_data), num_points)
        y_curve = self.evaluate(x_curve)

        if show_data:
            plt.scatter(x_data, y_data, label=data_label)

        plt.plot(x_curve, y_curve, label=curve_label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()


def linear_regression(x, y):
    """
    Compute the best-fit line for a set of data points.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    data = list(zip(x, y))
    model = PolynomialFit(data)
    return model.linear_regression()


def polynomial_regression(x, y, degree):
    """
    Fit a polynomial of specified degree to a set of data points.
    """
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    data = list(zip(x, y))
    model = PolynomialFit(data)
    return model.polynomial_regression(degree)