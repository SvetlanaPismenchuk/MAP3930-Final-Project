# Numerical Methods Toolkit

This project is a Python numerical methods library that includes tools for differentiation, integration, solving linear systems, regression/interpolation, and root-finding.

## Table of Contents

- [Files Included](#files-included)
- [Requirements](#requirements)
- [Differentiate](#differentiate)
- [Example with Data](#example-with-data)
- [Integration](#integration)
- [Linear Algebra](#linear-algebra)
- [Regression and Interpolation](#regression-and-interpolation)
- [Root Finding](#root-finding)
- [Testing](#testing)
- [Error Handling](#error-handling)
- [Author](#author)
- [Acknowledgment](#acknowledgment)

## Files Included

- [Differentiate](#differentiate)
- [Integration](#integration)
- [Linear System Solver](#linear-algebra)
- [Regression and Interpolation](#regression-and-interpolation)
- [Root Finding] (#Root-Finding)
- Test.py

---

## Requirements

This project uses Python 3 and the following libraries:

```bash
pip install numpy matplotlib
```

## Differentiate

The Differentiate.py module provides numerical differentiation methods for functions and tabulated data.

## Supported methods
- Forward difference
- Backward difference
- Central difference
- Five-point difference
- Differentiation of a list of ordered pairs

### Function
```python
diff(f, x0=None, h=0.01, mode=0)
```

### Modes
- 0 = forward difference
- 1 = backward difference
- 2 = central difference
- 3 = five-point difference

### Example
```python
from Calculus.Differentiate import diff

f = lambda x: x**2

print(diff(f, 2, h=0.001, mode=0))
print(diff(f, 2, h=0.001, mode=1))
print(diff(f, 2, h=0.001, mode=2))
print(diff(f, 2, h=0.001, mode=3))
```

## Example with Data
```python
from Calculus.Differentiate import diff

data = [(0, 0), (1, 1), (2, 4), (3, 9)]
print(diff(data))
```

## Integration

The Integrate.py module provides numerical integration methods for functions and tabulated data.

## Supported methods
- Left Riemann sum
- Right Riemann sum
- Midpoint rule
- Trapezoidal rule
- Simpson’s rule
- Integration of a list of ordered pairs

## Function
```python
integ(f, a=None, b=None, n=100, mode=3)
```

## Modes
- 0 = left Riemann sum
- 1 = right Riemann sum
- 2 = midpoint rule
- 3 = trapezoidal rule
- 4 = Simpson’s rule

## Example
```python
from Calculus.Integrate import integ

f = lambda x: x**2

print(integ(f, 0, 2, n=100, mode=0))
print(integ(f, 0, 2, n=100, mode=1))
print(integ(f, 0, 2, n=100, mode=2))
print(integ(f, 0, 2, n=100, mode=3))
print(integ(f, 0, 2, n=100, mode=4))
```

## Example with Data
```python
from integrate import integ

data = [(0, 0), (1, 1), (2, 4)]
print(integ(data))
```

## Linear Algebra

The Linear_System_Solver.py module solves linear systems using Gaussian elimination or least squares regression.

## Function
linear_system_solve(A, b)

## Behavior
If A is square, Gaussian elimination with scaled partial pivoting is used.
If A is overdetermined, least squares regression is used.
Underdetermined systems are not supported.

## Example
```python
from Linear_System_Solver import linear_system_solve

A = [[2, 1], [1, 3]]
b = [5, 6]

print(linear_system_solve(A, b))
```

## Least Squares Example
```python
from Linear_System_Solver import linear_system_solve

A = [[1, 1], [1, 2], [1, 3]]
b = [1, 2, 2]

print(linear_system_solve(A, b))
```

## Regression and Interpolation

The RegressionInterpolation.py module contains tools for polynomial regression, linear regression, and polynomial interpolation.

## Main Class
```python
PolynomialFit
```

## Features
- Linear regression
- Polynomial regression
- Polynomial interpolation
- Polynomial evaluation
- Plotting raw data and fitted curve
- Returning coefficients
- Returning the polynomial as a readable string

## Example: Linear Regression
```python
from RegressionInterpolation import PolynomialFit

data = [(0, 1), (1, 3), (2, 5), (3, 7)]
model = PolynomialFit(data)

model.linear_regression()
print(model.get_coefficients())
print(model.polynomial_string())
print(model.evaluate(4))
```

## Polynomial Regression Example
```python
from RegressionInterpolation import PolynomialFit

data = [(0, 1), (1, 2), (2, 5)]
model = PolynomialFit(data)

model.polynomial_regression(2)
print(model.get_coefficients())
print(model.polynomial_string())
print(model.evaluate(3))
```

## Polynomial Interpolation Example
```python
from RegressionInterpolation import PolynomialFit

data = [(0, 1), (1, 2), (2, 5)]
model = PolynomialFit(data)

model.polynomial_interpolation()
print(model.get_coefficients())
print(model.polynomial_string())
```

## Plot Fitted Curve Example
```python
from RegressionInterpolation import PolynomialFit

data = [(0, 1), (1, 2), (2, 5)]
model = PolynomialFit(data)

model.polynomial_regression(2)
model.plot(title="Polynomial Fit", xlabel="x", ylabel="y")
```

## Standalone Function Example
```python
from RegressionInterpolation import linear_regression, polynomial_regression

x = [0, 1, 2, 3]
y = [1, 3, 5, 7]

print(linear_regression(x, y))
print(polynomial_regression(x, y, 2))
```

## Root Finding

The Root.py module provides numerical root-finding methods.

## Supported methods
- Bisection method
- Newton’s method
- Secant method

## Function
```python
find_root(f, *args, tol=1e-6, max_iter=100)
```

## Example: bisection
```python
from Root import find_root

f = lambda x: x**2 - 4
print(find_root(f, 0, 3))
```

## Example: secant
```python
from Root import find_root

f = lambda x: x**2 - 4
print(find_root(f, 3))
```

## Example: Newton’s method
```python
from Root import find_root

f = lambda x: x**2 - 4
df = lambda x: 2*x

print(find_root(f, df, 3))
```

## Testing

The test.py script is used to test all of the modules in the project.

## Run the test script
```python
python test.py
```

## Error Handling

The project uses Python exceptions to handle invalid inputs. For example:

- invalid mode values
- singular matrices
- unsupported system sizes
- invalid regression/interpolation usage
- invalid root-finding arguments
- evaluating a polynomial before fitting

## Author
Svetlana Pismenchuk

## Acknowledgment
I would like to acknowledge Dr. Rumpf for his support and guidance throughout the development of this project. His instruction helped shape both the mathematical direction of the work and the refining that went into the final product.
