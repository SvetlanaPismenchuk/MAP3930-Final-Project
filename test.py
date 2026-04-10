"""Tests for the various modules included in this repo."""

import matplotlib.pyplot as plt

### Import modules here ###

#------------------------------------------------------------------------------

def differentiation_test():
    """Tests for the numerical differentiation module."""
    
    pass
    
    ### Tests with a test function
    
    ### Tests with numerical data

#------------------------------------------------------------------------------

def integration_test():
    """Tests for the numerical integration module."""
    
    pass
    
    ### Tests with a test function
    
    ### Tests with numerical data

#------------------------------------------------------------------------------

def root_test():
    """Tests for the numerical root-finding module."""
    
    # Define a test polynomial and its derivative
    f = lambda x: -x**3 + 6*x**2 - 9*x + 7
    fprime = lambda x: -3*x**2 + 12*x - 9
    
    ### Test the root-finding function for applying the bisection method on the
    ### search interval [4,5] with an error tolerance of 1e-10.
    
    ### Test the error-handling of the bisection method by repeating the test
    ### on the search interval [2,4].
    
    ### Test the root-finding function for applying Newton's method with an
    ### initial guess of 4 with an error tolerance of 1e-10.
    
    ### Test the error-handling of Newton's method by repeating the test with
    ### an initial guess of 3.
    
    ### Test the root-finding function for applying the secant method with an
    ### initial guess of 4 with an error tolerance of 1e-10.

#------------------------------------------------------------------------------

def linalg_test():
    """Tests for the numerical linear algebra module."""
    
    pass
    
    ### Tests with a square system
    
    ### Tests with an overdetermined system

#------------------------------------------------------------------------------

def regression_test():
    """Tests for the numerical regression and interpolation module."""
    
    pass
    
    ### Tests for finding an interpolating polynomial
    
    ### Tests for finding a best-fit polynomial

#==============================================================================

# Run scripts (comment out individual tests to skip)
differentiation_test()
integration_test()
root_test()
linalg_test()
regression_test()
