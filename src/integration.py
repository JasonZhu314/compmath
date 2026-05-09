import numpy as np
import matplotlib.pyplot as plt

# Numerical Integration Methods

# Midpoint Method
def midpoint(a, b, n, f):
    '''
    Computes the midpoint rule for numerical integration.
    
    Parameters
    ----------
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    n : int
        Number of subintervals
    f : function
        Function to be integrated
    
    Returns
    ----------
    float
        Approximation of the integral of f from a to b using the midpoint rule
    int
        Number of function evaluations
    '''
    h = (b - a) / n
    return h * np.sum(f(a + h/2 + np.arange(n) * h)), n

# Trapezoidal Method
def trapezoidal(a, b, n, f):
    '''
    Computes the trapezoidal rule for numerical integration.
    
    Parameters
    ----------
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    n : int
        Number of subintervals
    f : function
        Function to be integrated
    
    Returns
    ----------
    float
        Approximation of the integral of f from a to b using the trapezoidal rule
    int
        Number of function evaluations
    '''
    h = (b - a) / n
    return h * (0.5 * f(a) + 0.5 * f(b) + np.sum(f(a + np.arange(1, n) * h))), n + 1

# Simpson's Method
def simpson(a, b, n, f):
    '''
    Computes Simpson's rule for numerical integration.
    
    Parameters
    ----------
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    n : int
        Number of subintervals
    f : function
        Function to be integrated
    
    Returns
    ----------
    float
        Approximation of the integral of f from a to b using Simpson's rule
    int
        Number of function evaluations
    '''
    h = (b - a) / n
    return h/6 * (f(a) + f(b) + 2 * np.sum(f(a + np.arange(1, 2 * n) * h / 2)) + 2 * np.sum(f(a + h / 2 + np.arange(n) * h))), 2 * n + 1

# Romberg Integration
def romberg(a, b, f, tol=1e-12, max_iter=20):
    '''
    Computes the Romberg integration for numerical integration.
    
    Parameters
    ----------
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    f : function
        Function to be integrated
    tol : float, optional
        Tolerance for convergence (default is 1e-12)
    max_iter : int, optional
        Maximum number of iterations (default is 20)
    
    Returns
    ----------
    float
        Approximation of the integral of f from a to b using Romberg integration
    int
        Number of function evaluations
    '''
    R = np.zeros((max_iter, max_iter))
    h = b - a
    R[0, 0] = 0.5 * h * (f(a) + f(b))
    evals = 2
    for k in range(1, max_iter):
        h /= 2
        x_new = a + (2 * np.arange(1, 2**(k-1) + 1) - 1) * h
        sum_f = np.sum(f(x_new))
        evals += len(x_new)
        R[k, 0] = 0.5 * R[k-1, 0] + h * sum_f
        for j in range(1, k + 1):
            R[k, j] = R[k, j-1] + (R[k, j-1] - R[k-1, j-1]) / (4**j - 1)
        if abs(R[k, k] - R[k-1, k-1]) < tol:
            return R[k, k], evals
    return R[-1, -1], evals

# Adaptive Simpson's Method
def adaptive_simpson(a, b, f, tol):
    '''
    Computes the integral of f from a to b using adaptive Simpson's method.
    
    Parameters
    ----------
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    f : function
        Function to be integrated
    tol : float
        Tolerance for convergence
    
    Returns
    ----------
    float
        Approximation of the integral of f from a to b using adaptive Simpson's method
    int
        Number of function evaluations
    '''
    evals = [2]
    points = {a, b}
    def step(a, b, eps, whole, fa, fm, fb):
        mid = (a + b) / 2
        lm = (a + mid) / 2
        rm = (mid + b) / 2
        flm = f(lm)
        frm = f(rm)
        evals[0] += 2
        points.update([lm, mid, rm])
        left = (mid - a) / 6 * (fa + 4 * flm + fm)
        right = (b - mid) / 6 * (fm + 4 * frm + fb)
        if abs(left + right - whole) <= 15 * eps:
            return left + right + (left + right - whole) / 15
        return step(a, mid, eps/2, left, fa, flm, fm) + step(mid, b, eps/2, right, fm, frm, fb)
    mid = (a + b) / 2
    fa, fm, fb = f(a), f(mid), f(b)
    evals[0] += 1
    points.add(mid)
    initial_whole = (b - a) / 6 * (fa + 4 * fm + fb)
    val = step(a, b, tol, initial_whole, fa, fm, fb)
    return val, evals[0], sorted(list(points))
