import numpy as np
import matplotlib.pyplot as plt

# Newton Interpolation

def horner(c : np.array, x : np.array):
    '''
    Implements generalized Horner's algorithm.
    
    Parameters
    ----------
    c : np.array
    x : np.array
    
    Returns
    ----------
    float
        c[0] + c[1] * x[0] + ... + c[n] * x[0] * ... * x[n - 1]
    '''
    n = len(x)
    result = c[n]
    for i in range(n - 1, -1, -1):
        result = (result * x[i]) + c[i]
    return result

def newton_coeff(xs : np.array, ys : np.array):
    '''
    Implements Newton's interpolation algorithm, calculates the coefficients of the Newton polynomial.
    p(x) = c[0] + c[1] * (x - x[0]) + ... + c[n - 1] * (x - x[0]) * ... * (x - x[n - 2])
    
    Parameters
    ----------
    xs : np.array
        x-coordinates of the interpolation points
    ys : np.array
        y-coordinates of the interpolation points
    
    Returns
    ----------
    np.array
        Coefficients c of the Newton polynomial, where p(x) = c[0] + c[1] * (x - x[0]) + ... + c[n - 1] * (x - x[0]) * ... * (x - x[n - 2])
    '''
    n = len(xs)
    c = np.zeros(n)
    c[0] = ys[0]
    for k in range(1, n):
        p = c[k - 1]
        t = xs[k] -  xs[k - 1]
        for j in range(k - 2, -1, -1):
            p = (p * (xs[k] - xs[j])) + c[j]
            t *= (xs[k] - xs[j])
        c[k] = (ys[k] - p) / t
    return c

def newton(xs : np.array, ys : np.array, x : np.array):
    '''
    Implements Newton's interpolation algorithm, evaluates the Newton polynomial at each point in x.
    
    Parameters
    ----------
    xs : np.array
        x-coordinates of the interpolation points
    ys : np.array
        y-coordinates of the interpolation points
    x : np.array
        x-coordinates where the polynomial is evaluated
        
    Returns
    ----------
    np.array
        Values of the Newton polynomial at each point in x
    '''
    n = len(xs)
    c = newton_coeff(xs, ys)
    result = c[n - 1] * np.ones(len(x))
    for k in range(n - 2, -1, -1):
        result = result * (x - xs[k]) + c[k]
    return result

# Lagrange Interpolation

def lagrange(xs : np.array, ys : np.array, x : np.array):
    '''
    Implements Lagrange's interpolation algorithm, evaluates the Lagrange polynomial at each point in x.

    Parameters
    ----------
    xs : np.array
        x-coordinates of the interpolation points
    ys : np.array
        y-coordinates of the interpolation points
    x : np.array
        x-coordinates where the polynomial is evaluated
        
    Returns
    ----------
    np.array
        Values of the Lagrange polynomial at each point in x
    '''
    result = np.zeros(len(x))
    for i in range(len(xs)):
        temp = ys[i] * np.ones(len(x))
        for j in range(len(xs)):
            if j == i:
                continue
            temp *= ((x - xs[j]) / (xs[i] - xs[j]))
        result += temp
    return result

# Spline Interpolation

def spline(xs : np.array, ys : np.array, x : np.array):
    '''
    Implements cubic spline interpolation, evaluates the spline at each point in x.
    
    Parameters
    ----------
    xs : np.array
        x-coordinates of the interpolation points
    ys : np.array
        y-coordinates of the interpolation points
    x : np.array
        x-coordinates where the spline is evaluated
        
    Returns
    ----------
    np.array
        Values of the cubic spline at each point in x
    '''
    pass
