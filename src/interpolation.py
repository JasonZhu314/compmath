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

# Piecewise Linear Interpolation

def piecewise_linear(xs : np.array, ys : np.array, x : np.array):
    '''
    Implements piecewise linear interpolation, evaluates the spline at each point in x.
    
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
        Values of the piecewise linear function at each point in x
    '''
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < xs[0]:
            result[i] = ys[0]
        elif x[i] >= xs[-1]:
            result[i] = ys[-1]
        else:
            idx = np.searchsorted(xs, x[i], side='right')
            result[i] = (x[i] - xs[idx - 1]) / (xs[idx] - xs[idx - 1]) * ys[idx] + (xs[idx] - x[i]) / (xs[idx] - xs[idx - 1]) * ys[idx - 1]
    return result   

# Spline Interpolation

def spline_second_derivative(xs : np.array, ys : np.array):
    '''
    Implements cubic spline interpolation, calculates the second derivative at each interpolation point.
    
    Parameters
    ----------
    xs : np.array
        x-coordinates of the interpolation points
    ys : np.array
        y-coordinates of the interpolation points
        
    Returns
    ----------
    np.array
        Values of second derivative at each interpolation point
    '''
    n = len(xs)
    h = xs[1:] - xs[:-1]
    b = 6 * (ys[1:] - ys[:-1]) / h
    u = (h[:-1] + h[1:]) * 2
    v = (b[1:] - b[:-1])
    z = np.zeros_like(xs)
    
    for i in range(n - 3):
        u[i + 1] -= (h[i + 1] ** 2) / u[i]
        v[i + 1] -= (h[i + 1] * v[i]) / u[i]
        
    for i in range(n - 2, 0, -1):
        z[i] = (v[i - 1] - h[i] * z[i + 1]) / u[i - 1]
    
    return z

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
    n = len(xs)
    h = xs[1:] - xs[:-1]
    b = 6 * (ys[1:] - ys[:-1]) / h
    u = (h[:-1] + h[1:]) * 2
    v = (b[1:] - b[:-1])
    z = np.zeros_like(xs)
    for i in range(n - 3):
        u[i + 1] -= (h[i + 1] ** 2) / u[i]
        v[i + 1] -= (h[i + 1] * v[i]) / u[i]
    for i in range(n - 2, 0, -1):
        z[i] = (v[i - 1] - h[i] * z[i + 1]) / u[i - 1]
        
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < xs[0]:
            result[i] = ys[0]
        elif x[i] >= xs[-1]:
            result[i] = ys[-1]
        else:
            idx = np.searchsorted(xs, x[i], side='right') - 1
            result[i] = z[idx] * ((xs[idx + 1] - x[i]) ** 3) / (6 * h[idx]) + z[idx + 1] * ((x[i] - xs[idx]) ** 3) / (6 * h[idx]) + (ys[idx + 1] / h[idx] - (z[idx + 1] * h[idx]) / 6) * (x[i] - xs[idx]) + (ys[idx] / h[idx] - (z[idx] * h[idx]) / 6) * (xs[idx + 1] - x[i])
    return result

