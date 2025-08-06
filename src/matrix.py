import numpy as np
import matplotlib.pyplot as plt

# LU Decomposition

def LU(A : np.array):
    '''
    Implements LU decomposition.
    Decomposes matrix A into unit lower triangular matrix L and upper triangular matrix U such that A = LU.
    Store the factors in a single matrix, where L is stored in the lower triangular part and U in the upper triangular part.
    
    Parameters
    ----------
    A : np.array
        Matrix to be decomposed
        
    Returns
    ----------
    np.array
        Decomposed matrix with L in the lower triangular part and U in the upper triangular part
    '''
    n = A.shape[0]
    result = np.copy(A)
    for i in range(n - 1):
        np.divide(result[i + 1 : n, i], result[i, i], out=result[i + 1 : n, i])
        result[i + 1 : n, i + 1 : n] -= np.outer(result[i + 1 : n, i], result[i, i + 1 : n])
    return result

# Gaussian Elimination

# Cholesky Decomposition

# Solving Linear Systems

def Lxb(L, b):
    '''
    Solves the linear system Lx = b, where L is a unit lower triangular matrix.
    
    Parameters
    ----------
    L : np.array
        Unit lower triangular matrix
    b : np.array
        Right-hand side matrix
    
    Returns
    ----------
    np.array
        Solution matrix x such that Lx = b
    '''
    n = b.shape[0]
    x = np.zeros(b.shape)
    bt = np.copy(b)
    for i in range(n):
        x[i] = bt[i]
        bt[i + 1 : ] -= x[i].reshape(1, -1) * L[i + 1 : , i].reshape(-1, 1)
    return x

def Uxb(U, b):
    '''
    Solves the linear system Ux = b, where U is an upper triangular matrix.
    
    Parameters
    ----------
    U : np.array
        Upper triangular matrix
    b : np.array
        Right-hand side matrix
        
    Returns
    ----------
    np.array
        Solution matrix X such that UX = B
    '''
    n = b.shape[0]
    x = np.zeros(b.shape)
    bt = np.copy(b)
    for i in range(n - 1, -1, -1):
        x[i] = bt[i] / U[i, i]
        bt[: i] -= x[i].reshape(1, -1) * U[: i, i].reshape(-1, 1)
    return x

def Axb(A, b):
    '''
    Solves the linear system Ax = b using LU decomposition.
    Decomposes matrix A into L and U, then solves the systems Ly = b and Ux = y.
    
    Parameters
    ----------
    A : np.array
        Matrix to be decomposed
    b : np.array
        Right-hand side matrix
    
    Returns
    ----------
    np.array
        Solution matrix x such that Ax = b
    '''
    ALU = LU(A)
    y = Lxb(ALU, b)
    x = Uxb(ALU, y)
    return x
