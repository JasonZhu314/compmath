import numpy as np

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

def band_gaussian_elimination(A : np.array, b : np.array, k : int):
    '''
    Solves the linear system Ax = b using Gaussian elimination for band matrices.
    A is a band matrix with bandwidth k, meaning that A[i, j] = 0 for |i - j| > k.
    
    Parameters
    ----------
    A : np.array
        Band matrix to be solved
    b : np.array
        Right-hand side matrix
    k : int
        Bandwidth of the matrix A
    
    Returns
    ----------
    np.array
        Solution matrix x such that Ax = b
    '''
    n = A.shape[0]
    At = np.copy(A)
    res = np.copy(b)
    for i in range(n - 1):
        res[i + 1 : min(n, i + 1 + k)] -= res[i] / At[i, i] * At[i + 1 : min(n, i + 1 + k), [i]]
        At[i + 1 : min(n, i + 1 + k), i + 1 : min(n, i + 1 + k)] -= (At[i + 1 : min(n, i + 1 + k), [i]] / At[i, i]) * At[[i], i + 1 : min(n, i + 1 + k)]
    res[n - 1] /= At[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        res[i] -= np.dot(At[i, i + 1 : min(n, i + 1 + k)], res[i + 1 : min(n, i + 1 + k)])
        res[i] /= At[i, i]
    return res

def column_pivoting_gaussian_elimination(A : np.array, b : np.array):
    '''
    Solves the linear system Ax = b using Gaussian elimination with column pivoting.
    Parameters
    ----------
    A : np.array
        Matrix to be solved
    b : np.array
        Right-hand side matrix
    
    Returns
    ----------
    np.array
        Solution matrix x such that Ax = b
    '''
    n = A.shape[0]
    At = np.copy(A)
    res = np.copy(b)
    for i in range(n - 1):
        idx = i + np.argmax(abs(At[i :, i]))
        if idx != i:
            res[[i, idx]] = res[[idx, i]]
            At[[i, idx]] = At[[idx, i]]
        At[i + 1 :, i + 1 :] -= (At[i + 1 :, [i]] / At[i, i]) * (At[[i], i + 1 :])
        res[i + 1 :] -= res[i] / At[i, i] * At[i + 1 :, [i]]
    for i in range(n - 1, -1, -1):
        res[i] -= At[i, i + 1 :] @ res[i + 1 :]
        res[i] /= At[i, i]
    return res

# Cholesky Decomposition

def cholesky(A : np.array):
    '''
    Implements Cholesky decomposition.
    Decomposes symmetric matrix A into a product of a lower triangular matrix L and its transpose, such that A = LL^T.
    
    Parameters
    ----------
    A : np.array
        Symmetric matrix to be decomposed
        
    Returns
    ----------
    np.array
        Decomposed matrix L such that A = LL^T
    '''
    pass

# Improved Cholesky Decomposition

def cholesky_improved(A : np.array):
    '''
    Implements Cholesky decomposition.
    Decomposes symmetric matrix A into a product of the form A = LDL^T, where L is a unit lower triangular matrix and D is a diagonal matrix.
    
    Parameters
    ----------
    A : np.array
        Symmetric matrix to be decomposed
        
    Returns
    ----------
    [L, D] : list
        Decomposed matrix L and D such that A = LDL^T
    '''
    pass

# Solving Linear Systems

def Lxb(L : np.array, b : np.array):
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
    x = np.zeros_like(b)
    bt = np.copy(b)
    for i in range(n):
        x[i] = bt[i]
        bt[i + 1 : ] -= x[i].reshape(1, -1) * L[i + 1 : , i].reshape(-1, 1)
    return x

def Uxb(U : np.array, b : np.array):
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
    x = np.zeros_like(b)
    bt = np.copy(b)
    for i in range(n - 1, -1, -1):
        x[i] = bt[i] / U[i, i]
        bt[: i] -= x[i].reshape(1, -1) * U[: i, i].reshape(-1, 1)
    return x

def Axb(A : np.array, b : np.array):
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
