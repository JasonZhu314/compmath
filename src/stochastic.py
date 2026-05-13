import numpy as np
import matplotlib.pyplot as plt

def sample_spherical_sqrt(n):
    '''Sample points uniformly on the surface of a sphere using the square root method.'''
    z = np.random.uniform(low=-1, high=1, size=(n,))
    r = np.sqrt(1 - z ** 2)
    theta = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
    return np.column_stack((r * np.cos(theta), r * np.sin(theta), z))

def sample_spherical_arccos(n):
    '''Sample points uniformly on the surface of a sphere using the arccos method.'''
    theta1 = np.arccos(np.random.uniform(low=-1, high=1, size=(n,)))
    theta2 = np.random.uniform(low=0, high=2 * np.pi, size=(n,))
    return np.column_stack((np.sin(theta1) * np.cos(theta2), np.sin(theta1) * np.sin(theta2), np.cos(theta1)))

def sample_spherical_gaussian(n):
    '''Sample points uniformly on the surface of a sphere using the Gaussian method.'''
    result = np.random.randn(n, 3)
    result /= np.sqrt(np.sum(result ** 2, axis=1).reshape((n, 1)))
    return result

def metropolis(a, b, num_samples, pdf):
    '''
    Metropolis algorithm to sample from a distribution defined by the pdf function.
    
    Parameters:
    ----------
    a (float): The lower bound of the sampling range.
    b (float): The upper bound of the sampling range.
    num_samples (int): The number of samples to generate.
    pdf (function): The probability density function to sample from.
    
    Returns:
    ----------
    np.ndarray: An array of samples drawn from the distribution defined by the pdf.
    '''
    samples = np.zeros(num_samples)
    x_current = 0.0
    accepted = 0
    for i in range(num_samples):
        x_proposal = np.random.uniform(-10.0, 10.0)
        p_current = pdf(x_current, b)
        p_proposal = pdf(x_proposal, b)
        alpha = min(1.0, p_proposal / p_current)
        if np.random.rand() < alpha:
            x_current = x_proposal
            accepted += 1   
        samples[i] = x_current
    # print(f"Acceptance Rate: {accepted / num_samples:.2%}")
    return samples
