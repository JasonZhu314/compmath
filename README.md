# CompMath - Python implementations of computational mathematics algorithms

CompMath is a collection of my implementations of important computational mathematics algorithms when studying numerical analysis and numerical linear algebra. It uses Numpy for the algorithms, MatPlotLib for plotting and SciPy for testing results and benchmarking.

This code library is for educational purposes only and obviously not to be used anywhere else, since it contains from-scratch pure NumPy implementations and is not optimized for speed.

## Status

### Polynomials

- [ ] Chebyshev Polynomials
- [ ] Legendre Polynomials
- [ ] Hermite Polynomials
- [ ] Laguerre Polynomials
- [ ] Bernoulli Polynomials

### Interpolation

- [x] Horner's Algorithm
- [x] Newton Interpolation
- [x] Lagrange Interpolation
- [ ] Barycentric Lagrange Interpolation
- [x] Piecewise Linear Interpolation
- [x] Cubic Spline Interpolation
- [ ] B-spline Interpolation
- [ ] Bezier Curves

### Approximation

- [ ] Remez Algorithm
- [ ] Least Squares Approximation
- [ ] Mean Square Approximation
- [ ] Padé Approximation
- [ ] Sparse Grids

### Numerical Integration

- [x] Midpoint Rule
- [x] Trapezoidal Rule
- [x] Simpson's Rule
- [x] Romberg Integration
- [x] Adaptive Quadrature
- [ ] Gaussian Quadrature
  - [ ] Legendre-Gauss Quadrature
  - [ ] Chebyshev-Gauss Quadrature
  - [ ] Hermite-Gauss Quadrature
  - [ ] Laguerre-Gauss Quadrature

### Numerical Differentiation

- [ ] Forward Difference Method
- [ ] Backward Difference Method
- [ ] Central Difference Method
- [ ] Higher-Order Difference Methods
- [ ] Richardson Extrapolation
- [ ] Compact Finite Difference Schemes
- [ ] Implicit Differentiation Methods
- [ ] Interpolation-Based Differentiation Methods
- [ ] Spectral Differentiation Methods

### Nonlinear Equations

- [ ] Bisection Method
- [ ] Newton's Method
- [ ] Secant Method
- [ ] Steppen's Method
- [ ] Fixed-Point Iteration
- [ ] Nonlinear Jacobi's Method
- [ ] Nonlinear Gauss-Seidel Method
- [ ] Nonlinear SOR Method
- [ ] Broyden's Method
- [ ] Homotopy Continuation Method

### Transforms

- [ ] DFT
- [ ] DCT
- [ ] DST
- [ ] DQCT
- [ ] NTT
- [ ] Convolution
- [ ] Solving Circulant Systems
- [ ] Fast Gaussian Transform (FGT)
- [ ] Fast Multipole Method (FMM)

### ODEs

- [ ] Euler's Method
  - [ ] Forward Euler Method
  - [ ] Backward Euler Method
- [ ] Explicit Runge-Kutta Methods
  - [ ] Improved Euler Method
  - [ ] Heun's Method
  - [ ] RK4 Method
- [ ] Implicit Runge-Kutta Methods
  - [ ] Midpoint Method
- [ ] BVP Solvers
  - [ ] Shooting Method
  - [ ] Finite Difference Method
- [ ] Linear Multi-Step Methods
- [ ] Predictor-Corrector Methods
- [ ] Explicit Integration Methods
- [ ] Symplectic Schemes
- [ ] Verlet Integration Method
- [ ] Leapfrog Method
- [ ] Velocity Verlet Method

### PDEs

#### Finite Difference Methods

- [ ] Forward Difference Method
- [ ] Backward Difference Method
- [ ] Central Difference Method
- [ ] Crank-Nicolson Method
- [ ] Lax-Friedrichs Method
- [ ] Lax-Wendroff Method
- [ ] MacCormack Method
- [ ] Upwind Method
- [ ] WENO Method

#### Finite Element Methods

- [ ] Galerkin Method
- [ ] Petrov-Galerkin Method
- [ ] Discontinuous Galerkin Method
- [ ] Mixed Finite Element Method
- [ ] Spectral Element Method
- [ ] Isogeometric Analysis

#### Finite Volume Methods

- [ ] Godunov's Method
- [ ] MUSCL Scheme
- [ ] Roe's Approximate Riemann Solver
- [ ] HLLC Riemann Solver
- [ ] TVD Schemes
- [ ] ENO Schemes
- [ ] WENO Schemes

#### Spectral Methods

- [ ] Fourier Spectral Method
- [ ] Chebyshev Spectral Method
- [ ] Legendre Spectral Method
- [ ] Hermite Spectral Method
- [ ] Laguerre Spectral Method

#### Parabolic PDEs

- [ ] Heat Equation
- [ ] Diffusion Equation
- [ ] Black-Scholes Equation

#### Hyperbolic PDEs

- [ ] Transport Equation
- [ ] Burgers' Equation
- [ ] KdV Equation
- [ ] Wave Equation
- [ ] Advection Equation
- [ ] Euler Equations
- [ ] Navier-Stokes Equations

#### Elliptic PDEs

- [ ] Poisson's Equation
- [ ] Laplace's Equation
- [ ] Helmholtz Equation
- [ ] Schrödinger Equation

### Stochastic Methods

- [ ] Random Number Generation
  - [ ] Linear Congruential Generator
  - [ ] Middle Square Method
  - [ ] 16807 Generator
  - [ ] Van der Corput Sequence
  - [ ] Halton Sequence
  - [ ] Sobol Sequence
  - [ ] Faure Sequence
  - [ ] Transformation Methods
  - [ ] Box-Muller Transform
  - [ ] Acceptance-Rejection Method
- [ ] Sampling Methods
  - [ ] Inverse Transform Sampling
  - [ ] Box-Muller Transform
  - [ ] Rejection Sampling
  - [ ] Gibbs Sampling
  - [ ] Metropolis-Hastings Algorithm
  - [ ] Glauber Dynamics
- [ ] Markov Chains
  - [ ] Transition Matrix Construction
  - [ ] Stationary Distribution Calculation
  - [ ] Mixing Time Estimation
- [ ] Monte Carlo Method
- [ ] Quasi-Monte Carlo Method
- [ ] Variance Reduction Techniques
  - [ ] Antithetic Variates
  - [ ] Control Variates
  - [ ] Importance Sampling
- [ ] MCMC
- [ ] Simulated Annealing
- [ ] Ising Model Simulation
- [ ] Percolation Simulation
- [ ] Random Walk Simulation
- [ ] Brownian Motion Simulation

### SDE

- [ ] Euler-Maruyama Method

### Optimization

- [ ] Gradient Descent
  - [ ] Batch Gradient Descent
  - [ ] Stochastic Gradient Descent
  - [ ] Mini-Batch Gradient Descent
- [ ] Momentum-Based Gradient Descent
  - [ ] Nesterov Accelerated Gradient (NAG)
- [ ] Adagrad
- [ ] RMSProp
- [ ] Adam
- [ ] Adadelta
- [ ] AdamW
- [ ] AMSGrad
- [ ] L-BFGS
- [ ] Proximal Gradient Methods
  - [ ] Proximal Gradient Descent
  - [ ] Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
- [ ] Coordinate Descent
- [ ] Stochastic Variance Reduced Gradient (SVRG)
- [ ] SAGA
- [ ] Catalyst Acceleration
- [ ] Muon Acceleration

### Linear Algebra

#### Direct Methods

- [x] LU Decomposition
- [x] Solving Linear Systems
- [x] Cholesky Decomposition
- [x] Improved Cholesky Decomposition
- [x] Gaussian Elimination
- [x] Thomas Algorithm
- [x] Banded Gaussian Elimination
- [ ] Pivoting Strategies
  - [x] Column Pivoting
  - [ ] Complete Pivoting

#### Iterative Methods

- [ ] Jacobi Method
- [x] Gauss-Seidel Method
- [ ] Successive Over-Relaxation (SOR) Method
- [ ] Conjugate Gradient Method
- [ ] Preconditioned Conjugate Gradient Method
- [ ] V-Cycle Multigrid Method

#### Error Analysis

- [ ] Precision Estimation

#### Eigenvalues and Eigenvectors

- [ ] Power Iteration
- [ ] Inverse Power Iteration
- [ ] Schur Decomposition
- [ ] QR Algorithm
- [ ] Hessenberg Reduction
- [ ] SVD
