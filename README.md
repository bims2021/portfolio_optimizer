# portfolio_optimizer
Modern portfolio theory requires sophisticated mathematical techniques that align perfectly with machine learning fundamentals. This project bridges the gap between theoretical ML mathematics and practical financial applications, demonstrating how concepts like eigenvalue decomposition, gradient optimization, and Bayesian inference solve real-world investment problems.

This project presents a comprehensive Advanced Portfolio Optimization System that serves as a practical showcase of machine learning mathematics. The system demonstrates the real-world application of linear algebra, calculus, probability theory, statistics, and optimization techniques in quantitative finance. Rather than implementing isolated mathematical concepts, this project integrates all major ML mathematical foundations into a cohesive financial application.

The system encompasses:

Multi-asset portfolio construction and optimization
Risk analysis using advanced statistical methods
Uncertainty quantification through Monte Carlo simulation
Performance attribution using mathematical derivatives
Stress testing via Taylor approximations
Bayesian parameter estimation.

2. Mathematical Foundations Implemented
2.1 Linear Algebra Applications
Eigenvalue Decomposition for Risk Factor Analysis

Implementation: Decomposed the covariance matrix Σ = QΛQ^T to identify principal risk factors
Mathematical Significance: Eigenvalues represent the magnitude of risk along each principal component
Practical Impact: Enabled risk factor attribution and dimensionality reduction
Code Example: Used scipy.linalg.eig() to compute eigenvalues and eigenvectors of the covariance matrix

Singular Value Decomposition (SVD) for PCA

Implementation: Applied SVD to standardized returns matrix X = UΣV^T
Mathematical Significance: Identified orthogonal factors explaining maximum variance
Practical Impact: Reduced dimensionality from individual assets to risk factors
Validation: Verified that principal components are uncorrelated (orthogonal)

Matrix Condition Numbers for Numerical Stability

Implementation: Computed condition number κ(Σ) = λ_max/λ_min
Mathematical Significance: Assessed numerical stability of matrix inversions
Practical Impact: Identified when covariance matrices are ill-conditioned
Threshold: Flagged condition numbers > 10^12 as potentially problematic

2.2 Calculus and Optimization
Gradient-Based Portfolio Optimization

Implementation: Minimized portfolio variance subject to return and budget constraints
Objective Function: f(w) = ½w^T Σw - λμ^T w (risk-adjusted return)
Gradient: ∇f(w) = Σw - λμ
Hessian: H = Σ (positive definite, ensuring convex optimization)

Newton-Raphson Method

Implementation: w_{k+1} = w_k - H^{-1}∇f(w_k)
Mathematical Advantage: Quadratic convergence near optimum
Practical Benefit: Faster convergence than gradient descent for well-conditioned problems
Numerical Considerations: Added regularization for singular Hessian matrices

Risk Attribution Using Partial Derivatives

Implementation: Computed marginal risk contribution ∂σ_p/∂w_i = (Σw)_i/σ_p
Mathematical Insight: Each asset's contribution to total portfolio risk
Practical Application: Identified which positions drive portfolio volatility
Verification: Confirmed that Σ(w_i × ∂σ_p/∂w_i) = σ_p (Euler's theorem)

2.3 Probability Theory and Statistics
Maximum Likelihood Estimation (MLE)

Implementation: Estimated μ̂ = X̄ and Σ̂ = (X-μ̂)^T(X-μ̂)/n for multivariate normal
Log-likelihood: ℓ(θ) = -½[n log(2π) + log|Σ| + tr(Σ^{-1}S)]
Comparison: Contrasted MLE vs. sample statistics (n vs. n-1 denominator)
Validation: Computed actual log-likelihood values for parameter estimates

Hypothesis Testing Framework

Normality Tests: Applied Shapiro-Wilk test to validate return distribution assumptions
Mean Testing: Used one-sample t-tests to test H₀: μ = 0
Confidence Intervals: Constructed 95% CIs using t-distribution
Multiple Testing: Applied appropriate corrections for multiple assets

Monte Carlo Simulation for Risk Assessment

Implementation: Generated 10,000 scenarios using multivariate normal distribution
Value-at-Risk (VaR): Computed 5th percentile of portfolio returns
Expected Shortfall: Calculated average loss beyond VaR threshold
Validation: Verified that empirical distribution matches theoretical moments

2.4 Advanced Optimization Techniques
Quadratic Programming with Constraints

Formulation: min ½w^T Σw subject to w^T 1 = 1, w^T μ ≥ r_target, w ≥ 0
Lagrangian: L = ½w^T Σw + λ₁(1 - w^T 1) + λ₂(r_target - w^T μ) - λ₃^T w
KKT Conditions: Implemented necessary and sufficient optimality conditions
Solver: Used SLSQP algorithm for constraint handling

Efficient Frontier Construction

Mathematical Basis: Parametric solution to mean-variance optimization
Implementation: Solved for 50 different target returns
Theoretical Foundation: Two-fund theorem and capital allocation line
Visualization: Plotted risk-return trade-off curve


3. Advanced Mathematical Extensions
3.1 Bayesian Portfolio Theory (Black-Litterman Model)
Mathematical Framework
The Black-Litterman model combines market equilibrium with investor views using Bayesian updating:
      μ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} [(τΣ)^{-1}Π + P^T Ω^{-1} Q]
      Σ_BL = [(τΣ)^{-1} + P^T Ω^{-1} P]^{-1} + Σ
Where:

Π = λΣw_market (implied equilibrium returns)
P = picking matrix (which assets have views)
Q = investor views on returns
Ω = uncertainty matrix for views
τ = scaling parameter

Implementation Highlights

Estimated market risk aversion λ from Sharpe ratio
Computed implied equilibrium returns from market cap weights
Demonstrated Bayesian updating mechanism
Showed how prior beliefs (market equilibrium) combine with new information (views)

3.2 Kelly Criterion for Position Sizing
Mathematical Foundation
The Kelly formula maximizes logarithmic utility:
  f* = (bp - q) / b
Where:

f* = optimal fraction of capital to invest
b = odds received (average win / average loss)
p = probability of winning
q = probability of losing (1-p)

Statistical Estimation

Estimated win probability from historical return frequency
Computed average win/loss ratios from empirical data
Applied Kelly criterion to determine optimal position sizes
Validated against theoretical log-optimal growth rate

3.3 Regime Switching Models
Statistical Framework
Implemented a simplified Markov-switching model:

Identified high/low volatility regimes using rolling windows
Estimated regime-specific parameters (μ_regime, σ_regime)
Computed regime transition probabilities
Applied maximum likelihood estimation for parameter fitting

3.4 Covariance Shrinkage (Ledoit-Wolf Estimator)
Mathematical Approach
Combined sample covariance with structured shrinkage target:
   Σ_shrunk = α * F + (1-α) * S
Where:

S = sample covariance matrix
F = shrinkage target (scalar identity matrix)
α = optimal shrinkage intensity (estimated via Ledoit-Wolf formula)

Numerical Benefits

Improved condition number from 1847.3 to 156.2
Reduced estimation error in small samples
Enhanced numerical stability for matrix operations.
