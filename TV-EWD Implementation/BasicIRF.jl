using LinearAlgebra
# Function to compute residuals
"""
    compute_residuals(response_data::Vector{Float64}, regressor_data::Matrix{Float64}, coefficients::Vector{Float64}) -> Vector{Float64}

Compute the residuals from a simple linear model (through the origin).

# Arguments
- `response_data::Vector{Float64}`: A vector of observed dependent (response) values.
- `regressor_data::Matrix{Float64}`: A matrix whose columns correspond to regressors (predictor variables) and whose rows correspond to observations. Assumed to be conformable with `coefficients`.
- `coefficients::Vector{Float64}`: A vector of parameter estimates obtained (e.g., by ordinary least squares) for the model through the origin.

# Returns
- `residuals::Vector{Float64}`: A vector of the same length as `response_data`, containing the difference between observed responses and fitted values (`response_data - regressor_data * coefficients`).

# Example
```julia
y = [1.0, 2.0, 3.0]
X = [1.0 0.5;
     2.0 1.5;
     3.0 2.5]
β = [0.2, 0.8]
r = compute_residuals(y, X, β)  # returns y - X * β
"""
function compute_residuals(response_data::Vector{Float64}, regressor_data::Matrix{Float64}, coefficients::Vector{Float64})::Vector{Float64}
    residuals = response_data - regressor_data * coefficients
    return residuals
end


# Function to compute standard deviation of residuals
"""
    compute_standard_deviation(residuals::Vector{Float64}, max_lag::Int) -> Float64

Calculte the standard error of residuals, accounting for degrees of freedom (adjusted by a lag parameter).

# Arguments
- `residuals::Vector{Float64}`: A vector of residuals (errors) from a fitted model.
- `max_lag::Int`: The maximum lag used in fitted AR(p) model. Subtracts `max_lag` from the sample size to adjust the denominator when computing variance.

# Returns
- `σ::Float64`: The sample standard deviation of the `residuals`, computed as `sqrt((residuals' * residuals) / (length(residuals) - max_lag))`.

# Notes
- If `length(residuals) ≤ max_lag`, the denominator will be zero or negative; ensure that `length(residuals) > max_lag` before calling.

# Example
```julia
e = [0.5, -0.1, 0.3, -0.2]
σ = compute_standard_deviation(e, 1)  # computes sqrt((0.5^2 + (-0.1)^2 + 0.3^2 + (-0.2)^2) / (4 - 1))
"""
function compute_standard_deviation(residuals::Vector{Float64}, max_lag::Int)::Float64
    variance = (residuals' * residuals) / (length(residuals) - max_lag)
    return sqrt(variance)
end
# Function to compute Impulse Response Function (IRF) coefficients (alphas) using OLS estimates
"""
    IRF_alpha(
        response_data::Vector{Float64},
        regressor_data::Matrix{Float64},
        max_lag::Int,
        horizon::Int
    ) -> Tuple{Vector{Float64}, Vector{Float64}}

Compute impulse response function (IRF) coefficients and standardized residuals from an OLS regression through the origin.

This routine performs the following steps:
1. Estimate OLS coefficients β by solving `(X'X) "\" (X'y)`.
2. Compute residuals `ε = y - X * β`.
3. Estimate the residual standard deviation `σ = compute_standard_deviation(ε, max_lag)`.
4. Standardize the residuals by dividing by `σ`.
5. Build an IRF coefficient vector `α` of length `horizon`, where
   - `α[1] = σ`.
   - For each subsequent period `t ∈ 2:horizon`,  
     `α[t] = sum_{lag = max(t-1 - max_lag, 0)}^{t-2} α[lag + 1] * β[t-1 - lag]`.

# Arguments
- `response_data::Vector{Float64}`: Observed dependent variable (column vector) of length `n`.
- `regressor_data::Matrix{Float64}`: Design matrix `X` of size `n × p`, where each row corresponds to one observation and each column to one regressor. The model is assumed to pass through the origin (no intercept).
- `max_lag::Int`: The maximum lag to be used when estimating the residual standard deviation; also the maximum order of lags that the IRF recursion will reference.
- `horizon::Int`: The number of periods over which to compute IRF coefficients. Must be ≥ 1.

# Returns
- `IRFcoefficients::Vector{Float64}`: A length‐`horizon` vector of impulse response coefficients.  
  - `IRFcoefficients[1]` is set to `σ`.  
  - For `t = 2:horizon`, `IRFcoefficients[t]` is computed by convolving past IRF values with the OLS coefficients, truncated by `max_lag`.
- `standardized_residuals::Vector{Float64}`: The residuals divided by `σ`, i.e., `ε / σ`.

# Details
1. **OLS through the origin**  
   The regression coefficients are computed as  
   ```julia
   coefficients = (regressor_data' * regressor_data) "\" (regressor_data' * response_data)

"""
function IRF_alpha(response_data::Vector{Float64}, regressor_data::Matrix{Float64}, max_lag::Int, horizon::Int)::Tuple{Vector{Float64}, Vector{Float64}}
    # Estimate coefficients using OLS through origin
    coefficients = (regressor_data' * regressor_data) \ (regressor_data' * response_data)
    
    # Compute residuals
    residuals = compute_residuals(response_data, regressor_data, coefficients)
    
    # Estimate the standard deviation of residuals
    sigma = compute_standard_deviation(residuals, max_lag)
    
    # Normalize residuals (Eps)
    standardized_residuals = residuals ./ sigma

    # Initialize the alphaR vector for IRF coefficients
    IRFcoefficients = zeros(Float64, horizon)
    IRFcoefficients[1] = sigma

    # Calculate IRF for each period
    for period in 1:(horizon - 1)
        hstart = max(period - max_lag, 0)
        temp_sum = 0.0

        for lag in hstart:(period - 1)
            temp_sum += IRFcoefficients[lag + 1] * coefficients[period - lag]
        end

        IRFcoefficients[period + 1] = temp_sum
    end

    return (IRFcoefficients, standardized_residuals)
end