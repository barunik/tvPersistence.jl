using LinearAlgebra

include(joinpath(@__DIR__, "..", "models", "tvOLS.jl"))
using .tvOLS_estimator  # Importing local tvOLS module

# Helper function to compute residuals
function compute_residuals(response_data, regressor_data, coefficients::Matrix{Float64})::Vector{Float64}
    residuals = zeros(length(response_data))
    for i in 1:length(response_data)

        # Calculate residuals for each observation
        residuals[i] = response_data[i] - sum(coefficients[i, :] .* regressor_data[i, :])
    end
    return residuals
end

# Helper function to compute standard deviation of residuals
function compute_standard_deviation(residuals::Vector{Float64}, max_lag::Int)::Float64
    variance = (residuals' * residuals) ./ (length(residuals) - max_lag)
    return sqrt.(variance)
end

# General function to compute time-varying IRF coefficients
"""
    compute_time_varying_irf(
        coeffs::Matrix{Float64}, 
        sigma::Float64, 
        max_lag::Int, 
        horizon::Int, 
        period::Int = size(coeffs, 1)
    ) -> Vector{Float64}

Compute impulse response function (IRF) coefficients at a specific point in time using time-varying AR coefficients.

# Arguments
- `coeffs::Matrix{Float64}`: Matrix of time-varying AR coefficients (each row corresponds to time `t`). Those are obtained as output from tvOLS()
- `sigma::Float64`: Standard deviation of residuals used to scale the IRF.
- `max_lag::Int`: AR model lag order.
- `horizon::Int`: Length of IRF to compute.
- `period::Int`: Specific row (time point) of coefficients to use for IRF (default: last).

# Returns
- `irf_coefficients::Vector{Float64}`: Vector of IRF coefficients of length `horizon`, with initial value set to `sigma`.
"""
function compute_time_varying_irf(coeffs::Matrix{Float64}, sigma::Float64, max_lag::Int, horizon::Int, period::Int = size(coeffs, 1))::Vector{Float64}
    irf_coefficients = zeros(Float64, horizon)
    irf_coefficients[1] = sigma

    for n in 1:(horizon - 1)
        hstart = max(n - max_lag, 0)
        temp_sum = 0.0

        for h in hstart:(n - 1)
            temp_sum += irf_coefficients[h + 1] * coeffs[period, n - h]  # Using  OLS coefficients for specified period
        end

        irf_coefficients[n + 1] = temp_sum
    end

    return irf_coefficients
end

# Main function to compute time-varying IRF using localized linear LS estimation
"""
    IRFalpha_tvp(
        y::Vector, 
        x::Matrix, 
        max_lag::Int, 
        horizon::Int, 
        kernel_width::Float64, 
        kernel_type::String = "Gaussian"
    ) -> Tuple{Matrix{Float64}, Vector{Float64}}

Compute time-varying impulse response functions (IRFs) using custom local linear estimation.

# Arguments
- `y::Vector`: Dependent variable (response) vector.
- `x::Matrix`: Lag matrix or regressor matrix.
- `max_lag::Int`: Lag order of the AR model.
- `horizon::Int`: Number of periods to compute for each IRF.
- `kernel_width::Float64`: Bandwidth for local kernel regression.
- `kernel_type::String`: Kernel type to use ("Gaussian", "Epanechnikov", "one-sided").

# Returns
- `irf_matrix::Matrix{Float64}`: Matrix of IRFs, with one row per time `t` and one column per horizon `h`.
- `eps_tvp::Vector{Float64}`: Standardized residuals (residuals divided by σ̂).

# Notes
- This method estimates time-varying AR coefficients using custom `tvOLS`, then computes IRFs recursively at each time point.
- Used for analyzing dynamic persistence of shocks across time.
"""
function IRFalpha_tvp(
    y, 
    x, 
    max_lag::Int, 
    horizon::Int, 
    kernel_width::Float64, 
    kernel_type::String = "Gaussian"
)::Tuple{Matrix{Float64}, Vector{Float64}}

    # Estimate time-varying coefficients using custom tvOLS
    coeffs_tvp = tvOLS_estimator.tvOLS(x, y, kernel_width, kernel_type).coefficients

    # Compute residuals for each time step
    residuals_tvp = compute_residuals(y, x, coeffs_tvp)

    # Compute standard deviation of residuals
    sigma_tvp = compute_standard_deviation(residuals_tvp, max_lag)

    # Normalize residuals
    eps_tvp = residuals_tvp ./ sigma_tvp

    # Initialize matrix to store IRF results: rows for periods, columns for horizons
    irf_matrix = zeros(Float64, size(coeffs_tvp, 1), horizon)

    # Compute IRF coefficients for each period using the recursive function
    #irf_results = []
    for period in 1:size(coeffs_tvp, 1)
        irf_matrix[period, :] = compute_time_varying_irf(coeffs_tvp, sigma_tvp, max_lag, horizon, period)
    end

    return (irf_matrix, eps_tvp)
end