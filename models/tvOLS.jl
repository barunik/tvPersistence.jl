module tvOLS_estimator

using Pkg
using LinearAlgebra, Statistics

# Activate the project environment in the current directory (".")
Pkg.activate(".")

# Instantiate the environment, which installs exact versions of dependencies
Pkg.instantiate()

############################################################################
## Time-varying OLS model estimation functionality ## --------------------
############################################################################

# Standard OLS estimation
function OLSestimator(y,x)
    return (transpose(x)*x) \ (transpose(x)*y)
end

"""
    kernel(t::Vector{Float64}, bw::Float64, tkernel::String) -> Vector{Float64}

Compute kernel weights based on the specified kernel type and bandwidth.

# Arguments
- `t::Vector{Float64}`: A vector of distances from the target point. e.g: period 8 has a distance equal to 6 from period 14
- `bw::Float64`: The bandwidth parameter ( h ) that controls the width of the kernel function.
- `tkernel::String`: The type of kernel function to use. Options include `"Gaussian"` and `"Epanechnikov"`.

# Returns
- `Vector{Float64}`: A vector of kernel weights corresponding to the input distances `t`, with weights adjusted by the specified bandwidth and kernel type.

# Purpose
The kernel function computes the weights for each point based on its distance from a target point using a specified kernel type. This is used in local weighted regression to provide the weighting rule based on distance from the period of interest.
"""

function kernel(t, bw, tkernel)
    z = t / bw
    if tkernel == "Gaussian"
        return exp.(-0.5 * z.^2) / sqrt(2 * pi)
    elseif tkernel == "Epa" || tkernel == "Epanechnikov"
        return max.(0, 0.75 * (1 .- z.^2))
    elseif tkernel == "one-sided"
        return (z.<=0).*exp.(-0.5 * z.^2)
    else
        error("Unknown kernel type")
    end
end


"""
    tvOLS(x::Matrix{Float64}, y::Vector{Float64}, bw::Float64; tkernel::String = "Gaussian") -> NamedTuple

Estimate time-varying OLS regression coefficients using local linear regression.

# Arguments
- `x::Matrix{Float64}`: A matrix of size `(n, p)`, where `n` is the number of observations and `p` is the number of predictor variables. Each row corresponds to an observation, and each column corresponds to a predictor variable.
- `y::Vector{Float64}`: A vector of length `n` containing the response variable.
- `bw::Float64`: The bandwidth parameter ( h ) that determines the width of the kernel function.
- `tkernel::String`: The type of kernel function to use. Options include `"Gaussian"` and `"Epanechnikov"`. Default is `"Gaussian"`.

# Returns
- `NamedTuple`: A tuple containing:
  - `coefficients::Matrix{Float64}`: A matrix of size `(n, p)` where each row contains the time-varying OLS coefficients for each time period.
  - `fitted::Vector{Float64}`: A vector of length `n` containing the fitted values for each time period.
  - `residuals::Vector{Float64}`: A vector of length `n` containing the residuals (difference between observed and fitted values) for each time period.

# Purpose
The `tvOLS` function performs local linear regression to estimate time-varying coefficients. For each time period, it computes kernel weights for nearby observations, fits a weighted linear model, and returns the coefficients, fitted values, and residuals.
"""

function tvOLS(x, y, bw::Float64,
               tkernel::String = "Gaussian")

    # Ensure dimensions of x and y match
    obs = size(x, 1)
    if length(y) != obs
        error("Dimensions of 'x' and 'y' are not compatible.")
    end

    nvar = size(x, 2)  # Number of variables (columns of x)
    
    # Initialize containers for results
    theta = zeros(obs, nvar)  # Coefficients
    fitted = zeros(obs)  # Fitted values
    residuals = zeros(obs)  # Residuals
    
    # Calculate the rescaled distances of time points
    grid = collect(1:obs) / obs
    
    # Estimate the local regression coefficients for all periods
    for t in 1:obs

        # Compute the kernel weights for the current time point t
        tau0 = grid .- grid[t]  # Distance from time t to all other points
        kernel_weights = kernel(tau0, bw, tkernel)

        # Select only the points with non-zero kernel weights
        k_idx = findall(kernel_weights .> 0)

        if length(k_idx) < 1
            error("Bandwidth too small for 'bw'.")
        end

        # Weighted least squares regression for time t
        x_temp = x[k_idx, :]
        y_temp = y[k_idx]
        w_temp = kernel_weights[k_idx]

        # Perform the weighted OLS
        XW = x_temp .* sqrt.(w_temp)  # Apply square root of weights to X
        yW = y_temp .* sqrt.(w_temp)  # Apply square root of weights to y
        #coef = XW \ yW  # OLS: Solve for coefficients with alternative method
        coef = (XW' * XW) \ (XW' * yW)

        # Store the time-varying coefficients for time t
        theta[t, :] = coef'
        
        # Compute fitted values and residuals
        fitted[t] = dot(x[t, :], coef)
        residuals[t] = y[t] - fitted[t]
    end

    # Return the time-varying coefficients, fitted values, and residuals
    return (coefficients = theta, fitted = fitted, residuals = residuals)
end

"""
    forecast_tvAR(x::Matrix{Float64}, y::Vector{Float64}, bw::Float64, p::Int, n_ahead::Int; tkernel::String = "Gaussian") -> Vector{Float64}

Estimate a Time-Varying AR(p) model using tvOLS and generate n_ahead forecasts based on the latest estimated coefficients.

# Arguments
- `x::Matrix{Float64}`: A matrix of predictors for the AR model, typically consisting of lagged values.
- `y::Vector{Float64}`: A vector of response variables corresponding to `x`.
- `bw::Float64`: The bandwidth parameter for the kernel function used in tvOLS.
- `p::Int`: The order of the AR model (number of lags).
- `n_ahead::Int`: The number of future steps to forecast.
- `tkernel::String`: The type of kernel function to use. Options include `"Gaussian"` and `"Epanechnikov"`. Default is `"Gaussian"`.

# Returns
- `Vector{Float64}`: A vector containing the forecasted values for the next `n_ahead` periods.

# Purpose
The `forecast_tvAR` function fits a time-varying AR(p) model using `tvOLS`, extracts the latest set of AR coefficients, and uses these coefficients to generate forecasts for `n_ahead` future periods. It assumes that the AR coefficients remain constant during the forecast horizon.
"""
function forecast_tvAR(y::Vector{Float64}, p::Int, bw::Float64, n_ahead::Int; tkernel::String = "Gaussian")::Vector{Float64}

    T = length(y)

    if p >= T
        error("AR order p ($p) is too high for the data length ($T).")
    end

    # Construct the lagged design matrix for AR(p)
    # Each row of X_p contains [y[t-p], y[t-p+1], ..., y[t-1]]
    X_p = [y[t-p:t-1] for t in (p+1):T]

    # Convert to a matrix where each column is a lag
    X_p_matrix = hcat(X_p...)'  # Matrix of size (T-p) x p

    # Add a column of ones for the intercept
    X_p_with_intercept = hcat(ones(T-p), X_p_matrix)  # (T-p) x (p+1)

    # Fit the Time-Varying AR(p) model using tvOLS
    result = tvOLS(X_p_with_intercept, y[p+1:T], bw, tkernel)

    coefficients = result.coefficients  # (T-p) x (p+1)

    # Find the last set of valid (non-NaN) coefficients
    last_valid = findlast(row -> !any(isnan, row), eachrow(coefficients))

    if isnothing(last_valid)
        error("No valid coefficients found in tvOLS.")
    end

    latest_coeff = coefficients[last_valid, :]  # Vector of size (p+1)

    # Extract intercept and AR coefficients
    intercept = latest_coeff[1]
    ar_coeffs = latest_coeff[2:end]  # Vector of size p

    # Initialize the history with the last p observations
    history = copy(y[end-p+1:end])

    # Initialize the forecast vector
    forecast = Vector{Float64}(undef, n_ahead)

    for h in 1:n_ahead
        # Compute the forecasted value
        forecast[h] = intercept + dot(ar_coeffs, reverse(history))

        # Update the history with the new forecasted value
        push!(history, forecast[h])
        pop!(history)  # Ensure history remains of length p
    end

    return forecast
end

end