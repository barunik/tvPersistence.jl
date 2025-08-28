"""
TvPersistence — tv_ols.jl

Purpose:
  Local linear (kernel) OLS utilities + TV-AR forecasting.

Defines:
  - kernel(t, bw, kernel_type)
  - tvOLS(x, y, bw; kernel_type="Gaussian", singular_ok=true)
  - OLSestimator(y, X)
  - forecast_tvAR(y, p, bw, n_ahead; kernel_type="Gaussian")

Assumptions:
  - Rows are observations; x and y lengths match.
  - `kernel_type ∈ {"Gaussian","Epanechnikov" (or alias "Epa"),"one-sided","triweight"}`.
"""

# Standard OLS estimation
function replace_infs_and_nans_with_zeros(matrix)
    matrix[isinf.(matrix) .| isnan.(matrix)] .= 0
    return matrix
end

# Standard OLS estimator
function OLSestimator(y, x)
    return replace_infs_and_nans_with_zeros((transpose(x)*x)) \ (transpose(x)*y)
end


"""
    kernel(t::AbstractVector, bw::Real, tkernel::String) -> Vector{Float64}

Compute kernel weights based on the specified kernel type and bandwidth.

# Arguments
- `t::Vector{Float64}`: A vector of distances from the target point. e.g: period 8 has a distance equal to 6 from period 14
- `bw::Float64`: The bandwidth parameter ( h ) that controls the width of the kernel function.
- `tkernel::String`: The type of kernel function to use. Options include `"Gaussian"` and `"Epanechnikov"`.

# Returns
- `Vector{Float64}`: A vector of kernel weights corresponding to the input distances `t`, with weights adjusted by the specified bandwidth and kernel type.

# Purpose
The kernel function computes the weights for each point based on its distance from a target point using a specified kernel type. This is used in local linear regression to provide the weighting rule based on distance from the period of interest.
"""
function kernel(t::AbstractVector, bw::Real, tkernel::String)
    z = t / bw
    if tkernel == "Gaussian"
        return exp.(-0.5 * z.^2) / sqrt(2 * pi)
    elseif tkernel == "Epa" || tkernel == "Epanechnikov"
        return max.(0, 0.75 * (1 .- z.^2))
    elseif tkernel == "one-sided"
        return (z.<=0).*exp.(-0.5 * z.^2)
    elseif tkernel == "triweight"
        return (1 .-z.^2).^3
    else
        error("Unknown kernel type")
    end
end


"""
    tvOLS(x::AbstractMatrix, y::AbstractVector, bw::Float64; tkernel::String = "Gaussian", singular_ok::Bool = true) -> NamedTuple

Estimate time-varying OLS regression coefficients using local linear method.

# Arguments
- `x::Matrix{Float64}`: A matrix of size `(n, p)`, where `n` is the number of observations and `p` is the number of predictor variables. Each row corresponds to an observation, and each column corresponds to a predictor variable.
- `y::Vector{Float64}`: A vector of length `n` containing the response variable.
- `bw::Float64`: The bandwidth parameter ( h ) that determines the width of the kernel function.
- `tkernel::String`: The type of kernel function to use. Options include `"Gaussian"`, `"Epanechnikov"`, `"one-sided"`, and `"triweight"`. Default is `"Gaussian"`.
- `singular_ok::Bool = true: Whether to handle SingularException errors by replacing resulting values with NaN values

# Returns
- `NamedTuple`: A tuple containing:
  - `coefficients::Matrix{Float64}`: A matrix of size `(n, p)` where each row contains the time-varying OLS coefficients for each time period.
  - `fitted::Vector{Float64}`: A vector of length `n` containing the fitted values for each time period.
  - `residuals::Vector{Float64}`: A vector of length `n` containing the residuals (difference between observed and fitted values) for each time period.

# Purpose
The `tvOLS` function performs local linear regression to estimate time-varying coefficients. For each time period, it computes kernel weights for nearby observations, fits a weighted linear model, and returns the coefficients, fitted values, and residuals.
"""
function tvOLS(x::Union{AbstractMatrix, AbstractVector}, y::AbstractVector, bw::Float64,
               tkernel::String = "Gaussian";
               singular_ok::Bool = true,
               z::Union{Nothing, AbstractVector{<:Real}} = nothing,
               ez::Union{Nothing, AbstractVector{<:Real}} = nothing)

    obs = size(x, 1)
    if length(y) != obs
        error("Dimensions of 'x' and 'y' are not compatible.")
    end

    grid_vec = (z === nothing) ? (collect(1:obs) ./ obs) :
                (length(z) == obs ? collect(Float64.(z)) :
                error("vector z must have length = number of rows in x/y ($obs)"))

    

    ez_vec = (ez === nothing) ? grid_vec : collect(Float64.(ez))
    eobs   = length(ez_vec)

    nvar = size(x, 2)
    theta = zeros(eobs, nvar)
    fitted = zeros(eobs)
    residuals = zeros(eobs)



    for t in 1:eobs
        tau0 = grid_vec .- ez_vec[t]
        kernel_weights = kernel(tau0, bw, tkernel)
        k_idx = findall(kernel_weights .> 0)

        if length(k_idx) < 1
            error("Bandwidth too small for 'bw'.")
        end

        x_temp = x[k_idx, :]
        y_temp = y[k_idx]
        w_temp = kernel_weights[k_idx]

        XW = x_temp .* sqrt.(w_temp)
        yW = y_temp .* sqrt.(w_temp)

        try
            coef = (XW' * XW) \ (XW' * yW)
            θ = view(coef, 1:nvar)
            theta[t, :] .= θ
            row = obs - eobs + t
            xrow = view(x, row, :)
            valid = .!isnan.(θ)
            if any(valid)
                 fitted[t] = dot(xrow[valid], θ[valid])
                residuals[t] = y[row] - fitted[t]
            else
                fitted[t] = NaN
                residuals[t] = NaN
            end
            
        catch e
            if e isa LinearAlgebra.SingularException
                if singular_ok
                    theta[t, :] .= NaN
                    fitted[t] = NaN
                    residuals[t] = NaN
                else
                    rethrow(e)
                end
            else
                rethrow(e)
            end
        end
    end

    return (coefficients = theta, fitted = fitted, residuals = residuals)
end

"""
    forecast_tvAR(y::AbstractVector{<:Real}, p::Integer, bw::Real, n_ahead::Integer;
                  tkernel::String = "Triweight",
                  include_intercept::Bool = true,
                  window_size::Int = 0) -> Vector{Float64}

Multi-step forecast for a time-varying AR(p) model, aligning with the behavior of
`tvReg::forecast(tvAR(...))`:

- At each horizon step, re-fit time-varying OLS on the current (expanding by default) window.
- Extract the *last* available coefficient vector (boundary evaluation) and use it to forecast the next point.
- Append the forecast to the series and repeat.

Arguments
---------
- `y`: estimation sample (already preprocessed as you intend — e.g., centered or not).
- `p`: AR order.
- `bw`: kernel bandwidth for the time-varying fit.
- `n_ahead`: number of steps to forecast.
- `tkernel`: kernel name (default `"Triweight"` to match tvReg defaults).
- `include_intercept`: include an intercept column in the regression.
- `window_size`: if `> 0`, use a rolling window of this size (in observations);
                 if `0` (default), use an expanding window.

Notes
-----
- This function assumes a `tvOLS(X, y, bw, tkernel)` function that returns an object
  with a `coefficients::AbstractMatrix` field of size (T_eff × (p [+ 1])).
- Coefficient extraction uses the last non-NaN row as a proxy for evaluation at the
  end of the current sample (similar to tvReg's point-wise evaluation at a future ez).
"""
function forecast_tvAR(y::AbstractVector{<:Real},
                       p::Integer,
                       bw::Real,
                       n_ahead::Integer;
                       tkernel::String = "triweight",
                       include_intercept::Bool = true,
                       window_size::Int = 0)::Vector{Float64}

    T = length(y)
    if p >= T
        error("AR order p ($p) is too high for the data length ($T).")
    end
    if n_ahead <= 0
        error("n_ahead must be positive.")
    end
    if window_size < 0
        error("window_size must be >= 0.")
    end

    T0      = length(y)                     # original in-sample length
    totobs  = T0 + n_ahead                  # global grid size like R
    y_current = collect(y)
    forecasts = Vector{Float64}(undef, n_ahead)

    # Work on a copy because we expand the series as we forecast
    y_current = collect(y)
    forecasts = Vector{Float64}(undef, n_ahead)

    for h in 1:n_ahead
        obs = length(y_current)

        # Choose window start (expanding by default; rolling if window_size > 0)
        if window_size == 0
            start_idx = 1
        else
            start_idx = max(1, obs - window_size + 1)
        end

        # Subsample for current fit
        y_sub = @view y_current[start_idx:obs]
        n_sub = length(y_sub)

        if n_sub <= p
            error("Not enough observations in the current window (size=$n_sub) for AR($p) at step $h.")
        end

        # Build lag matrix: each row is [y[t-p], ..., y[t-1]] in oldest→newest order
        X_rows = [y_sub[t-p:t-1] for t in (p+1):n_sub]
        X = hcat(X_rows...)'  # (n_sub - p) × p
        y_dep = y_sub[p+1:end]

        # Add intercept if requested
        X_use = include_intercept ? hcat(ones(size(X, 1)), X) : X

        # Scale bandwidth from global [0,1] to local [0,1] grid:
        bw_eff = bw * (n_sub / totobs)

        # Fit tvOLS on the local window with bw_eff
        result = tvOLS(X_use, y_dep, bw_eff, tkernel)

        # Map R's future eval point u = (T0 + h)/totobs into local grid index
        u       = (T0 + h) / totobs
        t_star  = clamp(round(Int, u * (n_sub - p)), 1, (n_sub - p))   # coeff rows are (n_sub - p)
        beta    = collect(result.coefficients[t_star, :])

        # One-step-ahead forecast using last p ys (oldest→newest)
        intercept = include_intercept ? beta[1] : 0.0
        ar_coeffs = include_intercept ? beta[2:end] : beta
        history   = @view y_current[obs - p + 1 : obs]
        y_hat     = intercept + dot(ar_coeffs, history)

        # Save and append for the next step
        forecasts[h] = y_hat
        push!(y_current, y_hat)
    end

    return forecasts
end