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

function forecast_tvar(data, ar_order::Int, bw::Float64, horizon;
    kernel_type::String = "triweight",
    include_intercept = false,
    singular_ok::Bool = true)

    dependent_data, independent_data = ARlags_chron(data, ar_order)
    obs = length(dependent_data)
    independent_data = vcat(independent_data, zeros(horizon, ar_order))

    if include_intercept == true
        independent_data = hcat(ones(size(independent_data, 1)), independent_data)
    end

    dependent_data = vcat(dependent_data, zeros(horizon))
    totobs = obs + horizon # total number of observations
    predictions = Vector{Float64}(undef, horizon)

    grid_vec = collect(1:totobs)/totobs

    # generate forecasts
    for t = 1:horizon

        dependent_data_temp = dependent_data[1:(obs + t -1)] # sample of dependent data
        independent_data_temp = independent_data[1:(obs + t -1), :]
        temp_z = grid_vec[1:(obs + t -1)]
        temp_ez = Vector([grid_vec[obs + t]])

        theta_mat = tvOLS(independent_data_temp, dependent_data_temp, bw,
        kernel_type,
        singular_ok = singular_ok,
        z = temp_z,
        ez = temp_ez).coefficients

        theta = vec(theta_mat[1, :])

        if include_intercept == true
            new_x = vcat(ones(1), last(dependent_data_temp, ar_order))'
        else
            new_x =  last(dependent_data_temp, ar_order)'
        end

        predictions[t] = dot(new_x, theta)

        # update the data used for next step
        dependent_data[obs + t]      = predictions[t]
        independent_data[obs + t, :] = new_x'

    end

    return(predictions)

end