include("TV-EWD_forecast.jl")
include("Helper_functions.jl")
include(joinpath(@__DIR__, "..", "models", "tvOLS.jl"))
include("BasicIRF.jl")
using .tvOLS_estimator

# Autoregressive(p) rolling-window h-step-ahead forecast
"""
    ARp_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, p::Int) 
        -> Tuple{Vector, Vector, Vector}

Perform rolling-window h-step-ahead forecasts using an Autoregressive model of order `p`.

# Arguments
- `data0::Vector`: Original time series, ordered from past to present.
- `tt::Int`: Size of the rolling estimation window.
- `fcast_length::Int`: Number of forecasts to produce.
- `horizon::Int`: Forecast horizon `h` (average taken over next `h` periods).
- `p::Int`: Number of lags in the AR model.

# Returns
- `(forecasts, realized, errors)`: Tuple of vectors with:
    - Mean forecasts over horizon,
    - Realized horizon averages,
    - Forecast errors (forecast - realized).
"""
function ARp_forecast(
    data0,        # original series (chronological)
    tt::Int,                       # rolling window size
    fcast_length::Int,             # number of forecasts to produce
    horizon::Int,                  # forecast horizon h
    p::Int                         # AR order
)
    # Center the data
    μ = mean(data0)
    y = data0 .- μ
    T = length(y)

    # Containers
    forecasts = zeros(fcast_length)
    realized = zeros(fcast_length)
    errors   = zeros(fcast_length)

    for ii in 0:(fcast_length-1)
        # Define estimation window
        train_start = ii + 1
        train_end   = ii + tt
        # Build OLS design
        nobs = tt - p
        X = ones(nobs, p+1)
        Y = zeros(nobs)
        for j in 1:nobs
            t = train_start + p - 1 + j
            Y[j] = y[t]
            X[j, 2:end] = y[t-1:-1:t-p]
        end
        β = X \ Y  # OLS solution

        # Recursive h-step forecast
        history = copy(y[train_end-p+1:train_end])
        h_fore = zeros(horizon)
        for h in 1:horizon
            pred = β[1] + dot(β[2:end], reverse(history))
            h_fore[h] = pred
            push!(history, pred)
            popfirst!(history)
        end
        forecasts[ii+1] = mean(h_fore)
        # Realized mean over next h
        idx = train_end+1 : train_end+horizon
        realized[ii+1] = mean(y[idx])
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return forecasts, realized, errors
end

# Time-Varying AR(p) using local linear estimation
"""
    TVAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, p::Int, kernel_width::Float64;
                  kernel_type::String = "Gaussian")
        -> Tuple{Vector, Vector, Vector}

Forecast using a Time-Varying AR(p) model estimated via local linear regression with a kernel.

# Arguments
- `data0::Vector`: Original time series data.
- `tt::Int`: Estimation window size.
- `fcast_length::Int`: Number of forecasts to compute.
- `horizon::Int`: Horizon for the forecast average.
- `p::Int`: AR order.
- `kernel_width::Float64`: Bandwidth for kernel smoothing.
- `kernel_type::String`: Type of kernel function (default: "Gaussian").

# Returns
- Tuple `(forecasts, realized, errors)` as vectors for each forecasted point.
"""
function TVAR_forecast(
    data0,
    tt::Int,
    fcast_length::Int,
    horizon::Int,
    p::Int,
    kernel_width::Float64;
    kernel_type::String = "Gaussian"
)
    μ = mean(data0)
    y = data0 .- μ
    T = length(y)

    forecasts = zeros(fcast_length)
    realized  = zeros(fcast_length)
    errors    = zeros(fcast_length)

    for ii in 0:(fcast_length-1)
        train_start = ii + 1
        train_end   = ii + tt
        sample = y[train_start:train_end]
        # Forecast h-step ahead via provided function
        h_fore = tvOLS_estimator.forecast_tvAR(sample, p, kernel_width, horizon; tkernel=kernel_type)
        forecasts[ii+1] = mean(h_fore)
        idx = train_end+1 : train_end+horizon
        realized[ii+1] = mean(y[idx])
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return forecasts, realized, errors
end

# Heterogeneous Autoregressive (HAR) model
"""
    HAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int)
        -> Tuple{Vector, Vector, Vector}

Generate forecasts using the Heterogeneous Autoregressive (HAR) model with 1-day, 5-day, and 22-day lag components.

# Arguments
- `data0::Vector`: Time series data.
- `tt::Int`: Size of rolling estimation window.
- `fcast_length::Int`: Number of forecasts to produce.
- `horizon::Int`: h-step horizon to compute forecast averages.

# Returns
- Tuple `(forecasts, realized, errors)` with:
    - Forecasted means over horizon,
    - Realized horizon means,
    - Forecast errors.
"""
function HAR_forecast(
    data0,
    tt::Int,
    fcast_length::Int,
    horizon::Int
)
    μ = mean(data0)
    y = data0 .- μ
    T = length(y)

    forecasts = zeros(fcast_length)
    realized  = zeros(fcast_length)
    errors    = zeros(fcast_length)

    for ii in 0:(fcast_length-1)
        train_start = ii + 1
        train_end   = ii + tt
        # Fit HAR on last tt observations
        window = y[train_start:train_end]
        # Build history for recursion
        history = copy(y[train_end-21:train_end])  # need 22 lags
        # One-step regression to get coefficients
        # We use simple OLS on {t-22..t-1} -> t
        nobs = tt - 22
        X = ones(nobs, 4)
        Y = zeros(nobs)
        for j in 1:nobs
            t = train_start + 22 - 1 + j
            Y[j] = y[t]
            X[j,2] = y[t-1]
            X[j,3] = mean(y[t-5:t-1])
            X[j,4] = mean(y[t-22:t-1])
        end
        coefficients = X \ Y
        # Forecast recursion
        h_fore = zeros(horizon)
        for h in 1:horizon
            lag1  = history[end]
            lag5  = mean(history[end-4:end])
            lag22 = mean(history[end-21:end])
            pred = coefficients[1] + coefficients[2]*lag1 + coefficients[3]*lag5 + coefficients[4]*lag22
            h_fore[h] = pred
            push!(history, pred); popfirst!(history)
        end
        forecasts[ii+1] = mean(h_fore)
        idx = train_end+1 : train_end+horizon
        realized[ii+1] = mean(y[idx])
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return forecasts, realized, errors
end

# Time-Varying HAR (TV-HAR)
"""
    TVHAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int,
                   kernel_width::Float64; kernel_type::String = "Gaussian")
        -> Tuple{Vector, Vector, Vector}

Perform time-varying forecasting using a HAR model with locally estimated coefficients via kernel smoothing.

# Arguments
- `data0::Vector`: Time series input.
- `tt::Int`: Rolling window size.
- `fcast_length::Int`: Number of forecasts to generate.
- `horizon::Int`: Forecast horizon.
- `kernel_width::Float64`: Bandwidth for kernel regression.
- `kernel_type::String`: Kernel name (default is "Gaussian").

# Returns
- Tuple `(forecasts, realized, errors)` for forecast results and error diagnostics.
"""
function TVHAR_forecast(
    data0,
    tt::Int,
    fcast_length::Int,
    horizon::Int,
    kernel_width::Float64;
    kernel_type::String = "Gaussian"
)
    μ = mean(data0)
    y = data0 .- μ
    T = length(y)

    forecasts = zeros(fcast_length)
    realized  = zeros(fcast_length)
    errors    = zeros(fcast_length)

    for ii in 0:(fcast_length-1)
        train_start = ii + 1
        train_end   = ii + tt
        # Build local design for tvOLS
        nobs = tt - 22
        Rmat = ones(nobs, 4)
        Y    = zeros(nobs)
        for j in 1:nobs
            t = train_start + 22 - 1 + j
            Y[j] = y[t]
            Rmat[j,2] = y[t-1]
            Rmat[j,3] = mean(y[t-5:t-1])
            Rmat[j,4] = mean(y[t-22:t-1])
        end
        result = tvOLS_estimator.tvOLS(Rmat, Y, kernel_width, kernel_type)
        coefs  = result.coefficients[end, :]  # use latest TVP-coefs
        # Forecast recursion
        history = copy(y[train_end-21:train_end])
        h_fore = zeros(horizon)
        for h in 1:horizon
            lag1  = history[end]
            lag5  = mean(history[end-4:end])
            lag22 = mean(history[end-21:end])
            pred = coefs[1] + coefs[2]*lag1 + coefs[3]*lag5 + coefs[4]*lag22
            h_fore[h] = pred
            push!(history, pred); popfirst!(history)
        end
        forecasts[ii+1] = mean(h_fore)
        idx = train_end+1 : train_end+horizon
        realized[ii+1] = mean(y[idx])
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return forecasts, realized, errors
end

"""
    EWD_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, maxAR::Int, JMAX::Int)
        -> Tuple{Vector, Vector, Vector}

Forecast using the Extended Wold Decomposition (EWD) method proposed by Ortu et. al. (2020), which decomposes time series into multiple persistence scale components.

# Arguments
- `data0::Vector`: Input time series data.
- `tt::Int`: Size of the rolling estimation window.
- `fcast_length::Int`: Number of forecasts to compute.
- `horizon::Int`: Forecast horizon (averaging period).
- `maxAR::Int`: Maximum number of lags for AR-based estimation.
- `JMAX::Int`: Number of decomposition scales.

# Returns
- Tuple `(forecasts, realized, errors)` representing:
    - EWD forecasts,
    - Realized horizon means,
    - Forecast errors.
"""
function EWD_forecast(
    data0,
    tt::Int,
    fcast_length::Int,
    horizon::Int,
    maxAR::Int,
    JMAX::Int
)
    μ = mean(data0)
    y = data0 .- μ

    # pre-allocate
    forecasts = zeros(fcast_length)
    realized  = zeros(fcast_length)
    errors    = zeros(fcast_length)

    # compute realized ahead-means
    RVh = [mean(y[i:i+horizon-1]) for i in 1:(length(y)-horizon+1)]

    # compute M and KMAX
    KMAX    = Int(2^JMAX * (floor((tt - maxAR) / 2^JMAX) - 1))

    for ii in 0:(fcast_length-1)
        train_start = ii + 1
        train_end   = ii + tt
        sample = y[train_start:train_end]

        # IRF alphas & residuals via OLS
        (truncR, Rmat) = ARlags_chron(sample, maxAR)
        (alphaR, epsR) = IRF_alpha(truncR, Rmat, maxAR, KMAX)

        # build scale components in-sample
        g_scales = Vector{Vector{Float64}}(undef, JMAX)
        for j in 1:JMAX
            beta_scale_j    = compute_beta_scale(alphaR, j)
            eps_scale_j = compute_eps_scale(epsR, j)[1:tt-maxAR-2^j+1]
            g_scales[j] = compute_g_scale(length(eps_scale_j), maxAR, beta_scale_j, eps_scale_j, KMAX, j)[KMAX-2^j+1:end]
        end

        # regress sample on scales + intercept
        L = length(g_scales[end])
        Ych = sample[end-L+1:end]
        Xch = hcat(g_scales...)[end-L+1:end, :]
        Xfull = hcat(ones(L), Xch)
        coef  = tvOLS_estimator.OLSestimator(Ych, Xfull)

        # forecast each scale component
        g_cale_forecasts = zeros(JMAX)
        for j in 1:JMAX
            beta_plus_j = compute_betaPlus(alphaR, j, horizon)
            eps_plus_j = compute_eps_scale(epsR, j)[1:tt-maxAR-2^j+1]
            g_scale_forecast_j    = compute_gScale_forecast(length(eps_plus_j), beta_plus_j, eps_plus_j, j, KMAX, maxAR)
            g_cale_forecasts[j] = g_scale_forecast_j[end]
        end

        # assemble forecast row and compute
        Frow = vcat(horizon*coef[1], coef[2:end] .* g_cale_forecasts)
        forecasts[ii+1] = sum(Frow) / horizon

        # realized and error
        realized[ii+1] = RVh[train_end+1]
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return forecasts, realized, errors
end