include(joinpath(@__DIR__, "..", "models", "tvOLS.jl"))
include("TV-EWD.jl")
include("TimeVarying_IRF.jl")
include("Helper_functions.jl")
using .tvOLS_estimator
using RCall
R"library(tvReg)"
using GLMNet

using CSV, DataFrames,GLM
using Distributions, LinearAlgebra, Statistics
using Plots, StatsBase, StatsPlots, Colors

"""
    compute_betaPlus(alpha0::Vector{Float64}, J::Int, horizon::Int, KMAX::Int) -> Matrix{Float64}

Compute the matrix of multiscale impulse responses values needed for TV-EWD forecasting.

# Arguments
- `alpha0::Vector{Float64}`: Vector of classical Wold innovations.
- `J::Int`: Scale parameter.
- `horizon::Int`: Forecast horizon.
- `KMAX::Int`: Maximum lag on scales.

# Returns
- `Matrix{Float64}`: A matrix of size `(KMAX, horizon)` containing the betaPlus coefficients.
"""

function compute_betaPlus(
    alpha0::Vector{Float64},
    J::Int,
    horizon::Int;
    KMAX::Union{Int,String} = "Maximum",
)::Matrix{Float64}

    M = length(alpha0)
    # how many k’s?  floor((M-horizon)/2^J)-1
    leg_max_k = Int(floor((M - horizon) / 2^J)) - 1

    num_segments = if KMAX == "Maximum"
        leg_max_k
    elseif KMAX > leg_max_k
        error("The specified KMAX exceeds what the data length allows.")
    else
        KMAX
    end

    β = zeros(num_segments, horizon)
    for p in 1:horizon
        for k in 0:(num_segments-1)
            # exact legacy bounds:
            start1 = k*2^J + 1 + p
            mid    = k*2^J + 2^(J-1) + p
            end1   = k*2^J + 2^J     + p

            # first half includes the mid‐point, second half starts right after
            s1 = sum(alpha0[start1:mid])
            s2 = sum(alpha0[mid+1:end1])
            β[k+1, p] = (s1 - s2) / sqrt(2^J)
        end
    end

    return β
end

"""
    compute_g_scale_forecast_period(
        period::Int,
        beta_plus::Matrix{Float64},
        eps_scale::Vector{Float64},
        J::Int,
        max_shift::Int,
        p::Int
    ) -> Float64

Compute the value of the scale component "\"( g_t^{(j,p)} "\") at a given time `t = period`, for scale level `J` and horizon index `p`.

# Arguments
- `period::Int`: Time index `t` at which to compute the forecasted multiscale component.
- `beta_plus::Matrix{Float64}`: Matrix of impulse response coefficients for scale `J`, where each row corresponds to a shift `k` and each column to horizon index `p`.
- `eps_scale::Vector{Float64}`: Vector of scaled residuals (innovations) at scale `J`.
- `J::Int`: Scale component level.
- `max_shift::Int`: Maximum number of 2ʲ-period shifts considered in the forecast.
- `p::Int`: Horizon index used for selecting the appropriate IRF coefficient from `beta_plus`.

# Returns
- `gScale::Float64`: Forecasted value of the multiscale component "\"( g_t^{(j,p)} "\") for the given time and scale.
"""
function compute_g_scale_forecast_period(
    period::Int,
    beta_plus::Matrix{Float64},
    eps_scale::Vector{Float64},
    J::Int,
    max_shift::Int,
    p::Int,
)::Float64
    
    gScale = 0.0
    N = length(eps_scale)

    for k in 0:max_shift
        idx = period - k*2^J
        # skip out-of-bounds instead of zeroing everything
        if idx > 0
            gScale += beta_plus[k+1, p] * eps_scale[idx]
        else gScale = 0
        end
    end

    return gScale
end

"""
    compute_gScale_forecast(
        T::Int,
        betaPlus::Matrix{Float64},
        EpsScale::Vector{Float64},
        J::Int,
        KMAX::Int,
        maxAR::Int
    ) -> Vector{Float64}

Compute the full time series of multiscale component forecasts "\"( g_t^{(j)} "\") by aggregating over impulse response horizons.

# Arguments
- `T::Int`: Total number of observations in the original time series.
- `betaPlus::Matrix{Float64}`: Matrix of multiscale impulse response coefficients βₖ₊₁,ₚ for scale `J` and each horizon `p`.
- `EpsScale::Vector{Float64}`: Vector of scale-specific residuals "\"( "\"epsilon_t^{(j)} "\").
- `J::Int`: Scale index (dyadic level).
- `KMAX::Int`: Maximum shift parameter used in scale decomposition.
- `maxAR::Int`: AR model order used in the original IRF estimation.

# Returns
- `g_scale_vec::Vector{Float64}`: Time series of multiscale components "\"(" g_t^{(j)} "\"), where each value is the sum over all impulse horizons `p` at time `t`.

# Notes
- Internally calls `compute_g_scale_forecast_period` to evaluate each time-period and horizon-specific contribution.
- Final output collapses all horizon-specific IRFs into a single vector of multiscale component forecasts:
"""
function compute_gScale_forecast(
    T::Int,
    betaPlus::Matrix{Float64},
    EpsScale::Vector{Float64},
    J::Int,
    KMAX::Int,
    maxAR::Int,
)
    horizon  = size(betaPlus, 2)
    # same legacy formula: floor((KMAX-horizon)/2^J)-2
    max_shift = Int(floor((KMAX - horizon) / 2^J)) - 2

    # drop the first (maxAR + KMAX - 1) periods
    out_len   = T - maxAR - KMAX + 1
    Gmat      = zeros(T, horizon)

    for p in 1:horizon
        for t in 1:T
            Gmat[t, p] = compute_g_scale_forecast_period(
                t, betaPlus, EpsScale, J, max_shift, p
            )
        end
    end

    # collapse across p to get the final vector
    g_scale_vec = vec(sum(Gmat; dims=2))

    return g_scale_vec
end


# tvEWD forecasting function
"""
    tvEWD_forecast(
        data0::Vector,
        window_size::Int,
        horizon::Int,
        maxAR::Int,
        AR_lag_forecast::Int,
        JMAX::Int,
        kernel_width_for_const::Float64,
        kernel_width_IRF::Float64,
        kernel_width_forecast::Float64;
        kernel_type::String = "Epanechnikov",
        user_specified_scales::Union{Nothing, Vector{Int}} = nothing,
        LASSO_scale_selection::Bool = false,
        forecast_window_size::Union{String, Int} = "Maximum"
    ) -> Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}

Perform multi-scale time-varying forecasting using the time-varying Extended Wold Decomposition (TV-EWD) framework.

# Description
This function decomposes a univariate time series into a time-varying trend and multiple scale-specific components using recursive impulse response functions (IRFs), then forecasts both the trend and the scale components using localized kernel regression and TV-AR dynamics.

# Arguments
- `data0::Vector`: Input time series (chronologically ordered).
- `window_size::Int`: Number of past observations to use in each rolling estimation window.
- `horizon::Int`: Number of future periods over which to compute forecast averages (forecasting expected value).
- `maxAR::Int`: Lag order for TV-AR estimation later used for calculating scale components from impulse responses.
- `AR_lag_forecast::Int`: Lag order used to forecast the time-varying trend.
- `JMAX::Int`: Maximum dyadic scale for multiscale decomposition (e.g. J = 5 corresponds to shock component persistent for 32 days).
- `kernel_width_for_const::Float64`: Kernel bandwidth used to estimate time-varying trend.
- `kernel_width_IRF::Float64`: Kernel bandwidth for estimating IRF coefficients from the TV-AR model.
- `kernel_width_forecast::Float64`: Kernel bandwidth used in forecasting the trend component via TV-AR.
- `kernel_type::String`: Type of kernel function (default: `"Epanechnikov"`).
- `user_specified_scales::Union{Nothing, Vector{Int}}`: Optional subset of `JMAX` scales to include in forecasting (default: `nothing` = use all scales).
- `LASSO_scale_selection::Bool`: If `true`, use LASSO to select significant scale components; otherwise, use all scale components.
- `forecast_window_size::Union{String, Int}`: Length of the forecasting loop; if `"Maximum"`, use all allowable time points.

# Returns
- `forecasted_values::Vector{Float64}`: Forecasts of the de-meaned time series using the TV-EWD framework.
- `realized_values::Vector{Float64}`: Realized values (centered) against which forecasts are evaluated.
- `residuals::Vector{Float64}`: forecast errors; computed as `forecast - realized`.

# Key Steps
1. **Detrending**: Estimate a local trend from the data using `tvOLS`.
2. **TV-AR Modeling**: Fit a time-varying autoregressive model to the detrended series to obtain IRFs.
3. **Multiscale Decomposition**: Compute IRF-based components for each dyadic scale ( j in {1, ..., JMAX} )".
4. **Component Weighting**: Fit a linear model (OLS or LASSO) to determine the weights of the scale components.
5. **Forecasting**: Independently forecast the trend and the scale components `h` steps ahead, then combine them using the estimated weights.

# Notes
- Uses kernel-smoothed estimators for both the constant trend and AR coefficients.
- Dyadic decomposition via scale components follows the dynamic Wold decomposition logic.
- Scale forecasts rely on recursively computed IRFs at each scale.
- If `user_specified_scales` is provided, only those scales are included in both estimation and forecasting.
"""
function tvEWD_forecast(
    data0, # Univariate time series vector
    window_size::Int, # Rolling window size 
    horizon::Int, # Forecast horizon
    maxAR::Int, # TV-AR model order for our series
    AR_lag_forecast::Int, # TV-AR model order for trend forecasting (TV-AR(1) in the original paper)
    JMAX::Int, # Maximal component scale j
    kernel_width_for_const    ::Float64, 
    kernel_width_IRF          ::Float64,
    kernel_width_forecast     ::Float64;
    kernel_type::String = "Epanechnikov",
    user_specified_scales::Union{Nothing, Vector{Int}} = nothing, # Optional user-specified scales in ascending order
    LASSO_scale_selection::Bool = false,
    forecast_window_size::Union{String, Int} = "Maximum"
)   
    # Create containers for storing forecasts and realized values
    T = length(data0)

    if forecast_window_size == "Maximum"
        fcast_length = T-window_size-horizon
    else
        fcast_length = forecast_window_size
    end
    
    forecasted_values = zeros(fcast_length)
    realized_values = zeros(fcast_length)
    residuals = zeros(fcast_length)

    # Extract centered dataset from raw data
    centered_data = data0.-mean(data0)

    # Realized expected value estimation?
    RVh = zeros(length(centered_data)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(centered_data[i:i+horizon-1])
    end

    KMAX = Int.(2^(JMAX)*(floor((window_size-maxAR)/(2^(JMAX)))-1))

    # Option 2. Calculate all scales from 1 to JMAX
    
    for ii in 0:(fcast_length-1)

        sample_start = fcast_length + horizon - ii;
        sample_end = fcast_length + window_size + horizon - 1 - ii;

        chron_sample_start = T - sample_end + 1;
        chron_sample_end = T- sample_start + 1;
    
        ###### EXTRACTING SAMPLE FOR ESTIMATION ######
        sample_data = centered_data[chron_sample_start:chron_sample_end];

        ####### In-sample trend estimation ######
        x = ones(length(sample_data)) # vector of 1's for OLS estimation as independent variable
        tvp_trend = tvOLS_estimator.tvOLS(x, sample_data,
            kernel_width_for_const,
            kernel_type).coefficients;
        detrended_sample_data = sample_data.-tvp_trend; # Detrended data

        ####### IRF ALPHAS CALCULATION ######
        # Extract final data for TV-AR(maxAR) estimation
        (truncated_data, lagged_data_matrix) = ARlags_chron(detrended_sample_data, maxAR);

        # Calculate time-varying Impulse Response Function values (alpha_{T,t}(k) in the paper) matrix
        # Each row t of the matrix corresponds to IRF values calculated from the given period t
        (IRF_alpha_matrix ,epsilon_tvp) = IRFalpha_tvp(truncated_data, # Vector of contemporaneous observations
            lagged_data_matrix, # Matrix of lagged observations
            maxAR, # TV-AR order
            KMAX, # Value that dictates the maximal allowable time shift for a given period
            kernel_width_IRF, 
            kernel_type)
    
        IRF_alpha = IRF_alpha_matrix[end, :] # We want the IRFs from this period

        ####### ESTIMATION OF WEIGHTS FOR SCALE COMPONENTS ######
        IRF_beta_scales = []
        g_scales = []

        # Option 1. Calculate all scales from the user-specified list
        if !(user_specified_scales == nothing)

            # For each scale component, calculate the beta coefficients and scale components x^{j}_t
            for j in user_specified_scales
            
                # Calculations taken from test_Beta_Scale tests for replicability
                # Calculate IRF for scale j
                IRF_beta_scale_j = compute_beta_scale(IRF_alpha, j)

                sample_length_epsilon_scale = window_size - maxAR - 2^j + 1
            
                # estimate error components epsilon^{j}_t at scale j
                epsilon_scale_j = compute_eps_scale(epsilon_tvp, j)[1:sample_length_epsilon_scale]

                # estimate scale components x^{j}_t at scale j
                g_scale_start_index = KMAX - 2^j + 1
                g_scale_j = compute_g_scale(length(epsilon_scale_j), maxAR, IRF_beta_scale_j, epsilon_scale_j, KMAX, j)[g_scale_start_index:end]

                # Add the scale component calculations into our vector of vectors
                push!(IRF_beta_scales, IRF_beta_scale_j)
                push!(g_scales, g_scale_j)
            end

            # Sample length for estimation
            L = length(g_scales[end]) # common sample size across all components
    
            # Extract corresponding realized sample values to estimate importance of different j scale components
            Y_chron = sample_data[(end-L + 1):end];

            # Matrix of scale components along with the trend component, chronologically ordered
            X_chron = [tvp_trend[(end-L + 1):end] hcat(g_scales...)];

            # Estimate scale component weights (with optional LASSO variable selection)
            if LASSO_scale_selection == true
                # LASSO estimator using GLMNet
                fit = glmnetcv(X_chron, Y_chron; alpha = 1.0)  # alpha = 1.0 ⇒ pure LASSO
                
                # Extract best coefficients at λ that gives minimum cross-validated error
                coef_vec = GLMNet.coef(fit)
                scale_component_weights = coef_vec
            else
                scale_component_weights = tvOLS_estimator.OLSestimator(Y_chron, X_chron)
            end
    
            ####### TREND FORECAST CALCULATION ######
            # Vectorize estimated trend coefficients
            tvp_trend_vectorized = vec(tvp_trend)

            # Forecast the trend horizon-steps ahead
            forecast_trend_AR = tvOLS_estimator.forecast_tvAR(tvp_trend_vectorized,
            AR_lag_forecast,
            kernel_width_forecast, horizon,
            tkernel = kernel_type)

            # Average over forecasted periods
            forecast_constant = mean(forecast_trend_AR) # When horizon more than 1, this matters, otherwise it is just forecast_ar

            ####### x^{j}_T+h FORECAST CALCULATIONS ######
            IRF_beta_scales_forecast = []
            g_scales_forecast = []
    
            for j in user_specified_scales
                # shifted IRFs according to calculations Ortu 2020 Appendix
                betaPlus = compute_betaPlus(IRF_alpha, j, horizon)

                # error components at scale j
                epsScale = compute_eps_scale(epsilon_tvp, j)[1:(window_size - maxAR - 2^j + 1)]
                sample_length_g = length(epsScale)

                gScale_forecast_j_full = compute_gScale_forecast(sample_length_g, betaPlus, epsScale, j, KMAX, maxAR)
                gScale_forecast_j = gScale_forecast_j_full[(end - (window_size-maxAR-KMAX)):end] # Make it match the original values, only reversed

                # Insert the forecast components at the end of the containers
                push!(IRF_beta_scales_forecast, betaPlus)
                push!(g_scales_forecast, gScale_forecast_j)
            end

        else
            # For each scale component, calculate the beta coefficients and scale components x^{j}_t
            for j in 1:JMAX
            
                # Calculations taken from test_Beta_Scale tests for replicability
                # Calculate IRF for scale j
                IRF_beta_scale_j = compute_beta_scale(IRF_alpha, j)

                sample_length_epsilon_scale = window_size - maxAR - 2^j + 1
            
                # estimate error components epsilon^{j}_t at scale j
                epsilon_scale_j = compute_eps_scale(epsilon_tvp, j)[1:sample_length_epsilon_scale]

                # estimate scale components x^{j}_t at scale j
                g_scale_start_index = KMAX - 2^j + 1
                g_scale_j = compute_g_scale(length(epsilon_scale_j), maxAR, IRF_beta_scale_j, epsilon_scale_j, KMAX, j)[g_scale_start_index:end]

                # Add the scale component calculations into our vector of vectors
                push!(IRF_beta_scales, IRF_beta_scale_j)
                push!(g_scales, g_scale_j)
            end

            # Sample length for estimation
            L = length(g_scales[JMAX]) # common sample size across all components
    
            # Extract corresponding realized sample values to estimate importance of different j scale components
            Y_chron = sample_data[(end-L + 1):end];

            # Matrix of scale components along with the trend component, chronologically ordered
            X_chron = [tvp_trend[(end-L + 1):end] hcat(g_scales...)];
    
            # Estimate scale component weights (with optional LASSO variable selection)
            if LASSO_scale_selection == true
                # LASSO estimator using GLMNet
                fit = glmnetcv(X_chron, Y_chron; alpha = 1.0)  # alpha = 1.0 ⇒ pure LASSO
                
                # Extract best coefficients at λ that gives minimum cross-validated error
                coef_vec = GLMNet.coef(fit)
                scale_component_weights = coef_vec
            else
                scale_component_weights = tvOLS_estimator.OLSestimator(Y_chron, X_chron)
            end
    
            ####### TREND FORECAST CALCULATION ######
            # Vectorize estimated trend coefficients
            tvp_trend_vectorized = vec(tvp_trend)

            # Forecast the trend horizon-steps ahead
            forecast_trend_AR = tvOLS_estimator.forecast_tvAR(tvp_trend_vectorized,
            AR_lag_forecast,
            kernel_width_forecast, horizon,
            tkernel = kernel_type)

            # Average over forecasted periods
            forecast_constant = mean(forecast_trend_AR) # When horizon more than 1, this matters, otherwise it is just forecast_ar

            ####### x^{j}_T+h FORECAST CALCULATIONS ######
            IRF_beta_scales_forecast = []
            g_scales_forecast = []
    
            for j in 1:JMAX
                # shifted IRFs according to calculations Ortu 2020 Appendix
                betaPlus = compute_betaPlus(IRF_alpha, j, horizon)

                # error components at scale j
                epsScale = compute_eps_scale(epsilon_tvp, j)[1:(window_size - maxAR - 2^j + 1)]
                sample_length_g = length(epsScale)
    
                gScale_forecast_j_full = compute_gScale_forecast(sample_length_g, betaPlus, epsScale, j, KMAX, maxAR)
                gScale_forecast_j = gScale_forecast_j_full[(end - (window_size-maxAR-KMAX)):end] # Make it match the original values, only reversed

                # Insert the forecast components at the end of the containers
                push!(IRF_beta_scales_forecast, betaPlus)
                push!(g_scales_forecast, gScale_forecast_j)
            end
        end

        ####### Final forecast calculation ######
        # Matrix of forecast scale components along with trend forecast to calculate predicted values
        forecast_scale_component_data = [forecast_constant.*ones(size(hcat(g_scales_forecast...))[1]) hcat(g_scales_forecast...)];

        # Forecasted value calculation
        forecasted_values[ii+1]=(horizon^(-1).*forecast_scale_component_data[end,:])'*[horizon*scale_component_weights[1]; scale_component_weights[2:end]];
        realized_values[ii+1] = centered_data[chron_sample_end + 1] # We always forecast the value just outside the sample
        residuals[ii+1] = realized_values[ii+1] - forecasted_values[ii+1]
    end
    return (forecasted_values, realized_values, residuals)
end