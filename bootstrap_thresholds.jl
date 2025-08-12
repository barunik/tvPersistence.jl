# ================================================================
# File: bootstrap_thresholds.jl
# Purpose: Estimate significance thresholds for smoothed error differences (SED)
#          using AR(p)-based bootstrap simulations across various benchmark
#          and comparison forecast models (ARp, TVAR, HAR, TVHAR, tvEWD).
#
# Functions in this file:
#   - fit_ar_model: Fit AR(p) model using OLS.
#   - simulate_ar_bootstrap: Simulate pseudo-series from AR(p) with bootstrapped residuals.
#   - compute_global_threshold: Compute global (e.g., 95%) SED threshold from replicates.
#   - calculate_bootstrap_threshold: MAIN FUNCTION. Uses bootstrap and forecasting procedures
#     to estimate a one-sided statistical threshold on smoothed forecast error differences.
#
# Dependencies:
#   - Forecasting models (e.g., ARp_forecast, TVAR_forecast, tvEWD_forecast)
#   - SED smoothing utilities
#   - TV-OLS estimator module
# ================================================================

include("models/tvOLS.jl")
include("TV-EWD Implementation/Benchmark Forecasts.jl")
include("TV-EWD Implementation/Helper_functions.jl")
include("Pockets of Predictability.jl")
using .tvOLS_estimator
using Random
using Statistics

# Fit AR(p) by OLS, using ARlags_chron to build the lag matrix
"""
    fit_ar_model(series_vector::Vector{Float64}, ar_order::Int)
        -> Tuple{Vector{Float64}, Float64, Vector{Float64}}

Helper function to fit an AR(p) model to a univariate time series using OLS.

# Arguments
- `series_vector`: Time series data (chronological).
- `ar_order`: Number of lags in the AR model.

# Returns
- `coefficient_vector`: Estimated AR coefficients (including intercept).
- `residual_standard_deviation`: Standard deviation of OLS residuals.
- `residual_vector`: Vector of OLS residuals.

# Notes
Used as the base generator for bootstrap simulations in later steps.
"""
function fit_ar_model(
        series_vector::Vector{Float64},
        ar_order::Int
    )
    y, lag_matrix = ARlags_chron(series_vector, ar_order)
    T = length(y)  # = length(series_vector) - ar_order

    design_matrix = hcat(ones(T), lag_matrix)  # T × (ar_order+1)
    coefficient_vector = tvOLS_estimator.OLSestimator(y, design_matrix)

    fitted_values = design_matrix * coefficient_vector
    residual_vector = y .- fitted_values
    residual_standard_deviation = std(residual_vector)

    return coefficient_vector, residual_standard_deviation, residual_vector
end

# Bootstrap-simulate AR(p) pseudo-series
"""
    generate_AR_bootstrap_series(
        ar_coefficients::Vector{Float64},
        residual_standard_deviation::Float64,
        residual_vector::Vector{Float64},
        historical_series::Vector{Float64},
        ar_order::Int,
        burn_in_size::Int,
        seed_index_vector::Vector{Int}
    ) -> Vector{Float64}

Simulate a bootstrap series using an AR(p) process fit to historical data.

# Arguments
- `ar_coefficients`: Vector of AR model coefficients (intercept + lags).
- `residual_standard_deviation`: Std. deviation of model residuals.
- `residual_vector`: Residuals from fitted AR(p) model.
- `historical_series`: Original observed series to seed the bootstrap.
- `ar_order`: Number of lags in AR(p).
- `burn_in_size`: Number of leading samples to discard (if any).
- `seed_index_vector`: Indices to select starting values for simulation.

# Returns
- `simulated_series`: Bootstrap-generated time series of length `length(historical_series) + burn_in_size`.

# Notes
Draws standardized residuals with replacement and reconstructs the time series using AR recursion.
"""
function generate_AR_bootstrap_series(series, residuals, AR_coefficients, AR_order, burn_in_size = 0)
    # AR_order and size of AR_coefficients has to match
    # AR coefficients have to be ordered from the most recent lag to the most distant (not reversed!) + intercept first
    # burn_in_size refers to what?
    # length of series and residuals has to match!
    # Should we handle RNG within or outside the function? -> outside, because Random.seed!() works globally regardless?
    # Residuals are non-standardized!!

    # Check AR order and AR coefficient length match properly
    if (length(AR_coefficients) - 1) != AR_order
        throw(error("number of AR coefficients and AR_order must match"))
    end
    
    T = length(series)

    #if T != length(residuals)
    #    @warn "the length of residuals and input series differ"
    #end

    # Initialize a container for the simulated series
    simulated_series = vcat(series[rand(1:T, AR_order)], zeros(T-AR_order + burn_in_size))

    # Randomly resample residuals with replacement
    sampled_errors = rand(residuals, T + burn_in_size)

    # Calculate simulated values through the AR(p) formula
    for period = AR_order+1:T+burn_in_size
        lagged_values = [1;[simulated_series[period - k] for k in 1:AR_order]...] # append 1 for intercept
        simulated_series[period - 1] = dot(lagged_values, AR_coefficients) + sampled_errors[period]
    end

    return(simulated_series)
    
end

# Compute one-sided TV-OLS cutoff (same as before)
"""
    compute_global_threshold(
        list_of_sed_vectors::Vector{Vector{Float64}},
        cutoff_start_index::Int,
        alpha_level::Float64 = 0.05
    ) -> Float64

Compute a single global threshold for smoothed error differences using a bootstrap distribution.

# Arguments
- `list_of_sed_vectors`: List of smoothed SED vectors from B bootstrap replicates.
- `cutoff_start_index`: Index from which to begin considering SED values (e.g., end of in-sample).
- `alpha_level`: Significance level (default: 0.05 for a 95% threshold).

# Returns
- Global threshold as the median of the (1 - alpha_level) quantiles at each time point.
"""
function compute_global_threshold(
        list_of_sed_vectors::Vector{Vector{Float64}},
        cutoff_start_index::Int,
        alpha_level::Float64 = 0.05
    )::Float64

    B = length(list_of_sed_vectors)
    out_of_sample_length = length(list_of_sed_vectors[1])
    time_cutoffs = Float64[]

    for t in cutoff_start_index:out_of_sample_length
        vals = [ list_of_sed_vectors[b][t] for b in 1:B ]
        push!(time_cutoffs, quantile(vals, 1 - alpha_level))
    end

    return median(time_cutoffs)
end


# Main function: generate a single threshold for one benchmark vs. one comparison

"""
    calculate_bootstrap_threshold(
        series::Vector{Float64},
        ar_order::Int,
        in_sample_window_size::Int,
        forecast_horizon::Int,
        number_of_replicates::Int,
        smoothing_bandwidth::Float64,
        cutoff_start_index::Int,
        benchmark_method::Symbol,
        comparison_method::Symbol;
        forecast_length::Union{Int,String} = "Maximum",
        alpha_level::Float64 = 0.05,
        random_seed::Int = 0,
        tvp_kernel_width::Float64 = 0.4,
        kernel_type::String = "Gaussian",
        max_ar_order::Int = 15,
        jmax_scale::Int = 7,
        ar_lag_for_trend::Int = 1,
        tvp_constant_kernel_width::Float64 = 0.1,
        irf_kernel_width::Float64 = 0.2,
        forecast_kernel_width::Float64 = 0.4
    ) -> Float64

Estimate a one-sided bootstrap threshold for the smoothed error difference (SED) curve
comparing `benchmark_method` and `comparison_method`.

# Arguments
- `series`: Original univariate time series (chronological order).
- `ar_order`: AR(p) lag order for bootstrapping baseline data.
- `in_sample_window_size`: Size of rolling estimation window.
- `forecast_horizon`: Horizon (h-steps ahead) to average over forecast errors.
- `number_of_replicates`: Number of bootstrap samples B.
- `smoothing_bandwidth`: Bandwidth used for local smoothing of SED curves.
- `cutoff_start_index`: Index from which to start computing the quantile cutoff.
- `benchmark_method`: Forecast model to treat as baseline.
- `comparison_method`: Forecast model to test against the baseline.

# Keyword Arguments
- `forecast_length`: Number of forecast points to evaluate; can be `"Maximum"` or an `Int`.
- `alpha_level`: Desired significance level (e.g., 0.05 for 95% threshold).
- `random_seed`: Optional seed for reproducibility.
- `tvp_kernel_width`, `irf_kernel_width`, `forecast_kernel_width`: Kernel widths for time-varying forecast models.
- `kernel_type`: Kernel name (default: "Gaussian").
- `max_ar_order`, `jmax_scale`, `ar_lag_for_trend`, `tvp_constant_kernel_width`: Parameters for `tvEWD_forecast_test_4`.

# Returns
- `Float64`: A single global threshold for the smoothed error difference curve at the specified confidence level.

# Description
1. Fits an AR model to the original data.
2. Generates `number_of_replicates` bootstrap pseudo-series via simulation.
3. Runs forecast models (`ARp`, `TVAR`, `HAR`, `TVHAR`, `tvEWD`) on each series.
4. Computes smoothed SED for each pair of benchmark vs. comparison forecast errors.
5. Returns the specified quantile value from all replicates, starting from the out-of-sample index.

# Supported Forecast Models
- `:ARp` (vanilla AR(p)), `:TVAR` (time-varying AR(p)), `:HAR` (Heterogeneous Autoregressive), `:TVHAR`, `:tvEWD` (time-varying Extended Wold Decomposition)
# Note: Computationally intensive
"""

function calculate_bootstrap_threshold(
        series::Vector{Float64},
        ar_order::Int,
        in_sample_window_size::Int,
        forecast_horizon::Int,
        number_of_replicates::Int,
        smoothing_bandwidth::Float64,
        cutoff_start_index::Int,
        benchmark_method::Symbol,
        comparison_method::Symbol;
        forecast_length::Union{Int,String} = "Maximum",
        alpha_level::Float64 = 0.05,
        random_seed::Int = 0, # No random seed by default
        tvp_kernel_width::Float64 = 0.4,
        kernel_type::String = "Gaussian",
        max_ar_order::Int = 15,
        jmax_scale::Int = 7,
        ar_lag_for_trend::Int = 1,
        tvp_constant_kernel_width::Float64 = 0.1,
        irf_kernel_width::Float64 = 0.2,
        forecast_kernel_width::Float64 = 0.4
    )::Float64

    # Set random seed if provided
    if random_seed != 0
        Random.seed!(random_seed)
    end

    N = length(series)

    # Fit AR(ar_order)
    ar_coefficients, residual_sd, residual_vector = fit_ar_model(series, ar_order)
    in_sample_effective_length = N - ar_order

    # Pre-generate seed-index list
    seed_index_list = [ rand(1:in_sample_effective_length, ar_order) for _ in 1:number_of_replicates ]

    # Container for B smoothed-SED vectors
    smoothed_sed_collection = Vector{Vector{Float64}}(undef, number_of_replicates)

    @info "Running $number_of_replicates bootstrap replicates..."
    for i in 1:number_of_replicates
        @info "Performing boostrap simulation number $i"
        # Simulate pseudo-series (no burn-in)
        simulated_series = simulate_ar_bootstrap(
            ar_coefficients,
            residual_sd,
            residual_vector,
            series,
            ar_order,
            0,
            seed_index_list[i]
        )

        L = length(simulated_series)

        # Determine fcast_len by user input or maximum
        fcast_len = 0
        if forecast_length === "Maximum"
            # For each method, “maximum” means:
            #   ARp, TVAR: L - in_sample_window_size - ar_order - (forecast_horizon - 1)
            #   HAR, TVHAR: L - in_sample_window_size - (22 - 1) - (forecast_horizon - 1)
            #   tvEWD: handled internally by passing "Maximum" to forecast_window_size
            if benchmark_method == :tvEWD || comparison_method == :tvEWD
                # We won't set fcast_len here; tvEWD_forecast_test_4 will handle
                fcast_len = -1  # sentinel
            else
                if benchmark_method == :ARp || comparison_method == :ARp || 
                   benchmark_method == :TVAR || comparison_method == :TVAR
                    fcast_len = L - in_sample_window_size - ar_order - (forecast_horizon - 1)
                elseif benchmark_method == :HAR || comparison_method == :HAR ||
                       benchmark_method == :TVHAR || comparison_method == :TVHAR
                    fcast_len = L - in_sample_window_size - (22 - 1) - (forecast_horizon - 1)
                else
                    # If neither is ARp, TVAR, HAR, TVHAR, default to minimal:
                    fcast_len = L - in_sample_window_size - ar_order - (forecast_horizon - 1)
                end
            end
        elseif isa(forecast_length, Int)
            fcast_len = forecast_length
        else
            error("`forecast_length` must be an Integer or \"Maximum\"")
        end

        # Compute benchmark errors (using inline dispatch)
        bench_errors = begin
            if benchmark_method == :ARp
                _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
                errs
            elseif benchmark_method == :TVAR
                _, _, errs = TVAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    ar_order,
                    tvp_kernel_width;
                    kernel_type=kernel_type
                )
                errs
            elseif benchmark_method == :HAR
                _, _, errs = HAR_forecast_legacy(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
                errs
            elseif benchmark_method == :TVHAR
                _, _, errs = TVHAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    tvp_kernel_width;
                    kernel_type=kernel_type
                )
                errs
            elseif benchmark_method == :tvEWD
                # Pass forecast_window_size = forecast_length (Int or "Maximum")
                _, _, errs = tvEWD_forecast(
                    simulated_series,
                    in_sample_window_size,
                    forecast_horizon,
                    max_ar_order,
                    ar_lag_for_trend,
                    jmax_scale,
                    tvp_constant_kernel_width,
                    irf_kernel_width,
                    forecast_kernel_width;
                    kernel_type=kernel_type,
                    forecast_window_size=forecast_length
                )
                errs
            else
                error("Unsupported benchmark_method: $benchmark_method")
            end
        end

        # Compute comparison errors
        comp_errors = begin
            if comparison_method == :ARp
                _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
                errs
            elseif comparison_method == :TVAR
                _, _, errs = TVAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    ar_order,
                    tvp_kernel_width;
                    kernel_type=kernel_type
                )
                errs
            elseif comparison_method == :HAR
                _, _, errs = HAR_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
                errs
            elseif comparison_method == :TVHAR
                _, _, errs = TVHAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    tvp_kernel_width;
                    kernel_type=kernel_type
                )
                errs
            elseif comparison_method == :tvEWD
                _, _, errs = tvEWD_forecast(
                    simulated_series,
                    in_sample_window_size,
                    forecast_horizon,
                    max_ar_order,
                    ar_lag_for_trend,
                    jmax_scale,
                    tvp_constant_kernel_width,
                    irf_kernel_width,
                    forecast_kernel_width;
                    kernel_type=kernel_type,
                    forecast_window_size=forecast_length
                )
                errs
            else
                error("Unsupported comparison_method: $comparison_method")
            end
        end

        # Truncate both error‐vectors to the same minimum length
        min_len = min(length(bench_errors), length(comp_errors))
        bench_trunc = bench_errors[1:min_len]
        comp_trunc  = comp_errors[1:min_len]

        # SED estimation
        smoothed_sed_collection[i] = SED_smooth_one(
            bench_trunc,
            comp_trunc,
            smoothing_bandwidth,
            "triweight"
        )
        
    end

    @info "All bootstrap replicates generated."

    # Compute and return the single global threshold calculated as the 1-alpha quantile
    return compute_global_threshold(smoothed_sed_collection, cutoff_start_index, alpha_level)
end

# Bootstrap SED threshold calculations function

function calculate_bootstrap_threshold_parallel_V2(i,
        series::Vector{Float64},
        ar_order::Int,
        in_sample_window_size::Int,
        forecast_horizon::Int,
        smoothing_bandwidth::Float64,
        benchmark_method::Symbol,
        comparison_method::Symbol;
        fcast_len::Int,
        tvp_kernel_width::Float64 = 0.4,
        smoothing_kernel::String = "triweight",
        kernel_type_tvEWD::String = "Gaussian",
        kernel_type_tvHAR::String = "Gaussian",
        kernel_type_tvAR::String = "Gaussian",
        max_ar_order::Int = 1,
        jmax_scale::Int = 7,
        ar_lag_for_trend::Int = 1,
        tvp_constant_kernel_width::Float64 = 0.1,
        irf_kernel_width::Float64 = 0.2,
        forecast_kernel_width::Float64 = 0.4,
    )

    SED_list = []
    # Fit AR(ar_order)
    ar_coefficients, _, residual_vector = fit_ar_model(series, ar_order)
    #for i = 1:num_simulations

        @info "Performing boostrap simulation number $i"
        # Simulate pseudo-series (no burn-in)
        simulated_series = generate_AR_bootstrap_series(series,
            residual_vector,
            ar_coefficients,
            ar_order,
        )

        L = length(simulated_series)

        # Compute benchmark errors (using inline dispatch)
        bench_errors = begin
            if benchmark_method == :ARp
                _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
                errs
            elseif benchmark_method == :TVAR
                _, _, errs = TVAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    ar_order,
                    tvp_kernel_width;
                    kernel_type=kernel_type_tvAR
                )
                errs
            elseif benchmark_method == :HAR
                _, errs = HAR_forecast_legacy(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
                errs
            elseif benchmark_method == :TVHAR
                _, _, errs = TVHAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    tvp_kernel_width;
                    kernel_type=kernel_type_tvHAR
                )
                errs
            elseif benchmark_method == :tvEWD
                # Pass forecast_window_size = forecast_length (Int or "Maximum")
                _, _, errs = tvEWD_forecast(
                    simulated_series,
                    in_sample_window_size,
                    forecast_horizon,
                    max_ar_order,
                    ar_lag_for_trend,
                    jmax_scale,
                    tvp_constant_kernel_width,
                    irf_kernel_width,
                    forecast_kernel_width;
                    kernel_type=kernel_type_tvEWD,
                    forecast_window_size=fcast_len
                )
                errs
            else
                error("Unsupported benchmark_method: $benchmark_method")
            end
        end

        # Compute comparison errors
        comp_errors = begin
            if comparison_method == :ARp
                _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
                errs
            elseif comparison_method == :TVAR
                _, _, errs = TVAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    ar_order,
                    tvp_kernel_width;
                    kernel_type=kernel_type_tvAR
                )
                errs
            elseif comparison_method == :HAR
                _, _, errs = HAR_forecast_legacy(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
                errs
            elseif comparison_method == :TVHAR
                _, _, errs = TVHAR_forecast(
                    simulated_series,
                    in_sample_window_size,
                    fcast_len,
                    forecast_horizon,
                    tvp_kernel_width;
                    kernel_type=kernel_type_tvHAR
                )
                errs
            elseif comparison_method == :tvEWD
                _, _, errs = tvEWD_forecast(
                    simulated_series,
                    in_sample_window_size,
                    forecast_horizon,
                    max_ar_order,
                    ar_lag_for_trend,
                    jmax_scale,
                    tvp_constant_kernel_width,
                    irf_kernel_width,
                    forecast_kernel_width;
                    kernel_type=kernel_type_tvEWD,
                    forecast_window_size=fcast_len
                )
                errs
            else
                error("Unsupported comparison_method: $comparison_method")
            end
        end

        # SED estimation
        smoothed_sed = SED_smooth_one(
            bench_errors,
            comp_errors,
            smoothing_bandwidth,
            smoothing_kernel
        )
        push!(SED_list, smoothed_sed)
        @info "Bootstrap $i generated."
    #end

    # Compute and return the single global threshold calculated as the 1-alpha quantile
    return smoothed_sed
end