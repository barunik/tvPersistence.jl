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
    coefficient_vector = OLSestimator(y, design_matrix)

    fitted_values = design_matrix * coefficient_vector
    residual_vector = y .- fitted_values
    residual_standard_deviation = std(residual_vector)

    return coefficient_vector, residual_standard_deviation, residual_vector
end

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

    # Check AR order and AR coefficient length match properly
    if (length(AR_coefficients) - 1) != AR_order
        throw(error("number of AR coefficients and AR_order must match"))
    end
    
    T = length(series)

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
        # collect the B values at time t
        raw_vals = [ list_of_sed_vectors[b][t] for b in 1:B ]
        # drop any NaNs
        clean_vals = filter(!isnan, raw_vals)

        # if everything was NaN you might skip or push a NaN
        if isempty(clean_vals)
            continue
        end

        push!(time_cutoffs, quantile(clean_vals, 1 - alpha_level))
    end

    return median(time_cutoffs)
end

"""
    calculate_bootstrap_threshold_parallel(i,
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

Estimate a threshold for the smoothed error difference (SED) curve from bootstrapped simulated series
comparing `benchmark_method` and `comparison_method`.

# Arguments¨
- i: serial number for the simulation (for later parallelization)
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
- `random_seed`: Optional seed for reproducibility. -> handled for each worker in SED_threshold_example.ipynb
- `tvp_kernel_width`, `irf_kernel_width`, `forecast_kernel_width`: Kernel widths for time-varying forecast models.
- `kernel_type`: local linear estimation kernel. Options: "Gaussian" (default), "Epanechnikov", "one-sided", "triweight".
- `max_ar_order`, `jmax_scale`, `ar_lag_for_trend`, `tvp_constant_kernel_width`: Parameters for `tvEWD_forecast_test_4`.

# Returns
- `Float64`: A single global threshold for the smoothed error difference curve at the specified confidence level.

# Description
1. Fits an AR model to the original data.
2. Generates a bootstrap pseudo-series via simulation.
3. Runs forecast models (`ARp`, `TVAR`, `HAR`, `TVHAR`, `tvEWD`) on each series.
4. returns smoothed SED series for specified pair of benchmark vs. comparison forecast errors.


# Supported Forecast Models
- `:ARp` (vanilla AR(p)), `:TVAR` (time-varying AR(p)), `:HAR` (Heterogeneous Autoregressive), `:TVHAR`, `:tvEWD` (time-varying Extended Wold Decomposition)
# Note: This function is meant as an input to pmap() for faster calculation of the Threshold (computationally intensive)
"""
function calculate_bootstrap_threshold_parallel(i,
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

    @info "Performing boostrap simulation number $i"
    # Simulate pseudo-series (no burn-in)
    simulated_series = generate_AR_bootstrap_series(series,
        residual_vector,
        ar_coefficients,
        ar_order,
    )

    L = length(simulated_series)
    
    # benchmark forecast errors
    bench_errors = begin
        if benchmark_method == :ARp
            _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
            errs
        elseif benchmark_method == :TVAR
            _, errs = TVAR_forecast(
                simulated_series,
                in_sample_window_size,
                ar_order,
                fcast_len,
                forecast_horizon,
                tvp_kernel_width;
                kernel_type=kernel_type_tvAR
            )
            errs
        elseif benchmark_method == :HAR
            _, errs, _, _ = HAR_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
            errs
        elseif benchmark_method == :TVHAR
            _, errs = TVHAR_forecast(
                simulated_series,
                in_sample_window_size,
                fcast_len,
                forecast_horizon,
                tvp_kernel_width;
                kernel_type=kernel_type_tvHAR
            )
            errs
        elseif benchmark_method == :EWD
            _, _, errs = EWD_forecast(simulated_series,
            in_sample_window_size,
            max_ar_order,
            jmax_scale,
            forecast_horizon,
            fcast_len)
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

    # Compute comparison forecast errors
    comp_errors = begin
        if comparison_method == :ARp
            _, _, errs = ARp_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
            errs
        elseif comparison_method == :TVAR
            _, errs = TVAR_forecast(
                simulated_series,
                in_sample_window_size,
                ar_order,
                fcast_len,
                forecast_horizon,
                tvp_kernel_width;
                kernel_type=kernel_type_tvAR
            )
            errs
        elseif comparison_method == :HAR
            _, errs, _, _ = HAR_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
            errs
        elseif comparison_method == :TVHAR
            _, errs = TVHAR_forecast(
                simulated_series,
                in_sample_window_size,
                fcast_len,
                forecast_horizon,
                tvp_kernel_width;
                kernel_type=kernel_type_tvHAR
            )
            errs
        elseif comparison_method == :EWD
            _, _, errs = EWD_forecast(simulated_series,
            in_sample_window_size,
            max_ar_order,
            jmax_scale,
            forecast_horizon,
            fcast_len)
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

    return smoothed_sed
end