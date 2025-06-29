include("bootstrap_thresholds.jl")

function calculate_bootstrap_threshold_parallel(i,
        series::Vector{Float64},
        ar_order::Int,
        in_sample_window_size::Int,
        forecast_horizon::Int,
        smoothing_bandwidth::Float64,
        benchmark_method::Symbol,
        comparison_method::Symbol;
        forecast_length::Union{Int,String} = "Maximum",
        tvp_kernel_width::Float64 = 0.4,
        kernel_type::String = "Gaussian",
        max_ar_order::Int = 1,
        jmax_scale::Int = 7,
        ar_lag_for_trend::Int = 1,
        tvp_constant_kernel_width::Float64 = 0.1,
        irf_kernel_width::Float64 = 0.2,
        forecast_kernel_width::Float64 = 0.4
    )::Vector{Float64}

    N = length(series)

    # Fit AR(ar_order)
    ar_coefficients, residual_sd, residual_vector = fit_ar_model(series, ar_order)
    in_sample_effective_length = N - ar_order

    # Pre-generate seed-index list
    seed_index_list = rand(1:in_sample_effective_length, ar_order)

        @info "Performing boostrap simulation number $i"
        # Simulate pseudo-series (no burn-in)
        simulated_series = simulate_ar_bootstrap(
            ar_coefficients,
            residual_sd,
            residual_vector,
            series,
            ar_order,
            0,
            seed_index_list
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
                _, _, errs = HAR_forecast(simulated_series, in_sample_window_size, fcast_len, forecast_horizon)
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
        smoothed_sed = SED_smooth_one(
            bench_trunc,
            comp_trunc,
            smoothing_bandwidth,
            "one-sided"
        )

    @info "Bootstrap $i generated."

    # Compute and return the single global threshold calculated as the 1-alpha quantile
    return smoothed_sed
end