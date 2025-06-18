# bootstrap_thresholds_parallel.jl

using Distributed

 # Make sure we have exactly n_cores-1 workers every time we load this file:
let
    desired_workers = max(1, Sys.CPU_THREADS - 1)
    current_workers = nprocs() - 1
    if current_workers != desired_workers
        rmprocs(workers())            # kill all existing
        addprocs(desired_workers)     # start exactly one per core minus master
    end
end

@everywhere begin
    # 2) bring in all the original helpers & models
    using Random, Statistics
    include("bootstrap_thresholds.jl")   # defines fit_ar_model, simulate_ar_bootstrap,
                                         # compute_global_threshold, SED_smooth_one, forecasting functions, etc.

    # 3) helper to route a symbol → forecast-error vector
    function _forecast_errors(
            series::Vector{Float64},
            in_sample_window_size::Int,
            fcast_len::Int,
            forecast_horizon::Int,
            method::Symbol,
            ar_order::Int,
            tvp_kernel_width::Float64,
            kernel_type::String,
            max_ar_order::Int,
            jmax_scale::Int,
            ar_lag_for_trend::Int,
            tvp_constant_kernel_width::Float64,
            irf_kernel_width::Float64,
            forecast_kernel_width::Float64,
            forecast_length::Union{Int,String}
        )::Vector{Float64}

        if method == :ARp
            _, _, errs = ARp_forecast(series, in_sample_window_size, fcast_len, forecast_horizon, ar_order)
            return errs

        elseif method == :TVAR
            _, _, errs = TVAR_forecast(
                series, in_sample_window_size, fcast_len, forecast_horizon,
                ar_order, tvp_kernel_width; kernel_type=kernel_type
            )
            return errs

        elseif method == :HAR
            _, _, errs = HAR_forecast(series, in_sample_window_size, fcast_len, forecast_horizon)
            return errs

        elseif method == :TVHAR
            _, _, errs = TVHAR_forecast(
                series, in_sample_window_size, fcast_len, forecast_horizon,
                tvp_kernel_width; kernel_type=kernel_type
            )
            return errs

        elseif method == :tvEWD
            _, _, errs = tvEWD_forecast(
                series,
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
            return errs

        else
            error("Unsupported forecast method: $method")
        end
    end

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

    Exactly the same signature as before, but runs all B replicates in parallel
    and then calls `compute_global_threshold` on the full collection.
    """
    function calculate_bootstrap_threshold_parallel(
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
        )::Float64

        if random_seed != 0
            # seed the master RNG
            Random.seed!(random_seed)

            # seed each worker to the same value
            @sync for w in workers()
                @async remotecall(Random.seed!, w, random_seed)
            end
        end

        # 1) fit the bootstrap AR(p) generator
        N = length(series)
        ar_coefs, resid_sd, resid_vec = fit_ar_model(series, ar_order)
        in_sample_eff = N - ar_order

        # 2) pre‐draw all "seed" vectors for reproducibility
        seed_list = [ rand(1:in_sample_eff, ar_order) for _ in 1:number_of_replicates ]

        @info "Dispatching $number_of_replicates replicates across $n_workers workers…"

        # 3) map one replicate → one smoothed SED vector
        sed_collection = pmap(seed -> begin
            # simulate
            sim = simulate_ar_bootstrap(
                ar_coefs, resid_sd, resid_vec,
                series, ar_order, 0, seed
            )
            L = length(sim)

            # compute forecast‐length fcast_len
            fcast_len = forecast_length === "Maximum" ? (
                (benchmark_method in (:tvEWD) || comparison_method in (:tvEWD)) ? -1 :
                ((benchmark_method in (:ARp,:TVAR) || comparison_method in (:ARp,:TVAR)) ?
                   L - in_sample_window_size - ar_order - (forecast_horizon - 1) :
                ((benchmark_method in (:HAR,:TVHAR) || comparison_method in (:HAR,:TVHAR)) ?
                   L - in_sample_window_size - 21           - (forecast_horizon - 1) :
                   L - in_sample_window_size - ar_order      - (forecast_horizon - 1)))
            ) : forecast_length

            # benchmark vs comparison error series
            bench_err = _forecast_errors(
                sim, in_sample_window_size, fcast_len, forecast_horizon,
                benchmark_method, ar_order, tvp_kernel_width, kernel_type,
                max_ar_order, jmax_scale, ar_lag_for_trend,
                tvp_constant_kernel_width, irf_kernel_width,
                forecast_kernel_width, forecast_length
            )

            comp_err  = _forecast_errors(
                sim, in_sample_window_size, fcast_len, forecast_horizon,
                comparison_method, ar_order, tvp_kernel_width, kernel_type,
                max_ar_order, jmax_scale, ar_lag_for_trend,
                tvp_constant_kernel_width, irf_kernel_width,
                forecast_kernel_width, forecast_length
            )

            # align lengths & smooth
            m = min(length(bench_err), length(comp_err))
            SED_smooth_one(bench_err[1:m], comp_err[1:m], smoothing_bandwidth, "one-sided")
        end, seed_list)

        @info "All replicates complete; computing global threshold…"
        compute_global_threshold(sed_collection, cutoff_start_index, alpha_level)
    end

    export calculate_bootstrap_threshold
end
