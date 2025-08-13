"""
    SED_smooth_one(
        bench_err::AbstractVector{<:Real},
        model_err::AbstractVector{<:Real},
        kernel_width::Real;
        kernel_type::String = "triweight"
    ) -> Vector{Float64}

Compute the smoothed squared-error difference (SED) between two forecast-error series
using a local‐regression (tvOLS) smoother.

# Arguments
- `benchmark_forecast_error`: forecast errors from the benchmark model.
- `model_forecast_error`: forecast errors from the new model.
- `kernel_width`: bandwidth parameter for the local regression.
- `kernel_type`: smoothing kernel type (e.g., `"one-sided"` or `"two-sided"`).
- `include_intercept::Bool`: Whether to include intercept in the local linear estimation (true by default)

# Returns
- A vector of fitted (smoothed) SED values
"""
function SED_smooth_one(benchmark_forecast_error::AbstractVector{<:Real},
        model_forecast_error::AbstractVector{<:Real},
        kernel_width,
        kernel_type::String = "triweight",
        include_intercept::Bool = true)
    
     # Squared Error Difference as the dependent variable
    data = benchmark_forecast_error.^2 .- model_forecast_error.^2;
    
    if include_intercept == true
        regressor_data = hcat(ones(length(data)), [t for t in 1:length(data)])
    else
        regressor_data = [t for t in 1:length(data)]
    end

    result1 = tvOLS(regressor_data, data, kernel_width, kernel_type)
    return result1.fitted
end