include(joinpath(@__DIR__, "..", "models", "tvOLS.jl"))
include("TV-EWD.jl")
include("TimeVarying_IRF.jl")
include("Helper_functions.jl")

using RCall
R"library(tvReg)"

# Import packages
using .tvOLS_estimator
using CSV, DataFrames,GLM
using Distributions, LinearAlgebra, Statistics
using Plots, StatsBase, StatsPlots, Colors

"""
    tv_persistence_plot(
        data::Vector,
        maxAR::Int,
        JMAX::Int,
        kernel_width_for_const::Float64,
        kernel_width_IRF::Float64,
        kernel_type_for_const::String = "Gaussian",
        kernel_type_for_IRF::String = "Gaussian"
    ) -> Matrix{Float64}

Estimate multiscale time-varying persistence measures using a time-varying Extended Wold Decomposition.

This function computes the time-varying multiscale impulse response coefficients βₜ^{(j)} for `j = 1, ..., JMAX`, where each row corresponds to a point in time and each column to a decomposition scale.

# Arguments
- `data::Vector`: Univariate input time series (chronologically ordered).
- `maxAR::Int`: Maximum lag order for the TV-AR model.
- `JMAX::Int`: Maximal number of scale components (e.g. J = 5 corresponds to the IRF component with 32-day persistence).
- `kernel_width_for_const::Float64`: Bandwidth for local linear estimation when estimating the trend.
- `kernel_width_IRF::Float64`: Bandwidth for local linear estimation of time-varying IRF.
- `kernel_type_for_const::String`: Type of kernel used for trend estimation (default: `"Gaussian"`).
- `kernel_type_for_IRF::String`: Type of kernel used in IRF estimation (default: `"Gaussian"`).

- kernel type options: "Gaussian", "Epanechnikov", "one-sided"

# Returns
- `IRF_beta_scales_matrix::Matrix{Float64}`: A matrix of dimensions `(T - maxAR, JMAX)`, where each row corresponds to time `t` and each column to scale `j`. Each entry contains the first element of the scale-specific impulse response function value.

# Description
1. The input series is mean-centered and locally detrended using time-varying kernel regression.
2. The detrended series is then used to fit a rolling TV-AR(maxAR) model with local IRF estimation.
3. From the local IRFs (αₜ), the function computes multiscale IRF coefficients βₜ^{(j)} for each scale `j = 1, ..., JMAX`.
"""
function tv_persistence_plot(data,
        maxAR,
        JMAX,
        kernel_width_for_const,
        kernel_width_IRF,
        kernel_type_for_const::String = "Gaussian",
        kernel_type_for_IRF::String = "Gaussian")

    T=  length(data)
    centered_data = data.-mean(data)

    KMAX = Int.(2^(JMAX)*(floor((T-maxAR)/(2^(JMAX)))-1))

    ####### In-sample trend estimation ######
    x = ones(length(centered_data)) # vector of 1's for OLS estimation as independent variable
    tvp_trend = tvOLS_estimator.tvOLS(x, centered_data,
        kernel_width_for_const,
        kernel_type_for_const).coefficients;
    detrended_sample_data = centered_data.-tvp_trend; # Detrended data
    
    # Extract final data for TV-AR(maxAR) estimation
    (truncated_data, lagged_data_matrix) = ARlags_chron(detrended_sample_data, maxAR);

    # Calculate time-varying Impulse Response Function values (alpha_{T,t}(k) in the paper) matrix
    # Each row t of the matrix corresponds to IRF values calculated from the given period t
    (IRF_alpha_matrix ,epsilon_tvp) = IRFalpha_tvp(truncated_data, # Vector of contemporaneous observations
        lagged_data_matrix, # Matrix of lagged observations
        maxAR, # TV-AR order
        KMAX, # Value that dictates the maximal allowable time shift for a given period
        kernel_width_IRF, 
        kernel_type_for_IRF)

        # Initialize container for multiscale IRF functions
        IRF_beta_scales_matrix = zeros(size(IRF_alpha_matrix, 1), JMAX)
    
        for i=1:size(IRF_alpha_matrix,1)
            IRF_alpha = IRF_alpha_matrix[i,:]
        
            # Container for current multiscale IRF
            IRF_beta_scales = []
            # For each scale component, calculate the beta coefficients and scale components x^{j}_t
            for j in 1:JMAX
            
                # Calculations taken from test_Beta_Scale tests for replicability
                # Calculate IRF for scale j
                IRF_beta_scale_j = compute_beta_scale(IRF_alpha, j)
            
                # Add the scale component calculations into our vector of vectors
                push!(IRF_beta_scales, IRF_beta_scale_j)
            end

            for iii=1:JMAX
                IRF_beta_scales_matrix[i,iii] = IRF_beta_scales[iii][1]
            end
        end
    return(IRF_beta_scales_matrix)
end