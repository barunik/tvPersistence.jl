using LinearAlgebra

# Helper function to compute betaScale at a given scale J
"""
    compute_beta_scale(alpha0::Vector{Float64}, J::Int) -> Vector{Float64}

Compute the beta scale coefficients based on Wold innovations (alpha0) at a specified scale J.

# Arguments
- `alpha0::Vector{Float64}`: The vector of Wold innovations (in chronological order).
- `J::Int`: The scale at which betaScale coefficients are computed.

# Returns
- `Vector{Float64}`: A vector of computed betaScale coefficients in shift order (meaning the first element is equal to time shift k =0 etc...).

# Purpose
This function computes the betaScale coefficients by dividing the input alpha0 into segments of size 2^J 
and then taking differences of sums within each segment.
"""
function compute_beta_scale(alpha0, J::Int, KMAX = "Maximum")::Vector{Float64}
    M = length(alpha0) # Impulse response horizon computed
    segment_size = 2^J
    
    # If the maximum time shift is not provided, calculate as much as possible
    if KMAX == "Maximum"
        num_segments = Int(floor(M / segment_size))
    else
        num_segments = KMAX
    end
    
    # Initializea array for Beta coefficients at scale J
    betaScale = zeros(Float64, num_segments)
    
    for k in 0:(num_segments - 1)
        segment_start = k * segment_size + 1
        segment_mid = segment_start + 2^(J - 1)
        segment_end = segment_start + segment_size

        sum_first_half = sum(alpha0[segment_start : segment_mid - 1])
        sum_second_half = sum(alpha0[segment_mid : segment_end - 1])
        
        betaScale[k + 1] = (sum_first_half - sum_second_half) / sqrt(segment_size)
    end
    
    return betaScale
end

# Helper function to compute epsilon scale (EpsScale) at a given scale J
"""
    compute_eps_scale(T::Int, maxAR::Int, Eps::Vector{Float64}, J::Int) -> Vector{Float64}

Compute the EpsScale vector based on the input Wold innovations at a specified scale J.

# Arguments
- `T::Int`: Total number of time periods.
- `maxAR::Int`: Maximum lag length in the baseline AR model.
- `Eps::Vector{Float64}`: The vector of unit variance Wold innovations (in chronological order).
- `J::Int`: The scale at which EpsScale coefficients are computed.

# Returns
- `Vector{Float64}`: A vector of computed EpsScale coefficients.

# Purpose
This function computes the EpsScale coefficients at a specified scale J using cumulative sums 
and differences of segments in the input Wold innovations (Eps).
"""
function compute_eps_scale(eps::Vector{Float64}, J::Int)::Vector{Float64}
    # Length of the input innovations vector
    T = length(eps)
    
    # Check if we have enough data points to compute the scaled innovations at scale J
    if T < 2^J
        error("Not enough data points to compute the scaled innovations at scale J.")
    end
    
    # Number of points available for computing epsilon at scale J
    num_obs = T - 2^J + 1 # because in order to compute innovation at scale J at time t, we need data 2^J periods back
    
    # Initialize the vector to store the scaled epsilon values
    eps_scale = zeros(Float64, num_obs)
    
    # Loop over the available points to calculate each epsilon_t^{(J)}
    for t in 2^J:T
        # Compute the cumulative sums for the upper and lower windows
        cum_sum_lower = sum(eps[t - 2^J + 1 : t - 2^(J-1)])  # Sum over the first half of the window
        cum_sum_upper = sum(eps[t - 2^(J-1) + 1 : t])        # Sum over the second half of the window
        
        # Calculate the scaled epsilon value for the current period t
        eps_scale[t - 2^J + 1] = (cum_sum_upper - cum_sum_lower) / sqrt(2^J)
    end
    
    return eps_scale
end

# Helper function to compute gScale at a given scale J
"""
    compute_g_scale_for_period(period::Int, beta_scale::Vector{Float64}, eps_scale::Vector{Float64}, J::Int, max_shift::Int) -> Float64

Compute the  g-Scale value for a specific period `u` using the provided vectors of beta_scale and eps_scale.

# Arguments
- `period::Int`: The specific period for which to compute the gScale value.
- `beta_scale::Vector{Float64}`: The vector of beta-Scale values at the given scale J
- `eps_scale::Vector{Float64}`: The vector of epsilon-Scale values at the given scale J.
- `J::Int`: The scale J for which the calculations are being done.
- `max_shift::Int`: The maximum number of time shifts based on K_{MAX}.

# Returns
- `Float64`: The computed gScale value at period `u`.
"""
function compute_g_scale_for_period(period::Int, beta_scale, eps_scale, J::Int, max_shift::Int)::Float64
    # Initialize the g-scale value for the period
    gScale = 0.0
    
    # Calculate the contribution of each lagged component based on the shifts
    for k in 0:max_shift
        # Ensure that we do not exceed the bounds of eps_scale while using the lagged indices
        if period - k * 2^J > 0
            gScale += beta_scale[k + 1] * eps_scale[period - k * 2^J]
        # Add an exception handler when the shift is not possible
        else gScale = 0
        end

    end

    return gScale
end

"""
    compute_g_scale(T::Int, maxAR::Int, beta_scale::Vector{Float64}, eps_scale::Vector{Float64}, KMAX::Int, J::Int) -> Vector{Float64}

Compute the gScale values for all relevant periods using the provided vectors of beta_scale and eps_scale.

# Arguments
- `T::Int`: The total length of the dataset.
- `maxAR::Int`: The maximum lag in the baseline AR model.
- `beta_scale::Vector{Float64}`: The vector of beta-Scale values at the given scale J.
- `eps_scale::Vector{Float64}`: The vector of epsilon-Scale values at the given scale J.
- `KMAX::Int`: The maximum lag on scales, calculated as 2^{(J + 3)}.
- `J::Int`: The scale J for which the calculations are being done.

# Returns
- `Vector{Float64}`: A vector of computed gScale values for all relevant periods.
"""
function compute_g_scale(T::Int, maxAR::Int, beta_scale, eps_scale, KMAX::Int, J::Int)::Vector{Float64}
    # Calculate the number of valid observations for g-scale calculation
    #number_observations = T - maxAR - KMAX + 1

    # Determine the maximum shift based on KMAX and J
    max_shift = Int(floor(KMAX / (2^J))) - 1

    # Initialize a vector to store the g-scale values
    g_scale = zeros(Float64, T)

    # Loop through each relevant period and compute the g-scale using the helper function
    for t in 1:T
        g_scale[t] = compute_g_scale_for_period(t, beta_scale, eps_scale, J, max_shift)
    end

    return g_scale
end