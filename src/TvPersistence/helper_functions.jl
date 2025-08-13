"""
TvPersistence — helper_functions.jl

Purpose:
  Reusable helpers (lag matrices, etc.)

Defines:
  - ARlags_chron(X::AbstractVector, p::Integer) -> (y, Xlags)

Assumptions:
  - `X` is chronological (oldest → newest).
  - Returns `y` aligned with rows of `Xlags` (lags t-1 … t-p).

Notes:
  - Uses 1..p lag ordering by column (col 1 = t-1).
"""

"""
    ARlags_chron(X::Vector, p::Int) -> Tuple{Vector, Matrix}

Create a lagged regressor matrix for a univariate time series in chronological order (most recent value last).

# Arguments
- `X::Vector`: Input time series data, ordered from oldest to newest.
- `p::Int`: Number of lags to construct.

# Returns
- `(y, xx)`: A tuple where:
    - `y` is the target vector (from time `t = p+1` onward),
    - `xx` is a matrix where each row contains `p` lagged values of `X`, ordered from `t-1` to `t-p`.

# Example
```julia
y, Xlags = ARlags_chron(randn(100), 5) # Creates a 95x5 matrix of lagged values and a corresponding vector of length 95
"""
function ARlags_chron(X, p)

    # we lose first p observations if we want to construct lagged matrix
    y = X[p+1:end]

    # Initialize matrix of lagged values
    xx = zeros(length(y), p)

    # for each row, populate the matrix of lagged values with the correct lagged data
    for i in 1:length(y)
        for lag in 1:p
            xx[i, lag] = X[i + p - lag]
        end
    end
    return (y, xx)
end