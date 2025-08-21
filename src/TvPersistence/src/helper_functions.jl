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

# ========== 1) AR(p) fit (OLS with intercept) ==========

"""
    FitArOls(seriesVector::Vector{Float64}, arOrder::Int)
→ (intercept::Float64, phi::Vector{Float64})

Fits an AR(p) model with an intercept via OLS:
    y_t = intercept + phi[1]*y_{t-1} + ... + phi[p]*y_{t-p} + error_t

Uses `ARlags_chron` to build the lag matrix and a provided `OLSestimator`.
"""
function fit_ar_ols(seriesVector::Vector{Float64}, arOrder::Int)
    # y, Xlags: both in chronological order (helper prepares them)
    y, lagMatrix = ARlags_chron(seriesVector, arOrder)
    T = length(y)  # equals length(seriesVector) - arOrder

    # Design matrix: [1  Xlags]
    designMatrix = hcat(ones(T), lagMatrix)  # size: T × (arOrder + 1)
    coef = OLSestimator(y, designMatrix)

    intercept = coef[1]
    phi = coef[2:end]
    return intercept, phi
end

# ========== 2) h-step recursive forecast path for AR(p) ==========

"""
    ForecastPathAR(sample::AbstractVector{<:Real},
                   p::Integer,
                   horizon::Integer,
                   intercept::Real,
                   phi::AbstractVector{<:Real})
→ Vector{Float64} of length `horizon`

Builds the h-step-ahead *conditional mean* forecast path for an AR(p) with intercept.
Uses the last `p` observed values from `sample` as the state.
"""
function forecast_path_ar(sample::AbstractVector{<:Real},
                        p::Integer,
                        horizon::Integer,
                        intercept::Real,
                        phi::AbstractVector{<:Real})
    @assert length(phi) == p "Length of phi must equal p."

    # Gather last p observations: [y_{T-p+1}, ..., y_T]
    lastP = collect(@view sample[(end - p + 1):end])
    # Reorder to [y_T, y_{T-1}, ..., y_{T-p+1}] for dot-product convenience
    reverse!(lastP)

    forecasts = Vector{Float64}(undef, horizon)
    for k in 1:horizon
        # yhat_{t+1|t} = intercept + sum_{j=1..p} phi[j] * state[j]
        yhat = intercept
        @inbounds @simd for j in 1:p
            yhat += phi[j] * lastP[j]
        end
        forecasts[k] = yhat

        # Shift state forward in time: prepend yhat, drop oldest
        if p > 1
            @inbounds lastP = vcat(yhat, lastP[1:p-1])
        else
            lastP[1] = yhat
        end
    end
    return forecasts
end
# ========== 3) Simple trailing rolling mean ==========

"""
    RollingMean(x::AbstractVector{<:Real}, window::Integer)
→ Vector{Float64} of length `length(x) - window + 1`

Computes a trailing mean over a fixed window, aligned to the end of each window:
out[i] = mean(x[i : i + window - 1]).
"""
function calculate_rolling_mean(x::AbstractVector{<:Real}, window::Integer)
    n = length(x)
    @assert 1 <= window <= n "window must be in 1..length(x)"

    out = Vector{Float64}(undef, n - window + 1)
    s = 0.0

    # first window
    @inbounds for i in 1:window
        s += x[i]
    end
    out[1] = s / window

    # slide the window
    @inbounds for i in 2:(n - window + 1)
        s += x[i + window - 1] - x[i - 1]
        out[i] = s / window
    end
    return out
end

### HELPER FUNCTIONS FOR EWD FORECAST ###
function OLSestimatorconst(y,x)
    x=[ones(size(x)[1]) x]
    return (transpose(x)*x) \ (transpose(x)*y)
end

function IRFalpha(y,x,maxAR,M)

    b=OLSestimator(y,x)
    Eta=y-x*b;
    sigma2=(Eta'*Eta)./(length(y)-maxAR)
    sigma=sqrt.(sigma2)
    Eps=Eta./sigma;

    alphaR=zeros(M)
    alphaR[1]=sigma

    for n=1:(length(alphaR)-1) 
        hstart=max(n-maxAR,0);
        temp=0;
        for h=hstart:n-1 
            temp=temp+alphaR[h+1]*b[n-h]; 
        end
        alphaR[n+1]=temp;
    end
    return (alphaR,Eps)
end

function IRFscale(T,maxAR,alpha0,Eps,KMAX,J)
    # input:  vector alpha of classical Wold innovations
    #         with length 2^JMAX * constant
    #         T sample length
    #         maxAR max lag in the baseline AR
    #         Eps vector of unit variance classical Wold innovations in reverse order
    #         KMAX=2^(JMAX+3) maximum lag on scales
    #         J scale
    # output: vector betaScale of multiscale IRF at scale J with length length(alpha)/(2^J)
    #         vector EpsScale of details at scale J in reverse order with length T-maxAR-2^J+1
    #         vector gScale of component at scale J in reverse order with length T-maxAR-KMAX+1
    #         vector chronGScale is gScale in chronological order

    # all processes have ZERO MEAN
   
    M=length(alpha0);
    betaScale=zeros(Int(M./(2.0.^J)));
    for k=0:Int(floor(M/(2^J))-1)
        betaScale[k+1]=(sum(alpha0[k*2^J+1:k*2^J+2^(J-1)]) - sum(alpha0[k*2^J+2^(J-1)+1:k*2^J+2^J]))./sqrt(2^J);
    end
    
    EpsScale=zeros(T-maxAR-2^J+1);
    for t=0:1:(length(EpsScale)-1)
        EpsScale[t+1]= (sum(Eps[t+1:t+2^(J-1)])-sum(Eps[t+2^(J-1)+1:t+2^J]))./sqrt(2^J);
    end
    
    gScale=zeros(T-maxAR-KMAX+1); 
    for t=0:1:(T-maxAR-KMAX)
            for k=0:1:(Int(KMAX/(2^J)-1))
                gScale[t+1]+=betaScale[k+1].*EpsScale[t+k*2^J+1]
            end
    end 
    chronGScale=reverse(gScale);
    
    decimGScale=[];
    for i=1:1:length(gScale)
        if mod(i-1,2^J)==0
            push!(decimGScale,gScale[i]);
        end
    end
    
    return (betaScale,EpsScale,gScale,chronGScale,decimGScale)
end

function IRFforecast_horizon(T,maxAR,alpha0,Eps,KMAX,J,horizon)
    
    # input:  vector alpha of classical Wold innovations
    #         with length 2^JMAX * constant
    #         T sample length
    #         maxAR max lag in the baseline AR
    #         Eps vector of unit variance classical Wold innovations in reverse order
    #         KMAX=2^(JMAX+3) maximum lag on scales
    #         J scale
    #         horizon max lag in forecasts
    # output: matrix betaPlus of multiscale IRF Beta k,p at scale J with length length(alpha)/(2^J), p goes from 1 to horizon 
    #         (see Appendix of Ortu Severino Tamoni Tebaldi)
    #         vector gScale of forecast for the sum of following week values of scale J in reverse order with length T-maxAR-KMAX+1

    # all processes have ZERO MEAN
   
    M=length(alpha0);
    betaPlus=zeros(Int(floor((M-horizon)/(2^J)))-1,horizon); #collects betak,p
    for p=1:horizon #p are the steps ahead as in Appendix of Ortu Severino Tamoni Tebaldi
        for k=0:(Int(floor((M-horizon)/(2^J)))-2)
            betaPlus[k+1,p]=(sum(alpha0[k*2^J+1+p:k*2^J+2^(J-1)+p])-sum(alpha0[k*2^J+2^(J-1)+1+p:k*2^J+2^J+p]))/sqrt(2^J);
        end                     
    end
    
    EpsScale=zeros(T-maxAR-2^J+1);
    for t=0:1:(length(EpsScale)-1)
        EpsScale[t+1]= (sum(Eps[t+1:t+2^(J-1)])-sum(Eps[t+2^(J-1)+1:t+2^J]))./sqrt(2^J);
    end
    
    gScale=zeros(T-maxAR-KMAX+1,horizon); #now gscale has to contain all the p step ahead forecasts
    for p=1:horizon
        for t=0:1:(T-maxAR-KMAX)    
            for k=0:1:(Int(floor((KMAX-horizon)/(2^J)))-2)
                gScale[t+1,p]+=betaPlus[k+1,p]*EpsScale[t+k*2^J+1];
            end
        end
    end 
     gScale=sum(gScale,dims=2) #we make row-wise sums
    
    return (betaPlus, gScale)
end

# Same as ARlags_chron but for reversed data
function ARlags(X, p)
    #AR regression to estimate the AR coefficients
    y=X[1:end-p]
    xx = zeros(length(y),p)
    for i=1:length(y)
        xx[i,:]=X[i+1:i+p]; #fill the matrix with lags
    end
    return (y,xx)
end