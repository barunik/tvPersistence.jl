"""
    ARp_forecast(data0::Vector, trainWindow::Int, fcast_length::Int, horizon::Int, p::Int) 
        -> Tuple{Vector, Vector, Vector}

Perform rolling-window h-step-ahead forecasts using an Autoregressive model of order `p`.

# Arguments
- `data0::Vector`: Original time series, ordered from past to present.
- `tt::Int`: Size of the rolling estimation window.
- `fcast_length::Int`: Number of forecasts to produce.
- `horizon::Int`: Forecast horizon `h` (average taken over next `h` periods).
- `p::Int`: Number of lags in the AR model.

# Returns
- `(horizonForecastAR , errorVsRV)`: Tuple of vectors with:
    - Mean forecasts over horizon,
    - Forecast errors (forecast - realized).
"""

function ARp_forecast(data::Vector{Float64},
                  trainWindow::Int,
                  fcastLength::Int,
                  horizon::Int,
                  p::Int)

    T = length(data)
    mu = mean(data)
    centered = data .- mu
    revCentered = reverse(centered)

    # Rolling average target of reversed, centered data (window = horizon)
    rvH = calculate_rolling_mean(revCentered, horizon)

    horizonForecastAR = zeros(Float64, fcastLength)
    errorVsRV         = zeros(Float64, fcastLength)

    for ii in 0:(fcastLength - 1)
        # estimation window indices in reversed space
        lo = fcastLength + horizon - ii
        hi = fcastLength + trainWindow + horizon - 1 - ii
        @assert 1 <= lo <= hi <= length(revCentered)

        # put it back to chronological order
        estSample = reverse(@view revCentered[lo:hi])

        # Fit AR(p) with intercept
        intercept, phi = fit_ar_ols(estSample, p)

        # h-step recursive path and its mean
        fpath = forecast_path_ar(estSample, p, horizon, intercept, phi)
        horizonForecastAR[ii + 1] = mean(fpath)

        # error vs rolling-mean target at the matching index
        errorVsRV[ii + 1] = horizonForecastAR[ii + 1] - rvH[fcastLength - ii]
    end

    return horizonForecastAR, errorVsRV
end

# Time-Varying AR(p) using local linear estimation
"""
    TVAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, p::Int, kernel_width::Float64;
                  kernel_type::String = "Gaussian")
        -> Tuple{Vector, Vector, Vector}

Forecast using a Time-Varying AR(p) model estimated via local linear regression with a kernel.

# Arguments
- `data0::Vector`: Original time series data.
- `tt::Int`: Estimation window size.
- `fcast_length::Int`: Number of forecasts to compute.
- `horizon::Int`: Horizon for the forecast average.
- `p::Int`: AR order.
- `kernel_width::Float64`: Bandwidth for kernel smoothing.
- `kernel_type::String`: Type of kernel function (default: "Gaussian").

# Returns
- Tuple `(forecasts, realized, errors)` as vectors for each forecasted point.
"""
function TVAR_forecast(data0,tt, p, fcast_length,horizon,kernel_width_ARtvp; kernel_type = "triweight", include_intercept = true)
    
    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR;
    r=reverse(chronR);

    if fcast_length > (length(data0) - p - horizon)
        error("Maximum forecast length exceeded")
    end


    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end
    
    horizon_forecast_AR= zeros(fcast_length);
    Error_Jcomp_tvp=zeros(fcast_length);

    for ii=0:(fcast_length-1)
    
        est_sample_tvp = reverse(r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)])
        # Generate forecasts
        forecasts_ar1 = mean(forecast_tvAR(est_sample_tvp, p, kernel_width_ARtvp, horizon; tkernel = kernel_type,
        include_intercept = include_intercept))
        horizon_forecast_AR[ii+1] = forecasts_ar1

        Error_Jcomp_tvp[ii+1] = (horizon_forecast_AR[ii+1]-RVh[fcast_length-ii])
        
    end
    return (horizon_forecast_AR, Error_Jcomp_tvp)
end

# Heterogeneous Autoregressive (HAR) model
"""
    HAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int)
        -> Tuple{Vector, Vector, Vector}

Generate forecasts using the Heterogeneous Autoregressive (HAR) model with 1-day, 5-day, and 22-day lag components.

# Arguments
- `data0::Vector`: Time series data.
- `tt::Int`: Size of rolling estimation window.
- `fcast_length::Int`: Number of forecasts to produce.
- `horizon::Int`: h-step horizon to compute forecast averages.

# Returns
- Tuple `(forecasts, realized, errors)` with:
    - Forecasted means over horizon,
    - Realized horizon means,
    - Forecast errors.
"""
function HAR_forecast(data0,tt,fcast_length,horizon)

    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR
    r=reverse(chronR);

    RVh = zeros(length(r)-horizon+1,1);
        for i=1:length(RVh)
            RVh[i]=mean(r[i:i+horizon-1])
        end

    RVd = r[1:end-22];
    RVw = zeros(length(RVd));
    for i=1:length(RVd)
       RVw[i]= (r[i]+r[i+1]+r[i+2]+r[i+3]+r[i+4])/5;
    end

    RVm = zeros(length(RVd));
    for i=1:length(RVd) 
       temp=0;
       for h=0:21
           temp = temp + r[i+h];
       end
       RVm[i]= temp./22;
    end

    horizon_forecast_corsi=zeros(fcast_length);
    daily_onestep=zeros(fcast_length,horizon);
    const_HAR=zeros(fcast_length,horizon);
    realized_HAR = zeros(fcast_length);

    Error_HAR=zeros(fcast_length);

    for ii=0:(fcast_length-1)

        RVd_estim = RVd[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];  #is the moving window of TT obs, moves towards RVd(0)
        RVw_estim = RVw[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];
        RVm_estim = RVm[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];

        Rmat = [ones(length(RVd_estim)-1) RVd_estim[2:end] RVw_estim[2:end] RVm_estim[2:end]]
        betaHAR = OLSestimator(RVd_estim[1:end-1],Rmat)

        #One-step ahead out of sample forecast, forecasts RVd(fcast_length+4)
        daily_onestep[ii+1,1]= betaHAR[1] + betaHAR[2]*RVd_estim[1] + betaHAR[3]*RVw_estim[1] + betaHAR[4]*RVm_estim[1]; 

        if horizon >=2
            for p=2:min(5,horizon)
                daily_onestep[ii+1,p]= betaHAR[1] + betaHAR[2]*daily_onestep[ii+1,p-1] + betaHAR[3]*0.2*(sum(daily_onestep[ii+1,1:p-1])+sum(RVd_estim[1:5-p+1])) + betaHAR[4]*(1/22)*(sum(daily_onestep[ii+1,1:p-1])+ sum(RVd_estim[1:22-p+1]));
            end
        end

        if horizon >= 6
            for p=6:min(22,horizon)
                daily_onestep[ii+1,p]= betaHAR[1] + betaHAR[2]*daily_onestep[ii+1,p-1] + betaHAR[3]*0.2*(sum(daily_onestep[ii+1,p-5:p-1])) + betaHAR[4]*(1/22)*(sum(daily_onestep[ii+1,1:p-1])+ sum(RVd_estim[1:22-p+1]));
            end
        end

        if horizon>= 23
            for p=23:horizon
                daily_onestep[ii+1,p]= betaHAR[1] + betaHAR[2]*daily_onestep[ii+1,p-1] + betaHAR[3]*0.2*(sum(daily_onestep[ii+1,p-5:p-1])) + betaHAR[4]*(1/22)*(sum(daily_onestep[ii+1,p-22:p-1]));
            end
        end

        horizon_forecast_corsi[ii+1] = horizon^(-1)*sum(daily_onestep[(ii+1),:]); 
        Error_HAR[ii+1]=(horizon_forecast_corsi[ii+1]-RVh[fcast_length-ii])
        const_HAR[ii+1]=betaHAR[1]
        realized_HAR[ii+1] = RVh[ii+1]
        
    end
    return (horizon_forecast_corsi,Error_HAR,reverse(realized_HAR),const_HAR)
end

# Time-Varying HAR (TV-HAR)
"""
    TVHAR_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int,
                   kernel_width::Float64; kernel_type::String = "Gaussian")
        -> Tuple{Vector, Vector, Vector}

Perform time-varying forecasting using a HAR model with locally estimated coefficients via kernel smoothing.

# Arguments
- `data0::Vector`: Time series input.
- `tt::Int`: Rolling window size.
- `fcast_length::Int`: Number of forecasts to generate.
- `horizon::Int`: Forecast horizon.
- `kernel_width::Float64`: Bandwidth for kernel regression.
- `kernel_type::String`: Kernel name (default is "Gaussian").

# Returns
- Tuple `(forecasts, realized, errors)` for forecast results and error diagnostics.
"""
function TVHAR_forecast(data0, tt, fcast_length, horizon,kernel_width; kernel_type::String = "triweight")

    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR
    r=reverse(chronR);

    RVh = zeros(length(r)-horizon+1,1);
        for i=1:length(RVh)
            RVh[i]=mean(r[i:i+horizon-1])
        end

    RVd = r[1:end-22];
    RVw = zeros(length(RVd));
    for i=1:length(RVd)
       RVw[i]= (r[i]+r[i+1]+r[i+2]+r[i+3]+r[i+4])/5;
    end

    RVm = zeros(length(RVd));
    for i=1:length(RVd) 
       temp=0;
       for h=0:21
           temp = temp + r[i+h];
       end
       RVm[i]= temp./22;
    end

    horizon_forecast_corsiTVP=zeros(fcast_length);
    daily_onestepTVP=zeros(fcast_length,horizon);
    Error_HARTVP=zeros(fcast_length);

    for ii=0:(fcast_length-1)

        RVd_estim = RVd[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];  #is the moving window of TT obs, moves towards RVd(0)
        RVw_estim = RVw[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];
        RVm_estim = RVm[fcast_length+horizon-ii: fcast_length + tt+horizon-1-ii];
    
        Rmat = [ones(length(RVd_estim)-1) RVd_estim[2:end] RVw_estim[2:end] RVm_estim[2:end]]

        # new TVP estimation
        y=RVd_estim[1:end-1]
        tvp_ols = tvOLS(Rmat, y, kernel_width, kernel_type).coefficients

        #One-step ahead out of sample forecast, forecasts RVd(fcast_length+4)
        daily_onestepTVP[ii+1,1]= tvp_ols[1,1] + tvp_ols[1,2]*RVd_estim[1] + tvp_ols[1,3]*RVw_estim[1] + tvp_ols[1,4]*RVm_estim[1]; 

        if horizon >=2
            for p=2:min(5,horizon)
                daily_onestepTVP[ii+1,p]= tvp_ols[1,1] + tvp_ols[1,2]*daily_onestepTVP[ii+1,p-1] + tvp_ols[1,3]*0.2*(sum(daily_onestepTVP[ii+1,1:p-1])+sum(RVd_estim[1:5-p+1])) + tvp_ols[1,4]*(1/22)*(sum(daily_onestepTVP[ii+1,1:p-1])+ sum(RVd_estim[1:22-p+1]));
            end
        end

        if horizon >= 6
            for p=6:min(22,horizon)
                daily_onestepTVP[ii+1,p]= tvp_ols[1,1] + tvp_ols[1,2]*daily_onestepTVP[ii+1,p-1] + tvp_ols[1,3]*0.2*(sum(daily_onestepTVP[ii+1,p-5:p-1])) + tvp_ols[1,4]*(1/22)*(sum(daily_onestepTVP[ii+1,1:p-1])+ sum(RVd_estim[1:22-p+1]));
            end
        end

        if horizon>= 23
            for p=23:horizon
                daily_onestepTVP[ii+1,p]= tvp_ols[1,1] + tvp_ols[1,2]*daily_onestepTVP[ii+1,p-1] + tvp_ols[1,3]*0.2*(sum(daily_onestepTVP[ii+1,p-5:p-1])) + tvp_ols[1,4]*(1/22)*(sum(daily_onestepTVP[ii+1,p-22:p-1]));
            end
        end

        horizon_forecast_corsiTVP[ii+1] = horizon^(-1)*sum(daily_onestepTVP[(ii+1),:]); 
        Error_HARTVP[ii+1]=(horizon_forecast_corsiTVP[ii+1]-RVh[fcast_length-ii])
 end
    return (horizon_forecast_corsiTVP,Error_HARTVP)
end

"""
    EWD_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, maxAR::Int, JMAX::Int)
        -> Tuple{Vector, Vector, Vector}

Forecast using the Extended Wold Decomposition (EWD) method proposed by Ortu et. al. (2020), which decomposes time series into multiple persistence scale components.

# Arguments
- `data0::Vector`: Input time series data.
- `tt::Int`: Size of the rolling estimation window.
- `fcast_length::Int`: Number of forecasts to compute.
- `horizon::Int`: Forecast horizon (averaging period).
- `maxAR::Int`: Maximum number of lags for AR-based estimation.
- `JMAX::Int`: Number of decomposition scales.

# Returns
- Tuple `(forecasts, realized, errors)` representing:
    - EWD forecasts,
    - Realized horizon means,
    - Forecast errors.
"""
function EWD_forecast(data0,tt,maxAR,JMAX,horizon, fcast_length = "Maximum")

    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR
    r=reverse(chronR);

    if fcast_length == "Maximum"
        fcast_length = T-tt-horizon; #length of forecast sample
    end

    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end

    M=Int.(2^(JMAX)*(floor((tt -maxAR)/(2^(JMAX)))-1));
    KMAX = M;

    horizon_forecast_Jcomp= zeros(fcast_length);
    Error_Jcomp=zeros(fcast_length);
    RV_h=zeros(fcast_length);

    for ii=0:(fcast_length-1)

        est_sample = r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)]

        # AR regression to estimate the AR coefficients
        # alpha coefficients and the Eps for the current sample
        (truncR,Rmat)=ARlags(est_sample, maxAR)
        (alphaR,Eps)=IRFalpha(truncR,Rmat,maxAR,M)

        # Estimate the decomposition
        betaRj=[]
        rj=[]
        for j in 1:JMAX
            (betaR, _ , r0, _ , _)=IRFscale(tt,maxAR,alphaR,Eps,KMAX,j);
            push!(betaRj,betaR)
            push!(rj,r0)
        end

        # Estimate coefficients of regression model
        b_Jcomp=OLSestimatorconst(est_sample[1:length(rj[JMAX])],hcat(rj...))

        # We forecast the sum horizon-step ahead 
        betaFj=[]
        rfj=[]
        for j in 1:JMAX
            (betaF,rf)=IRFforecast_horizon(tt,maxAR,alphaR,Eps,KMAX,j,horizon);
            push!(betaFj,betaF)
            push!(rfj,rf)
        end

        RFmat_Jcomp=[ones(size(hcat(rfj...))[1]) hcat(rfj...)];
        horizon_forecast_Jcomp[ii+1]=(horizon^(-1).*RFmat_Jcomp[1,:])'*[horizon*b_Jcomp[1]; b_Jcomp[2:end]];

        # error with J components
        RV_h[ii+1]=RVh[fcast_length-ii];
        Error_Jcomp[ii+1]=  (horizon_forecast_Jcomp[ii+1]-RV_h[ii+1]);

    end
    
    return (horizon_forecast_Jcomp, RV_h, Error_Jcomp)
end
