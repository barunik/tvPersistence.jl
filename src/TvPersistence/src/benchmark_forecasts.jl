"""
    ARp_forecast(data0::Vector, tt::Int, fcast_length::Int, horizon::Int, p::Int) 
        -> Tuple{Vector, Vector, Vector}

Perform rolling-window h-step-ahead forecasts using an Autoregressive model of order `p`.

# Arguments
- `data0::Vector`: Original time series, ordered from past to present.
- `tt::Int`: Size of the rolling estimation window.
- `fcast_length::Int`: Number of forecasts to produce.
- `horizon::Int`: Forecast horizon `h` (average taken over next `h` periods).
- `p::Int`: Number of lags in the AR model.

# Returns
- `(forecasts, realized, errors)`: Tuple of vectors with:
    - Mean forecasts over horizon,
    - Realized horizon averages,
    - Forecast errors (forecast - realized).
"""
function ARp_forecast(
    data0,        # original series (chronological)
    tt::Int,                       # rolling window size
    fcast_length::Int,             # number of forecasts to produce
    horizon::Int,                  # forecast horizon h
    p::Int                         # AR order
)
    # Center the data
    μ = mean(data0)
    y = data0 .- μ
    T = length(y)

    # Containers
    forecasts = zeros(fcast_length)
    realized = zeros(fcast_length)
    errors   = zeros(fcast_length)

    for ii in 0:(fcast_length-1)
        # Define estimation window
        train_start = ii + 1
        train_end   = ii + tt
        # Build OLS design
        nobs = tt - p
        X = ones(nobs, p+1)
        Y = zeros(nobs)
        for j in 1:nobs
            t = train_start + p - 1 + j
            Y[j] = y[t]
            X[j, 2:end] = y[t-1:-1:t-p]
        end
        β = X \ Y  # OLS solution

        # Recursive h-step forecast
        history = copy(y[train_end-p+1:train_end])
        h_fore = zeros(horizon)
        for h in 1:horizon
            pred = β[1] + dot(β[2:end], reverse(history))
            h_fore[h] = pred
            push!(history, pred)
            popfirst!(history)
        end
        forecasts[ii+1] = mean(h_fore)
        # Realized mean over next h
        idx = train_end+1 : train_end+horizon
        realized[ii+1] = mean(y[idx])
        errors[ii+1]   = forecasts[ii+1] - realized[ii+1]
    end

    return (forecasts, realized, errors)
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
function TVAR_forecast(data0,tt, p, fcast_length,horizon,kernel_width_ARtvp; kernel_type = "triweight")
    
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
    
        #!!!!! REVERSE ? !!!!! 
        est_sample_tvp = reverse(r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)])
        # Generate forecasts
        forecasts_ar1 = mean(forecast_tvAR(est_sample_tvp, p, kernel_width_ARtvp, horizon; tkernel = kernel_type,
        include_intercept = true))
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
