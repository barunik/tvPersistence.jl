
# nohup julia Inflation_Restat_R1.jl > out_nohup_v1"`date +%FT%H%M`".txt &@e´´@@
using Pkg
Pkg.activate(".")
using CSV, DataFrames, GLM, Distributions, LinearAlgebra, Statistics
using BSON, Dates
#using XLSX

data = CSV.read("PCEpi.csv", DataFrame) 
data = data .* 100;
rvfx=data[:,1];

println("All things loaded. ", now())
flush(stdout)


########## HORIZON 1 ##########################

horizon=1

### Chose parameters
maxAR					= 2;   # original with AR=5
kernel_width_for_const 	= 0.05; # deterministic (constant) forecast
kernel_width_IRF		= 0.2;  # IRFs on errors
kernel_width_forecast 	= 0.6;  # forecast of constant
AR_lag_forecast 		= 1;	# AR lag choice for forecast of constant
kernel_width_HAR		= 0.3; 	# kernel for TVP-AR3 TVP-HAR model 
JMAX=5;  

tt=610;

import TvPersistence: ARp_forecast, TVHAR_forecast, HAR_forecast, tvEWD_forecast, EWD_forecast, TVAR_forecast

function RW(data0,tt,fcast_length,horizon)
    
    T=length(data0)
    muR=mean(data0)
    chronR=data0.-muR;
    r=reverse(chronR);

    RVh = zeros(length(r)-horizon+1,1);
    for i=1:length(RVh)
        RVh[i]=mean(r[i:i+horizon-1])
    end

    
    horizon_forecast_AR= zeros(fcast_length);
    Error_Jcomp_tvp=zeros(fcast_length);
    errs0_last=zeros(fcast_length);
    forecasted_constant=zeros(fcast_length);

    for ii=0:(fcast_length-1)
        #!!!!! REVERSE ? !!!!! 
        est_sample_tvp = reverse(r[(fcast_length+horizon - ii): (fcast_length+tt+horizon-1-ii)])
        
        forecasts_ar1 =[]
        forecasts_ar1 = est_sample_tvp[end];
        
        horizon_forecast_AR[ii+1] = forecasts_ar1;
 
        # error with J components
        Error_Jcomp_tvp[ii+1]=  (horizon_forecast_AR[ii+1]-RVh[fcast_length-ii])
 
    end
    return (horizon_forecast_AR,Error_Jcomp_tvp)
end

errors=[]
	println(now()," h=",horizon,"   tt=",tt)
    flush(stdout)

	fcast_length = length(rvfx)-tt-21-horizon; #length of forecast sample

	(horizon_forecast_RW,Error_RW) = RW(rvfx,tt,fcast_length,horizon);

	(horizon_forecast_AR2,Error_AR2) = ARp_forecast(rvfx,tt,fcast_length,horizon,2); # AR2
	(horizon_forecast_AR7,Error_AR7) = ARp_forecast(rvfx,tt,fcast_length,horizon,7);

	#=out0=hcat(pmap(i -> EWD_parallel(i,rvfx,tt,maxAR,JMAX,horizon), 0:(fcast_length-1))...);
	horizon_forecast_EWD=out0[1,:];
	RV_h=out0[2,:];
	Error_EWD=out0[3,:];=#

	# EWD forecast
	horizon_forecast_EWD, RV_h, Error_EWD = EWD_forecast(rvfx, tt, maxAR, JMAX, horizon, fcast_length)

	#out=hcat(pmap(i -> TV(i,rvfx,tt,fcast_length,horizon,kernel_width_HAR,2), 0:(fcast_length-1))...);
	#Error_AR2tv=out[2,:];h

	horizon_forecast_AR2tv, Error_AR2tv = TVAR_forecast(rvfx, tt, 2, fcast_length, horizon, kernel_width_HAR)

	#=out=hcat(pmap(i -> EWD_tvLS_parallel(i,rvfx,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast), 0:(fcast_length-1))...);
	horizon_forecast_EWDtv=out[1,:];
	Error_EWDtv=out[2,:];=#
    horizon_forecast_EWDtv, _, Error_EWDtv = tvEWD_forecast(rvfx, tt, horizon, maxAR, AR_lag_forecast, JMAX, kernel_width_for_const, kernel_width_IRF,
	kernel_width_forecast; kernel_type = "triweight", forecast_window_size = fcast_length)


	res=[Error_RW Error_AR2 Error_AR7 Error_EWD Error_AR2tv Error_EWDtv]

    forecasts=[RV_h horizon_forecast_RW horizon_forecast_AR2 horizon_forecast_AR7 horizon_forecast_EWD horizon_forecast_AR2tv horizon_forecast_EWDtv]

	push!(errors,res)

output_bson = Dict("errors" => errors, "tt" => tt,"horizon" => horizon, "forecasts"=> forecasts)
BSON.bson("inflation_results_replication/results_Inflation_$tt-$horizon.bson", output_bson)

println("Horizon 1 completed. ", now())
flush(stdout)



########## HORIZON 2 ##########################

horizon=2

### Chose parameters
maxAR					= 2;   # original with AR=5
kernel_width_for_const 	= 0.05; # deterministic (constant) forecast
kernel_width_IRF		= 0.2;  # IRFs on errors
kernel_width_forecast 	= 0.6;  # forecast of constant
AR_lag_forecast 		= 1;	# AR lag choice for forecast of constant
kernel_width_HAR		= 0.3; 	# kernel for TVP-AR3 TVP-HAR model 
JMAX=5;  

tt=610;


errors=[]
	println(now()," h=",horizon,"   tt=",tt)
    flush(stdout)

	fcast_length = length(rvfx)-tt-21-horizon; #length of forecast sample

	(horizon_forecast_RW,Error_RW) = RW(rvfx,tt,fcast_length,horizon);

	(horizon_forecast_AR2,Error_AR2) = ARp_forecast(rvfx,tt,fcast_length,horizon,2); # AR2
	(horizon_forecast_AR7,Error_AR7) = ARp_forecast(rvfx,tt,fcast_length,horizon,7);

	#=out0=hcat(pmap(i -> EWD_parallel(i,rvfx,tt,maxAR,JMAX,horizon), 0:(fcast_length-1))...);
	horizon_forecast_EWD=out0[1,:];
	RV_h=out0[2,:];
	Error_EWD=out0[3,:];=#

	# EWD forecast
	horizon_forecast_EWD, RV_h, Error_EWD = EWD_forecast(rvfx, tt, maxAR, JMAX, horizon, fcast_length)

	#out=hcat(pmap(i -> TV(i,rvfx,tt,fcast_length,horizon,kernel_width_HAR,2), 0:(fcast_length-1))...);
	#Error_AR2tv=out[2,:];h

	horizon_forecast_AR2tv, Error_AR2tv = TVAR_forecast(rvfx, tt, 2, fcast_length, horizon, kernel_width_HAR)

	#=out=hcat(pmap(i -> EWD_tvLS_parallel(i,rvfx,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast), 0:(fcast_length-1))...);
	horizon_forecast_EWDtv=out[1,:];
	Error_EWDtv=out[2,:];=#
    horizon_forecast_EWDtv, _, Error_EWDtv = tvEWD_forecast(rvfx, tt, horizon, maxAR, AR_lag_forecast, JMAX, kernel_width_for_const, kernel_width_IRF,
	kernel_width_forecast; kernel_type = "triweight", forecast_window_size = fcast_length)


	res=[Error_RW Error_AR2 Error_AR7 Error_EWD Error_AR2tv Error_EWDtv]

    forecasts=[RV_h horizon_forecast_RW horizon_forecast_AR2 horizon_forecast_AR7 horizon_forecast_EWD horizon_forecast_AR2tv horizon_forecast_EWDtv]

	push!(errors,res)

output_bson = Dict("errors" => errors, "tt" => tt,"horizon" => horizon, "forecasts"=> forecasts)
BSON.bson("inflation_results_replication/results_Inflation_$tt-$horizon.bson", output_bson)

println("Horizon 2 completed. ", now())
flush(stdout)


########## HORIZON 6 ##########################

horizon=6

### Chose parameters

maxAR					= 2;  
kernel_width_for_const 	= 0.05; # deterministic (constant) forecast
kernel_width_IRF		= 0.2;  # IRFs on errors
kernel_width_forecast 	= 0.6;  # forecast of constant (K3)
AR_lag_forecast 		= 1;	# AR lag choice for forecast of constant
kernel_width_HAR		= 0.3; 	# kernel for TVP-AR3 TVP-HAR model 
JMAX=5;  

tt=610;


errors=[]
	println(now()," h=",horizon,"   tt=",tt)
    flush(stdout)

	fcast_length = length(rvfx)-tt-21-horizon; #length of forecast sample

	(horizon_forecast_RW,Error_RW) = RW(rvfx,tt,fcast_length,horizon);

	(horizon_forecast_AR2,Error_AR2) = ARp_forecast(rvfx,tt,fcast_length,horizon,2); # AR2
	(horizon_forecast_AR7,Error_AR7) = ARp_forecast(rvfx,tt,fcast_length,horizon,7);

	#=out0=hcat(pmap(i -> EWD_parallel(i,rvfx,tt,maxAR,JMAX,horizon), 0:(fcast_length-1))...);
	horizon_forecast_EWD=out0[1,:];
	RV_h=out0[2,:];
	Error_EWD=out0[3,:];=#

	# EWD forecast
	horizon_forecast_EWD, RV_h, Error_EWD = EWD_forecast(rvfx, tt, maxAR, JMAX, horizon, fcast_length)

	#out=hcat(pmap(i -> TV(i,rvfx,tt,fcast_length,horizon,kernel_width_HAR,2), 0:(fcast_length-1))...);
	#Error_AR2tv=out[2,:];h

	horizon_forecast_AR2tv, Error_AR2tv = TVAR_forecast(rvfx, tt, 2, fcast_length, horizon, kernel_width_HAR)

	#=out=hcat(pmap(i -> EWD_tvLS_parallel(i,rvfx,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast), 0:(fcast_length-1))...);
	horizon_forecast_EWDtv=out[1,:];
	Error_EWDtv=out[2,:];=#
    horizon_forecast_EWDtv, _, Error_EWDtv = tvEWD_forecast(rvfx, tt, horizon, maxAR, AR_lag_forecast, JMAX, kernel_width_for_const, kernel_width_IRF,
	kernel_width_forecast; kernel_type = "triweight", forecast_window_size = fcast_length)


	res=[Error_RW Error_AR2 Error_AR7 Error_EWD Error_AR2tv Error_EWDtv]

    forecasts=[RV_h horizon_forecast_RW horizon_forecast_AR2 horizon_forecast_AR7 horizon_forecast_EWD horizon_forecast_AR2tv horizon_forecast_EWDtv]

	push!(errors,res)

output_bson = Dict("errors" => errors, "tt" => tt,"horizon" => horizon, "forecasts"=> forecasts)
BSON.bson("inflation_results_replication/results_Inflation_$tt-$horizon.bson", output_bson)

println("Horizon 6 completed. ", now())
flush(stdout)

########## HORIZON 12 ##########################

horizon=12

### Chose parameters

maxAR					= 3;  
kernel_width_for_const 	= 0.05; # deterministic (constant) forecast
kernel_width_IRF		= 0.2;  # IRFs on errors
kernel_width_forecast 	= 0.6;  # forecast of constant (K3)
AR_lag_forecast 		= 1;	# AR lag choice for forecast of constant
kernel_width_HAR		= 0.3; 	# kernel for TVP-AR3 TVP-HAR model 
JMAX=5;  

tt=610;


errors=[]
	println(now()," h=",horizon,"   tt=",tt)
    flush(stdout)

	fcast_length = length(rvfx)-tt-21-horizon; #length of forecast sample

	(horizon_forecast_RW,Error_RW) = RW(rvfx,tt,fcast_length,horizon);

	(horizon_forecast_AR2,Error_AR2) = ARp_forecast(rvfx,tt,fcast_length,horizon,2); # AR2
	(horizon_forecast_AR7,Error_AR7) = ARp_forecast(rvfx,tt,fcast_length,horizon,7);

	#=out0=hcat(pmap(i -> EWD_parallel(i,rvfx,tt,maxAR,JMAX,horizon), 0:(fcast_length-1))...);
	horizon_forecast_EWD=out0[1,:];
	RV_h=out0[2,:];
	Error_EWD=out0[3,:];=#

	# EWD forecast
	horizon_forecast_EWD, RV_h, Error_EWD = EWD_forecast(rvfx, tt, maxAR, JMAX, horizon, fcast_length)

	#out=hcat(pmap(i -> TV(i,rvfx,tt,fcast_length,horizon,kernel_width_HAR,2), 0:(fcast_length-1))...);
	#Error_AR2tv=out[2,:];h

	horizon_forecast_AR2tv, Error_AR2tv = TVAR_forecast(rvfx, tt, 2, fcast_length, horizon, kernel_width_HAR)

	#=out=hcat(pmap(i -> EWD_tvLS_parallel(i,rvfx,tt,maxAR,JMAX,horizon,kernel_width_for_const,kernel_width_IRF,kernel_width_forecast,AR_lag_forecast), 0:(fcast_length-1))...);
	horizon_forecast_EWDtv=out[1,:];
	Error_EWDtv=out[2,:];=#
    horizon_forecast_EWDtv, _, Error_EWDtv = tvEWD_forecast(rvfx, tt, horizon, maxAR, AR_lag_forecast, JMAX, kernel_width_for_const, kernel_width_IRF,
	kernel_width_forecast; kernel_type = "triweight", forecast_window_size = fcast_length)


	res=[Error_RW Error_AR2 Error_AR7 Error_EWD Error_AR2tv Error_EWDtv]

    forecasts=[RV_h horizon_forecast_RW horizon_forecast_AR2 horizon_forecast_AR7 horizon_forecast_EWD horizon_forecast_AR2tv horizon_forecast_EWDtv]

	push!(errors,res)

output_bson = Dict("errors" => errors, "tt" => tt,"horizon" => horizon, "forecasts"=> forecasts)
BSON.bson("inflation_results_replication/results_Inflation_$tt-$horizon.bson", output_bson)

println("Horizon 12 completed. ", now())
flush(stdout)





