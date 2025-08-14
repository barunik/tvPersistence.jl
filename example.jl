using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Core TV-EWD functionality module
include("src/TvPersistence/TvPersistence.jl")
using .TvPersistence

# SED Threshold calculations and Pockets of Predictability functionality
include("src/SED_Thresholds/SEDThresholds.jl")
using .SEDThresholds

# External packages
using CSV, DataFrames, BSON, Random, Dates, Plots, StatsBase
using BSON: @save, @load

# Load example data
data_read=CSV.File("data/example_data.csv",missingstring=["NA"],header=true) |> DataFrame;

############ Replication for TV-EWD forecast ##################
data0=100.0.*data_read.A[ismissing.(data_read.A).==false];
date_vector = data_read[ismissing.(data_read.A).==false,:dates];
date_vector = Date.(date_vector, "dd.mm.yyyy")

################################# TV-EWD forecast #####################################################
forecast_test, actual_test, forecast_error_test = tvEWD_forecast(data0, 1000, 1, 2, 1, 5, 0.05, 0.2, 0.5, 
    kernel_type = "Epa",
    LASSO_scale_selection = false, forecast_window_size = 100);

display(plot([actual_test forecast_test], label=["Data" "Forecast"],frame=:box))

############ PERSISTENCE PLOT ##################
decomp_new = tv_persistence_plot(data0,5,7,0.15,0.02, "Gaussian", "Gaussian");
yearfirstb_new=decomp_new./sum(decomp_new,dims=2);

year_ticks = unique(year.(date_vector[6:end]))
xtick_dates = Date.(year_ticks,1,1)
myrainbow=reverse(cgrad(:RdYlBu_7, 7, categorical = true));

plot(date_vector[6:end],yearfirstb_new,size=(700,700/1.6666),color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],frame=:box,
    linestyle=:dot,linealpha=0.7,label=false,legend=:topleft,yaxis="A") 
xticks!(Dates.value.(xtick_dates), string.(year_ticks))
scatter!(date_vector[6:12:end],yearfirstb_new[1:12:size(yearfirstb_new,1),:],color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],
    label=["2 days" "4" "8" "16" "32" "64" "128+"],msc=:white,markersize=3,markershape=[:circle :diamond :utriangle :+ :x :heptagon :dtriangle])

#––– Forecast Parameters –––
tt           = 1000 # Fisrt 1000 days for model fitting
fcast_length = 2258 # rolling-window forecasts until the end
horizon      = 1
bw           = 0.3 # Kernel bandwidth for TV-OLS based models
p            = 1   # AR order for AR and TV‐AR

#––– Generate forecasts –––
TV_EWD_f, TV_EWD_r, TV_EWD_e = TvPersistence.tvEWD_forecast(data0, tt, 1, 2, 1, 5, 0.05, 0.2, 0.5, 
    kernel_type = "Epa",
    LASSO_scale_selection = false,
    forecast_window_size = fcast_length); # Scales 1-7

EWD_f,    EWD_r,    EWD_e    = TvPersistence.EWD_forecast(data0, tt, fcast_length, 1, 1,5); # Scales 1-7
ar1_f,    ar1_r,    ar1_e    = TvPersistence.ARp_forecast(data0, tt, fcast_length, horizon, p);
ar3_f,    ar3_r,    ar3_e    = TvPersistence.ARp_forecast(data0, tt, fcast_length, horizon, 3);
tvar1_f,  tvar1_r,  tvar1_e  = TvPersistence.TVAR_forecast(data0, tt, fcast_length, horizon, p, bw);
har_f, har_e, har_r,_    = TvPersistence.HAR_forecast_legacy(data0, tt, fcast_length, horizon);
tvhar_f,  tvhar_r,  tvhar_e  = TvPersistence.TVHAR_forecast(data0, tt, fcast_length, horizon, bw);

# Save the corresponding date vector for Pockets plotting
forecast_dates = date_vector[tt+1:tt+fcast_length]

forecasts = (
  TV_EWD_f = TV_EWD_f,
  TV_EWD_r = TV_EWD_r,
  TV_EWD_e = TV_EWD_e,

  EWD_f    = EWD_f,
  EWD_r    = EWD_r,
  EWD_e    = EWD_e,

  ar1_f    = ar1_f,
  ar1_r    = ar1_r,
  ar1_e    = ar1_e,

  ar3_f    = ar3_f,
  ar3_r    = ar3_r,
  ar3_e    = ar3_e,

  tvar1_f  = tvar1_f,
  tvar1_r  = tvar1_r,
  tvar1_e  = tvar1_e,

  har_f    = har_f,
  har_r    = har_r,
  har_e    = har_e,

  tvhar_f  = tvhar_f,
  tvhar_r  = tvhar_r,
  tvhar_e  = tvhar_e,
)
@save "all_forecasts_V2.bson" forecasts

#––– Build subplots –––
p1 = plot([ar1_r ar1_f],
    label = ["Data" "Forecast"],
    title = "AR(1)",
    frame = :box)

p2 = plot([tvar1_r tvar1_f],
    label = ["Data" "Forecast"],
    title = "TV-AR(1)",
    frame = :box)

p3 = plot([har_r har_f],
    label = ["Data" "Forecast"],
    title = "HAR",
    frame = :box)

p4 = plot([tvhar_r tvhar_f],
    label = ["Data" "Forecast"],
    title = "TV-HAR",
    frame = :box)

p5 = plot([ar3_r ar3_f],
    label = ["Data" "Forecast"],
    title = "AR(3)",
    frame = :box)

p6 = plot([EWD_f EWD_r],
    label = ["Data" "Forecast"],
    title = "EWD",
    frame = :box)

#––– Combine into a 2×2 grid –––
plot(p1, p2, p3, p4, p5, p6,
     layout = (2, 3),
     size   = (900, 700))

#––– Single overlay plot –––
plot(
    1:fcast_length,
    [ar1_r ar1_f ar3_f tvar1_f har_f tvhar_f EWD_f],
    label     = ["Realized" "AR(1)" "AR(3)" "TV-AR(1)" "HAR" "TV-HAR" "EWD"],
    linestyle = [:solid :dash :dot :dashdot :dashdotdot :dash],
    frame     = :box,
    xlabel    = "Rolling‐window step",
    ylabel    = "Centered RV",
    title     = "All Forecasts vs. Realized"
)

mycolor=[colorant"rgb(222,102,62)",colorant"rgb(255,145,43)",colorant"rgb(76,144,186)",colorant"rgb(43,194,194)",colorant"rgb(244,184,17)"]

### Pockets of Predictability ######

ar_order = 1
in_sample_window = 1000
forecast_horizon = 1
num_replicates = 10
smoothing_bw = 0.05
cutoff_start = 100
forecast_length = fcast_length

# Choose one benchmark and one comparison method:
bench = :HAR
comp  = :tvEWD  # or :TVHAR, or :tvEWD

@load "all_forecasts_V2.bson" forecasts
har_e    = forecasts.har_e
TV_EWD_e  = forecasts.TV_EWD_e

# Threshold calculated from SED_threshold_example.ipynb
threshold_fixed = 2.066830453726511e-6
println("95%-threshold for smoothed dSED (", bench, " vs ", comp, "): ", threshold_fixed)

# Imported from outside, possibly not correct thresholds
tvEWD_vs_HAR_pockets = SEDThresholds.plot_pockets(Float64.(winsor(har_e,prop=0.05)), Float64.(winsor(TV_EWD_e,prop=0.05)), forecast_dates, 0.01,threshold_fixed; title = "TV-EWD vs. HAR (h=1)")
display(tvEWD_vs_HAR_pockets)

savefig("pockets_of_predictability_example.svg")