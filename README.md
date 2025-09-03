# tvPersistence.jl

The code has been developed in Julia as a code accompanying the Barunik and Vacha (2023) and Barunik and Vacha (2024) papers, and provides estimation of time-varying persistence using *localised heterogeneous persistence*:

Baruník, J. and Vacha, L. (2025): *The Dynamic Persistence of Economic Shocks*, forthcoming in the Review of Economics and Statistics,  [link](https://ideas.repec.org/p/arx/papers/2306.01511.html)

Baruník, J., & Vácha, L. (2024). *Predicting the volatility of major energy commodity prices: The dynamic persistence model*. Energy Economics, 140, 107982 [link](https://doi.org/10.1016/j.eneco.2024.107982)

*Package created by Jiří Mikulenka based on the original codes by J.Barunik and L.Vacha*

## Software requirements

Install [Julia](http://julialang.org/) version 1.10.5 or newer and with the first use of this code install the same version of packages with which the projects is built and work in the environment of the project as

```julia
using Pkg
Pkg.activate(".") # activating project in its directory
Pkg.instantiate() # installing packages with which versions the project is built
```

## Example usage

This example iillustrates how to obtain the decomposition of dynamic persistence as well as forecasts on a sample series of Realized Volatility of returns on Agilent Technologies stock.

Load packages:

```julia
using CSV, DataFrames, BSON, Random, Dates, Plots, StatsBase
using BSON: @save, @load
```

Load modules containing core functions:

```julia
# Core TV-EWD functionality module
include("src/TvPersistence/TvPersistence.jl")
using .TvPersistence
# SED Threshold calculations and Pockets of Predictability functionality
include("src/SED_Thresholds/SEDThresholds.jl")
using .SEDThresholds
```

Load example data:

```julia
data_read=CSV.File("example_data.csv",missingstring=["NA"],header=true) |> DataFrame;
data0=100.0.*data_read.A[ismissing.(data_read.A).==false];
date_vector = data_read[ismissing.(data_read.A).==false,:dates];
date_vector = Date.(date_vector, "dd.mm.yyyy");
```

### Part 1: Plot persistence decomposition
Here we calculate the multiscale impulse response functions $\beta^{\{j\}}(u,k)$ and plot the ratios $\frac{\beta^{\{j\}}(u,k)}{\sum_j \beta^{\{j\}}(u,k)}$ over time to provide a visual overview of the relative importance of shock components persistent at different scales. Specifically, we can see persistence components at horizons of 2,4,8,16,32,64,128 days.
```julia
# Calculate multiscale impulse response functions
decomp_new = tv_persistence_plot(data0,5,7,0.15,0.02, "Gaussian", "Gaussian");
# Rescale the decomposition
yearfirstb_new=decomp_new./sum(decomp_new,dims=2);

# Generate the plot
year_ticks = unique(year.(date_vector[6:end]))
xtick_dates = Date.(year_ticks,1,1)
myrainbow=reverse(cgrad(:RdYlBu_7, 7, categorical = true));

plot(date_vector[6:end],yearfirstb_new,size=(700,700/1.6666),color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],frame=:box,
    linestyle=:dot,linealpha=0.7,label=false,legend=:topleft,yaxis="A") 
xticks!(Dates.value.(xtick_dates), string.(year_ticks))
scatter!(date_vector[6:12:end],yearfirstb_new[1:12:size(yearfirstb_new,1),:],color=[myrainbow[1] myrainbow[2] myrainbow[3] cgrad(:grayC, 7, categorical = true)[2] myrainbow[5] myrainbow[6] myrainbow[7]],
    label=["2 days" "4" "8" "16" "32" "64" "128+"],msc=:white,markersize=3,markershape=[:circle :diamond :utriangle :+ :x :heptagon :dtriangle])
```

![svg](/readme_files/Persistence_plot_example.svg)

### Part 2: Generate TV-EWD forecasts
The function tvEWD_forecast allows the user to generate rolling-window forecasts using the TV-EWD approach:

```julia
forecast, actual, error = tvEWD_forecast(
    data0, # Univariate time series vector
    window_size::Int, # Rolling window size 
    horizon::Int, # Forecast horizon
    maxAR::Int, # TV-AR model order for our series
    AR_lag_forecast::Int, # TV-AR model order for trend forecasting (TV-AR(1) in the original paper)
    JMAX::Int, # Maximal component scale j
    kernel_width_for_const    ::Float64, # Bandwidth of the kernel used for the trend estimation
    kernel_width_IRF          ::Float64, # Bandwidth of the kernel used for the TVP IRF estimation
    kernel_width_forecast     ::Float64; # Bandwidth of the kernel used for forecasting of the constant
    kernel_type::String = "Epanechnikov", # Kernel type for all local estimations
    user_specified_scales::Union{Nothing, Vector{Int}} = nothing, # Optional user-specified scales in ascending order
    LASSO_scale_selection::Bool = false, # Optional to use LASSO regression for automatic scale selection
    forecast_window_size::Union{String, Int} = "Maximum" # Out-of-sample size, set to maximal length by default
);
```

Example of h = 1 step ahead forecast of Realized Volatility of Agilent Technologies using scales j = 1,...5:

```julia
# Load example data
data_read=CSV.File("example_data.csv",missingstring=["NA"],header=true) |> DataFrame;
data0=100.0.*data_read.A[ismissing.(data_read.A).==false];

############ TV-EWD forecast ##################
forecast_test, actual_test, forecast_error_test = tvEWD_forecast(data0, 1000, 1, 2, 1, 5, 0.05, 0.2, 0.5, 
    kernel_type = "Epa",
    LASSO_scale_selection = false, forecast_window_size = 100);
```

```julia
# Plot the forecasted values against realized values
display(plot([actual_test forecast_test], label=["Data" "Forecast"],frame=:box))
```

![svg](/readme_files/TV-EWD_forecast_example.svg)

#### Inflation forecasts against benchmarks
These results can be replicated through the code in "inflation_results_replication/Revision_inflation_REPLICATION.ipynb"
<p><strong>Forecast errors across horizons (in months) relative to Random Walk forecasts</strong></p>
<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Model</th>
      <th align="center" colspan="2">h = 1</th>
      <th align="center" colspan="2">h = 2</th>
      <th align="center" colspan="2">h = 6</th>
      <th align="center" colspan="2">h = 12</th>
    </tr>
    <tr>
      <th align="right">RMSE</th><th align="right">MAE</th>
      <th align="right">RMSE</th><th align="right">MAE</th>
      <th align="right">RMSE</th><th align="right">MAE</th>
      <th align="right">RMSE</th><th align="right">MAE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>AR</td>
      <td align="right">0.8933</td><td align="right">0.9316</td>
      <td align="right">0.8306</td><td align="right">0.8897</td>
      <td align="right">0.7570</td><td align="right">0.8726</td>
      <td align="right">0.8756</td><td align="right">1.0953</td>
    </tr>
    <tr>
      <td>EWD</td>
      <td align="right">0.9110</td><td align="right">0.9408</td>
      <td align="right">0.8378</td><td align="right">0.8847</td>
      <td align="right">0.7151</td><td align="right">0.7446</td>
      <td align="right">0.7833</td><td align="right">0.7520</td>
    </tr>
    <tr>
      <td>TV-AR</td>
      <td align="right">0.9781</td><td align="right">1.0384</td>
      <td align="right">0.7898</td><td align="right">0.8579</td>
      <td align="right">0.6997</td><td align="right">0.7070</td>
      <td align="right">0.8025</td><td align="right">0.7377</td>
    </tr>
    <tr>
      <td>EWDtv</td>
      <td align="right">0.8452</td><td align="right">0.8830</td>
      <td align="right">0.7602</td><td align="right">0.7922</td>
      <td align="right">0.6546</td><td align="right">0.6887</td>
      <td align="right">0.7528</td><td align="right">0.7199</td>
    </tr>
  </tbody>
</table>

### Part 3: Find and plot Pockets of Predictability
Here we compare the TV-EWD forecasting approach with the benchmark HAR model through Pockets of Predictability, generating a plot that clearly shows non-spurious pockets given a 95% confidence threshold obtained through bootstrap simulations:

#### Step 1: Calculate the threshold

Open the notebook SED_threshold_example.ipynb (Make sure you have Julia Kernel installed and connected to Jupyter). The main function handling the calculations of the thresholds allows to parallelize the computation among multiple cores (see documentation)

```julia
function calculate_bootstrap_threshold_parallel(i, # iteration number
        series::Vector{Float64}, # univariate time-series
        ar_order::Int, # AR order for bootstrap-resampling
        in_sample_window_size::Int,
        forecast_horizon::Int,
        smoothing_bandwidth::Float64, # badnwidth for smoothed SED regression
        benchmark_method::Symbol, # Method with which we compare TV-EWD forecasting performance (:HAR in our case)
        comparison_method::Symbol; # Method we are interested in (:tvEWD in our case)
        fcast_len::Int,
        tvp_kernel_width::Float64 = 0.4, # Kernel width for TV-AR and TV-HAR forecasting (irrelevant if using HAR and TV-EWD)
        smoothing_kernel::String = "triweight", # SED regression kernel type
        kernel_type_tvEWD::String = "Gaussian",
        kernel_type_tvHAR::String = "Gaussian",
        kernel_type_tvAR::String = "Gaussian",
        max_ar_order::Int = 1, # TV-EWD AR order for Impulse Response Function calculations
        jmax_scale::Int = 7, # Maximal scale we are interested in
        ar_lag_for_trend::Int = 1, # Trend forecasting for TV-EWD AR order
        tvp_constant_kernel_width::Float64 = 0.1,
        irf_kernel_width::Float64 = 0.2,
        forecast_kernel_width::Float64 = 0.4,
    )
```

##### Usage

We use median volatility of 496 S&P500 stocks as our data series to calculate the threshold. First, import necessary packages:

```julia
using Distributed
using CSV, DataFrames, BSON, Random
```

Next, set the number of bootstrap simulations and number of cores you want to use in your computation:

```julia
num_workers = 4
num_replicates = 100
addprocs(num_workers)
```

Export necessary information for the calculation to all cores:

```julia
@everywhere begin
    data_file                    = "data/median_RV.csv"
    data_column                  = "x1"
    missingstring                = "NA"

    ar_order                     = 1
    in_sample_window_size        = 1000
    forecast_horizon             = 1
    forecast_length              = 2130
    random_seed                  = 1234

    smoothing_bandwidth          = 0.05
    cutoff_start_index           = 100

    benchmark_method             = "HAR"
    comparison_method            = "tvEWD"

    tvp_kernel_width             = 0.4
    kernel_type                  = "Epanechnikov"
    max_ar_order                 = 1
    jmax_scale                   = 5
    ar_lag_for_trend             = 1
    tvp_constant_kernel_width    = 0.1
    irf_kernel_width             = 0.2
    forecast_kernel_width        = 0.5
    smoothing_kernel             = "one-sided"
    kernel_type_tvEWD            = "Epanechnikov"
    kernel_type_tvHAR            = "Epanechnikov"
    kernel_type_tvAR             = "Epanechnikov"

    alpha_level                  = 0.05 # 1- confidence level we want (95% in this case)
end
```

Export the module containing core functionality for SED threshold calculations:

```julia
const SED_PATH = abspath("src/SED_Thresholds/SEDThresholds.jl")
@everywhere include($SED_PATH)        # <— absolute path shipped to workers
@everywhere using .SEDThresholds
```

Load the data:

```julia
df = CSV.File(data_file, missingstring=[missingstring], header=true) |> DataFrame;

# turn column name into a Symbol, drop missings & scale
col_sym = Symbol(data_column);
series  = Float64.(df[.!ismissing.(df[!, col_sym]), col_sym]);
```

Obtain vectors of fitted smoothed SED values:

```julia
sed_vals = pmap(1:num_replicates) do i
    # re-seed for reproducibility
    Random.seed!(random_seed + i)

    calculate_bootstrap_threshold_parallel(
        i, series,
        ar_order, in_sample_window_size, forecast_horizon,
        smoothing_bandwidth,
        Symbol(benchmark_method), Symbol(comparison_method);
        fcast_len                  = forecast_length,
        tvp_kernel_width           = tvp_kernel_width,
        kernel_type_tvEWD          = kernel_type_tvEWD,
        kernel_type_tvHAR          = kernel_type_tvHAR,
        kernel_type_tvAR           = kernel_type_tvAR,
        smoothing_kernel           = smoothing_kernel,
        max_ar_order               = max_ar_order,
        jmax_scale                 = jmax_scale,
        ar_lag_for_trend           = ar_lag_for_trend,
        tvp_constant_kernel_width  = tvp_constant_kernel_width,
        irf_kernel_width           = irf_kernel_width,
        forecast_kernel_width      = forecast_kernel_width
    )
end;

# remove working processes
rmprocs(workers())
```

Finally, calculate the threshold and (optionally) save the SED vectors in a BSON file so as to not have to run the calculation again if needed:

```julia
thr = SEDThresholds.compute_global_threshold(sed_vals, cutoff_start_index, alpha_level)
println("SED threshold: ", thr)
# Save the SED values into BSON file
BSON.@save "sed_thresholds.bson" sed_vals thr
```

#### Step 2: Generate forecasts of TV-EWD and the benchmark model, while saving the dates of forecasted values

To this, first load the module containing TV-EWD and benchmark model forecasting functions (or use example.jl)

```julia
include("src/TvPersistence/TvPersistence.jl")
using .TvPersistence
```

From here, we can generate HAR and TV-EWD forecasts

```julia
#––– Parameters –––
tt           = 1000 # Fisrt 1000 days for model fitting
fcast_length = 2258 # rolling-window forecasts until the end
horizon      = 1
bw           = 0.3 # Kernel bandwidth for TV-OLS based models
p            = 1   # AR order for AR and TV‐AR

#––– Generate forecasts –––
TV_EWD_f, TV_EWD_r, TV_EWD_e = tvEWD_forecast(data0, tt, 1, 2, 1, 5, 0.05, 0.2, 0.5, 
    kernel_type = "Epa",
    LASSO_scale_selection = false,
    forecast_window_size = fcast_length); # Scales 1-7
har_f,    har_r,    har_e    = HAR_forecast_legacy(data0, tt, fcast_length, horizon);

# Alternatively, load from the BSON file attached
@load "all_forecasts_V2.bson" forecasts
har_e = forecasts.har_e
TV_EWD_e = forecasts.TV_EWD_e

# Save the corresponding date vector for Pockets plotting
forecast_dates = date_vector[tt+1:tt+fcast_length]
```

#### Step 3: Plot the Pockets of Predictability
The plot_pockets() function generates a plot showcasing periods where the model of interest (TV-EWD in our case) achieves a better forecasting performance than the benchmark model (HAR in our case)

```julia
mycolor=[colorant"rgb(222,102,62)",colorant"rgb(255,145,43)",colorant"rgb(76,144,186)",colorant"rgb(43,194,194)",colorant"rgb(244,184,17)"]
plot_pockets(
    err_benchmark, # forecast errors of benchmark model
    err_model, # forecast errors of the model of interest
    date_vector, # date vector corresponding to forecast errors in length
    smoothing_bandwidth, # for local linear estimation of out of sample SED
    thresholds; # thresholds for identifying spurious Pockets of Predictability contained in an array. By default, 0.0 is included as well
    auto_xticks     = true, # automatically extract date ticks for the plot
    user_xticks     = nothing, # user-defined ticks as an array of positions in the error vector
    title           = "", # plot title
    pocket_colors   = [mycolor[3], mycolor[4]], # pockets colouring. Length of this array needs to coincide with length of "thresholds"
    pocket_alphas   = [0.6, 0.3],
    base_line_color = :white,
    sed_line_color  = mycolor[1],
    hline_color     = mycolor[3],
    hline_style     = :dash,
    plot_size       = (1000,200),
    framestyle      = :box,
xtickfontsize  = xtick_fontsize,
        ytickfontsize  = ytick_fontsize,
        ylabelfontsize = ylabel_fontsize)
```

The threshold used in the paper has been computed on severs as median of all 496 stocks and hence we here use the value to reproduce the plot

```julia

threshold_fixed = 2.066830453726511e-6

# Winsorize forecast errors
har_e = Float64.(winsor(forecasts.har_e, prop=0.05))
TV_EWD_e = Float64.(winsor(forecasts.TV_EWD_e, prop=0.05))

# plot the pockets
tvEWD_vs_HAR_pockets = plot_pockets(
    har_e, TV_EWD_e, forecast_dates,
    0.01, threshold_fixed;
    title = "TV-EWD vs HAR (h = 1)"
)
display(tvEWD_vs_HAR_pockets)
```

![svg](/readme_files/pockets_of_predictability_example.svg)
