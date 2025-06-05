# tvPersistence.jl

The code has been developed in Julia as a code accompanying the Barunik and Vacha (2023) and Barunik and Vacha (2024) papers, and provides estimation of time-varying persistence using *localised heterogeneous persistence*:

Baruník, J. and Vacha, L. (2023): *The Dynamic Persistence of Economic Shocks*, forthcoming in the Review of Economics and Statistics,  [link](https://ideas.repec.org/p/arx/papers/2306.01511.html)

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
using CSV, DataFrames,GLM
using Distributions, LinearAlgebra, Statistics
using Plots, StatsBase, StatsPlots, Colors
using Random
using BSON: @save, @load
```

Load files containing core functions:

```julia
include("TV-EWD Implementation/TV-EWD_forecast.jl") # Forecasting function
include("TV-EWD Implementation/persistence_plot.jl") # Dynamic Persistence decomposition plots
include("Pockets of Predictability.jl") # Pockets of Predictability extraction and plots
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

### Part 3: Find and plot Pockets of Predictability
Here we compare the TV-EWD forecasting approach with the benchmark HAR model through Pockets of Predictability, generating a plot that clearly shows non-spurious pockets given a confidence threshold obtained through bootstrap simulations:

#### Step 1: Calculate the threshold
```julia
include("bootstrap_thresholds.jl") # file containing the function


include("bootstrap_thresholds.jl")

ar_order = 22
in_sample_window = 1000
forecast_horizon = 22
num_replicates = 30
smoothing_bw = 0.05
cutoff_start = 100
forecast_length = 100

# Choose one benchmark and one comparison method:
bench = :HAR
comp  = :tvEWD  # or :TVHAR, or :tvEWD

@load "all_forecasts_V2.bson" forecasts
har_e    = forecasts.har_e
TV_EWD_e  = forecasts.TV_EWD_e

threshold_fixed = calculate_bootstrap_threshold(
    data0,
    ar_order,
    in_sample_window,
    forecast_horizon,
    num_replicates,
    smoothing_bw,
    cutoff_start,
    bench,
    comp;
    forecast_length,
    random_seed = 42,
    tvp_kernel_width = 0.4,
    kernel_type = "Gaussian",
    max_ar_order = 15,
    jmax_scale = 7,
    ar_lag_for_trend = 1,
    tvp_constant_kernel_width = 0.1,
    irf_kernel_width = 0.2,
    forecast_kernel_width = 0.4
)
```

#### Step 2: Generate forecasts of TV-EWD and the benchmark model, while saving the dates of forecasted values

```julia
# Benchmark Forecasting functions
include("TV-EWD Implementation/Benchmark Forecasts.jl")

#––– Parameters –––
tt           = 1000 # Fisrt 1000 days for model fitting
fcast_length = 2280 # rolling-window forecasts until the end
horizon      = 1
bw           = 0.3 # Kernel bandwidth for TV-OLS based models
p            = 1   # AR order for AR and TV‐AR

#––– Generate forecasts –––
TV_EWD_f, TV_EWD_r, TV_EWD_e = tvEWD_forecast(data0, tt, 1, 2, 1, 5, 0.05, 0.2, 0.5, 
    kernel_type = "Epa",
    LASSO_scale_selection = false,
    forecast_window_size = fcast_length); # Scales 1-7
har_f,    har_r,    har_e    = HAR_forecast(data0, tt, fcast_length, horizon);

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

```julia
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
