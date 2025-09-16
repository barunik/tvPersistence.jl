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

## Example usage (Inflation)

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

## Example usage (Realized Volatility)

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


