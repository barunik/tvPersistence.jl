module TvPersistence

# imports and uses of external packages
using LinearAlgebra, Statistics, Distributions, StatsBase, GLM, GLMNet, MultivariateStats

# Core functionality files
include("tv_ols.jl") # Local linear regression estimation and TV-AR forecasts
include("basic_irf.jl") # Basic impulse response function calculations
include("helper_functions.jl") # Lagged data matrix creation
include("time_varying_irf.jl") # Impulse Response Function calculations based on TV-AR models
include("tv_ewd.jl") # calculation of components for TV-EWD

# TV-EWD visualization
include("persistence_plot.jl")

# Forecasting files
include("benchmark_forecasts.jl") # HAR, TV-AR, AR, EWD forecasting functions
include("tv_ewd_forecast.jl") # TV-EWD forecasting function -> main output function

export tvEWD_forecast, tv_persistence_plot, ARp_forecast, HAR_forecast_legacy, TVHAR_forecast, EWD_forecast, TVARp_forecast

end