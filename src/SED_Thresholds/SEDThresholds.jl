module SEDThresholds

using Random, Statistics, Dates, Colors, LinearAlgebra, StatsBase
using Plots: plot, plot!, vspan!, hline!

# Bring in TvPersistence via a relative include
import TvPersistence: ARp_forecast, TVAR_forecast, HAR_forecast, TVHAR_forecast, RW_forecast,
                      EWD_forecast, tvEWD_forecast, ARlags_chron, tvOLS, OLSestimator

include("sed_smoother.jl")
include("pockets_of_predictability.jl")
include("bootstrap_thresholds.jl")

export compute_global_threshold, calculate_bootstrap_threshold_parallel, SED_smooth_one, find_intervals, plot_pockets

end