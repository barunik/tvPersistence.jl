#!/usr/bin/env bash
set -euo pipefail

# run_sed_threshold.sh - calculate SED threshold via bootstrap simulations
#
# Usage:
#   ./run_sed_threshold.sh -n NUM_SIMULATIONS -c NUM_CORES -f CONFIG_FILE
#
# Options:
#   -n    Number of bootstrap replicates to run
#   -c    Number of worker processes (cores) to launch
#   -f    Path to a key=value config file with all other arguments
#   -h    Show this help message and exit
#
# CONFIG_FILE format (one key=value per line, comments with #):
#   data_file=path/to/example_data.csv
#   data_column=A
#   scale_multiplier=100.0
#   missingstring=NA
#   ar_order=5
#   in_sample_window_size=1000
#   forecast_horizon=1
#   forecast_length=500
#   smoothing_bandwidth=0.4
#   cutoff_start_index=100
#   benchmark_method=HAR
#   comparison_method=tvEWD
#   random_seed=1234
#   tvp_kernel_width=0.4
#   kernel_type=Gaussian
#   max_ar_order=2
#   jmax_scale=5
#   ar_lag_for_trend=1
#   tvp_constant_kernel_width=0.05
#   irf_kernel_width=0.2
#   forecast_kernel_width=0.5
#   alpha_level=0.15

usage() {
  sed -n '2,16p' "$0"
  exit 1
}

# handle help
if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
  usage
fi

# parse flags
while getopts "hn:c:f:" opt; do
  case $opt in
    h) usage ;;
    n) NUM_SIM=$OPTARG ;;
    c) NUM_CORES=$OPTARG ;;
    f) CONFIG_FILE=$OPTARG ;;
    *) usage ;;
  esac
done

# ensure required args
if [ -z "${NUM_SIM:-}" ] || [ -z "${NUM_CORES:-}" ] || [ -z "${CONFIG_FILE:-}" ]; then
  echo "Error: missing required arguments." >&2
  usage
fi

if [ ! -r "$CONFIG_FILE" ]; then
  echo "Error: Cannot read config file '$CONFIG_FILE'" >&2
  exit 1
fi

random_seed=$(grep '^random_seed=' "$CONFIG_FILE" | cut -d= -f2 | tr -d '[:space:]')

# run Julia—UNQUOTED heredoc so Bash expands $NUM_SIM, $NUM_CORES, $CONFIG_FILE
julia --project=. <<JULIA
using Pkg; Pkg.activate("."); Pkg.instantiate();
using CSV, DataFrames, Distributed, Random, Statistics;

# embed Bash variables directly
const num_replicates = $NUM_SIM
const num_workers    = $NUM_CORES
const config_file    = raw"$CONFIG_FILE"

# read config
cfg = Dict{String,String}()
for line in eachline(config_file)
    l = strip(line)
    if isempty(l) || startswith(l, "#")
        continue
    end
    k, v = split(l, "=", limit=2)
    cfg[strip(k)] = strip(v)
end

# load & preprocess
data_file       = cfg["data_file"];
col             = cfg["data_column"];
scale           = parse(Float64, cfg["scale_multiplier"]);
missingstr      = cfg["missingstring"];
df              = CSV.File(data_file, missingstring=[missingstr], header=true) |> DataFrame;
series          = scale .* Float64.(df[.!ismissing.(getproperty(df, Symbol(col))), Symbol(col)]);

# assign all required arguments
ar_order            = parse(Int,    cfg["ar_order"]);
in_sample_window    = parse(Int,    cfg["in_sample_window_size"]);
forecast_horizon    = parse(Int,    cfg["forecast_horizon"]);
smoothing_bw        = parse(Float64,cfg["smoothing_bandwidth"]);
cutoff_idx          = parse(Int,    cfg["cutoff_start_index"]);
benchmark_method    = Symbol(cfg["benchmark_method"]);
comparison_method   = Symbol(cfg["comparison_method"]);
forecast_length     = cfg["forecast_length"] == "Maximum" ? "Maximum" : parse(Int, cfg["forecast_length"]);
tvp_kernel_width    = parse(Float64,cfg["tvp_kernel_width"]);
kernel_type         = cfg["kernel_type"];
max_ar_order        = parse(Int,    cfg["max_ar_order"]);
jmax_scale          = parse(Int,    cfg["jmax_scale"]);
ar_lag_for_trend    = parse(Int,    cfg["ar_lag_for_trend"]);
tvp_const_bw        = parse(Float64,cfg["tvp_constant_kernel_width"]);
irf_kernel_width    = parse(Float64,cfg["irf_kernel_width"]);
forecast_kernel_w   = parse(Float64,cfg["forecast_kernel_width"]);
alpha_level         = parse(Float64,cfg["alpha_level"]);

# launch workers and run parallel bootstrap
addprocs(num_workers)
@everywhere using Random, Statistics;
@everywhere include("bootstrap_thresholds.jl");
@everywhere include("bootstrap_thresholds_parallel.jl");


rng_list = [MersenneTwister($random_seed + i) for i in 1:num_replicates]

sed_vals = pmap(i -> calculate_bootstrap_threshold_parallel(rng_list[i],
    i, series,
    ar_order, in_sample_window, forecast_horizon,
    smoothing_bw,
    benchmark_method, comparison_method;
    forecast_length        = forecast_length,
    tvp_kernel_width       = tvp_kernel_width,
    kernel_type            = kernel_type,
    max_ar_order           = max_ar_order,
    jmax_scale             = jmax_scale,
    ar_lag_for_trend       = ar_lag_for_trend,
    tvp_constant_kernel_width = tvp_const_bw,
    irf_kernel_width       = irf_kernel_width,
    forecast_kernel_width  = forecast_kernel_w
), 1:num_replicates);

# compute and print final threshold
thr = compute_global_threshold(sed_vals, cutoff_idx, alpha_level);
println("SED threshold: ", thr)
JULIA
