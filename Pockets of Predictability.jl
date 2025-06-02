# Import necessary modules
include("models/tvOLS.jl")
include("TV-EWD Implementation/Benchmark Forecasts.jl")
using .tvOLS_estimator
using Dates
using Colors

"""
    find_intervals(vec::AbstractVector{Bool}) -> Base.Iterators.Zip{Tuple{Vector{Int},Vector{Int}}}

Identify contiguous runs of `true` values in a boolean vector.

# Arguments
- `vec::AbstractVector{Bool}`  
  A boolean mask where `true` indicates the region of interest.

# Returns
An iterator over `(start, stop)` index pairs (inclusive) for each contiguous block of `true`.

# Examples
```julia
julia> collect(find_intervals([false, true, true, false, true]))
2-element Array{Tuple{Int,Int},1}:
 (2, 3)
 (5, 5)
 """
function find_intervals(vec)
    diff_x = diff([0; vec; 0]) # Add 0 at start and end to detect edges
    starts = findall(x -> x == 1, diff_x) # Find rising edges
    ends = findall(x -> x == -1, diff_x).- 1 # Find falling edges
    return zip(starts, ends)
end

"""
    SED_smooth_one(
        bench_err::AbstractVector{<:Real},
        model_err::AbstractVector{<:Real},
        kernel_width::Real;
        kernel_type::String = "one-sided"
    ) -> Vector{Float64}

Compute the smoothed squared-error difference (SED) between two forecast-error series
using a local‐regression (tvOLS) smoother.

# Arguments
- `bench_err`: forecast errors from the benchmark model.
- `model_err`: forecast errors from the new model.
- `kernel_width`: bandwidth parameter for the local regression.
- `kernel_type`: smoothing kernel type (e.g., `"one-sided"` or `"two-sided"`).

# Returns
- A vector of fitted (smoothed) SED values
"""
function SED_smooth_one(benchmark_forecast_error,# Vector of benchmark model forecast errors
        model_forecast_error, # Vector of new model forecast errors
        kernel_width,
        kernel_type::String = "one-sided")
    data=benchmark_forecast_error.^2 .- model_forecast_error.^2; # Squared Error Difference as the dependent variable
    
    t = [t for t in 1.0:1.0:length(data)] # time periods as independent variables
    result1 = tvOLS_estimator.tvOLS(t, data, kernel_width, kernel_type) # we need fitted values to estimate the SED
    return result1.fitted
end

# Helper: derive sensible tick positions & labels from a date vector
function derive_date_ticks(date_vector; max_ticks=10)
    # assume date_vector is sorted
    first_date = first(date_vector)
    last_date  = last(date_vector)
    span_years = year(last_date) - year(first_date)

    # choose frequency
    tick_dates = Date[]
    if span_years >= 5
        # yearly
        yrs = unique(year.(date_vector))
        for year in yrs
            push!(tick_dates, Date(year, 1, 1))
        end
    elseif span_years >= 2
        # semi-annual Jan & Jul
        yrs = unique(year.(date_vector))
        for y in yrs
            push!(tick_dates, Date(y, 1, 1))
            push!(tick_dates, Date(y, 7, 1))
        end
    else
        # quarterly
        yrs = unique(year.(date_vector))
        for y in yrs, m in (1,4,7,10)
            push!(tick_dates, Date(y, m, 1))
        end
    end

    # keep only those within our range
    tick_dates = filter(d -> d ≥ first_date && d ≤ last_date, tick_dates)

    # find the first data‐index at or after each tick_date
    positions = Int[]
    labels    = String[]
    for d in tick_dates
        idx = findfirst(x -> x ≥ d, date_vector)
        if idx !== nothing
            push!(positions, idx)
            push!(labels, string(d))
        end
    end

    return positions, labels
end

"""
    plot_pockets(
        err_benchmark,
        err_model,
        date_vector,
        smoothing_bandwidth,
        thresholds;
        auto_xticks     = true,
        user_xticks     = nothing,
        title           = "",
        pocket_colors   = [mycolor[3], mycolor[4]],
        pocket_alphas   = [0.6, 0.3],
        base_line_color = :white,
        sed_line_color  = mycolor[1],
        hline_color     = mycolor[3],
        hline_style     = :dash,
        plot_size       = (1000, 200),
        framestyle      = :box,
        fontfamily      = "serif-roman",
        title_fontsize  = 10,
        xtick_fontsize  = 10,
        ytick_fontsize  = 10,
        ylabel_fontsize = 10
    ) -> Plot

Plot smoothed squared error difference (SED) between two forecasting models with visual "pockets" indicating regions where one model significantly outperforms the benchmark.

# Arguments
- `err_benchmark`: Vector of forecast errors (e.g., squared) from the benchmark model.
- `err_model`: Vector of forecast errors from the competing model.
- `date_vector`: Vector of date labels or time indices, used for x-axis ticks.
- `smoothing_bandwidth`: Bandwidth parameter for smoothing the SED curve.
- `thresholds`: A scalar or vector of threshold levels. Regions where SED exceeds each threshold are shaded.

# Keyword Arguments
- `auto_xticks::Bool`: Automatically determine x-axis ticks from `date_vector` (default: `true`).
- `user_xticks`: Optional custom x-axis tick positions and labels as a tuple `(positions, labels)`.
- `title::String`: Plot title.
- `pocket_colors::Vector`: Colors for the shaded pockets corresponding to each threshold level. Default: colorant"rgb(76,144,186)",colorant"rgb(43,194,194)" assuming only 2 thresholds
- `pocket_alphas::Vector`: Opacity (alpha) levels for each pocket.
- `base_line_color`: Color of the base (initial) plot before overlaying SED.
- `sed_line_color`: Color of the SED curve. Default: colorant"rgb(222,102,62)"
- `hline_color`: Color of the horizontal threshold lines. Default: colorant"rgb(76,144,186)"
- `hline_style`: Line style for horizontal thresholds (e.g., `:dash`).
- `plot_size::Tuple`: Tuple specifying plot dimensions (width, height).
- `framestyle::Symbol`: Plot frame style (`:box`, `:none`, etc.).
- `fontfamily::String`: Font used for all labels and text.
- `title_fontsize`, `xtick_fontsize`, `ytick_fontsize`, `ylabel_fontsize`: Font sizes for various elements.

# Returns
- `Plot`: A `Plots.jl` object displaying the smoothed SED curve and shaded threshold regions ("Pockets of Predictability").

# Description
1. Computes the smoothed squared error difference between the benchmark and model errors.
2. Plots the SED curve over time.
3. Shades "pockets" (vertical bands) where the SED exceeds each threshold.
4. Allows optional x-axis control and customization of colors, styles, and fonts.
"""
mycolor=[colorant"rgb(222,102,62)",colorant"rgb(255,145,43)",colorant"rgb(76,144,186)",colorant"rgb(43,194,194)",colorant"rgb(244,184,17)"]
function plot_pockets(
    err_benchmark,
    err_model,
    date_vector,
    smoothing_bandwidth,
    thresholds;
    auto_xticks     = true,
    user_xticks     = nothing,
    title           = "",
    pocket_colors   = [mycolor[3], mycolor[4]],
    pocket_alphas   = [0.6, 0.3],
    base_line_color = :white,
    sed_line_color  = mycolor[1],
    hline_color     = mycolor[3],
    hline_style     = :dash,
    plot_size       = (1000,200),
    framestyle      = :box,
    fontfamily      = "serif-roman",
    title_fontsize  = 10,
    xtick_fontsize  = 10,
    ytick_fontsize  = 10,
    ylabel_fontsize = 10
)
    # 1) compute the SED curve once
    sed_curve = SED_smooth_one(err_benchmark, err_model, smoothing_bandwidth)

    # 2) determine x‐axis ticks
    if auto_xticks == true
        xticks_positions, xticks_labels = derive_date_ticks(date_vector)
    elseif user_xticks !== nothing
        xticks_positions, xticks_labels = user_xticks
    else
        xticks_positions, xticks_labels = (1:length(sed_curve), nothing)
    end

    # 3) normalize thresholds to a vector
    threshold_levels = isa(thresholds, AbstractVector) ? thresholds : [0.0, thresholds]

    # 4) start the base plot
    p = plot(
        sed_curve;
        color        = base_line_color,
        legend       = false,
        framestyle   = framestyle,
        size         = plot_size,
        title        = title,
        xticks       = (xticks_positions, xticks_labels),
        fontfamily   = fontfamily,
        titlefontsize  = title_fontsize,
        xtickfontsize  = xtick_fontsize,
        ytickfontsize  = ytick_fontsize,
        ylabelfontsize = ylabel_fontsize
    )

    # 5) shade each pocket band (skip zero if you like)
    for (i, thr) in enumerate(threshold_levels)
        mask = sed_curve .> thr
        for (start, stop) in find_intervals(mask)
            vspan!(
                p,
                start:stop;
                color     = pocket_colors[i],
                alpha     = pocket_alphas[i],
                linecolor = :transparent,
                label     = false
            )
        end
    end

    # 6) overlay the SED line and horizontal thresholds
    plot!(p, sed_curve, color = sed_line_color)
    for thr in threshold_levels
        hline!(
            p,
            [thr];
            linestyle = hline_style,
            color     = hline_color,
            label     = false
        )
    end

    return p
end

