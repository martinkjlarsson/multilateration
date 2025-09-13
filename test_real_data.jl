using CSV, DataFrames, JSON, Statistics, Plots, ColorSchemes, Printf

include("proposed.jl")
include("previous_works.jl")

@enum Method linear ml_local_opt proposed proposed_iter

c = 299792458 # Speed of light (m/s).
uez = 1.05    # User equipment height (m).
ns = 128      # Number of samples in CSI.
fs = 184.32e6 # CSI sampling frequency (Hz).
Ts = 1 / fs   # CSI sampling period (s).

function multilat_linear_known_height(s, z, height)
    m = size(s, 2)
    A = -2 * [s[1:(end - 1), :]' -z ones(m)]
    b = z .^ 2 .- vec(sum(abs2, s; dims=1)) .- height^2 .+ 2 * height * s[end, :]
    y = A \ b
    return [y[1:(end - 2)]; height], y[end - 1]
end

function print_errors(est_pos, test_pos)
    errors = sqrt.(vec(sum(abs2, est_pos .- test_pos; dims=1)))

    println("Mean error:     ", mean(errors))
    println("Std error:      ", std(errors))
    println("Median error:   ", median(errors))
    println("75% percentile: ", quantile(errors, 0.75))
    println()

    return errors
end

function print_table(results, variant)
    for iresult in 1:length(results)
        result = results[iresult]
        if variant == 1
            @printf(
                "%s & %.2f & %.2f & %.2f & %.2f \\\\\n",
                result[1],
                mean(result[2]),
                std(result[2]),
                mean(result[3]),
                std(result[3])
            )
            # println(result[1], " & ", mean(result[2]), " & ", std(result[2]), " & ", mean(result[3]), " & ", std(result[3]), "\\\\")
        else
            @printf(
                "%s & %.2f & %.2f & %.2f & %.2f \\\\\n",
                result[1],
                median(result[2]),
                quantile(result[2], 0.75),
                median(result[3]),
                quantile(result[3], 0.75)
            )
            # println(result[1], " & ", median(result[2]), " & ", quantile(result[2], 0.75), " & ", median(result[3]), " & ", quantile(result[3], 0.75), "\\\\")
        end
    end

    return nothing
end

function kalman_update(x, P, z, R, dt, velocity_variance)
    # Using the notation from https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/

    if dt <= 0.0
        dt = 0.0
        @warn "Negative time delta in Kalman filter."
    end

    # Predict.
    F = [1 0 dt 0; 0 1 0 dt; 0 0 1 0; 0 0 0 1]
    Q = velocity_variance * [
        dt*dt 0 dt 0
        0 dt*dt 0 dt
        dt 0 1 0
        0 dt 0 1
    ]
    x = F * x # No external influences.
    P = F * P * F' + Q

    # Update.
    if norm(x[1:2] - z) < 100 # Disregard outlier measurements.
        H = [1 0 0 0; 0 1 0 0]
        K = (P * H') / (H * P * H' + R)
        x = x + K * (z - H * x)
        P = (I - K * H) * P
    else
        # @warn "Outlier measurement norm(x[1:2] - z) = $(norm(x[1:2] - z))"
    end

    return x, P
end

function load_anchors(anchorPath)
    anchors_csv = CSV.read(anchorPath, DataFrame)
    rename!(anchors_csv, [:id, :px, :py, :pz])
    anchors = [anchors_csv.px'; anchors_csv.py'; anchors_csv.pz']
    return anchors
end

function perform_kalman_filtering(df, useKalman=false, method=proposed)
    gdf = groupby(df, :burst_id; sort=true)

    x0 = [first(df.ref_x), first(df.ref_y), 0.0, 0.0]
    P0 = [
        0.013 0.0 0.004 0.0
        0.0 0.013 0.0 0.004
        0.004 0.0 0.0029 0.0
        0.0 0.004 0.0 0.0029
    ]

    x = x0
    P = P0
    prev_time = first(df.rec_time)

    est_pos = zeros(2, length(gdf))
    ref_pos = zeros(2, length(gdf))

    for (i, sdf) in enumerate(gdf)
        sort!(sdf, :anch_id)
        toad = (sdf.toa) * c
        if method == linear
            xest, _ = multilat_linear_known_height(anchors, toad, uez)
        elseif method == proposed
            xest, _ = multilat_known_height(anchors, toad, uez)
        elseif method == ml_local_opt
            d_gt = [
                norm([first(sdf.ref_x); first(sdf.ref_y); uez] - si) for
                si in eachcol(anchors)
            ]
            xest, _ = multilat_known_height_local_opt(
                anchors, toad, uez, [first(sdf.ref_x); first(sdf.ref_y)], mean(toad - d_gt)
            )
        elseif method == proposed_iter
            xest, _ = multilat_known_height_iter(anchors, toad, uez; iters=2)
        end
        xest = xest[1:2]
        R = 2 * I(2)

        curr_time = first(sdf.rec_time)
        dt = curr_time - prev_time

        if useKalman
            x, P = kalman_update(x, P, xest, R, dt, 1e-3)
        else
            x = xest
        end

        ref_pos[:, i] = [first(sdf.ref_x), first(sdf.ref_y)]
        est_pos[:, i] = x[1:2]

        prev_time = curr_time
    end

    return est_pos, ref_pos
end

function create_plot(max_ind, est_pos, ref_pos)
    gr()
    p = plot(;
        aspect_ratio=1,
        size=(500, 340),
        xlims=(8, 26),
        ylims=(7, 20),
        legend=:bottomleft,
        xticks=0:2:30,
        yticks=0:2:30,
        minorgrid=true,
        minorticks=2,
        xlabel="x (m)",
        ylabel="y (m)",
        palette=ColorSchemes.tab10,
    )
    plot!(est_pos[1, 1:max_ind], est_pos[2, 1:max_ind]; label="Estimate", linewidth=2)
    plot!(ref_pos[1, 1:max_ind], ref_pos[2, 1:max_ind]; label="Ground Truth", linewidth=2)

    return p
end

anchors = load_anchors("data/anchors.txt")

ann_data_path = "data/exp_trial_ann.csv"
fraun_data_path = "data/exp_trial.csv"

test_df = CSV.read(fraun_data_path, DataFrame)
test_ann_df = CSV.read(ann_data_path, DataFrame)

useLinear = false
useKalman = true

solvers = Vector{Tuple{String,Function}}()
push!(
    solvers, ("Proposed", (test_df) -> perform_kalman_filtering(test_df, false, proposed))
)
push!(
    solvers,
    ("Proposed 2x", (test_df) -> perform_kalman_filtering(test_df, false, proposed_iter)),
)
push!(
    solvers,
    ("ML - local opt", (test_df) -> perform_kalman_filtering(test_df, false, ml_local_opt)),
)

results = Vector{Tuple{String,Vector{Float64},Vector{Float64}}}()
for isolver in 1:length(solvers)
    solver = solvers[isolver]
    est_pos, ref_pos = solver[2](test_df)
    est_pos_ann, ref_pos_ann = solver[2](test_ann_df)
    errors = sqrt.(vec(sum(abs2, est_pos .- ref_pos; dims=1)))
    errors_ann = sqrt.(vec(sum(abs2, est_pos_ann .- ref_pos_ann; dims=1)))
    push!(results, (solver[1], errors, errors_ann))
    println(solver[1])
    print_errors(est_pos, ref_pos)
end

# print_table(results, 1)
print_table(results, 2)
# print_errors(est_pos, ref_pos);
# est_pos, ref_pos = perform_kalman_filtering(test_ann_df, true, proposed_iter)
# p = create_plot(size(est_pos, 2), est_pos, ref_pos)

# savefig(p, "figs/fraunhofer_path.pdf")
