using StatsPlots
using Plots.PlotMeasures
using DataFrames
using ColorSchemes

include("proposed.jl")
include("previous_works.jl")

m = 6
dim = 3
room_scale = 10
iters = 1000
# iters = 10000 # Used in the paper.
sigmas = [0.01, 0.1, 1]

W_beck = (1.0 * I(m - 1)) / (ones(m - 1, m - 1) + 1.0 * I(m - 1))

solvers = Vector{Tuple{String,Function}}()
push!(solvers, ("Linear", (s, z, W, x0, o0) -> multilat_linear(s, z)))
push!(solvers, ("Chan", (s, z, W, x0, o0) -> multilat_chan_W(s, z)))
push!(solvers, ("Heidari", (s, z, W, x0, o0) -> multilat_heidari_W(s, z)))
push!(solvers, ("Zeng", (s, z, W, x0, o0) -> multilat_zeng_W(s, z)))
push!(solvers, ("Beck", (s, z, W, x0, o0) -> multilat_beck_srls(s, z, W_beck; tol=0)))
push!(
    solvers,
    (
        "Ismailova",
        (s, z, W, x0, o0) -> multilat_beck_srls_iter(s, z, W_beck; tol=0, irls_iters=2),
    ),
)
push!(solvers, ("SOLVIT", (s, z, W, x0, o0) -> solvit(s, z)))
push!(solvers, ("Proposed", (s, z, W, x0, o0) -> multilat(s, z)))
push!(solvers, ("Proposed 2x", (s, z, W, x0, o0) -> multilat_iter(s, z; iters=2)))
push!(solvers, ("Proposed W", (s, z, W, x0, o0) -> multilat(s, z, W)))

push!(solvers, ("ML", (s, z, W, x0, o0) -> multilat_local_opt(s, z, x0, o0)))

mean_errors = fill(Inf, length(sigmas), length(solvers))
median_errors = fill(Inf, length(sigmas), length(solvers))
errors = DataFrame(; solver=Int[], sigma=[], error=[])
for isolver in 1:length(solvers)
    solver = solvers[isolver]
    println("Performing tests for $(solver[1])")
    for isigma in eachindex(sigmas)
        sigma = sigmas[isigma]

        errors_iter = fill(Inf, iters)
        for iter in 1:iters
            x = room_scale * rand(dim)
            o = room_scale * rand()
            s = room_scale * rand(dim, m)
            d = [norm(x - s[:, j]) for j in 1:m]
            z = d .+ o
            z += sigma * randn(m)

            W = Diagonal(1 ./ d .^ 2)

            xo_est = solver[2](s, z, W, x, o)
            x_est = isa(xo_est, Tuple) ? first(xo_est) : xo_est

            if !isnothing(x_est)
                err = minimum(norm(x_esti - x) for x_esti in eachcol(x_est))
                errors_iter[iter] = err
                push!(errors, [isolver sigma err])
            else
                @error "Solver failed! Error is not saved!"
            end
        end
        mean_errors[isigma, isolver] = mean(errors_iter)
        median_errors[isigma, isolver] = median(errors_iter)
    end
end

for isolver in 1:length(solvers)
    solver = solvers[isolver]
    for isigma in eachindex(sigmas)
        sigma = sigmas[isigma]
        println(
            "$(solver[1]) - Median error - sigma = $(sigma) - $(median_errors[isigma, isolver]/sigma)",
        )
    end
end

## Plot.
names = reshape([solver[1] for solver in solvers], 1, :)
cs = [colorschemes[:tab10].colors; RGB(0.8, 0.8, 0.8)]

gr()

p = groupedboxplot(
    errors[:, 2],
    errors[:, 3] ./ errors[:, 2];
    group=errors[:, 1],
    outliers=false,
    whisker_range=0.0001,
    whiskerlinewidth=0.0001,
    whisker_width=0.001,
    legend=:outerbottomright,
    palette=cs,
    grid=:y,
    label=names,
    xlabel="Noise σ",
    ylabel="Normalized error (error/σ)",
    size=(1000, 300),
    left_margin=5mm,
    bottom_margin=5mm,
)
display(p)

# savefig(p, "figs/numerics_wide.pdf")
