using StatsPlots
using DataFrames
using ColorSchemes

include("proposed.jl")
include("previous_works.jl")

## Setup.
m = 6
dim = 3
iters = 100
# iters = 1000 # Used in the paper.
tol = 1e-16
success_tol = 1e-6

xscales = exp10.(-12:0.5:0) # Scale of sender x-coordinate.
sigma = 0.0

W_beck = (1.0 * I(m - 1)) / (ones(m - 1, m - 1) + 1.0 * I(m - 1))

## Define solvers.
solvers = Vector{Tuple{String,Function}}()
push!(solvers, ("Linear", (s, z, W, x0, o0) -> multilat_linear(s, z)))
push!(solvers, ("Chan", (s, z, W, x0, o0) -> multilat_chan_W(s, z)))
push!(solvers, ("Heidari", (s, z, W, x0, o0) -> multilat_heidari_W(s, z)))
push!(solvers, ("Zeng", (s, z, W, x0, o0) -> multilat_zeng_W(s, z)))
push!(solvers, ("Beck", (s, z, W, x0, o0) -> multilat_beck_srls(s, z, W_beck; tol=0)))
push!(solvers, ("SOLVIT", (s, z, W, x0, o0) -> solvit(s, z)))
push!(solvers, ("Proposed", (s, z, W, x0, o0) -> multilat(s, z)))
# push!(solvers, ("ML", (s, z, W, x0, o0) -> multilat_local_opt(s, z, x0, o0)))

## Test.
median_errors = fill(Inf, length(xscales), length(solvers))
success_rate = zeros(length(xscales), length(solvers))
errors = DataFrame(; solver=Int[], scale=[], error=[])
for isolver in 1:length(solvers)
    solver = solvers[isolver]
    println("Performing tests for $(solver[1]).")
    for (iscale, xscale) in enumerate(xscales)
        errors_iter = fill(Inf, iters)
        for iter in 1:iters
            x = randn(dim)
            o = randn()
            s = randn(dim, m)
            s[1, :] *= xscale
            d = [norm(x - s[:, j]) for j in 1:m]
            z = d .+ o
            z += sigma * randn(m)

            W = Diagonal(1 ./ d .^ 2)

            xo_est = solver[2](s, z, W, x, o)
            x_est = isa(xo_est, Tuple) ? first(xo_est) : xo_est

            if isnothing(x_est)
                err = Inf
            else
                err = minimum(norm(x_esti - x) for x_esti in eachcol(x_est))
            end

            errors_iter[iter] = err
            push!(errors, [isolver xscale err])
        end
        median_errors[iscale, isolver] = median(errors_iter)
        success_rate[iscale, isolver] = count(errors_iter .< success_tol) / iters
    end
end

## Plot.
names = reshape([solvers[i][1] for i in 1:length(solvers)], 1, :)

tab10_colors = palette(:tab10)
line_colors = tab10_colors[[1, 2, 3, 4, 5, 7, 8]]

gr()

psuccess_rate = plot(
    xscales,
    success_rate;
    lw=3,
    legend=:outerright,
    label=names,
    xaxis=:log,
    palette=line_colors,
    gridalpha=1.0,
    minorgrid=false,
    grid=true,
    foreground_color_grid=:lightgray,
    xticks=10.0 .^ (-12:4:0),
    yticks=0:0.2:1,
    xlabel="Scaling factor",
    ylabel="Success rate",
    ylims=(-0.02, 1.02),
    size=(500, 250),
)
display(psuccess_rate)

pmedian = plot(
    xscales,
    median_errors;
    lw=3,
    legend=:outerright,
    label=names,
    xaxis=:log,
    yaxis=:log,
    palette=line_colors,
    gridalpha=1.0,
    minorgrid=false,
    grid=true,
    foreground_color_grid=:lightgray,
    xticks=10.0 .^ (-12:4:0),
    yticks=10.0 .^ (-16:4:0),
    xlabel="Scaling factor",
    ylabel="Median error",
    ylims=(5e-17, 1e1),
    size=(500, 250),
)
display(pmedian)

# savefig(psuccess_rate, "figs/success_rate.pdf")
# savefig(pmedian, "figs/degenerate.pdf")
