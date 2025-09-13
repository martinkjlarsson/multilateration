using Optimization, OptimizationOptimJL

include("multilat_heidari.jl")
include("multilat_zeng.jl")
include("multilat_solvit.jl")
include("multilat_beck.jl")
include("multilat_chan.jl")

"""
Minimizes ∑ⱼwⱼ(α - 2xᵀsⱼ + sⱼᵀsⱼ - zⱼ² + 2zⱼo)² over (x, α) and returns x.
The constraint α = xᵀx - o² is ignored.
"""
function multilat_linear(s, z, W=I(size(s, 2)))
    # || r - sⱼ ||² = (zⱼ-o)² ⇔ rᵀr - 2rᵀsⱼ + sⱼᵀsⱼ = zⱼ² - 2zⱼo + o² ⇔
    # rᵀr - o² - 2rᵀsⱼ + 2zⱼo = zⱼ² - sⱼᵀsⱼ
    # This is linear in r, o, and rᵀr-o².
    m = size(s, 2)
    A = -2 * [s' -z ones(m)]
    b = z .^ 2 - vec(sum(abs2, s; dims=1))
    # y = A \ b # Ignore weights.
    y = (A' * W * A) \ (A' * W * b)
    return y[1:(end - 2)], y[end - 1]
end

function multilat_local_opt(s, z, x0=zeros(size(s, 1)), o0=0.0; loss=abs2, maxiters=100)
    function multilat_cost(ro, (s, z))
        r = ro[1:(end - 1)]
        o = ro[end]
        return sum(loss, norm(r - s[:, i]) - (z[i] - o) for i in eachindex(z))
    end
    f = OptimizationFunction(multilat_cost, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, [x0; o0], (s, z))
    sol = solve(prob, BFGS(); maxiters=maxiters)
    ropt = sol.u[1:(end - 1)]
    oopt = sol.u[end]
    return ropt, oopt
end

function multilat_local_opt2(s, z, x0=zeros(size(s, 1)), o0=0.0; loss=abs2, maxiters=100)
    function multilat_cost(ro, (s, z))
        r = ro[1:(end - 1)]
        o = ro[end]
        return sum(loss, norm(r - s[:, i])^2 - (z[i] - o)^2 for i in eachindex(z))
    end
    f = OptimizationFunction(multilat_cost, Optimization.AutoForwardDiff())
    prob = OptimizationProblem(f, [x0; o0], (s, z))
    sol = solve(prob, BFGS(); maxiters=maxiters)
    ropt = sol.u[1:(end - 1)]
    oopt = sol.u[end]
    return ropt, oopt
end

function multilat_known_height_local_opt(
    s, z, height, x0=zeros(size(s, 1)), o0=0.0; loss=abs2, maxiters=100
)
    function multilat_cost(ro, (s, z, height))
        r = [ro[1:(end - 1)]; height]
        o = ro[end]
        return sum(loss, norm(r - s[:, i]) - (z[i] - o) for i in eachindex(z))
    end
    f = OptimizationFunction(multilat_cost, Optimization.AutoForwardDiff())
    x0 = x0[1:(size(s, 1) - 1), 1]
    prob = OptimizationProblem(f, [x0; o0], (s, z, height))
    sol = solve(prob, BFGS(); maxiters=maxiters)
    ropt = [sol.u[1:(end - 1)]; height]
    oopt = sol.u[end]
    return ropt, oopt
end
