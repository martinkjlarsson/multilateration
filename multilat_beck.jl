using LinearAlgebra
using PolynomialRoots
using Optimization, OptimizationOptimJL

function multilat_beck_srls_iter(s, z, W=I; tol=1e-5, maxiter=300, irls_iters=1)
    (n, m) = size(s)

    x = multilat_beck_srls(s, z, W; tol=tol, maxiter=maxiter)

    # Use first sender as "sensor 0".
    ss = s[:, 2:end] .- s[:, 1] # aⱼ in the paper.
    zz = z[2:end] .- z[1] # dⱼ in the paper.

    for _ in 2:irls_iters
        if isnothing(x)
            return nothing
        end
        xx = x .- s[:, 1]
        dd = [norm(xx - ss[:, j]) for j in 1:(m - 1)]
        W_1 = Diagonal(1 ./ (zz .+ norm(xx) .+ dd))
        x = multilat_beck_srls(s, z, W_1' * W * W_1; tol=tol, maxiter=maxiter)
    end

    return x
end

# A. Beck, P. Stoica, and J. Li, “Exact and Approximate Solutions of Source Localization Problems,” IEEE Transactions on Signal Processing, vol. 56, no. 5, pp. 1770–1778, May 2008, doi: 10.1109/TSP.2007.909342.
#
# Example from the paper:
# s = [0 -5 -12 -1 -9 -3; 0 -13 1 -5 -12 -12]
# x = [-5, 11]
# z = [0, 11.8829, 0.1803, 4.6399, 11.2402, 10.8183]
#
# xest = [-4.97976, 10.27864]
#
# Note W is an (m-1) x (m-1) matrix, not m x m.
# Minimizes Σⱼ wⱼ(-2aⱼᵀx - 2dⱼ ||x|| - gⱼ)² where gⱼ = aⱼ² - ||dⱼ||².
# and aⱼ = sⱼ - s₀, dⱼ = zⱼ - z₀.
function multilat_beck_srls(s, z, W=I; tol=1e-5, maxiter=300)
    (n, m) = size(s)

    # Calculate B, C. Use first sender as "sensor 0".
    ss = s[:, 2:end] .- s[:, 1] # aⱼ in the paper.
    zz = z[2:end] .- z[1] # dⱼ in the paper.
    B = [-2 * ss' -2 * zz]
    g = zz .^ 2 .- vec(sum(abs2, ss; dims=1))
    C = Diagonal([ones(n); -1])

    # Precalculations.
    BB = Symmetric(B' * W * B)
    Bg = B' * W * g

    # Definte φ(λ), which is strictly decreasing on the interval considered.
    function φ(λ)
        y = (BB + λ * C) \ Bg
        return y' * C * y
    end

    # Find bounds (interval I₁) on lambda.
    lambdas = eigvals(C, BB)
    alphas = -1 ./ lambdas
    sort!(alphas; rev=true)
    lb = alphas[2]
    ub = alphas[1]

    if !isfinite(lb)
        # A finite lower bound could possibly be found, but may then result in
        # singularity issues in the linear systems further down.
        @warn "Lower bound is not finite. (lb, ub) = ($lb, $ub)"
        return nothing
    end

    # Use bisection to find the Lagrange multiplier λ.
    if tol <= 0.0
        tol = max(eps(lb), eps(ub))
    end
    iter = 0
    while ub - lb > tol
        mb = (lb + ub) / 2
        if φ(mb) < 0
            ub = mb
        else
            lb = mb
        end

        iter += 1
        if iter >= maxiter
            @info "Bisection reached maximum number of iterations ($maxiter)."
            break
        end
    end

    # Find final y and x.
    λ = (lb + ub) / 2
    y = (BB + λ * C) \ Bg

    if y[end] >= 0
        return y[1:n] + s[:, 1]
    end

    # Diagonalize BB and C such that PᵀBᵀB*P = I ⇒ γₖ = 1 for all k.
    F = cholesky(BB)
    L = F.L
    Δ, Q = eigen(L \ C / L')
    f = Q' * (L \ Bg)
    # P = L' \ Q
    # f = P' * Bg
    # Γ = P'*BB*P == I

    # Solve polynomial.
    poly = beck_coeffs(n, f, Δ)
    lambdas = roots(poly)

    # Find all roots λ ∈ I₀ ∪ I₂.
    good_lambdas = Float64[]
    for λ in lambdas
        if !isreal(λ)
            continue
        end
        λ = real(λ)
        if λ >= alphas[1] || alphas[3] <= λ <= alphas[2] # I₀ or I₂
            push!(good_lambdas, λ)
        end
    end

    # Find solution y with lowest cost.
    best_y = zeros(n + 1)
    best_cost = beck_cost(ss, zz, W, best_y[1:n])
    for λ in good_lambdas
        y = (BB + λ * C) \ Bg
        if y[end] < 0
            continue
        end
        cost = beck_cost(ss, zz, W, y[1:n])
        if cost < best_cost
            best_y = y
            best_cost = cost
        end
    end

    return best_y[1:n] + s[:, 1]
end

function beck_cost(ss, zz, W, x)
    res = -2 .* ss' * x .- 2 .* zz .* norm(x) .- zz .^ 2 .+ vec(sum(abs2, ss; dims=1))
    return res' * W * res
end

function beck_coeffs(n, f, Δ)
    if n == 2
        poly = [
            Δ[1] * (f[1]^2) + Δ[2] * (f[2]^2) + Δ[3] * (f[3]^2),
            2Δ[1] * Δ[2] * (f[1]^2) +
            2Δ[1] * Δ[2] * (f[2]^2) +
            2Δ[1] * Δ[3] * (f[1]^2) +
            2Δ[1] * Δ[3] * (f[3]^2) +
            2Δ[2] * Δ[3] * (f[2]^2) +
            2Δ[2] * Δ[3] * (f[3]^2),
            (Δ[1]^2) * Δ[2] * (f[2]^2) +
            (Δ[1]^2) * Δ[3] * (f[3]^2) +
            Δ[1] * (Δ[2]^2) * (f[1]^2) +
            4Δ[1] * Δ[2] * Δ[3] * (f[1]^2) +
            4Δ[1] * Δ[2] * Δ[3] * (f[2]^2) +
            4Δ[1] * Δ[2] * Δ[3] * (f[3]^2) +
            Δ[1] * (Δ[3]^2) * (f[1]^2) +
            (Δ[2]^2) * Δ[3] * (f[3]^2) +
            Δ[2] * (Δ[3]^2) * (f[2]^2),
            2(Δ[1]^2) * Δ[2] * Δ[3] * (f[2]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[3] * (f[3]^2) +
            2Δ[1] * (Δ[2]^2) * Δ[3] * (f[1]^2) +
            2Δ[1] * (Δ[2]^2) * Δ[3] * (f[3]^2) +
            2Δ[1] * Δ[2] * (Δ[3]^2) * (f[1]^2) +
            2Δ[1] * Δ[2] * (Δ[3]^2) * (f[2]^2),
            (Δ[1]^2) * (Δ[2]^2) * Δ[3] * (f[3]^2) +
            (Δ[1]^2) * Δ[2] * (Δ[3]^2) * (f[2]^2) +
            Δ[1] * (Δ[2]^2) * (Δ[3]^2) * (f[1]^2),
        ]
    elseif n == 3
        poly = [
            Δ[1] * (f[1]^2) + Δ[2] * (f[2]^2) + Δ[3] * (f[3]^2) + Δ[4] * (f[4]^2)
            2 * Δ[1] * Δ[2] * (f[1]^2) +
            2 * Δ[1] * Δ[2] * (f[2]^2) +
            2 * Δ[1] * Δ[3] * (f[1]^2) +
            2 * Δ[1] * Δ[3] * (f[3]^2) +
            2 * Δ[1] * Δ[4] * (f[1]^2) +
            2 * Δ[1] * Δ[4] * (f[4]^2) +
            2 * Δ[2] * Δ[3] * (f[2]^2) +
            2 * Δ[2] * Δ[3] * (f[3]^2) +
            2 * Δ[2] * Δ[4] * (f[2]^2) +
            2 * Δ[2] * Δ[4] * (f[4]^2) +
            2 * Δ[3] * Δ[4] * (f[3]^2) +
            2 * Δ[3] * Δ[4] * (f[4]^2)
            (Δ[1]^2) * Δ[2] * (f[2]^2) +
            (Δ[1]^2) * Δ[3] * (f[3]^2) +
            (Δ[1]^2) * Δ[4] * (f[4]^2) +
            Δ[1] * (Δ[2]^2) * (f[1]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (f[1]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (f[2]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (f[3]^2) +
            4 * Δ[1] * Δ[2] * Δ[4] * (f[1]^2) +
            4 * Δ[1] * Δ[2] * Δ[4] * (f[2]^2) +
            4 * Δ[1] * Δ[2] * Δ[4] * (f[4]^2) +
            Δ[1] * (Δ[3]^2) * (f[1]^2) +
            4 * Δ[1] * Δ[3] * Δ[4] * (f[1]^2) +
            4 * Δ[1] * Δ[3] * Δ[4] * (f[3]^2) +
            4 * Δ[1] * Δ[3] * Δ[4] * (f[4]^2) +
            Δ[1] * (Δ[4]^2) * (f[1]^2) +
            (Δ[2]^2) * Δ[3] * (f[3]^2) +
            (Δ[2]^2) * Δ[4] * (f[4]^2) +
            Δ[2] * (Δ[3]^2) * (f[2]^2) +
            4 * Δ[2] * Δ[3] * Δ[4] * (f[2]^2) +
            4 * Δ[2] * Δ[3] * Δ[4] * (f[3]^2) +
            4 * Δ[2] * Δ[3] * Δ[4] * (f[4]^2) +
            Δ[2] * (Δ[4]^2) * (f[2]^2) +
            (Δ[3]^2) * Δ[4] * (f[4]^2) +
            Δ[3] * (Δ[4]^2) * (f[3]^2)
            2(Δ[1]^2) * Δ[2] * Δ[3] * (f[2]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[3] * (f[3]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[4] * (f[2]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[4] * (f[4]^2) +
            2(Δ[1]^2) * Δ[3] * Δ[4] * (f[3]^2) +
            2(Δ[1]^2) * Δ[3] * Δ[4] * (f[4]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[3] * (f[1]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[3] * (f[3]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[4] * (f[1]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[4] * (f[4]^2) +
            2 * Δ[1] * Δ[2] * (Δ[3]^2) * (f[1]^2) +
            2 * Δ[1] * Δ[2] * (Δ[3]^2) * (f[2]^2) +
            8Δ[1] * Δ[2] * Δ[3] * Δ[4] * (f[1]^2) +
            8Δ[1] * Δ[2] * Δ[3] * Δ[4] * (f[2]^2) +
            8Δ[1] * Δ[2] * Δ[3] * Δ[4] * (f[3]^2) +
            8Δ[1] * Δ[2] * Δ[3] * Δ[4] * (f[4]^2) +
            2 * Δ[1] * Δ[2] * (Δ[4]^2) * (f[1]^2) +
            2 * Δ[1] * Δ[2] * (Δ[4]^2) * (f[2]^2) +
            2 * Δ[1] * (Δ[3]^2) * Δ[4] * (f[1]^2) +
            2 * Δ[1] * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            2 * Δ[1] * Δ[3] * (Δ[4]^2) * (f[1]^2) +
            2 * Δ[1] * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            2(Δ[2]^2) * Δ[3] * Δ[4] * (f[3]^2) +
            2(Δ[2]^2) * Δ[3] * Δ[4] * (f[4]^2) +
            2 * Δ[2] * (Δ[3]^2) * Δ[4] * (f[2]^2) +
            2 * Δ[2] * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            2 * Δ[2] * Δ[3] * (Δ[4]^2) * (f[2]^2) +
            2 * Δ[2] * Δ[3] * (Δ[4]^2) * (f[3]^2)
            (Δ[1]^2) * (Δ[2]^2) * Δ[3] * (f[3]^2) +
            (Δ[1]^2) * (Δ[2]^2) * Δ[4] * (f[4]^2) +
            (Δ[1]^2) * Δ[2] * (Δ[3]^2) * (f[2]^2) +
            4(Δ[1]^2) * Δ[2] * Δ[3] * Δ[4] * (f[2]^2) +
            4(Δ[1]^2) * Δ[2] * Δ[3] * Δ[4] * (f[3]^2) +
            4(Δ[1]^2) * Δ[2] * Δ[3] * Δ[4] * (f[4]^2) +
            (Δ[1]^2) * Δ[2] * (Δ[4]^2) * (f[2]^2) +
            (Δ[1]^2) * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            (Δ[1]^2) * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            Δ[1] * (Δ[2]^2) * (Δ[3]^2) * (f[1]^2) +
            4 * Δ[1] * (Δ[2]^2) * Δ[3] * Δ[4] * (f[1]^2) +
            4 * Δ[1] * (Δ[2]^2) * Δ[3] * Δ[4] * (f[3]^2) +
            4 * Δ[1] * (Δ[2]^2) * Δ[3] * Δ[4] * (f[4]^2) +
            Δ[1] * (Δ[2]^2) * (Δ[4]^2) * (f[1]^2) +
            4 * Δ[1] * Δ[2] * (Δ[3]^2) * Δ[4] * (f[1]^2) +
            4 * Δ[1] * Δ[2] * (Δ[3]^2) * Δ[4] * (f[2]^2) +
            4 * Δ[1] * Δ[2] * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (Δ[4]^2) * (f[1]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (Δ[4]^2) * (f[2]^2) +
            4 * Δ[1] * Δ[2] * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            Δ[1] * (Δ[3]^2) * (Δ[4]^2) * (f[1]^2) +
            (Δ[2]^2) * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            (Δ[2]^2) * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            Δ[2] * (Δ[3]^2) * (Δ[4]^2) * (f[2]^2)
            2(Δ[1]^2) * (Δ[2]^2) * Δ[3] * Δ[4] * (f[3]^2) +
            2(Δ[1]^2) * (Δ[2]^2) * Δ[3] * Δ[4] * (f[4]^2) +
            2(Δ[1]^2) * Δ[2] * (Δ[3]^2) * Δ[4] * (f[2]^2) +
            2(Δ[1]^2) * Δ[2] * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[3] * (Δ[4]^2) * (f[2]^2) +
            2(Δ[1]^2) * Δ[2] * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            2 * Δ[1] * (Δ[2]^2) * (Δ[3]^2) * Δ[4] * (f[1]^2) +
            2 * Δ[1] * (Δ[2]^2) * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[3] * (Δ[4]^2) * (f[1]^2) +
            2 * Δ[1] * (Δ[2]^2) * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            2 * Δ[1] * Δ[2] * (Δ[3]^2) * (Δ[4]^2) * (f[1]^2) +
            2 * Δ[1] * Δ[2] * (Δ[3]^2) * (Δ[4]^2) * (f[2]^2)
            (Δ[1]^2) * (Δ[2]^2) * (Δ[3]^2) * Δ[4] * (f[4]^2) +
            (Δ[1]^2) * (Δ[2]^2) * Δ[3] * (Δ[4]^2) * (f[3]^2) +
            (Δ[1]^2) * Δ[2] * (Δ[3]^2) * (Δ[4]^2) * (f[2]^2) +
            Δ[1] * (Δ[2]^2) * (Δ[3]^2) * (Δ[4]^2) * (f[1]^2)
        ]
    else
        error("Unsuported dimension $n")
    end
end

# Use this to generate the coefficients for the Beck polynomials.
# using Symbolics
# function create_beck_polynomials(n)
#     λ = Symbolics.variable(:λ)
#     f = Symbolics.variables(:f, 1:(n + 1))
#     # gamma = Symbolics.variables(:gamma, 1:(n + 1))
#     gamma = ones(Int, n + 1)
#     delta = Symbolics.variables(:delta, 1:(n + 1))

#     P = 0
#     for j in 1:(n + 1)
#         p = prod((gamma[k] + λ * delta[k])^2 for k in 1:(n + 1) if k != j)
#         P += f[j]^2 * delta[j] * p
#     end
#     P = expand(P)
#     coeffs = [
#         substitute(P, λ => 0)
#         [Symbolics.coeff(P, λ^i) for i in 1:Symbolics.degree(P, λ)]
#     ]

#     return coeffs
# end
