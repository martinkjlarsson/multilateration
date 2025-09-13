using LinearAlgebra

include("jacobi.jl")

# K must be a signature matrix, i.e., K⁻¹ = K.
function gentrilat(s, d2, W=I(size(s, 2)), K::Diagonal=I(size(s, 1)))
    (n, m) = size(s)

    # Normalize weights.
    W = W ./ sum(W)

    # Translate senders.
    t = s * sum(W; dims=2)
    st = s .- t
    Kst = K * st

    # Construct A and g such that (x'*x)*x - A*x + g = 0.
    ws2md2 = vec(W * (sum(st .* Kst; dims=1)' .- d2))
    A = Symmetric(-2 * Kst * W * Kst' .- sum(ws2md2) * K)
    g = -Kst * ws2md2

    # Rotate senders.
    DA, Q = jacobi(A, K)
    D = K * DA # D is the eigenvalues of K * A = K \ A.
    perm = sortperm(D; rev=true)
    D = D[perm]
    Q = Q[:, perm]
    b = Q' * g
    Λ = Diagonal(K.diag[perm])

    # We now have D and b such that (y'*Λi*y)*y - diagm(D)*y + b = 0.

    # Monomial basis = [x^2,y^2,z^2,...,x,y,z,...,1].
    M = [
        diagm(D) diagm(-b) zeros(n, 1)
        zeros(n, n) diagm(D) -b
        diag(Λ)' zeros(1, n + 1)
    ]

    # Find eigenvalues and sort in descending order.
    lambdas = eigvals(M; sortby=λ -> -real(λ))
    p = count(diag(K) .< 0) # The number of negative eigenvalues of K.
    λ = real(lambdas[2 * p + 1]) # CONJECTURE: The eigenvalue corresponding to the global minimum. 

    rnk = rank(Diagonal(λ .- D); rtol=n * sqrt(eps()))

    # Find receiver position.
    index = argmin(abs.(λ .- D))
    if rnk == n
        y = -b ./ (λ .- D)
        sy = y[index] >= 0 ? 1 : -1
        y[index] = 0
        y[index] = sy * sqrt(max((λ - y' * Λ * y) / Λ[index, index], 0))
    elseif rnk == n - 1
        y = -b ./ (λ .- D)
        y[index] = 0
        yindex = sqrt(max((λ - y' * Λ * y) / Λ[index, index], 0))
        y = [y y]
        y[index, 1] = yindex
        y[index, 2] = -yindex
    else
        @warn "Generalized trilateration did not have finitely many solutions."
        return nothing
    end

    # Undo transformations.
    x = Q * Λ * y .+ t

    return real(x)
end

function multilat(s, z, W=I(size(s, 2)))
    (n, m) = size(s)
    sz = [s; z']
    d2 = zeros(m)
    K = Diagonal(ones(n + 1))
    K[end, end] = -1

    xo = gentrilat(sz, d2, W, K)

    if isnothing(xo)
        return nothing
    end

    return xo[1:(end - 1), :], xo[end, :]
end

function multilat_iter(s, z, W=I(size(s, 2)); iters=1)
    (n, m) = size(s)
    sz = [s; z']
    K = Diagonal(ones(n + 1))
    K[end, end] = -1

    xo = gentrilat(sz, zeros(m), W, K)
    if isnothing(xo)
        return nothing
    end

    for _ in 2:iters
        x = xo[1:(end - 1), 1]
        d = [norm(x - si) for si in eachcol(s)]
        W = Diagonal(1 ./ d .^ 2)
        xo = gentrilat(sz, zeros(m), W, K)
        if isnothing(xo)
            return nothing
        end
    end

    return xo[1:(end - 1), :], xo[end, :]
end

function multilat_known_height(s, z, height, W=I(size(s, 2)))
    (n, m) = size(s)
    sz = [s[1:(n - 1), :]; z']
    d2 = -(s[n, :] .- height) .^ 2
    K = Diagonal(ones(n))
    K[end, end] = -1

    xo = gentrilat(sz, d2, W, K)
    if isnothing(xo)
        return nothing
    end

    return [xo[1:(end - 1), :]; fill(height, size(xo, 2))'], xo[end, :]
end

function multilat_known_height_iter(s, z, height, W=I(size(s, 2)); iters=1)
    (n, m) = size(s)
    sz = [s[1:(n - 1), :]; z']
    d2 = -(s[n, :] .- height) .^ 2
    K = Diagonal(ones(n))
    K[end, end] = -1

    xo = gentrilat(sz, d2, W, K)
    if isnothing(xo)
        return nothing
    end

    for _ in 2:iters
        x = [xo[1:(end - 1), 1]; height]
        d = [norm(x - si) for si in eachcol(s)]
        W = Diagonal(1 ./ d .^ 2)
        xo = gentrilat(sz, d2, W, K)
        if isnothing(xo)
            return nothing
        end
    end

    return [xo[1:(end - 1), :]; fill(height, size(xo, 2))'], xo[end, :]
end
