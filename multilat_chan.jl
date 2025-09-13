using LinearAlgebra

function multilat_chan(s, z)
    r = z[2:end] .- z[1]
    return multilat_chan(s, r, I)
end

function multilat_chan_W(s, z)
    r = z[2:end] .- z[1]
    m = size(s, 2)
    W = (ones(m - 1, m - 1) + 1.0 * I(m - 1))
    return multilat_chan(s, r, W)
end

# Chan, Y. T., and K. C. Ho. “A Simple and Efficient Estimator for Hyperbolic Location.” IEEE Transactions on Signal Processing 42, no. 8 (August 1994): 1905–15. https://doi.org/10.1109/78.301830.
# See https://cisp.ece.missouri.edu/code.html for the original MATLAB code.
function multilat_chan(S, r, Q)
    # S: Dim x M
    # r: (M-1) x 1 vector
    # Q: Covariance matrix of r

    iters = 3
    n, m = size(S)

    @assert m >= n + 2 "Number of sensors must be at least $(n + 2)"
    @assert rank(S) == n "The sensors should not lie in one plane or line!"

    R = sqrt.(vec(sum(abs2, S; dims=1)))

    # =========== Construct vector and matrix ============
    h1 = r .^ 2 .- R[2:end] .^ 2 .+ R[1]^2
    G1 = -2 * [S[:, 2:end]' .- S[:, 1]' r]

    # ============ First Stage ===========================
    B = I(m - 1)
    W1 = inv(B * Q * B')
    u1 = (G1' * W1 * G1) \ (G1' * W1 * h1)

    for _ in 1:iters
        diff = S .- u1[1:(end - 1)]
        ri_hat = sqrt.(vec(sum(abs2, diff; dims=1)))
        B = 2 * Diagonal(ri_hat[2:end])
        W1 = inv(B * Q * B')
        u1 = (G1' * W1 * G1) \ (G1' * W1 * h1)
    end

    u1p = u1 .- [S[:, 1]; 0.0]

    # ============ Second Stage ==========================
    h2 = u1p .^ 2
    G2 = [I(length(u1p) - 1); ones(1, length(u1p) - 1)]

    B2 = 2 * Diagonal(u1p)
    B2i = inv(B2)
    W2 = B2i' * G1' * W1 * G1 * B2i
    u2 = (G2' * W2 * G2) \ (G2' * W2 * h2)

    # ============ Mapping ===============================
    result = sign.(u1p[1:length(u2)]) .* sqrt.(abs.(u2)) .+ S[:, 1]

    if u1[end] < 0 || minimum(u2) < 0
        result = u1[1:length(u2)]
    end

    return result
end
