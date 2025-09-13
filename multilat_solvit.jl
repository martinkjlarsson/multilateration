using LinearAlgebra
using Statistics

function solvit(y, r::AbstractVector)
    return solvit(y, r .- r')
end

function solvit(y, r)
    f = x -> errorFun(x, y, r)
    k = 0
    xk = mean(y; dims=2)
    maxIters = 10000
    while k < maxIters
        k = k + 1
        x_next = computeNextXAcc(xk, y, r, f)
        if abs((f(x_next) - f(xk)) / f(xk)) < 1e-6
            xk = x_next
            break
        end

        xk = x_next
    end
    return xk
end

function computeNextXAcc(x0, y, r, f)
    x1 = computeNextX(x0, y, r)
    x2 = computeNextX(x1, y, r)
    rr = x1 - x0
    v = (x2 - x1) - rr
    alpha = -norm(rr) / norm(v)
    x_sq = x0 - 2 * alpha * rr + alpha^2 * v
    if f(x_sq) < f(x2)
        x_next = computeNextX(x_sq, y, r)
    else
        x_next = x2
    end
    return x_next
end

function computeNextX(xk, y, r)
    dim = size(y)[1]
    m = size(y)[2]
    numCols = m * (m - 1)
    d_val = 0
    U = zeros(dim, numCols)
    V = zeros(dim, numCols)
    b = zeros(dim, 1)
    colid = 1
    for i in 1:m
        for j in 1:m
            if r[i, j] < 0 || i == j
                continue
            end
            yi = y[:, i:i]
            yj = y[:, j:j]
            wi = (xk - yi) / computeDist(xk, yi)
            sij = r[i, j] / computeDist(xk, yj)
            d_val += 2 + sij
            Qij = (xk - yj) * (xk - yi)' / (computeDist(xk, yj) * computeDist(xk, yi))
            pij = yi + yj + r[i, j] * wi + sij * yj - Qij * yi - Qij' * yj
            uj = (xk - yj) / computeDist(xk, yj)
            vi = (xk - yi) / computeDist(xk, yi)
            U[:, colid] = uj
            V[:, colid] = vi
            b += pij

            colid += 1
        end
    end
    D = d_val * I(dim)
    D_inv = I(dim) / D
    B_inv = D_inv + D_inv * U * ((I(numCols) - V' * D_inv * U) \ V') * D_inv
    A = B_inv + B_inv * V * ((I(numCols) - U' * B_inv * V) \ U') * B_inv
    x_next = A * b
    return x_next
end

function computeDist(x, y)
    return norm(x - y)
end

function errorFun(x, y, r)
    m = size(y)[2]
    error = 0
    for i in 1:m
        for j in 1:m
            if r[i, j] < 0 || i == j
                continue
            end
            error += (computeDist(x, y[:, i:i]) - computeDist(x, y[:, j:j]) - r[i, j])^2
        end
    end
    return error
end
