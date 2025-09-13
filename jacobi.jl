using LinearAlgebra

"""
    jacobi(A, S; tol=ϵ, max_iter=4*n^2)

Compute the simultaneous diagonalization of the real symmetric matrix `A` and the signature matrix
`S` using the Jacobi method, returning a vector `D` and matrix `Q` such that `Q'*A*Q = diagm(D)`
and `Q'*S*Q = S`. `S` must be nonsingular. If the diagonal of `S` contains elements other than `-1`
and `1`, the constraint `Q'*S*Q = S` might not be satisfied.

The optimization stops when the magnitude of the off-diagonal elements in `Q'*A*Q` are smaller than
`tol` or when the maximum number of iterations `max_iter` is reached.

See Veselié, K. A Jacobi eigenreduction algorithm for definite matrix pairs. Numer. Math. 64,
241-269 (1993). https://doi.org/10.1007/BF01388689
"""
function jacobi(
    A::Symmetric{AT}, S::Diagonal{ST}; tol=eps(eltype(A)), max_iter=4 * length(A)
) where {AT<:Real,ST<:Real}
    Base.require_one_based_indexing(S)

    n = size(A, 1)
    Q = Matrix{eltype(A)}(I, n, n)
    J = Matrix{eltype(A)}(I, n, n) # TODO: A sparse Givens matrix could yield improvements.
    temp = Matrix{eltype(A)}(undef, n, n)
    D = Matrix(A)

    tmax = eltype(A)(0.99)

    prevp = -1
    prevq = -1
    for iter in 1:max_iter
        max_val = zero(eltype(A))
        p, q = -1, -1
        for i in 1:(n - 1), j in (i + 1):n
            if abs(D[i, j]) > max_val
                max_val = abs(D[i, j])
                p, q = i, j
            end
        end

        # println("Iteration $iter\tp,q = ($p, $q)\tmax_val=$max_val")

        if max_val <= tol
            # println("Tolerance $tol reached. Exiting.")
            break
        end
        if p == prevp && prevq == q
            # println("Maximum pivot cannot be reduced. Exiting.")
            break
        end
        prevp = p
        prevq = q

        App = D[p, p]
        Aqq = D[q, q]
        Apq = D[p, q]

        Sprod = S[p, p] * S[q, q]
        if Sprod > 0 # Elliptic
            # Trigonometric functions can be avoided.
            # x = atan(2 * Apq / (App - Aqq)) / 2
            # c2 = cos(x)
            # s2 = sin(x)

            tau = (App - Aqq) / (2 * Apq)
            t = copysign(1 / (abs(tau) + sqrt(1 + tau^2)), tau)
            c = 1 / sqrt(1 + t^2)
            s = c * t

            J[p, p] = c
            J[q, q] = c
            J[p, q] = -s
            J[q, p] = s
        elseif Sprod < 0 # Hyperbolic
            t = -2 * Apq / (App + Aqq)
            t = clamp(t, -tmax, tmax)

            # Trigonometric functions can be avoided.
            # y = atanh(t) / 2
            # c = cosh(y)
            # s = sinh(y)

            sec2 = 2 * sqrt(1 - t^2)
            c = sqrt(1 / sec2 + eltype(A)(0.5))
            s = t / sqrt(2 * (1 - t^2) + sec2)

            J[p, p] = c
            J[q, q] = c
            J[p, q] = s
            J[q, p] = s
        else
            error("S is singular")
        end

        # D = J' * D * J
        mul!(temp, J', D)
        mul!(D, temp, J)
        # Q = Q * J
        mul!(temp, Q, J)
        Q, temp = temp, Q

        # Restore J to identity matrix.
        J[p, p] = one(eltype(A))
        J[q, q] = one(eltype(A))
        J[p, q] = zero(eltype(A))
        J[q, p] = zero(eltype(A))
    end

    return diag(D), Q
end
