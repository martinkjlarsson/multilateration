using JuMP
using Hypatia
using LinearAlgebra

function multilat_heidari(s, d; maxiter=1000)
    target_position = multilat_heidari_inner(
        s[:, 2:end] .- s[:, 1], d[2:end] .- d[1]; maxiter
    )
    return target_position + s[:, 1]
end

function multilat_heidari_inner(s, d; maxiter=1000)
    n = size(s)[2]
    dim = size(s)[1]
    s_abs = zeros(n)
    for i in 1:n
        s_abs[i] = norm(s[:, i])
    end
    A = [s' d]
    b = (s_abs .^ 2 - d .^ 2) / 2

    A_tilde = [A'*A -A'*b; -b'*A b'*b]
    C = zeros(dim + 2, dim + 2)
    C[1:dim, 1:dim] = I(dim)
    C[dim + 1, dim + 1] = -1

    model = Model(optimizer_with_attributes(
        Hypatia.Optimizer,
        "iter_limit" => maxiter,
        # "tol_rel_opt" => tol,
        # "tol_abs_opt" => tol,
    ))
    set_silent(model)
    @variable(model, Y[1:(dim + 2), 1:(dim + 2)], PSD)
    @objective(model, Min, tr(A_tilde * Y))
    @constraint(model, Y[dim + 2, dim + 2] == 1)
    @constraint(model, tr(C * Y) == 0)

    # Solve SDP problem.
    optimize!(model)

    # Disregard optimizer status and return what was found.
    res = value.(Y[1:dim, dim + 2])
    return res
end

function multilat_heidari_W(s, d; maxiter=1000)
    target_position = multilat_heidari_inner_W(
        s[:, 2:end] .- s[:, 1], d[2:end] .- d[1]; maxiter
    )
    return target_position + s[:, 1]
end

function multilat_heidari_inner_W(s, d; maxiter=1000)
    n = size(s)[2]
    W = (1.0 * I(n)) / (ones(n, n) + 1.0 * I(n))
    dim = size(s)[1]
    s_abs = zeros(n)
    for i in 1:n
        s_abs[i] = norm(s[:, i])
    end
    A = [s' d]
    b = (s_abs .^ 2 - d .^ 2) / 2

    A_tilde = [A'*W*A -A'*W*b; -b'*W*A b'*W*b]
    C = zeros(dim + 2, dim + 2)
    C[1:dim, 1:dim] = I(dim)
    C[dim + 1, dim + 1] = -1

    model = Model(optimizer_with_attributes(
        Hypatia.Optimizer,
        "iter_limit" => maxiter,
        # "tol_rel_opt" => tol,
        # "tol_abs_opt" => tol,
    ))
    set_silent(model)
    @variable(model, Y[1:(dim + 2), 1:(dim + 2)], PSD)
    @objective(model, Min, tr(A_tilde * Y))
    @constraint(model, Y[dim + 2, dim + 2] == 1)
    @constraint(model, tr(C * Y) == 0)

    # Solve SDP problem.
    optimize!(model)

    # Disregard optimizer status and return what was found.
    res = value.(Y[1:dim, dim + 2])
    return res
end
