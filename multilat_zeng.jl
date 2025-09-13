using LinearAlgebra
using PolynomialRoots

function multilat_zeng(a, d)
    target_position, exitflag = CLSsolver_inner((a[:, 2:end] .- a[:, 1])', d[2:end] .- d[1])
    if exitflag < 0
        return nothing
    end
    return target_position + a[:, 1]
end

function CLSsolver_inner(a, d)
    target_position = Float64[]
    exitflag = 0
    n = length(d)
    m = size(a)[2]
    b = zeros(n)
    A = zeros(n, m + 1)
    D = Diagonal([1; -ones(m)])
    for i in 1:n
        b[i] = ((norm(a[i, :]))^2 - d[i]^2) / 2
        A[i, :] = [d[i] a[i:i, :]]
    end

    # Algorithm starts
    y = zeros(m + 1, 0)
    if norm(b) < 0.001
        target_position = y[2:(m + 1)]
        exitflag = 0
        return target_position, exitflag
    else
        eig_lambda = eigen(D, A' * A).values
        lamuda_l = -1 / maximum(eig_lambda)
        lamuda_u = -1 / minimum(eig_lambda)

        # Searching in the positive-definite interval
        y_l = (A' * A + (lamuda_l + 0.001) * D) \ A' * b
        h_l = y_l' * D * y_l
        y_u = (A' * A + (lamuda_u - 0.001) * D) \ A' * b
        h_u = y_u' * D * y_u
        if (h_l > 0) && (h_u < 0)
            while ((lamuda_u - lamuda_l) > 0.000001)
                lamuda_m = (lamuda_u + lamuda_l) / 2
                y_m = (A' * A + lamuda_m * D) \ A' * b
                h_m = y_m' * D * y_m
                if h_m < 0
                    lamuda_u = lamuda_m
                else
                    lamuda_l = lamuda_m
                end
            end
            lamuda_optimal = (lamuda_u + lamuda_l) / 2
            y_optimal = (A' * A + lamuda_optimal * D) \ A' * b
            if y_optimal[1] > 0
                y = y_optimal
                target_position = y[2:(m + 1)]
                exitflag = 1
                return target_position, exitflag
            end

            # Searching on the singular points
        elseif (h_l < 0) && (h_u < 0)
            y_star = (A' * A + (lamuda_l + 0.001) * D) \ A' * b
            O = A' * A + lamuda_l * D
            tezhengxl = eigen(O).vectors
            TEMP = norm(O * tezhengxl[:, 1])
            INDEX = 1
            for k in 2:(m + 1)
                if norm(O * tezhengxl[:, k]) < TEMP
                    TEMP = norm(O * tezhengxl[:, k])
                    INDEX = k
                end
            end
            z_negative = tezhengxl[:, INDEX]
            z_negative = z_negative / norm(z_negative)
            aaa = z_negative' * D * z_negative
            bbb = 2 * y_star' * D * z_negative
            ccc = y_star' * D * y_star
            alfa1 = (-bbb + (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
            alfa2 = (-bbb - (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
            y_G_star1 = y_star + alfa1 * z_negative
            y_G_star2 = y_star + alfa2 * z_negative
            if y_G_star1[1] > 0
                target_position = y_G_star1[2:(m + 1)]
                exitflag = 2
                return target_position, exitflag
            else
                target_position = y_G_star2[2:(m + 1)]
                exitflag = 2
                return target_position, exitflag
            end
        elseif (h_l > 0) && (h_u > 0)
            temp = (A' * A) \ A' * b
            if temp[1] > 0
                # Save temp temp
                y_star = (A' * A + (lamuda_u - 0.001) * D) \ A' * b
                O = A' * A + lamuda_u * D
                tezhengxl = eigen(O).vectors
                TEMP = norm(O * tezhengxl[:, 1])
                INDEX = 1
                for k in 2:(m + 1)
                    if norm(O * tezhengxl[:, k]) < TEMP
                        TEMP = norm(O * tezhengxl[:, k])
                        INDEX = k
                    end
                end
                z_positive = tezhengxl[:, INDEX]
                z_positive = z_positive / norm(z_positive)
                aaa = z_positive' * D * z_positive
                bbb = 2 * y_star' * D * z_positive
                ccc = y_star' * D * y_star
                alfa1 = (-bbb + (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
                y_G_star = y_star + alfa1 * z_positive
                target_position = y_G_star[2:(m + 1)]
                exitflag = 3
                return target_position, exitflag
            end
        end

        # Searching in the indefinite interval
        tempEigenResult = eigen(A' * A)
        if any(tempEigenResult.values .<= 0)
            # A'*A is not positive definite.
            return target_position, -1 # Positioning failed.
        end

        Q = tempEigenResult.vectors
        LAMUDA = Diagonal(tempEigenResult.values)
        # [Q,LAMUDA]=eig(A'*A);
        R = Q * LAMUDA^(-0.5)

        tempEigenResult2 = eigen(R' * D * R)
        V = tempEigenResult2.vectors
        SIGAMA = Diagonal(tempEigenResult2.values)
        # [V,SIGAMA]=eig(R'*D*R);
        P = R * V
        F = P' * A' * b
        if m == 3
            coeff =
                F[1]^2 *
                SIGAMA[1, 1] *
                myConv(
                    myConv(
                        [SIGAMA[2, 2]^2; 2 * SIGAMA[2, 2]; 1],
                        [SIGAMA[3, 3]^2; 2 * SIGAMA[3, 3]; 1],
                    ),
                    [SIGAMA[4, 4]^2; 2 * SIGAMA[4, 4]; 1],
                ) +
                F[2]^2 *
                SIGAMA[2, 2] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                    ),
                    [SIGAMA[4, 4]^2, 2 * SIGAMA[4, 4], 1],
                ) +
                F[3]^2 *
                SIGAMA[3, 3] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    ),
                    [SIGAMA[4, 4]^2, 2 * SIGAMA[4, 4], 1],
                ) +
                F[4]^2 *
                SIGAMA[4, 4] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    ),
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                )
        else
            coeff =
                F[1]^2 *
                SIGAMA[1, 1] *
                myConv(
                    [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                ) +
                F[2]^2 *
                SIGAMA[2, 2] *
                conv(
                    [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                ) +
                F[3]^2 *
                SIGAMA[3, 3] *
                conv(
                    [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                    [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                )
        end
        ROOTS = roots(coeff)
        for i in eachindex(ROOTS)
            if isreal(ROOTS[i]) && (real(ROOTS[i]) < lamuda_l)
                y_star = (A' * A + real(ROOTS[i]) * D) \ A' * b
                if y_star[1] > 0
                    target_position = y_star[2:(m + 1)]
                    exitflag = 4
                    return target_position, exitflag
                end
            end
        end
        target_position = zeros(m, 1)
        exitflag = 5
        return target_position, exitflag
    end
end

# Assumes columns
function myConv(a, b)
    b_padded = [zeros(length(a) - 1); b; zeros(length(a) - 1)]
    c = zeros(length(a) + length(b) - 1)
    for i in eachindex(c)
        c[i] = a' * b_padded[i:(i + length(a) - 1)]
    end
    return c
end

function multilat_zeng_W(a, d)
    target_position, exitflag = CLSsolver_inner_W(
        (a[:, 2:end] .- a[:, 1])', d[2:end] .- d[1]
    )
    if exitflag < 0
        return nothing
    end
    return target_position + a[:, 1]
end

function CLSsolver_inner_W(a, d)
    target_position = Float64[]
    exitflag = 0
    n = length(d)
    m = size(a)[2]
    b = zeros(n)
    A = zeros(n, m + 1)
    D = Diagonal([1; -ones(m)])
    for i in 1:n
        b[i] = ((norm(a[i, :]))^2 - d[i]^2) / 2
        A[i, :] = [d[i] a[i:i, :]]
    end
    W = sqrt((1.0 * I(n)) / (ones(n, n) + 1.0 * I(n)))
    A = W * A
    b = W * b

    # Algorithm starts
    y = zeros(m + 1, 0)
    if norm(b) < 0.001
        target_position = y[2:(m + 1)]
        exitflag = 0
        return target_position, exitflag
    else
        eig_lambda = eigen(D, A' * A).values
        lamuda_l = -1 / maximum(eig_lambda)
        lamuda_u = -1 / minimum(eig_lambda)

        # Searching in the positive-definite interval
        y_l = (A' * A + (lamuda_l + 0.001) * D) \ A' * b
        h_l = y_l' * D * y_l
        y_u = (A' * A + (lamuda_u - 0.001) * D) \ A' * b
        h_u = y_u' * D * y_u
        if (h_l > 0) && (h_u < 0)
            while ((lamuda_u - lamuda_l) > 0.000001)
                lamuda_m = (lamuda_u + lamuda_l) / 2
                y_m = (A' * A + lamuda_m * D) \ A' * b
                h_m = y_m' * D * y_m
                if h_m < 0
                    lamuda_u = lamuda_m
                else
                    lamuda_l = lamuda_m
                end
            end
            lamuda_optimal = (lamuda_u + lamuda_l) / 2
            y_optimal = (A' * A + lamuda_optimal * D) \ A' * b
            if y_optimal[1] > 0
                y = y_optimal
                target_position = y[2:(m + 1)]
                exitflag = 1
                return target_position, exitflag
            end

            # Searching on the singular points
        elseif (h_l < 0) && (h_u < 0)
            y_star = (A' * A + (lamuda_l + 0.001) * D) \ A' * b
            O = A' * A + lamuda_l * D
            tezhengxl = eigen(O).vectors
            TEMP = norm(O * tezhengxl[:, 1])
            INDEX = 1
            for k in 2:(m + 1)
                if norm(O * tezhengxl[:, k]) < TEMP
                    TEMP = norm(O * tezhengxl[:, k])
                    INDEX = k
                end
            end
            z_negative = tezhengxl[:, INDEX]
            z_negative = z_negative / norm(z_negative)
            aaa = z_negative' * D * z_negative
            bbb = 2 * y_star' * D * z_negative
            ccc = y_star' * D * y_star
            alfa1 = (-bbb + (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
            alfa2 = (-bbb - (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
            y_G_star1 = y_star + alfa1 * z_negative
            y_G_star2 = y_star + alfa2 * z_negative
            if y_G_star1[1] > 0
                target_position = y_G_star1[2:(m + 1)]
                exitflag = 2
                return target_position, exitflag
            else
                target_position = y_G_star2[2:(m + 1)]
                exitflag = 2
                return target_position, exitflag
            end
        elseif (h_l > 0) && (h_u > 0)
            temp = (A' * A) \ A' * b
            if temp[1] > 0
                # Save temp temp
                y_star = (A' * A + (lamuda_u - 0.001) * D) \ A' * b
                O = A' * A + lamuda_u * D
                tezhengxl = eigen(O).vectors
                TEMP = norm(O * tezhengxl[:, 1])
                INDEX = 1
                for k in 2:(m + 1)
                    if norm(O * tezhengxl[:, k]) < TEMP
                        TEMP = norm(O * tezhengxl[:, k])
                        INDEX = k
                    end
                end
                z_positive = tezhengxl[:, INDEX]
                z_positive = z_positive / norm(z_positive)
                aaa = z_positive' * D * z_positive
                bbb = 2 * y_star' * D * z_positive
                ccc = y_star' * D * y_star
                alfa1 = (-bbb + (bbb^2 - 4 * aaa * ccc)^0.5) / (2 * aaa)
                y_G_star = y_star + alfa1 * z_positive
                target_position = y_G_star[2:(m + 1)]
                exitflag = 3
                return target_position, exitflag
            end
        end

        # Searching in the indefinite interval
        tempEigenResult = eigen(A' * A)
        if any(tempEigenResult.values .<= 0)
            # A'*A is not positive definite.
            return target_position, -1 # Positioning failed.
        end

        Q = tempEigenResult.vectors
        LAMUDA = Diagonal(tempEigenResult.values)
        # [Q,LAMUDA]=eig(A'*A);
        R = Q * LAMUDA^(-0.5)

        tempEigenResult2 = eigen(R' * D * R)
        V = tempEigenResult2.vectors
        SIGAMA = Diagonal(tempEigenResult2.values)
        # [V,SIGAMA]=eig(R'*D*R);
        P = R * V
        F = P' * A' * b
        if m == 3
            coeff =
                F[1]^2 *
                SIGAMA[1, 1] *
                myConv(
                    myConv(
                        [SIGAMA[2, 2]^2; 2 * SIGAMA[2, 2]; 1],
                        [SIGAMA[3, 3]^2; 2 * SIGAMA[3, 3]; 1],
                    ),
                    [SIGAMA[4, 4]^2; 2 * SIGAMA[4, 4]; 1],
                ) +
                F[2]^2 *
                SIGAMA[2, 2] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                    ),
                    [SIGAMA[4, 4]^2, 2 * SIGAMA[4, 4], 1],
                ) +
                F[3]^2 *
                SIGAMA[3, 3] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    ),
                    [SIGAMA[4, 4]^2, 2 * SIGAMA[4, 4], 1],
                ) +
                F[4]^2 *
                SIGAMA[4, 4] *
                myConv(
                    myConv(
                        [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                        [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    ),
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                )
        else
            coeff =
                F[1]^2 *
                SIGAMA[1, 1] *
                myConv(
                    [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                ) +
                F[2]^2 *
                SIGAMA[2, 2] *
                conv(
                    [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                    [SIGAMA[3, 3]^2, 2 * SIGAMA[3, 3], 1],
                ) +
                F[3]^2 *
                SIGAMA[3, 3] *
                conv(
                    [SIGAMA[1, 1]^2, 2 * SIGAMA[1, 1], 1],
                    [SIGAMA[2, 2]^2, 2 * SIGAMA[2, 2], 1],
                )
        end
        ROOTS = roots(coeff)
        for i in eachindex(ROOTS)
            if isreal(ROOTS[i]) && (real(ROOTS[i]) < lamuda_l)
                y_star = (A' * A + real(ROOTS[i]) * D) \ A' * b
                if y_star[1] > 0
                    target_position = y_star[2:(m + 1)]
                    exitflag = 4
                    return target_position, exitflag
                end
            end
        end
        target_position = zeros(m, 1)
        exitflag = 5
        return target_position, exitflag
    end
end
