using BenchmarkTools
using Random
using DataStructures
using Plots
using Printf

include("proposed.jl")
include("previous_works.jl")

function create_benchmark_suite(m, dim)
    x = zeros(dim)
    s = zeros(dim, m)
    z = zeros(m)

    function setup!(x, s, z)
        randn!(x)
        randn!(s)
        o = rand(m)
        for j in 1:m
            z[j] = norm(x - s[:, j]) + o[j]
        end
    end

    suite = BenchmarkGroup()
    suite["linear"] = @benchmarkable multilat_linear($s, $z) setup = ($setup!($x, $s, $z))
    suite["chan"] = @benchmarkable multilat_chan($s, $z) setup = ($setup!($x, $s, $z))
    suite["heidari"] = @benchmarkable multilat_heidari($s, $z) setup = ($setup!($x, $s, $z))
    suite["zeng"] = @benchmarkable multilat_zeng($s, $z) setup = ($setup!($x, $s, $z))
    suite["beck"] = @benchmarkable multilat_beck_srls($s, $z) setup = ($setup!($x, $s, $z))
    suite["solvit"] = @benchmarkable solvit($s, $z) setup = ($setup!($x, $s, $z))
    suite["proposed"] = @benchmarkable multilat($s, $z) setup = ($setup!($x, $s, $z))

    return suite
end

function benchmark_all(m=10, dim=3)
    suite = create_benchmark_suite(m, dim)

    # Warm up for compilation purposes.
    BenchmarkTools.run(suite; evals=1, samples=1)

    # Actual benchmark.
    Random.seed!(0)
    results = BenchmarkTools.run(suite; verbose=true, evals=1, seconds=10, samples=10000)

    return results
end

function create_table()
    ms = [5, 10, 100]

    results = [benchmark_all(m) for m in ms]

    names = OrderedDict{String,String}()
    names["linear"] = "Linear"
    names["chan"] = "Chan"
    names["heidari"] = "Heidari"
    names["zeng"] = "Zeng"
    names["beck"] = "Beck"
    names["solvit"] = "SOLVIT"
    names["proposed"] = "Proposed"

    for (id, name) in names
        if !haskey(results[1], id)
            println("Missing result for solver \"$name\"")
            continue
        end
        print(name)
        for i in 1:length(ms)
            microseconds = median(results[i][id]).time / 1000
            if microseconds >= 1000000
                @printf " & \\SI{%.1f}{\\second}" microseconds / 1000000
            elseif microseconds >= 1000
                @printf " & \\SI{%.1f}{\\milli\\second}" microseconds / 1000
            else
                @printf " & \\SI{%.1f}{\\micro\\second}" microseconds
            end
        end
        println(" \\\\")
    end
    return results
end

# Create table in paper.
results = create_table()
