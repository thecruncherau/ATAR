using ATAR
using Test
using DataFrames

# 3 students × 3 subjects, all students sitting all subjects
test_data = DataFrame(
    student_id=[1, 1, 1, 2, 2, 2, 3, 3, 3],
    subject_id=[1, 2, 3, 1, 2, 3, 1, 2, 3],
    result=[85.0, 70.0, 60.0, 90.0, 75.0, 65.0, 80.0, 68.0, 58.0]
)
N_STUDENTS = 3
N_SUBJECTS = 3
N_RESULTS = 9   # one unique (subject, result) pair per row in test_data

@testset "ATAR.jl" begin

    @testset "Return structure" begin
        result = run_atar(test_data, 10)

        @test haskey(result, :r)
        @test haskey(result, :p)
        @test haskey(result, :pdash)
        @test haskey(result, :max_rank_changes)

        @test isa(result.r, DataFrame)
        @test isa(result.p, Vector{Float64})
        @test isa(result.pdash, Vector{Float64})
        @test isa(result.max_rank_changes, Vector{Float64})
    end

    @testset "Scaled scores DataFrame (r)" begin
        result = run_atar(test_data, 5)
        df = result.r

        @test hasproperty(df, :subject_id)
        @test hasproperty(df, :result)
        @test hasproperty(df, :percentile)

        # One row per unique (subject, result) pair
        @test nrow(df) == N_RESULTS

        # Percentiles must be valid probabilities
        @test all(0.0 .<= df.percentile .<= 1.0)

        # Every subject appears in r
        @test sort(unique(df.subject_id)) == [1, 2, 3]
    end

    @testset "Polyscores (p)" begin
        result = run_atar(test_data, 10)

        # One score per student
        @test length(result.p) == N_STUDENTS

        # Polyscores are averages of percentiles → must lie in (0, 1)
        @test all(0.0 .< result.p .< 1.0)
    end

    @testset "Polyrank (pdash)" begin
        result = run_atar(test_data, 10)

        # One rank per student
        @test length(result.pdash) == N_STUDENTS

        # Polyranks are fractions of n → must lie in (0, 1]
        @test all(0.0 .< result.pdash .<= 1.0)

        # Higher polyscore should yield higher (or equal) polyrank
        order_p = sortperm(result.p, rev=true)
        order_pdash = sortperm(result.pdash, rev=true)
        @test order_p == order_pdash
    end

    @testset "max_rank_changes" begin
        iterations = 10
        result = run_atar(test_data, iterations)

        # At most one entry per iteration (may be fewer if converged early)
        @test length(result.max_rank_changes) <= iterations

        # Every recorded change is non-negative
        @test all(result.max_rank_changes .>= 0.0)
    end

    @testset "Early stopping with L" begin
        # L larger than any possible swing on 3 students (max swing = 1 position = 1/3)
        # so convergence should trigger on iteration 1
        result_early = run_atar(test_data, 100, 2)
        @test length(result_early.max_rank_changes) < 100

        # L=0 disables early stopping → always runs all iterations
        result_full = run_atar(test_data, 10, 0)
        @test length(result_full.max_rank_changes) == 10
    end

    @testset "Determinism" begin
        r1 = run_atar(test_data, 20)
        r2 = run_atar(test_data, 20)
        @test r1.p == r2.p
        @test r1.pdash == r2.pdash
        @test r1.r.percentile == r2.r.percentile
    end



end
