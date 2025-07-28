using ATAR
using Test
using DataFrames

@testset "ATAR.jl" begin
    # Create test data
    test_data = DataFrame(
        student_id = [1, 1, 1, 2, 2, 2, 3, 3, 3],
        subject_id = [1, 2, 3, 1, 2, 3, 1, 2, 3],
        result = [85.0, 70.0, 60.0, 90.0, 75.0, 65.0, 80.0, 68.0, 58.0]
    )
    
    @testset "Basic functionality" begin
        result = run_atar(test_data, 10)
        
        @test haskey(result, :r)
        @test haskey(result, :p)
        @test haskey(result, :pdash)
        @test isa(result.r, DataFrame)
        @test isa(result.p, Vector)
        @test isa(result.pdash, Vector)
    end
    
    # @testset "DataFrame structure" begin
    #     result = run_atar(test_data, 5)
    #     df = result.r
        
    #     @test haskey(df, :subject_id)
    #     @test haskey(df, :result)
    #     @test haskey(df, :percentile)
    #     @test all(0 .<= df.percentile .<= 1)
    # end
end