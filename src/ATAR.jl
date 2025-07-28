module ATAR

using DataFrames
using Statistics
using StatsBase

export run_atar

function calculate_raw_percentile_rank(d::DataFrame, subject_id::Int, total_students::Int)
    subsetted_data = subset(d, :subject_id => ByRow(isequal(subject_id)))
    by_grade = combine(groupby(subsetted_data, :result), nrow => :count)
    sort!(by_grade, :result)
    
    return DataFrame(
        :subject_id => fill(subject_id, nrow(by_grade)),
        :result => by_grade.result,
        :percentile => [by_grade.count[ix] / 2 + sum(by_grade.count[1:ix-1]) for ix in eachindex(by_grade.result)] ./ total_students
    )
end

function calculate_raw_percentile_rank(d::DataFrame)
    student_ids = unique(d.student_id)
    subject_ids = unique(d.subject_id)
    total_students = length(student_ids)
    
    return vcat([calculate_raw_percentile_rank(d, subject_id, total_students) for subject_id in subject_ids]...)
end

function calculate_polyscore(d::DataFrame, r::DataFrame, student_id::Int)
    subsetted_d = subset(d, :student_id => ByRow(isequal(student_id)))
    
    return mean([
        r[(r.subject_id .== subsetted_d.subject_id[ix]) .&& (r.result .== subsetted_d.result[ix]), :percentile][1] for ix in 1:nrow(subsetted_d)
    ])
end

function calculate_polyscore(d::DataFrame, r::DataFrame)
    student_ids = unique(d.student_id)
    return [calculate_polyscore(d, r, student_id) for student_id in student_ids]
end

function calculate_polyrank(p::Vector{T}) where T <: Number
    total_students = length(p)
    ranks = total_students .+ 1 .- competerank(p, rev=true) 
    return ranks ./ length(p)
end

logit(p::Number) = log(p / (1 - p))
expit(x::Number) = 1 / (1 + exp(-x))

function fit_logistic(d::DataFrame, r::DataFrame, p::Vector{T}, subject_id::Int, student_ids::Vector{Int}, eps=1e-6) where T <: Number
    subsetted_d = subset(d, :subject_id => ByRow(isequal(subject_id)))
    subsetted_r = subset(r, :subject_id => ByRow(isequal(subject_id)))

    subject_student_ids = subsetted_d.student_id
    polyrank_indices = [findfirst(isequal(subject_student_id), student_ids) for subject_student_id in subject_student_ids]

    X = hcat(ones(nrow(subsetted_d)), subsetted_d.result)
    y = clamp.(p[polyrank_indices], eps, 1 - eps) .|> logit

    betas = X \ y

    subsetted_r.percentile = subsetted_r.result .|> (j -> betas[1] + betas[2] * j) .|> expit
    return subsetted_r
end

function fit_logistic(d::DataFrame, r::DataFrame, p::Vector{T}, eps=1e-6) where T <: Number
    student_ids = unique(d.student_id)
    subject_ids = unique(d.subject_id)
    return vcat([fit_logistic(d, r, p, subject_id, student_ids, eps) for subject_id in subject_ids]...)
end

function run_atar(raw_data::DataFrame, iterations::Int=100)
    r = calculate_raw_percentile_rank(raw_data)
    p = calculate_polyscore(raw_data, r)
    pdash = calculate_polyrank(p)

    for _ in 1:iterations
        r = fit_logistic(raw_data, r, p) 
        p = calculate_polyscore(raw_data, r)
        pdash = calculate_polyrank(p)
    end

    return (r=r, p=p, pdash=pdash)
end

end