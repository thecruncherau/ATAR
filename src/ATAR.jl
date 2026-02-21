module ATAR
using DataFrames
using Statistics
using StatsBase

export run_atar

logit(p::Number) = log(p / (1 - p))
expit(x::Number) = 1 / (1 + exp(-x))

# Core data structure: pre-indexed lookup tables built once from the DataFrame
struct IndexedData
    # subject_id => sorted unique results
    subject_results::Dict{Int, Vector{Float64}}
    # (subject_id, result) => percentile (mutable dict so we can update in-place)
    percentile::Dict{Tuple{Int,Float64}, Float64}
    # student_id => index in the student array
    student_index::Dict{Int, Int}
    # student_id => Vector of (subject_id, result) pairs
    student_subjects::Dict{Int, Vector{Tuple{Int,Float64}}}
    # subject_id => Vector of (student_index, result) pairs
    subject_students::Dict{Int, Vector{Tuple{Int,Float64}}}
    student_ids::Vector{Int}
    subject_ids::Vector{Int}
    n_students::Int
end

function build_index(raw_data::DataFrame)::IndexedData
    student_ids = unique(raw_data.student_id)
    subject_ids = unique(raw_data.subject_id)
    n_students  = length(student_ids)

    student_index = Dict(id => i for (i, id) in enumerate(student_ids))

    student_subjects = Dict{Int, Vector{Tuple{Int,Float64}}}(id => Tuple{Int,Float64}[] for id in student_ids)
    subject_students = Dict{Int, Vector{Tuple{Int,Float64}}}(id => Tuple{Int,Float64}[] for id in subject_ids)
    subject_results  = Dict{Int, Vector{Float64}}()

    for row in eachrow(raw_data)
        sid, subj, res = row.student_id, row.subject_id, Float64(row.result)
        push!(student_subjects[sid], (subj, res))
        push!(subject_students[subj], (student_index[sid], res))
    end

    for subj in subject_ids
        subject_results[subj] = sort(unique(r for (_, r) in subject_students[subj]))
    end

    percentile = Dict{Tuple{Int,Float64}, Float64}()

    return IndexedData(
        subject_results, percentile,
        student_index, student_subjects, subject_students,
        student_ids, subject_ids, n_students
    )
end

function calculate_raw_percentile_rank!(idx::IndexedData)
    n = idx.n_students
    for subj in idx.subject_ids
        pairs    = idx.subject_students[subj]
        results  = [r for (_, r) in pairs]
        counts   = countmap(results)
        sorted   = idx.subject_results[subj]  # already sorted
        cumcount = 0
        for res in sorted
            c = counts[res]
            idx.percentile[(subj, res)] = (cumcount + c / 2) / n
            cumcount += c
        end
    end
end

function calculate_polyscores(idx::IndexedData)::Vector{Float64}
    p = Vector{Float64}(undef, idx.n_students)
    for (i, sid) in enumerate(idx.student_ids)
        subjects = idx.student_subjects[sid]
        p[i] = mean(idx.percentile[(subj, res)] for (subj, res) in subjects)
    end
    return p
end

function calculate_polyrank(p::Vector{Float64})::Vector{Float64}
    n     = length(p)
    ranks = n .+ 1 .- competerank(p, rev=true)
    return ranks ./ n
end

function fit_logistic!(idx::IndexedData, polyrank::Vector{Float64}, eps::Float64=1e-6)
    for subj in idx.subject_ids
        pairs = idx.subject_students[subj]
        n     = length(pairs)

        # Build X and y for OLS in logit space
        results = [r for (_, r) in pairs]
        ys      = [clamp(polyrank[si], eps, 1 - eps) |> logit for (si, _) in pairs]

        # Analytic OLS: X = [1 | result], solve for betas
        mean_x  = mean(results)
        mean_y  = mean(ys)
        ss_xx   = sum((x - mean_x)^2 for x in results)
        ss_xy   = sum((results[i] - mean_x) * (ys[i] - mean_y) for i in 1:n)
        beta1   = ss_xx > 0 ? ss_xy / ss_xx : 0.0
        beta0   = mean_y - beta1 * mean_x

        # Update percentiles for every unique result in this subject
        for res in idx.subject_results[subj]
            idx.percentile[(subj, res)] = expit(beta0 + beta1 * res)
        end
    end
end

# Returns a DataFrame in the same shape as the original r output for compatibility
function to_dataframe(idx::IndexedData)::DataFrame
    rows = [(subj, res, idx.percentile[(subj, res)])
            for subj in idx.subject_ids
            for res  in idx.subject_results[subj]]
    return DataFrame(
        subject_id  = [r[1] for r in rows],
        result      = [r[2] for r in rows],
        percentile  = [r[3] for r in rows],
    )
end

"""
    run_atar(raw_data, iterations, L)

Run the ATAR scaling algorithm.

# Arguments
- `raw_data`:   DataFrame with columns `student_id`, `subject_id`, `result`.
- `iterations`: Maximum number of iterations `M`.
- `L`:          Convergence tolerance expressed as a **position swing** (number of student
                positions). For example, `L=1` means the algorithm stops when the maximum
                change in polyrank percentile rank is less than `1 / n_students` — equivalent
                to at most ±1 student position changing rank. `L=0` (default) disables early
                stopping and always runs all iterations.

The per-iteration threshold is `ε_{m+1} = L / n_students`, matching equation (10) and
Table 4 of the specification (e.g. 50,000 students, L=1 → ε = 0.002%).
"""
function run_atar(raw_data::DataFrame, iterations::Int=100, L::Real=0)
    idx         = build_index(raw_data)
    n_students  = idx.n_students
    # Convert position swing → polyrank fraction (ε in equation 10)
    # L=0 means no early stopping (ε=0 can never be triggered on finite data)
    eps_threshold = L / n_students

    calculate_raw_percentile_rank!(idx)
    p     = calculate_polyscores(idx)
    pdash = calculate_polyrank(p)

    max_rank_changes = Float64[]

    for m in 1:iterations
        fit_logistic!(idx, pdash)
        p          = calculate_polyscores(idx)
        pdash_new  = calculate_polyrank(p)
        # max_s |P'_{s,m} - P'_{s,(m+1)}|  (equation 10)
        max_change = maximum(abs.(pdash_new .- pdash))
        push!(max_rank_changes, max_change)
        pdash = pdash_new

        # Express max_change back as a position swing for legible logging
        max_swing = max_change * n_students
        @info "Iteration $m: max swing = ±$(round(max_swing, digits=3)) positions (ε = ±$L positions)"

        if L > 0 && max_change < eps_threshold
            @info "Converged after $m iterations (max swing ±$(round(max_swing, digits=3)) < ±$L positions)"
            break
        end
    end

    return (r=to_dataframe(idx), p=p, pdash=pdash, max_rank_changes=max_rank_changes)
end

end
