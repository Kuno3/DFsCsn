using LinearAlgebra

function logsumexp(values::Array{Float64})
    max_val = maximum(values)
    sum_exp = sum(exp.(values .- max_val))
    return log(sum_exp) + max_val
end

function calculate_inv(eigen_values, eigen_vector)
    return Symmetric(eigen_vector * Diagonal(1.0 ./ eigen_values) * eigen_vector')
end

function calculate_sqrt(eigen_values, eigen_vector)
    return eigen_vector * Diagonal(1.0 ./ sqrt.(eigen_values)) * eigen_vector'
end