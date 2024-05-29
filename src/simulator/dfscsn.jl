using Random, Distributions, LinearAlgebra

struct DFsCsnSimulator
    T::Int
    W::Array{Float64,2}
    K::Int
    feature::Array{Float64,3}
    dim::Int
    beta::Array{Float64,1}
    sigma2::Float64
    tau2::Float64
    rhoS::Float64
    rhoT::Float64
    l::Float64
    theta_init::Array{Float64,1}
    theta::Array{Float64,2}
    y::Array{Float64,2}
    alpha::Array{Float64,2}

    function DFsCsnSimulator(T, W, feature; beta=nothing, sigma2=nothing, tau2=nothing, rhoS=nothing, rhoT=nothing, l=nothing, seed=1234, theta_init=nothing)
        Random.seed!(seed)
        K = size(W, 1)
        dim = size(feature, 3)
        beta = isnothing(beta) ? randn(dim) : beta
        sigma2 = isnothing(sigma2) ? rand(InverseGamma(1, 0.01)) : sigma2
        tau2 = isnothing(tau2) ? rand(InverseGamma(1, 0.01)) : tau2
        rhoS = isnothing(rhoS) ? rand() : rhoS
        rhoT = isnothing(rhoT) ? randn() : rhoT
        l = isnothing(l) ? randn() : l
        theta_init = isnothing(theta_init) ? zeros(K) : theta_init

        new(T, W, K, feature, dim, beta, sigma2, tau2, rhoS, rhoT, l, theta_init, zeros(T, K), zeros(T, K), zeros(T, K))
    end
end

function simulate(simulator::DFsCsnSimulator)
    Q = Symmetric(simulator.rhoS * (Diagonal(sum(simulator.W, dims=2)[:]) - simulator.W) + (1-simulator.rhoS) * I(simulator.K))
    invQ = inv(Q)
    invQ_half = sqrt(invQ)
    c = sqrt(pi*(1+simulator.l^2) / ((pi-2)*simulator.l^2+pi))

    simulator.alpha[1, :] = abs.(rand(Normal(0, sqrt(1+simulator.l^2)), simulator.K))
    conditional_mu_theta = c * simulator.l * sqrt(simulator.tau2) / (1+simulator.l^2) * invQ_half * (simulator.alpha[1, :] - sqrt(2*(1+simulator.l^2)/pi) * ones(simulator.K))
    conditional_cov_theta = (c^2 * simulator.tau2 * invQ) / (1+simulator.l^2)
    simulator.theta[1, :] = simulator.rhoT * simulator.theta_init + rand(MvNormal(conditional_mu_theta, conditional_cov_theta))
    simulator.y[1, :] = simulator.feature[1, :, :] * simulator.beta + simulator.theta[1, :] + sqrt(simulator.sigma2) * randn(simulator.K)

    for t in 2:simulator.T
        simulator.alpha[t, :] = abs.(rand(Normal(0, sqrt(1+simulator.l^2)), simulator.K))
        conditional_mu_theta = c * simulator.l * sqrt(simulator.tau2) / (1+simulator.l^2) * invQ_half * (simulator.alpha[t, :] - sqrt(2*(1+simulator.l^2)/pi) * ones(simulator.K))
        conditional_cov_theta = (c^2 * simulator.tau2 * invQ) / (1+simulator.l^2)
        simulator.theta[t, :] = simulator.rhoT * simulator.theta[t-1, :] + rand(MvNormal(conditional_mu_theta, conditional_cov_theta))
        simulator.y[t, :] = simulator.feature[t, :, :] * simulator.beta + simulator.theta[t, :] + sqrt(simulator.sigma2) * randn(simulator.K)
    end
end