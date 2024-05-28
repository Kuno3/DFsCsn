using LinearAlgebra, Random, Distributions, ProgressMeter

struct DCarPostSampler
    y::Array{Float64, 2}
    W::Array{Float64, 2}
    feature::Array{Float64, 3}
    nu2_beta::Float64
    a_sigma2::Float64
    b_sigma2::Float64
    a_tau2::Float64
    b_tau2::Float64
    T::Int
    K::Int
    dim::Int
    y_flatten::Array{Float64, 1}
    feature_reshaped::Array{Float64, 2}

    function DCarPostSampler(
        y,
        W,
        feature;
        nu2_beta=100.0,
        a_sigma2=1.0,
        b_sigma2=0.01,
        a_tau2=1.0,
        b_tau2=0.01,
        seed=1234
    )
        Random.seed!(seed)
        T = size(y, 1)
        K = size(W, 1)
        dim = size(feature, 3)
        y_flatten = vec(y)
        y_flatten = y'[:]
        feature_reshaped = reshape(permutedims(feature, [2, 1, 3]), T * K, dim)

        new(
            y,
            W,
            feature,
            nu2_beta,
            a_sigma2,
            b_sigma2,
            a_tau2,
            b_tau2,
            nu2_rhoT,
            T,
            K,
            dim,
            y_flatten,
            feature_reshaped
        )
    end
end

function log_acceptance_ratio_theta(sampler::DCarPostSampler, theta, tau2, rhoS, rhoT)
    try
        Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))

        log_acceptance_ratio = sampler.T * (- sampler.K / 2 * log(tau2) + 0.5 * logdet(Q))
        log_acceptance_ratio += - 1 / (2 * tau2) * dot(theta[1, :], Q * theta[1, :])
        
        for t in 2:sampler.T
            log_acceptance_ratio += - 1 / (2 * tau2) * dot(theta[t, :] - rhoT * theta[t-1, :], Q * (theta[t, :] - rhoT * theta[t-1, :]))
        end
        return log_acceptance_ratio
    catch
        return -Inf
    end
end

function theta_sampler(sampler::DCarPostSampler, beta, sigma2, tau2, rhoS, rhoT)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    invQ = inv(Q)
    
    mean_theta_pred = zeros(sampler.T, sampler.K)
    cov_theta_pred = zeros(sampler.T, sampler.K, sampler.K)
    mean_theta_filt = zeros(sampler.T, sampler.K)
    cov_theta_filt = zeros(sampler.T, sampler.K, sampler.K)
    
    cov_theta_pred[1, :, :] = tau2 * invQ
    mean_theta_pred[1, :] = zeros(sampler.K)
    cov_theta_filt[1, :, :] = (cov_theta_pred[1, :, :] + sigma2 * I) \ cov_theta_pred[1, :, :] * sigma2
    mean_theta_filt[1, :] = mean_theta_pred[1, :] + cov_theta_filt[1, :, :] * (sampler.y[1, :] - sampler.feature[1, :, :] * beta - mean_theta_pred[1, :]) / sigma2
    
    for t in 2:sampler.T
        cov_theta_pred[t, :, :] = rhoT^2 * cov_theta_filt[t-1, :, :] + tau2 * invQ
        mean_theta_pred[t, :] = rhoT * mean_theta_filt[t-1, :]
        cov_theta_filt[t, :, :] = (cov_theta_pred[t, :, :] + sigma2 * I) \ cov_theta_pred[t, :, :] * sigma2
        mean_theta_filt[t, :] = mean_theta_pred[t, :] + cov_theta_filt[t, :, :] * (sampler.y[t, :] - sampler.feature[t, :, :] * beta - mean_theta_pred[t, :]) / sigma2
    end
    
    theta_sample = zeros(sampler.T, sampler.K)
    theta_sample[sampler.T, :] = rand(MvNormal(mean_theta_filt[sampler.T, :], Symmetric(cov_theta_filt[sampler.T, :, :])))
    for t in sampler.T-1:-1:1
        gain = rhoT * cov_theta_filt[t, :, :] / cov_theta_pred[t+1, :, :]
        mean_theta_ffbs = mean_theta_filt[t, :] + gain * (theta_sample[t+1, :] - mean_theta_pred[t+1, :])
        cov_theta_ffbs = cov_theta_filt[t, :, :] - gain * cov_theta_pred[t+1, :, :] * gain'
        theta_sample[t, :] = rand(MvNormal(mean_theta_ffbs, Symmetric(cov_theta_ffbs)))
    end

    return theta_sample
end

function beta_sampler(sampler::DCarPostSampler, theta, sigma2)
    theta_flatten = theta'[:]
    cov_beta = inv(sampler.feature_reshaped' * sampler.feature_reshaped / sigma2 + I / sampler.nu2_beta)
    mu_beta = cov_beta * (sampler.feature_reshaped' * (sampler.y_flatten - theta_flatten)) / sigma2
    beta = rand(MvNormal(mu_beta, Symmetric(cov_beta)))
    return beta
end

function sigma2_sampler(sampler::DCarPostSampler, theta, beta)
    theta_flatten = theta'[:]
    an = sampler.a_sigma2 + sampler.K * sampler.T / 2
    bn = sampler.b_sigma2 + sum((sampler.y_flatten - theta_flatten - sampler.feature_reshaped * beta).^2) / 2
    return rand(InverseGamma(an, bn))
end

function tau2_sampler(sampler::DCarPostSampler, theta, rhoS, rhoT)
    theta_flatten = theta'[:]
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    D = create_D(sampler.T, rhoT)
    an = sampler.a_tau2 + sampler.K * sampler.T / 2
    bn = sampler.b_tau2 + (theta_flatten' * (kron(D, Q) * theta_flatten)) / 2

    return rand(InverseGamma(an, bn))
end

function rhoS_sampler(sampler::DCarPostSampler, theta, tau2, rhoS, rhoT, rhoS_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = Beta(rhoS_prop_scale * rhoS + 1e-4, rhoS_prop_scale * (1 - rhoS) + 1e-4)
    rhoS_prop = rand(dst_curr)
    dst_prop = Beta(rhoS_prop_scale * rhoS_prop + 1e-4, rhoS_prop_scale * (1 - rhoS_prop) + 1e-4)

    lp_new = log_acceptance_ratio_theta(sampler, theta, tau2, rhoS_prop, rhoT)
    lp_old = isnothing(lp_old) ? log_acceptance_ratio_theta(sampler, theta, tau2, rhoS, rhoT) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio -= logpdf(dst_curr, rhoS_prop) - logpdf(dst_prop, rhoS)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (rhoS_prop, lp_new) : (rhoS, lp_old)
end

function rhoT_sampler(sampler::DCarPostSampler, theta, tau2, rhoS)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))

    prec_rhoT = 0
    mu_rhoT = 0
    for t in 1:sampler.T-1
        prec_rhoT += theta[t, :]' * Q * theta[t, :] / tau2
        mu_rhoT += theta[t+1, :]' * Q * theta[t, :] / tau2
    end
    var_rhoT = 1 / prec_rhoT
    mu_rhoT /= prec_rhoT
    return rand(Truncated(Normal(mu_rhoT, sqrt(var_rhoT)), 0, 1))
end

function log_posterior(sampler::DCarPostSampler, theta, beta, sigma2, tau2, rhoS, rhoT, lp_old::Union{Float64, Nothing}=nothing)
    theta_flatten = theta'[:]
    lp = isnothing(lp_old) ? log_acceptance_ratio_theta(sampler, theta, tau2, rhoS, rhoT) : lp_old
    lp += sum(logpdf.(Normal.(sampler.feature_reshaped * beta + theta_flatten, sqrt(sigma2)), sampler.y_flatten))
    lp += sum(logpdf.(Normal(0.0, sqrt(sampler.nu2_beta)), beta))
    lp += logpdf(InverseGamma(sampler.a_sigma2, sampler.b_sigma2), sigma2)
    lp += logpdf(InverseGamma(sampler.a_tau2, sampler.b_tau2), tau2)
    lp += logpdf(Normal(0, sqrt(sampler.nu2_rhoT)), rhoT)
    return lp
end

function sampling(
    sampler::DCarPostSampler,
    num_sample;
    burn_in=0,
    thinning=1,
    rhoS_prop_scale=0.1,
    theta_init=nothing,
    beta_init=nothing,
    sigma2_init=nothing,
    tau2_init=nothing,
    rhoS_init=nothing,
    rhoT_init=nothing
)
    rhoS_prop_scale = (1 - rhoS_prop_scale^2) / rhoS_prop_scale^2
    
    # Initialize variables
    theta = isnothing(theta_init) ? zeros(sampler.T, sampler.K) : theta_init
    beta = isnothing(beta_init) ? zeros(sampler.dim) : beta_init
    sigma2 = isnothing(sigma2_init) ? 1.0 : sigma2_init
    tau2 = isnothing(tau2_init) ? 1.0 : tau2_init
    rhoS = isnothing(rhoS_init) ? 0.5 : rhoS_init
    rhoT = isnothing(rhoT_init) ? 0.5 : rhoT_init
    lp = -Inf

    # Arrays to store samples
    theta_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    beta_samples = zeros(Float64, (num_sample, sampler.dim))
    sigma2_samples = zeros(Float64, num_sample)
    tau2_samples = zeros(Float64, num_sample)
    rhoS_samples = zeros(Float64, num_sample)
    rhoT_samples = zeros(Float64, num_sample)
    lp_list = zeros(Float64, num_sample)

    @showprogress for i in 1:(burn_in + num_sample)
        for _ in 1:thinning
            theta = theta_sampler(sampler, beta, sigma2, tau2, rhoS, rhoT)
            beta = beta_sampler(sampler, theta, sigma2)
            sigma2 = sigma2_sampler(sampler, theta, beta)
            tau2 = tau2_sampler(sampler, theta, rhoS, rhoT)
            rhoT = rhoT_sampler(sampler, theta, tau2, rhoS)
            rhoS, lp = rhoS_sampler(sampler, theta, tau2, rhoS, rhoT, rhoS_prop_scale)
        end

        if i > burn_in
            theta_samples[i-burn_in, :, :] = theta
            beta_samples[i-burn_in, :] = beta
            sigma2_samples[i-burn_in] = sigma2
            tau2_samples[i-burn_in] = tau2
            rhoS_samples[i-burn_in] = rhoS
            rhoT_samples[i-burn_in] = rhoT
            lp_list[i-burn_in] = log_posterior(sampler, theta, beta, sigma2, tau2, rhoS, rhoT, lp)
        end
    end

    return Dict(
        "theta" => theta_samples,
        "beta" => beta_samples,
        "sigma2" => sigma2_samples,
        "tau2" => tau2_samples,
        "rhoS" => rhoS_samples,
        "rhoT" => rhoT_samples,
        "lp" => lp_list
    )
end

function log_likelihood_normal(y, W, feature, theta_init, beta, sigma2, tau2, rhoS, rhoT)
    T, K, dim = size(feature)
    y_flatten = y'[:]
    feature_reshaped = reshape(permutedims(feature, [2, 1, 3]), T * K, dim)

    Q = Symmetric(rhoS * (Diagonal(sum(W, dims=2)[:]) - W) + (1-rhoS) * I(K))
    invQ = inv(Q)
    D = Symmetric(create_D(T, rhoT))
    invD = inv(D)

    lp = 0

    cov_y = tau2 * kron(invD, invQ) + sigma2 * I(T * K)
    mu_y = feature_reshaped * beta + repeat(theta_init, T)
    lp -= 0.5 * logdet(cov_y)
    lp -= 0.5 * (y_flatten - mu_y)' * (cov_y \ (y_flatten - mu_y))
    return lp
end