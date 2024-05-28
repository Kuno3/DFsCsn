using LinearAlgebra, Distributions, Random, ProgressMeter, MvNormalCDF
import ..DFsCsn

struct DFsCsnPostSampler
    y::Array{Float64,2}
    W::Array{Float64,2}
    feature::Array{Float64,3}
    nu2_beta::Float64
    a_sigma2::Float64
    b_sigma2::Float64
    a_tau2::Float64
    b_tau2::Float64
    nu2_l::Float64
    T::Int
    K::Int
    dim::Int
    y_flatten::Array{Float64,1}
    feature_reshaped::Array{Float64,2}
    eigen_values_raw::Vector{Float64}
    eigen_vector::Matrix{Float64}

    function DFsCsnPostSampler(
        y,
        W,
        feature;
        nu2_beta=100.0,
        a_sigma2=1.0,
        b_sigma2=0.01,
        a_tau2=1.0,
        b_tau2=0.01,
        nu2_l=9.0,
        seed=1234
    )
        Random.seed!(seed)
        T = size(y, 1)
        K = size(W, 1)
        dim = size(feature, 3)
        y_flatten = y'[:]
        feature_reshaped = reshape(permutedims(feature, [2, 1, 3]), T * K, dim)
        eigen_decomp = eigen(Diagonal(sum(W, dims=2)[:]) - W)
        eigen_values_raw = eigen_decomp.values
        eigen_vector = eigen_decomp.vectors

        new(
            y,
            W,
            feature,
            nu2_beta,
            a_sigma2,
            b_sigma2,
            a_tau2,
            b_tau2,
            nu2_l,
            T,
            K,
            dim,
            y_flatten,
            feature_reshaped,
            eigen_values_raw,
            eigen_vector
        )
    end
end

function log_acceptance_ratio_theta(sampler::DFsCsnPostSampler, theta, alpha, tau2, rhoS, rhoT, l)
    try
        Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
        invQ_half = calculate_sqrt(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
        c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))

        log_acceptance_ratio = sampler.T * (- sampler.K * (log(c) + 0.5 * log(tau2) - 0.5 * log(1+l^2)) + 0.5 * logdet(Q))
        conditional_mu_theta = c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[1, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
        log_acceptance_ratio += - (1+l^2) / (2 * tau2 * c^2) * dot(theta[1, :]-conditional_mu_theta, Q * (theta[1, :]-conditional_mu_theta))
        
        for t in 2:sampler.T
            conditional_mu_theta = rhoT * theta[t-1, :] + c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[t, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
            log_acceptance_ratio += - (1+l^2) / (2 * tau2 * c^2) * dot(theta[t, :]-conditional_mu_theta, Q * (theta[t, :]-conditional_mu_theta))
        end
        return log_acceptance_ratio
    catch
        return -Inf
    end
end

function d_log_acceptance_ratio_theta_dtau2_and_dl(sampler::DFsCsnPostSampler, theta, alpha, tau2, rhoS, rhoT, l)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    invQ_half = calculate_sqrt(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))
    
    dlp = hcat(
        - sampler.T * sampler.K / (2 * tau2),
        sampler.T * sampler.K * (pi-2) * l / ((pi - 2) * l^2 + pi)
    )
    conditional_mu_theta = c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[1, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
    dlp += (theta[1, :] - conditional_mu_theta)' * Q * hcat(
        ((pi-2) * l^2 + pi) / (2*pi*tau2^2) * theta[1, :],
        - ((pi - 2) * l / (pi*tau2) * theta[1, :] - conditional_mu_theta / (c^2*tau2*l) + sqrt(2/pi) * l^2 / (c*sqrt(tau2*(1+l^2))) * invQ_half * ones(sampler.K))
    )
    
    for t in 2:sampler.T
        conditional_mu_theta = c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[t, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
        dlp += (theta[t, :] - rhoT * theta[t-1, :] - conditional_mu_theta)' * Q * hcat(
            ((pi-2) * l^2 + pi) / (2*pi*tau2^2) * (theta[t, :] - rhoT * theta[t-1, :]),
            - ((pi - 2) * l / (pi*tau2) * (theta[t, :] - rhoT * theta[t-1, :]) - conditional_mu_theta / (c^2*tau2*l) + sqrt(2/pi) * l^2 / (c*sqrt(tau2*(1+l^2))) * invQ_half * ones(sampler.K))
        )
    end
    return dlp
end

function log_acceptance_ratio_alpha(sampler::DFsCsnPostSampler, alpha, l)
    return sum(logpdf.(Normal.(zeros(sampler.T*sampler.K), sqrt(1+l^2)), alpha[:]))
end

function d_log_acceptance_ratio_alpha_dl(sampler::DFsCsnPostSampler, alpha, l)
    return l * sum(alpha[:].^2) / (1+l^2)^2 - sampler.T * sampler.K * l / (1+l^2)
end

function alpha_sampler!(alpha, sampler::DFsCsnPostSampler, theta, tau2, rhoS, rhoT, l)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    invQ_half = calculate_sqrt(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    inv_invQ_half = invQ_half * Q
    c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))
    conditional_mu_alpha = l / (c * sqrt(tau2)) * inv_invQ_half * theta[1, :] + l^2 * sqrt(2 / (pi*(1+l^2))) * ones(sampler.K)
    alpha[1, :] = conditional_mu_alpha + rand.(Truncated.(Normal(0, 1), -conditional_mu_alpha, Inf))
    for t in 2:sampler.T
        conditional_mu_alpha = l / (c * sqrt(tau2)) * inv_invQ_half * (theta[t, :]-rhoT*theta[t-1, :]) + l^2 * sqrt(2 / (pi*(1+l^2))) * ones(sampler.K)
        alpha[t, :] = conditional_mu_alpha + rand.(Truncated.(Normal(0, 1), -conditional_mu_alpha, Inf))
    end
end

function theta_sampler!(theta, sampler::DFsCsnPostSampler, alpha, beta, sigma2, tau2, rhoS, rhoT, l)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    invQ = calculate_inv(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    invQ_half = calculate_sqrt(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))
    
    mean_theta_pred = zeros(sampler.T, sampler.K)
    cov_theta_pred = zeros(sampler.T, sampler.K, sampler.K)
    mean_theta_filt = zeros(sampler.T, sampler.K)
    cov_theta_filt = zeros(sampler.T, sampler.K, sampler.K)
    
    cov_theta_pred[1, :, :] = c^2 * tau2 / (1+l^2) * invQ
    mean_theta_pred[1, :] = c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[1, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
    cov_theta_filt[1, :, :] = inv((1+l^2) / (c^2 * tau2) * Q + I / sigma2)
    mean_theta_filt[1, :] = mean_theta_pred[1, :] + cov_theta_filt[1, :, :] * (sampler.y[1, :] - sampler.feature[1, :, :] * beta - mean_theta_pred[1, :]) / sigma2
    
    for t in 2:sampler.T
        cov_theta_pred[t, :, :] = rhoT^2 * cov_theta_filt[t-1, :, :] + c^2 * tau2 / (1+l^2) * invQ
        mean_theta_pred[t, :] = rhoT * mean_theta_filt[t-1, :] + c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[t, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
        cov_theta_filt[t, :, :] = (cov_theta_pred[t, :, :] + sigma2 * I) \ cov_theta_pred[t, :, :] * sigma2
        mean_theta_filt[t, :] = mean_theta_pred[t, :] + cov_theta_filt[t, :, :] * (sampler.y[t, :] - sampler.feature[t, :, :] * beta - mean_theta_pred[t, :]) / sigma2
    end
    
    theta[sampler.T, :] = rand(MvNormal(mean_theta_filt[sampler.T, :], Symmetric(cov_theta_filt[sampler.T, :, :])))
    for t in sampler.T-1:-1:1
        gain = rhoT * cov_theta_filt[t, :, :] / cov_theta_pred[t+1, :, :]
        mean_theta_ffbs = mean_theta_filt[t, :] + gain * (theta[t+1, :] - mean_theta_pred[t+1, :])
        cov_theta_ffbs = cov_theta_filt[t, :, :] - gain * cov_theta_pred[t+1, :, :] * gain'
        theta[t, :] = rand(MvNormal(mean_theta_ffbs, Symmetric(cov_theta_ffbs)))
    end
end

function beta_sampler(sampler::DFsCsnPostSampler, theta, sigma2)
    theta_flatten = theta'[:]
    cov_beta = inv(sampler.feature_reshaped' * sampler.feature_reshaped / sigma2 + I / sampler.nu2_beta)
    mu_beta = cov_beta * (sampler.feature_reshaped' * (sampler.y_flatten - theta_flatten)) / sigma2
    beta = rand(MvNormal(mu_beta, Symmetric(cov_beta)))
    return beta
end

function sigma2_sampler(sampler::DFsCsnPostSampler, theta, beta)
    theta_flatten = theta'[:]
    an = sampler.a_sigma2 + sampler.K * sampler.T / 2
    bn = sampler.b_sigma2 + sum((sampler.y_flatten - theta_flatten - sampler.feature_reshaped * beta).^2) / 2
    return rand(InverseGamma(an, bn))
end

function rhoS_sampler(sampler::DFsCsnPostSampler, theta, alpha, tau2, rhoS, rhoT, l, rhoS_prop_scale, lp_old::Union{Float64, Nothing}=nothing)
    dst_curr = Beta(rhoS_prop_scale * rhoS + 1e-4, rhoS_prop_scale * (1 - rhoS) + 1e-4)
    rhoS_prop = rand(dst_curr)
    dst_prop = Beta(rhoS_prop_scale * rhoS_prop + 1e-4, rhoS_prop_scale * (1 - rhoS_prop) + 1e-4)

    lp_new = log_acceptance_ratio_theta(sampler, theta, alpha, tau2, rhoS_prop, rhoT, l)
    lp_old = isnothing(lp_old) ? log_acceptance_ratio_theta(sampler, theta, alpha, tau2, rhoS, rhoT, l) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio -= logpdf(dst_curr, rhoS_prop) - logpdf(dst_prop, rhoS)
    threshold = log(rand())
    return log_accept_ratio > threshold ? (rhoS_prop, lp_new) : (rhoS, lp_old)
end

function rhoT_sampler(sampler::DFsCsnPostSampler, theta, alpha, tau2, rhoS, l)
    Q = Symmetric(rhoS * (Diagonal(sum(sampler.W, dims=2)[:]) - sampler.W) + (1-rhoS) * I(sampler.K))
    invQ = calculate_inv(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    invQ_half = calculate_sqrt(rhoS * sampler.eigen_values_raw .+ (1-rhoS), sampler.eigen_vector)
    c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))

    prec_rhoT = 0 # 1 / sampler.nu2_rhoT
    mu_rhoT = 0
    for t in 2:sampler.T
        mu_theta = c * l * sqrt(tau2) / (1+l^2) * invQ_half * (alpha[t, :] - sqrt(2*(1+l^2)/pi) * ones(sampler.K))
        prec_rhoT += (1+l^2) * theta[t-1, :]' * Q * theta[t-1, :] / (c^2 * tau2)
        mu_rhoT += (1+l^2) * (theta[t, :] - mu_theta)' * Q * theta[t-1, :] / (c^2 * tau2)
    end
    var_rhoT = 1 / prec_rhoT
    mu_rhoT /= prec_rhoT
    return rand(Truncated(Normal(mu_rhoT, sqrt(var_rhoT)), 0, 1))
end

function tau2_and_l_sampler(sampler::DFsCsnPostSampler, theta, alpha, tau2, rhoS, rhoT, l, step_size=0.1, num_step=1, lp_old::Union{Float64, Nothing}=nothing)
    dst_prior_tau2 = InverseGamma(sampler.a_tau2, sampler.b_tau2)
    dst_prior_l = Normal(0.0, sqrt(sampler.nu2_l))

    r_init = randn(2)'
    r = r_init - step_size / 2 * (hcat(tau2, 1) .* (- d_log_acceptance_ratio_theta_dtau2_and_dl(sampler, theta, alpha, tau2, rhoS, rhoT, l)))
    r += - step_size / 2 * hcat(
        0,
        - d_log_acceptance_ratio_alpha_dl(sampler, alpha, l)
    )
    r += - step_size / 2 * (hcat(tau2, 1) .* hcat(
        (sampler.a_tau2 + 1) / tau2 - sampler.b_tau2 / tau2^2,
        l / sampler.nu2_l
    ))
    log_tau2_prop, l_prop = hcat(log(tau2), l) + step_size * r
    tau2_prop = exp(log_tau2_prop)
    for _ in 1:num_step-1
        r += - step_size * (hcat(tau2_prop, 1) .* (- d_log_acceptance_ratio_theta_dtau2_and_dl(sampler, theta, alpha, tau2_prop, rhoS, rhoT, l_prop)))
        r += - step_size * hcat(
            0,
            - d_log_acceptance_ratio_alpha_dl(sampler, alpha, l_prop)
        )
        r += - step_size * (hcat(tau2_prop, 1) .* hcat(
            (sampler.a_tau2 + 1) / tau2_prop - sampler.b_tau2 / tau2_prop^2,
            l_prop / sampler.nu2_l
        ))
        log_tau2_prop, l_prop = hcat(log_tau2_prop, l_prop) + step_size * r
        tau2_prop = exp(log_tau2_prop)
    end
    r += - step_size / 2 * (hcat(tau2_prop, 1) .* (- d_log_acceptance_ratio_theta_dtau2_and_dl(sampler, theta, alpha, tau2_prop, rhoS, rhoT, l_prop)))
    r += - step_size / 2 * hcat(
        0,
        - d_log_acceptance_ratio_alpha_dl(sampler, alpha, l_prop)
    )
    r += - step_size / 2 * (hcat(tau2_prop, 1) .* hcat(
        (sampler.a_tau2 + 1) / tau2_prop - sampler.b_tau2 / tau2_prop^2,
        l_prop / sampler.nu2_l
    ))

    lp_new = log_acceptance_ratio_theta(sampler, theta, alpha, tau2_prop, rhoS, rhoT, l_prop)
    lp_old = isnothing(lp_old) ? log_acceptance_ratio_theta(sampler, theta, alpha, tau2, rhoS, rhoT, l) : lp_old
    log_accept_ratio = lp_new - lp_old
    log_accept_ratio += log_acceptance_ratio_alpha(sampler, alpha, l_prop)
    log_accept_ratio -= log_acceptance_ratio_alpha(sampler, alpha, l)
    log_accept_ratio += logpdf(dst_prior_tau2, tau2_prop)
    log_accept_ratio -= logpdf(dst_prior_tau2, tau2)
    log_accept_ratio += logpdf(dst_prior_l, l_prop)
    log_accept_ratio -= logpdf(dst_prior_l, l)
    log_accept_ratio += - sum(r.^2) / 2
    log_accept_ratio -= - sum(r_init.^2) / 2
    threshold = log(rand())
    return log_accept_ratio > threshold ? (tau2_prop, l_prop, lp_new) : (tau2, l, lp_old)
end

function log_posterior(sampler::DFsCsnPostSampler, theta, alpha, beta, sigma2, tau2, rhoS, rhoT, l, lp_old::Union{Float64, Nothing}=nothing)
    theta_flatten = theta'[:]
    lp = isnothing(lp_old) ? log_acceptance_ratio_theta(sampler, theta, alpha, tau2, rhoS, rhoT, l) : lp_old
    lp += log_acceptance_ratio_alpha(sampler, alpha, l)
    lp += sum(logpdf.(Normal.(sampler.feature_reshaped * beta + theta_flatten, sqrt(sigma2)), sampler.y_flatten))
    lp += sum(logpdf.(Normal(0.0, sqrt(sampler.nu2_beta)), beta))
    lp += logpdf(InverseGamma(sampler.a_sigma2, sampler.b_sigma2), sigma2)
    lp += logpdf(InverseGamma(sampler.a_tau2, sampler.b_tau2), tau2)
    lp += logpdf(Normal(0, sqrt(sampler.nu2_rhoT)), rhoT)
    lp += logpdf(Normal(0, sqrt(sampler.nu2_l)), l)
    return lp
end

function sampling(
    sampler::DFsCsnPostSampler,
    num_sample::Int;
    burn_in::Int=0,
    thinning::Int=1,
    step_size::Float64=0.1,
    num_step::Int=1,
    rhoS_prop_scale::Float64=0.1,
    theta_init::Union{Matrix{Float64}, Nothing}=nothing,
    alpha_init::Union{Matrix{Float64}, Nothing}=nothing,
    beta_init::Union{Vector{Float64}, Nothing}=nothing,
    sigma2_init::Union{Float64, Nothing}=nothing,
    tau2_init::Union{Float64, Nothing}=nothing,
    rhoS_init::Union{Float64, Nothing}=nothing,
    rhoT_init::Union{Float64, Nothing}=nothing,
    l_init::Union{Float64, Nothing}=nothing
)
    rhoS_prop_scale = (1 - rhoS_prop_scale^2) / rhoS_prop_scale^2
    
    theta = isnothing(theta_init) ? zeros(sampler.T, sampler.K) : theta_init
    alpha = isnothing(alpha_init) ? zeros(sampler.T, sampler.K) : alpha_init
    beta = isnothing(beta_init) ? zeros(sampler.dim) : beta_init
    sigma2 = isnothing(sigma2_init) ? 1.0 : sigma2_init
    tau2 = isnothing(tau2_init) ? 1.0 : tau2_init
    rhoS = isnothing(rhoS_init) ? 0.5 : rhoS_init
    rhoT = isnothing(rhoT_init) ? 0.5 : rhoT_init
    l = isnothing(l_init) ? randn() : l_init
    lp = -Inf

    alpha_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    theta_samples = zeros(Float64, (num_sample, sampler.T, sampler.K))
    beta_samples = zeros(Float64, (num_sample, sampler.dim))
    sigma2_samples = zeros(Float64, num_sample)
    tau2_samples = zeros(Float64, num_sample)
    rhoS_samples = zeros(Float64, num_sample)
    rhoT_samples = zeros(Float64, num_sample)
    l_samples = zeros(Float64, num_sample)
    lp_list = zeros(Float64, num_sample)

    @showprogress for i in 1:(burn_in + num_sample)
        for _ in 1:thinning
            alpha_sampler!(alpha, sampler, theta, tau2, rhoS, rhoT, l)
            theta_sampler!(theta, sampler, alpha, beta, sigma2, tau2, rhoS, rhoT, l)
            beta = beta_sampler(sampler, theta, sigma2)
            sigma2 = sigma2_sampler(sampler, theta, beta)
            rhoT = rhoT_sampler(sampler, theta, alpha, tau2, rhoS, l)
            rhoS, lp = rhoS_sampler(sampler, theta, alpha, tau2, rhoS, rhoT, l, rhoS_prop_scale)
            try
                tau2, l, lp = tau2_and_l_sampler(sampler, theta, alpha, tau2, rhoS, rhoT, l, step_size, num_step, lp)
            catch
                nothing
            end
        end

        if i > burn_in
            alpha_samples[i-burn_in, :, :] = alpha
            theta_samples[i-burn_in, :, :] = theta
            beta_samples[i-burn_in, :] = beta
            sigma2_samples[i-burn_in] = sigma2
            tau2_samples[i-burn_in] = tau2
            rhoS_samples[i-burn_in] = rhoS
            rhoT_samples[i-burn_in] = rhoT
            l_samples[i-burn_in] = l
            lp_list[i-burn_in] = log_posterior(sampler, theta, alpha, beta, sigma2, tau2, rhoS, rhoT, l, lp)
        end
    end

    return Dict(
        "alpha" => alpha_samples,
        "theta" => theta_samples,
        "beta" => beta_samples,
        "sigma2" => sigma2_samples,
        "tau2" => tau2_samples,
        "rhoS" => rhoS_samples,
        "rhoT" => rhoT_samples,
        "l" => l_samples,
        "lp" => lp_list
    )
end

function log_likelihood_sun(y, W, feature, theta_init, beta, sigma2, tau2, rhoS, rhoT, l)
    T, K, dim = size(feature)
    y_flatten = y'[:]
    feature_reshaped = reshape(permutedims(feature, [2, 1, 3]), T * K, dim)

    Q = Symmetric(rhoS * (Diagonal(sum(W, dims=2)[:]) - W) + (1-rhoS) * I(K))
    invQ = inv(Q)
    invQ_half = sqrt(invQ)
    D = Symmetric(create_D(T, rhoT))
    invD = inv(D)
    invD_half = cholesky(invD).L
    c = sqrt(pi * (1 + l^2) / ((pi - 2) * l^2 + pi))

    lp = 0

    cov_y = c^2 * tau2 * kron(invD, invQ) + sigma2 * I(T * K)
    mu_y = feature_reshaped * beta + (repeat(theta_init, T) - c * l * sqrt(2 * tau2 / (pi * (1 + l^2))) * kron(invD_half, invQ_half) * ones(T * K))
    lp -= 0.5 * logdet(cov_y)
    lp -= 0.5 * (y_flatten - mu_y)' * (cov_y \ (y_flatten - mu_y))

    cov_y_and_alpha = c * l * sqrt(tau2) * (kron(invD_half, invQ_half))'
    conditional_mean_alpha = cov_y_and_alpha * (cov_y \ (y_flatten - mu_y))
    conditional_cov_alpha = (1 + l^2) * I(T * K) - cov_y_and_alpha * (cov_y \ cov_y_and_alpha')

    lp += mvnormcdf(
        conditional_mean_alpha,
        conditional_cov_alpha,
        zeros(T * K),
        Inf * ones(T * K)
    )[1]

    lp += T * K * log(2)
    return lp
end