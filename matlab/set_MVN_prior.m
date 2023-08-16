% set_MVN_prior Create structure specifying MVN prior for theta_x
%   prior = set_MVN_prior(has_mu, theta_mean, theta_variance)
%   populates the structure "prior" with the hyper-parameters theta_mean
%   and theta_variance defining a prior for
%      theta = (log omega, tanh^{-1} phi, mu),
%   where theta ~ N(theta_mean, theta_variance).
%   The structure has a field log_eval which returns the log unnormalized
%   prior density, its gradient vector and its Hessian matrix.
function prior = set_MVN_prior(has_mu, theta_mean, theta_variance)

    prior.has_mu = has_mu;
    prior.theta_mean = theta_mean;
    prior.theta_variance = theta_variance;
    prior.H = -inv(theta_variance);
    prior.R = chol(-prior.H);
    prior.log_eval = @MVN_log_eval;
end

function [lnp_th, lnp_th_g, lnp_th_H] = MVN_log_eval(prior, theta)

	u = prior.R * (theta - prior.theta_mean);
	lnp_th = -0.5*(u'*u);
	lnp_th_g = -(prior.R'*u);
	lnp_th_H = prior.H;
end
