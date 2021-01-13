function [lnp_th, lnp_th_g, lnp_th_H] = log_prior_eval(prior, theta)

	[lnp_th, lnp_th_g, lnp_th_H] = prior.log_prior_eval(prior.hyper, theta);
end