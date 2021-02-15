function [lnp_th, lnp_th_g, lnp_th_H] = MVN_log_prior_eval(hyper, theta)

	u = hyper.R * (theta.th - hyper.mean);
	lnp_th = -0.5*(u'*u);
	lnp_th_g = -(hyper.R'*u);
	lnp_th_H = hyper.H;
end
