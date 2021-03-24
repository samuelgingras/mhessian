function [lnp_th, lnp_th_g, lnp_th_H] = GaBeN_log_prior_eval(hyper, theta)

	[v1, g1, H11] = Ga_sigma2(exp(-theta.th(1)), hyper.Ga_al, hyper.Ga_be);
	[v2, g2, H22] = Be_phi(tanh(theta.th(2)), hyper.Be_al, hyper.Be_be);
	if hyper.has_mu
		u = theta.th(3) - hyper.N_mu;
		g3 = -hyper.N_h * u;
		H33 = -hyper.N_h;
		lnp_th = v1 + v2 - 0.5 * hyper.N_h * u^2;
		lnp_th_g = [g1; g2; g3];
		lnp_th_H = diag([H11; H22; H33]);
	else
		lnp_th = v1 + v2;
		lnp_th_g = [g1; g2];
		lnp_th_H = diag([H11; H22]);
	end
end

function [lnf, dlnf_dth, d2lnf_dth2] = Ga_sigma2(sigma2, alpha, beta)

	% Value of unnormalized gamma(alpha, beta) log density on (0,infty),
	% and two derivatives, with respect to theta = -log sigma2,
	% evaluated at sigma2.
	lnf = alpha * log(sigma2) - beta * sigma2;
	dlnf_dth = -alpha + beta*sigma2;
	d2lnf_dth2 = -beta*sigma2;
end

function [lnf, dlnf_dth, d2lnf_dth2] = Be_phi(phi, alpha, beta)

	% Value of unnormalized beta(alpha, beta) log density on (-1,1),
	% and two derivatives, with respect to theta = atanh(phi),
	% evaluated at phi.
	lnf = alpha * log(1+phi) + beta * log(1-phi);
	dlnf_dth = (alpha - beta) - phi * (alpha + beta);
	d2lnf_dth2 = -(alpha + beta) * (1-phi^2);
end
