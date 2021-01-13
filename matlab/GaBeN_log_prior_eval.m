function [lnp_th, lnp_th_g, lnp_th_H] = GaBeN_log_prior_eval(hyper, theta)

	[v1, g1, H11] = Ga_sigma2(exp(-theta.th(1)), hyper.Ga_al, hyper.Ga_be);
	[v2, g2, H22] = Be_phi(tanh(theta(2)), hyper.Be_al, hyper.Be_be);
	u = theta.th(3) - hyper.N_mu;
	g3 = -hyper.N_h * u;
	H33 = -hyper.N_h;
	lnp_th = v1 + v2 - 0.5 * hyper.N_h * u^2;
	lnp_th_g = [g1; g2; g3];
	lnp_th_H = diag([H11; H22; H33]);
end

function [lnf, dlnf_dth, d2lnf_dth2] = Ga_sigma2(sigma2, alpha, beta)

	% Value of log gamma(alpha, beta) density on (0,infty), and two derivatives, at sigma2
	lnf_sigma2 = (alpha-1)*log(sigma2) - beta*sigma2;
	dlnf_dsigma2 = (alpha-1)./sigma2 - beta;
	d2lnf_dsigma2 = -(alpha-1)./(sigma2.^2);
	
	lnf = lnf_sigma2 + log(sigma2);
	dlnf_dth = -dlnf_dsigma2 .* sigma2 - 1;
	d2lnf_dth2 = d2lnf_dsigma2 .* sigma2.^2 + dlnf_dsigma2 .* sigma2;
end

function [lnf, dlnf_dth, d2lnf_dth2] = Be_phi(phi, alpha, beta)

	% Value of log beta(alpha, beta) density on (-1,1), and two derivatives, at phi
	lnf_phi = (alpha-1)*log(1+phi) + (beta-1)*log(1-phi);
	dlnf_dphi = (alpha-1)./(1+phi) - (beta-1)./(1-phi);
	d2lnf_dphi2 = -(alpha-1)./(1+phi).^2 - (beta-1)./(1-phi).^2;
	
	lnf = lnf_phi + log(1-phi.^2);
	dlnf_dth = dlnf_dphi .* (1-phi.^2) - 2*phi;
	d2lnf_dth2 = d2lnf_dphi2 .* (1-phi.^2).^2 - dlnf_dphi .* (2*phi).*(1-phi.^2) - 2*(1-phi.^2);
end
