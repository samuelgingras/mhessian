% set_GaBeN_prior Create structure specifying GaBeN prior for theta_x
%   prior = set_GaBeN_prior(has_mu, Ga_al, Ga_be, Be_al, Be_be, N_mu, N_h)
%   populates the structure "prior" with the hyper-parameters Ga_al, Ga_be,
%   Be_al, Be_be, N_mu, N_h defining a prior for
%      theta = (log omega, tanh^{-1} phi, mu),
%   where sigma2 = omega^{-1}, phi and mu are independent,
%   * sigma2 ~ Ga(Ga_al, Ga_be)
%   * 2*phi-1 ~ Be(Be_al, Be_be)
%   * mu ~ N(N_mu, N_h^{-1}.
%   The structure has a field log_eval which returns the log unnormalized
%   prior density, its gradient vector and its Hessian matrix.
function prior = set_GaBeN_prior(has_mu, Ga_al, Ga_be, Be_al, Be_be, N_mu, N_h)

    prior.Ga_al = Ga_al;
    prior.Ga_be = Ga_be;
    prior.Be_al = Be_al;
    prior.Be_be = Be_be;
    prior.has_mu = has_mu;
    if has_mu
        prior.N_mu = N_mu;
        prior.N_h = N_h;
    end
    prior.log_eval = @GaBeN_log_eval;
end

function [lnp_th, lnp_th_g, lnp_th_H] = GaBeN_log_eval(prior, th)

	[v1, g1, H11] = Ga_sigma2(exp(-th(1)), prior.Ga_al, prior.Ga_be);
	[v2, g2, H22] = Be_phi(tanh(th(2)), prior.Be_al, prior.Be_be);
	if prior.has_mu
		u = th(3) - prior.N_mu;
		g3 = -prior.N_h * u;
		H33 = -prior.N_h;
		lnp_th = v1 + v2 - 0.5 * prior.N_h * u^2;
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
