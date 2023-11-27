% set_LNBeN_prior Create structure specifying LNBeN prior for theta_x
%   prior = set_LNBeN_prior(has_mu, LN_mu, LN_h, Be_al, Be_be, N_mu, N_h)
%   populates the structure "prior" with the hyper-parameters LN_mu, LN_h,
%   Be_al, Be_be, N_mu, N_h defining a prior for
%      theta = (log omega, tanh^{-1} phi, mu),
%   where sigma = omega^{-1/2}, phi and mu are independent,
%   * sigma ~ LN(LN_mu, LN_h^{-1})
%   * 2*phi-1 ~ Be(Be_al, Be_be)
%   * mu ~ N(N_mu, N_h^{-1}.
%   The structure has a field log_eval which returns the log unnormalized
%   prior density, its gradient vector and its Hessian matrix.
function prior = set_LNBeN_prior(has_mu, LN_mu, LN_h, Be_al, Be_be, N_mu, N_h)

    prior.LN_mu = LN_mu;
    prior.LN_h = LN_h;
    prior.Be_al = Be_al;
    prior.Be_be = Be_be;
    prior.has_mu = has_mu;
    if has_mu
        prior.N_mu = N_mu;
        prior.N_h = N_h;
    end
    prior.log_eval = @LNBeN_log_eval;
    prior.mean = @LNBeN_mean;
end

function [lnp_th, lnp_th_g, lnp_th_H] = LNBeN_log_eval(prior, th)
	u = th(1) + 2*prior.LN_mu;
	H11 = -0.25 * prior.LN_h;
	v1 = 0.5 * H11 * u^2;
	g1 = H11 * u;
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

function [lnf, dlnf_dth, d2lnf_dth2] = Be_phi(phi, alpha, beta)

	% Value of log beta(alpha, beta) density on (-1,1), and two derivatives, at phi
	lnf_phi = (alpha-1)*log(1+phi) + (beta-1)*log(1-phi);
	dlnf_dphi = (alpha-1)./(1+phi) - (beta-1)./(1-phi);
	d2lnf_dphi2 = -(alpha-1)./(1+phi).^2 - (beta-1)./(1-phi).^2;
	
	lnf = lnf_phi + log(1-phi.^2);
	dlnf_dth = dlnf_dphi .* (1-phi.^2) - 2*phi;
	d2lnf_dth2 = d2lnf_dphi2 .* (1-phi.^2).^2 - dlnf_dphi .* (2*phi).*(1-phi.^2) - 2*(1-phi.^2);
end

function theta = LNBeN_mean(prior)

	theta.phi = (prior.Be_al - prior.Be_be) / (prior.Be_al + prior.Be_be);
    theta.omega = exp(-2*prior.LN_mu + 2/prior.LN_h);
	if prior.has_mu
		theta.mu = prior.N_mu;
	    theta.th = [log(theta.omega); atanh(theta.phi); theta.mu];
	else
		theta.th = [log(theta.omega); atanh(theta.phi)];
	end
end

