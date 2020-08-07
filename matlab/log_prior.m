function [v, g, H] = log_prior(prior, theta)	
	switch prior.type
		case 'GaBeN'
			[v1, g1, H11] = Ga_sigma2(exp(-theta(1)), prior.Ga_al, prior.Ga_be);
			[v2, g2, H22] = Be_phi(tanh(theta(2)), prior.Be_al, prior.Be_be);
			u = theta(3) - prior.N_mu;
			g3 = -prior.N_h * u;
			H33 = -prior.N_h;
			v = v1 + v2 - 0.5 * prior.N_h * u^2;
			g = [g1; g2; g3];
			H = diag([H11; H22; H33]);
		case 'LNBeN'
			[v1, g1, H11] = LN_sigma(exp(-0.5*theta(1)), prior.LN_mu, prior.LN_h);
			[v2, g2, H22] = Be_phi(tanh(theta(2)), prior.Be_al, prior.Be_be);
			u = theta(3) - prior.N_mu;
			g3 = -prior.N_h * u;
			H33 = -prior.N_h;
			v = v1 + v2 - 0.5 * prior.N_h * u^2;
			g = [g1; g2; g3];
			H = diag([H11; H22; H33]);
		case 'MVN'
			u = prior.R * (theta - prior.theta);
			v = -0.5*(u'*u);
			g = -(prior.R'*u);
			H = prior.H;
		otherwise
			error('Unavailable prior distribution')
	end
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

function [lnf, dlnf_dth, d2lnf_dth2] = LN_sigma(sigma, mu, h)

	% Value of log Log-Normal(mu, h^-1) density on (0,infty), and two derivatives, at sigma
	lnf_sigma = -log(sigma) - 0.5*h*(log(sigma) - mu)^2;
	dlnf_dsigma = sigma^(-1) * (1 + h*(log(sigma) - mu));
	d2lnf_dsigma = sigma^(-2) * (1 + h*(log(sigma) - mu - 1));

	lnf = lnf_sigma + log(sigma);
	dlnf_dth = -0.5 * (dlnf_dsigma*sigma + 1);
	d2lnf_dth2 = 0.25*sigma * (d2lnf_dsigma*sigma + dlnf_dsigma);  
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
