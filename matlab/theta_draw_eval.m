function [lnq_thSt, varargout] = ...
	theta_draw_eval(model, prior, y, mode_sh, theta, hmout, varargin)

	% Simulation parameters
	global hmctrl;
	if exist('hmctrl', 'var')
		th2_rho = hmctrl.th2_rho;
		th2_mag = hmctrl.th2_mag;
		mu_mag = hmctrl.mu_mag;
		nu = hmctrl.nu;
	else
		th2_rho = [0.1; 0.1];
		th2_mag = [1.2; 1.2];
		mu_mag = 1.05;
		nu = 10;
	end

	% See if proposal thetaSt is to be drawn (is_draw) or not.
	if nargin == 7 && nargout == 1
		is_draw = false;
		thetaSt = varargin{1};
	elseif nargin == 6 && nargout == 2
		is_draw = true;
	else
		error("Incorrect combination of number of inputs and outputs");
	end

	% See if there is a mu in theta or not
	has_mu = prior.has_mu;
	n = theta.N;

	if ~isfield(hmout, 'sh')
		hmout.sh = compute_proposal_params(model, prior, y, mode_sh, theta, hmout);
	end
	H = hmout.sh.params.H2;
	g = hmout.sh.params.g2;
	th2_unc_mean = hmout.sh.params.mean2;

	% Compute eigenvalues for epsilon precision
	[V, D] = eig(H);
	D_prec_eps = -diag(1./(th2_mag .* (1-th2_rho.^2))) * D;  % Innovation precision eigenvalues
	D_prec_eps_root = sqrt(D_prec_eps);                 % Square roots of these

	% Unconditional mean and conditional mean for AR draw
	th2_con_mean = th2_rho .* theta.th(1:2) + (1 - th2_rho) .* th2_unc_mean;

	% Bivariate normal code
	% {
	H = hmout.sh.params.H2_3rd;
	[V, D] = eig(H);
	D_prec_eps = -diag(1./(th2_mag .* (1-th2_rho.^2))) * D;  % Innovation precision eigenvalues	
	D_prec_eps_root = sqrt(D_prec_eps);
	if min(abs(D_prec_eps_root(1,1)), abs(D_prec_eps_root(2,2))) < 0.01
		display(D_prec_eps)
		display(hmout.sh.params.H2)
		display(D)
		display(H)
	end
	if is_draw
		uSt = trnd([nu;nu]); %randn(2, 1);
		delta_th2 = V*inv(D_prec_eps_root)*V' * uSt;
		[L_plus_res, L] = directional_3rd(n, H, th2_con_mean, delta_th2);
		L_plus_res = sign(L_plus_res) * min(abs(L_plus_res), 3);
		if (rand < exp(L_plus_res) / (exp(L_plus_res) + exp(-L_plus_res)))
			thSt2 = th2_con_mean + delta_th2;
			lnq_thSt = L_plus_res - log(cosh(L_plus_res));
		else
			thSt2 = th2_con_mean - delta_th2;
			lnq_thSt = -L_plus_res - log(cosh(L_plus_res));
		end
	else
		thSt = thetaSt.th;
		thSt2 = thSt(1:2);
		delta_th2 = thSt2 - th2_con_mean;
		uSt = V*D_prec_eps_root*V' * delta_th2;
		[L_plus_res, L] = directional_3rd(n, H, th2_con_mean, delta_th2);
		L_plus_res = sign(L_plus_res) * min(abs(L_plus_res), 3);
		lnq_thSt = L_plus_res - log(cosh(L_plus_res));
	end
	%lnq_thSt = lnq_thSt + 0.5 * (log(det(D_prec_eps)) - uSt'*uSt);
	lnq_thSt = lnq_thSt + 0.5 * (log(det(D_prec_eps)) ...
		- (nu-1) * log( (nu+uSt(1)^2) * (nu+uSt(2)^2) ));
	% }

	% Double gamma code
	%{
	is_chol = false;
	if is_draw
		[lnq_thSt, thSt2] = double_gamma(-V*D_prec_eps*V', is_chol);
		thSt2 = thSt2 + th2_con_mean;
	else
		thSt = thetaSt.th;
		thSt2 = thSt(1:2);
		lnq_thSt = double_gamma(-V*D_prec_eps*V', is_chol, thSt2 - th2_con_mean);
	end
	%}

	% Conditional draw/eval of th3St given thSt
	if has_mu
		% Conditional precision mu|y,omega,phi
		phiSt = tanh(thSt2(2));
		omegaSt = exp(thSt2(1));
		n = length(y);
		om_q_iotaSt = omegaSt * (2*(1-phiSt) + (n-2)*(1-phiSt)^2);

		% Conditional precision of mu at new values of omega, phi
		muSt_ml_prec = om_q_iotaSt - hmout.sh.V33;
		delta = thSt2 - hmout.sh.th2;
		muSt_ml_prec = muSt_ml_prec - delta' * hmout.sh.V33_th;
		muSt_ml_prec = muSt_ml_prec - 0.5 * delta' * hmout.sh.V33_th_th * delta;
		muSt_ml_prec = max(muSt_ml_prec, 0.1 * om_q_iotaSt);

		% Values of mu maximizing likelihood for (omega, phi) and
		% proposal (omegaSt, phiSt)
		mu = theta.th(3);
		v = hmout.q_theta.Var(3,1:2);
		mu_mle = mu - hmout.sh.params.like_g3 / hmout.sh.params.like_H33;
		muSt_mle = mu_mle - v * (thSt2 - theta.th(1:2)) / hmout.sh.params.like_H33;

		% Compute prior mean and precision at (omega, phi)
		% Could recompute but they are usually constant and usually 
		% not very important.
		mu_prior_prec = -hmout.sh.params.prior_H33;
		mu_prior_mode = mu + hmout.sh.params.prior_g3 / mu_prior_prec;

		% posterior values
		mu_post_prec = muSt_ml_prec + mu_prior_prec;
		thSt3_mean = (muSt_ml_prec*muSt_mle + mu_prior_prec*mu_prior_mode)/mu_post_prec;
		thSt3_sd = sqrt(mu_mag/mu_post_prec);

		if is_draw
			uSt3 = randn(1);
			thSt3 = thSt3_mean + uSt3 * thSt3_sd;
			thSt = [thSt2; thSt3];
		else
			uSt3 = (thSt(3) - thSt3_mean)/thSt3_sd;
		end
		lnq_thSt = lnq_thSt - log(thSt3_sd) - 0.5*uSt3^2;
	end

	if is_draw
		thetaSt = fill_theta_from_th(theta, thSt, has_mu);
		varargout{1} = thetaSt;
	end
end
