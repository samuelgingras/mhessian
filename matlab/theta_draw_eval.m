function [lnq_thSt, varargout] = ...
	theta_draw_eval(model, prior, y, mode_sh, theta, hmout, varargin)

	% Simulation parameters
	global hmctrl;
	if exist('hmctrl', 'var')
		mu_mag = hmctrl.mu_mag;
		nu = hmctrl.nu;
	else
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
	th2_unc_mean = hmout.sh.params.th_hat;
	th2_con_mean = th2_unc_mean;

	% Bivariate normal code
	H = hmout.sh.params.H2_3rd;
	[V, D] = eig(H);
	prec_eps = -D;  % Innovation precision eigenvalues	
	prec_eps_root = sqrt(prec_eps);
	if is_draw
		uSt = trnd([nu;nu]); %randn(2, 1);
		delta_th2 = V*inv(prec_eps_root)*V' * uSt;
		L_plus = directional_3rd(n, H, th2_con_mean, delta_th2);
		L_plus = sign(L_plus) * min(abs(L_plus), 3);
		if (rand < exp(L_plus) / (exp(L_plus) + exp(-L_plus)))
			thSt2 = th2_con_mean + delta_th2;
			lnq_thSt = L_plus - log(cosh(L_plus));
		else
			thSt2 = th2_con_mean - delta_th2;
			lnq_thSt = -L_plus - log(cosh(L_plus));
		end
	else
		thSt = thetaSt.th;
		thSt2 = thSt(1:2);
		delta_th2 = thSt2 - th2_con_mean;
		uSt = V*prec_eps_root*V' * delta_th2;
		L_plus = directional_3rd(n, H, th2_con_mean, delta_th2);
		L_plus = sign(L_plus) * min(abs(L_plus), 3);
		lnq_thSt = L_plus - log(cosh(L_plus));
	end
	lnq_thSt = lnq_thSt + 0.5 * (log(det(prec_eps)) ...
		- (nu-1) * log( (nu+uSt(1)^2) * (nu+uSt(2)^2) ));

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
		V33 = hmout.sh.V33;
		V33_th = hmout.sh.V33_th;
		V33_th_th = hmout.sh.V33_th_th;
		log_V33 = log(V33);
		log_V33_th = (1/V33) * V33_th;
		log_V33_th_th = -(1/V33^2) * V33_th * V33_th' + (1/V33) * V33_th_th;
		log_V33_pred = log_V33 + delta' * log_V33_th + 0.5 * delta' * log_V33_th_th * delta;
		
		muSt_ml_prec = om_q_iotaSt - exp(log_V33_pred);
		muSt_ml_prec = max(muSt_ml_prec, 0.5 * om_q_iotaSt);

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
