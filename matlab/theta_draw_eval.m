function [lnq_thSt, varargout] = ...
	theta_draw_eval(prior, theta, q_theta, varargin)

	% See if 
	if( isfield(theta,'x') )
		is_simple = false;
		old_theta = theta;
		theta = theta.x;
	else
		is_simple = true;
	end

	% See if thetaSt (theta star) needs to be drawn (is_draw) or not.
    if nargin == 4 && nargout == 1
		is_draw = false;
		thetaSt = varargin{1};
		if( ~is_simple )
			thetaSt = thetaSt.x;
		end
		thSt = thetaSt.th;
	elseif nargin == 3 && nargout == 2
		is_draw = true;
	else
		error("Incorrect combination of number of inputs and outputs");
    end
    long_th = prior.hyper.has_mu;


	[v_prior, g_prior, H_prior] = log_prior_eval(prior, theta);
	th = theta.th;

	% Construct gradient g and Hessian H
	g_y = q_theta.grad;
	g = q_theta.grad + g_prior;
	H_y = q_theta.Hess + q_theta.Var;
	H = H_y + H_prior;

	% Compute (WJM: new H12_post correction)
	if long_th
		phi = tanh(th(2));
		gth_grad = [1; -2*(1+phi)];
		gth_Hess = [1, -2*(1+phi); -2*(1+phi), 2*(1+phi)*(1+3*phi)];
		gth_opg = gth_grad * gth_grad';
		h_bar = -H_prior(3, 3);
		h_bar2_diff2 = h_bar^2 * (g_prior(3)/h_bar + g(3)/H_y(3, 3));
		lambda_g = -H_y(3, 3);
		coeff_opg = 0.5 * lambda_g^2 / (h_bar + lambda_g)^2;
		coeff_opg = coeff_opg + h_bar2_diff2 * lambda_g^2 / (h_bar + lambda_g)^3;
		term_opg = coeff_opg * gth_opg;
		coeff_Hess = -0.5 * lambda_g / (h_bar + lambda_g);
		coeff_Hess = coeff_Hess - 0.5 * h_bar2_diff2 * lambda_g / (h_bar + lambda_g)^2;
		term_Hess = coeff_Hess * gth_Hess;
		coeff_grad = -0.5 * lambda_g / (h_bar + lambda_g);
		coeff_grad = coeff_grad - 0.5 * h_bar2_diff2 * lambda_g / (h_bar + lambda_g)^2;
		term_grad = coeff_grad * gth_grad;

		g12_prior = g_prior(1:2);
		g12_post = q_theta.grad(1:2);
		g12 = g12_prior + g12_post;
		H12_prior = H_prior(1:2, 1:2);
		H12_post = H_y(1:2, 1:2);
	end
	g12 = g_prior(1:2) + q_theta.grad(1:2);
	
	if long_th
		%H12_post = H12_post - 0.5*(1+g(3)^2/H_y(3,3)) * gth_Hess;
		%g12_post = g12_post - 0.5*(1+g(3)^2/H_y(3,3)) * gth_grad;

		H12_post = H12_post - 0.5*g_y(3)^2/H_y(3,3) * gth_Hess;
		H12_post = H12_post + term_opg + term_Hess;
		g12_post = g12_post - 0.5*g_y(3)^2/H_y(3,3) * gth_grad;
		g12_post = g12_post + term_grad; 
	end

	res = mode_guess(q_theta, theta, prior);

	theta.th(1:2) = res.mode;
	theta.omega = exp(res.mode(1));
	theta.phi = tanh(res.mode(2));
	[v_prior, g_prior, H_prior] = log_prior_eval(prior, theta);
	g12_prior = g_prior(1:2);
	Om12_prior = -H_prior(1:2, 1:2);
	Om12_post = -res.H_mode;
	g12_post = [0; 0];

	R = chol(Om12_prior + Om12_post);

	% Construct guess of variance, streched by lambda, of (theta_1, theta_2).
	% Compute lower Chol factor
	I = eye(2);
	Lambda = [1.2, 0; 0, 1.2];
	Sigma_th = Lambda * (R\(R'\I)) * Lambda';
	L = chol(Sigma_th, 'lower');

	% Specify first order PAC Psi, construct Phi and Sigma for step
	Psi = 0.2 * I;
	Phi = L * Psi / L;
	Sigma_eps = (Sigma_th - L * Psi * Psi' * L');
	R_eps = chol(Sigma_eps);

	% Draw leading 2x1 subvector of thetaSt or evaluate uSt from thetaSt
	thSt12_mean = th(1:2) + (I-Phi) * R\(R'\g12);
	if is_draw
		% Draw thSt (theta*)
		uSt = randn(2, 1);
		thSt = thSt12_mean + R_eps' * uSt;
	else
		% Compute uSt
		uSt = R_eps'\(thSt(1:2) - thSt12_mean);
	end

	% Compute log density (up to normalization constant) at thSt.
	lnq_thSt = -log(det(R_eps)) - 0.5*(uSt'*uSt);

	% Conditional draw/eval of th3St given thSt
	if long_th
		phi = tanh(th(2)); omega = exp(th(1));
		phiSt = tanh(thSt(2)); omegaSt = exp(thSt(1));
		c = omegaSt*(1-phiSt)^2/(omega*(1-phi)^2);
		phi_mid = tanh((th(2)+thSt(2))/2); omega_mid = exp((th(1)+thSt(1))/2);
		c_mid = omega_mid*(1-phi_mid)^2/(omega*(1-phi)^2);

		v = H(1:2, 3) * c_mid;
		h = H(3,3) * c_mid;
		g3 = g(3) + (thSt(1:2) - th(1:2))' * v;
		thSt3_mean = th(3) - g3 / h;
		thSt3_sd = 1.0/sqrt(-H(3,3) * c);

		if is_draw
			uSt3 = randn(1);
			thSt3 = thSt3_mean + uSt3 * thSt3_sd;
			thSt = [thSt; thSt3];
		else
			uSt3 = (thSt(3) - thSt3_mean)/thSt3_sd;
		end
		lnq_thSt = lnq_thSt - log(thSt3_sd) - 0.5*uSt3^2;
	end

	if is_draw
		thetaSt = theta;
		thetaSt.th = thSt;
		thetaSt.omega = exp(thSt(1));
		thetaSt.phi = tanh(thSt(2));
		if long_th
			thetaSt.mu = thSt(3);
		end

		if( ~is_simple )
			tmp.x = thetaSt;
			fields = fieldnames(old_theta);
			for f=1:length(fields)
				if( ~strcmp(fields{f},'x') )
					tmp.(fields{f}) = old_theta.(fields{f});
				end
			end
			thetaSt = tmp;
		end
		varargout{1} = thetaSt;
	end
end
