function [lnq_thSt, varargout] = ...
	theta_draw_eval(prior, theta, q_theta, varargin)

	% See if thetaSt (theta star) needs to be drawn (is_draw) or not.
	if nargin == 4 && nargout == 1
		is_draw = false;
		thetaSt = varargin{1};
		thSt = thetaSt.th;
	elseif nargin == 3 && nargout == 2
		is_draw = true;
	else
		error("Incorrect combination of number of inputs and outputs");
    end
    long_th = true;

	[v_prior, g_prior, H_prior] = log_prior_eval(prior, theta);
	th = theta.th;

	% Construct gradient g and Hessian H
	n_th = length(q_theta.grad);
	g_y = q_theta.grad;
	g = q_theta.grad + g_prior(1:n_th);
	H_y = q_theta.Hess + q_theta.Var;
	H = H_y + H_prior(1:n_th, 1:n_th);
	[neg_H11, neg_H22, neg_Det, neg_def] = get_flags(H_y);

	% Compute (WJM: new H12_post correction)
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
	if long_th
		%H12_post = H12_post - 0.5*(1+g(3)^2/H_y(3,3)) * gth_Hess;
		%g12_post = g12_post - 0.5*(1+g(3)^2/H_y(3,3)) * gth_grad;

		H12_post = H12_post - 0.5*g_y(3)^2/H_y(3,3) * gth_Hess;
		H12_post = H12_post + term_opg + term_Hess;
		g12_post = g12_post - 0.5*g_y(3)^2/H_y(3,3) * gth_grad;
		g12_post = g12_post + term_grad; 
	end

	c_max = chi2inv(0.95, 2);
	c_turn = chi2inv(0.75, 2);

	res = mode_guess(q_theta, theta, prior);

	theta.th(1:2) = res.mode;
	theta.omega = exp(res.mode(1));
	theta.phi = tanh(res.mode(2));
	[v_prior, g_prior, H_prior] = log_prior_eval(prior, theta);
	g12_prior = g_prior(1:2);
	Om12_prior = -H_prior(1:2, 1:2);
	Om12_post = -res.H_mode;
	g12_post = [0; 0];

	%%%Om12_prior = robust_Om(H12_prior, g12_prior, 0.5, c_turn, c_max);
	%%%Om12_post = robust_Om(H12_post, g12_post, 0.5, c_turn, c_max);
	%Om12_prior = robust_Om_rotate(H12_prior, g12_prior, 0.80, 0.95, 0.25);
	%Om12_post = robust_Om_rotate(H12_post, g12_post, 0.80, 0.95, 0.25);
	R = chol(Om12_prior + Om12_post);

	% Construct guess of variance, streched by lambda, of (theta_1, theta_2).
	% Compute lower Chol factor
	I = eye(2);
	Lambda = [1.2, 0; 0, 1.2];
	Sigma_th = Lambda * (R\(R'\I)) * Lambda';
	L = chol(Sigma_th, 'lower');

	% Specify first order PAC Psi, construct Phi and Sigma for step
	Psi = 0.2 * I; %diag([0.2, 0.2]);
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
	elseif is_draw
		thSt = [thSt; th(3)];
	end

	if is_draw
		thetaSt = theta;
		thetaSt.th = thSt;
		thetaSt.omega = exp(thSt(1));
		thetaSt.phi = tanh(thSt(2));
		thetaSt.mu = thSt(3);
		varargout{1} = thetaSt;
	end
end

function Om = min_curve(H, Om_bar)
	[V, D] = eig(H + Om_bar);
	Om = Om_bar + V*abs(D)*V';
end

function [neg_H11, neg_H22, pos_det, nd] = get_flags(H)
	neg_H11 = H(1, 1) < 0;
	neg_H22 = H(2, 2) < 0;
	pos_det = (H(1, 1) * H(2, 2) - H(1, 2)^2) > 0;
	nd = neg_H11 && neg_H22 && pos_det;
end

function Om = robust_Om_rotate(H, g, q_turn, q_max, alpha)
	[V, D] = eig(H);
	c_turn = chi2inv(q_turn, 1);
	c_max = chi2inv(q_max, 1);
	for i=1:2
		d_i = D(i,i);
		gamma_i2 = (g'*V(:,i))^2;
		if (d_i<0)
			c_in = -gamma_i2 / d_i;
			if c_in > c_turn
				c_out = c_turn + (c_max - c_turn) * (1-exp(-(c_in-c_turn)/(c_max-c_turn)));
				D(i,i) = gamma_i2 / c_out;
			else
				D(i,i) = -d_i;
			end
		else
			D(i,i) = 0.5 * d_i + 0.5 * alpha * gamma_i2;
		end
	end
	Om = V*D*V';
end

function Om = robust_Om(H, g, min_eig, c_turn, c_max)
	[Om, Om_inv] = clean_eigs_min(H, min_eig);
	c_Om = g' * Om_inv * g;
	if c_Om > c_turn
		c_out = c_turn + (c_max - c_turn) * (1-exp(-(c_Om-c_turn)/(c_max-c_turn)));
		alpha = (c_Om - c_out)/(c_Om * c_out);
		Om = Om + alpha * g * g';
	end
end

function [Om, Om_inv] = clean_eigs_abs(H, min_eig)
	[dim_n, ~] = size(H);
	[V, D] = eig(H);
	d = diag(D);
	for i=1:dim_n
		d(i) = soft_abs(d(i), min_eig);
	end
	D = diag(d);
	Om = V*D*V';
	Om_inv = V*inv(D)*V';
end

function [Om, Om_inv] = clean_eigs_min(H, min_eig)
	[dim_n, ~] = size(H);
	[V, D] = eig(H);
	d = diag(D);
	for i=1:dim_n
		d(i) = soft_abs(abs(d(i)), min_eig);
	end
	D = diag(d);
	Om = V*D*V';
	Om_inv = V*inv(D)*V';
end

function soft_abs_x = soft_abs(x, zero_value)
	if abs(x) < 100
		soft_abs_x = log(exp(x) + exp(-x) - 2 + exp(zero_value));
	else
		soft_abs_x = abs(x);
	end
end
