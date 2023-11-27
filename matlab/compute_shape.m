function sh = compute_shape(prior, hmout, theta)

	[sh.prior.v, sh.prior.g, sh.prior.H] = prior.log_eval(prior, theta.th);
	q_theta = hmout.q_theta;

	sh.th = theta.th;
	sh.like.v = hmout.lnp_y__x + hmout.lnp_x - hmout.lnq_x__y;
	sh.like.g = q_theta.grad;
	sh.like.H = q_theta.Hess + q_theta.Var;

	sh.post.v = sh.prior.v + sh.like.v;
	sh.post.g = sh.prior.g + sh.like.g;
	sh.post.H = sh.prior.H + sh.like.H;

	omega = theta.omega;
	phi = theta.phi;
	n = theta.N;

	% Compute approximations of shape after marginalizing out mu.
	% See Appendix on marginalizing out mu.
	if prior.has_mu
		sh.th2 = theta.th(1:2);
		h = omega * (1-phi) * ((n-2)*(1-phi) + 2);
		hp = -omega * (1-phi^2) * (2*(n-2)*(1-phi) + 2);
		hpp = 2*omega * (1-phi^2) * ((n-2)*(1+3*phi)*(1-phi) + 2*phi);
		
		% New November 1-3 stuff to obtain L_opt_th, L_opt_th_th
		% Quantities obtained without further approximation
		omq_Ee = sh.like.g(3);
		omqp_Ee = q_theta.Hess(2,3);
		omq_Ee1 = q_theta.Var(1,3);
		omq_Ee2 = q_theta.Var(2,3);
		omq_Eemu = sh.like.H(3,3);
		V33 = q_theta.Var(3,3);
		% Quantities requiring further approximations
		omqpp_Ee = -2*((1+phi)^2 * omq_Ee + (1+2*phi) * omqp_Ee);  % u_3 in notes
		omqp_Ee1 = -2*(1+phi) * omq_Ee1;                           % u_2 in notes
		omqp_Ee2 = -2*(1+phi) * omq_Ee2;                           % u_5 in notes
		V33_outer = omega * (1-phi) * q_theta.xp(1);
		V33_inner = V33 - V33_outer;
		V33_11 = V33;
		V33_12 = -2*(1+phi) * V33_inner - (1+phi) * V33_outer;
		V33_22 = 2*(1+phi)*(1+3*phi) * V33_inner + 2*(1+phi)*phi * V33_outer;
		omqp_Eemu = V33_12 - hp;                                   % u_1 in notes
		omqpp_Eemu = V33_22 - hpp;                                 % u_4 in notes

		sh.V33 = V33;
		sh.V33_th = [V33_11; V33_12];            % Extracts V33 derivs from L_mu_mu_th ...
		sh.V33_th_th = [V33, V33_12; V33_12, V33_22]; % ... and L_mu_mu_th_th

		L_mu = sh.like.g(3);
		L_mu_mu = sh.like.H(3,3);
		L_mu_mu_mu = q_theta.L_mu_mu_mu;

		L_mu_th = sh.like.H(1:2,3);
		L_mu_mu_th = [omq_Eemu; omqp_Eemu];
		L_mu_th_th = [omq_Ee + 2*omq_Ee1, omqp_Ee + omqp_Ee1 + omq_Ee2; ...
					  omqp_Ee + omqp_Ee1 + omq_Ee2, omqpp_Ee + 2*omqp_Ee2];
		L_mu_mu_th_th = [omq_Eemu, omqp_Eemu; omqp_Eemu, omqpp_Eemu];
		mu_diff = -L_mu / (L_mu_mu - 0.5 * L_mu_mu_mu * L_mu / L_mu_mu);
		sh.like2.v = sh.like.v + L_mu * mu_diff ...
			+ L_mu_mu * mu_diff^2 / 2 + L_mu_mu_mu * mu_diff^2 / 6;
		sh.like2.g = sh.like.g(1:2) + L_mu_th * mu_diff + 0.5 * L_mu_mu_th * mu_diff^2;
		sh.like2.H = sh.like.H(1:2, 1:2) ...
			+ L_mu_th_th * mu_diff + 0.5 * L_mu_mu_th_th * mu_diff^2;

		h_bar = -sh.prior.H(3,3);
		g_bar = sh.prior.g(3);
		mu_less_m = sh.like.g(3)/sh.like.H(3,3);
		mu_less_mu_bar = g_bar/h_bar;
		mu_bar_less_m = mu_less_m - mu_less_mu_bar;
		h_bar2_delta2 = (h_bar * mu_bar_less_m)^2;

		% Common computations for all versions of I(theta) derivatives
		c1 = h_bar - sh.like.H(3,3);
		c2 = 1/c1;
		c3 = h_bar2_delta2 * c2;
		f = 0.5*(-log(c1) + c3);         % f_1 + f_2 in paper
		fp = -0.5*c2*(1 + c3);           % f_1' + f_2' in paper
		fpp = c2^2*(0.5 + c3);           % f_1'' + f_2'' in paper

		I = 0.5 * log(h_bar) + f;
		I_th = fp * -L_mu_mu_th;
		I_th_th = fpp * -L_mu_mu_th * -L_mu_mu_th' + fp * -L_mu_mu_th_th;

		% Some shapes to check
		sh.I.v = I;
		sh.I.g = I_th;
		sh.I.H = I_th_th;

		sh.int2.v = L_mu_mu;
		sh.int2.g = L_mu_mu_th;
		sh.int2.H = L_mu_mu_th_th;

		sh.int3.v = L_mu;
		sh.int3.g = L_mu_th;
		sh.int3.H = L_mu_th_th;

		sh.post2.v = sh.like2.v + sh.prior.v + I;
		sh.post2.g = sh.like2.g + sh.prior.g(1:2) + I_th;
		sh.post2.H = sh.like2.H + sh.prior.H(1:2, 1:2) + I_th_th;

		sh.L_mu_mu_mu = L_mu_mu_mu;
	else
		sh.th2 = sh.th;
		sh.prior2 = sh.prior;
		sh.like2 = sh.like;
		sh.post2 = sh.post;
	end

	% New stuff related to third derivatives
	sh.L12_norm = sh.like2.H(1,2) / hp;
	sh.L11_const = sh.like2.H(1,1) - sh.L12_norm * h;
	sh.L22_norm = sh.like2.H(2,2) / hpp;
	sh.L22_const = sh.like2.H(2,2) - sh.L12_norm * hpp;

end
