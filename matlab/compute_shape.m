function sh = compute_shape(prior, hmout, theta)

	[sh.prior.v, sh.prior.g, sh.prior.H] = prior.log_eval(prior, theta.th);
	q_theta = hmout.q_theta;

	sh.th = theta.th;
	sh.like.v = hmout.lnp_y__x + hmout.lnp_x - hmout.lnq_x__y;
	sh.like.g = q_theta.grad;
	sh.like.H = q_theta.Hess + q_theta.Var;
	sh.like.Var = q_theta.Var;
	sh.like.Hess = q_theta.Hess;

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
		g3 = q_theta.grad(3);
		V13 = q_theta.Var(1,3);
		V23 = q_theta.Var(2,3);
		V33 = q_theta.Var(3,3);
		H23 = q_theta.Hess(2,3);
		H33 = q_theta.Hess(3,3);
		
		% New November 1-3 stuff to obtain L_opt_th, L_opt_th_th
		% Quantities obtained without further approximation

		% Quantities requiring further approximations
		mean_223 = -2*((1+phi)^2 * g3 + (1+2*phi) * H23);  % u_3 in notes
		Cov_1_23 = -2*(1+phi) * V13;
		Cov_2_23 = -2*(1+phi) * V23;
		V33_outer = omega * (1-phi) * q_theta.xp(1);
		V33_inner = V33 - V33_outer;
		V33_12 = -2*(1+phi) * V33_inner - (1+phi) * V33_outer;
		V33_22 = 2*(1+phi)*(1+3*phi) * V33_inner + 2*(1+phi)*phi * V33_outer;

		L_mu = g3;
		sh.V_mu_mu = V33;
		L_mu_mu = sh.like.H(3,3);
		L_mu_mu_mu = q_theta.L_mu_mu_mu;
 		L_mu_th = sh.like.H(1:2,3);

 		H_mu_mu_th = [-h; -hp];
 		sh.V_mu_mu_th = [2*V33; 2*V33_12];
		L_mu_mu_th = H_mu_mu_th + sh.V_mu_mu_th;

		H_mu_th_th = [g3 + V13, H23 + V23; H23 + V23, mean_223 + Cov_2_23];
		V_mu_th_th = [2*V13, Cov_1_23 + V23; Cov_1_23 + V23, Cov_2_23];
		L_mu_th_th = H_mu_th_th + V_mu_th_th;

		H_mu_mu_th_th = [-h, -hp; -hp, -hpp];
		sh.V_mu_mu_th_th = [4*V33, 4*V33_12; 4*V33_12, 2*V33_22 + 0*8*(1+phi)^2*V33];
		L_mu_mu_th_th = H_mu_mu_th_th + sh.V_mu_mu_th_th;

		mu_diff = -L_mu / (L_mu_mu - 0.5 * L_mu_mu_mu * L_mu / L_mu_mu);
		sh.like2.v = sh.like.v + L_mu * mu_diff ...
			+ L_mu_mu * mu_diff^2 / 2 + L_mu_mu_mu * mu_diff^2 / 6;
		sh.like2.g = sh.like.g(1:2) + L_mu_th * mu_diff + 0.5 * L_mu_mu_th * mu_diff^2;
		sh.like2.H = sh.like.H(1:2, 1:2) ...
			+ L_mu_th_th * mu_diff + 0.5 * L_mu_mu_th_th * mu_diff^2;

		sh.like2.Hess = sh.like.Hess(1:2, 1:2) ...
			+ H_mu_th_th * mu_diff + 0.5 * H_mu_mu_th_th * mu_diff^2;
		sh.like2.Var = sh.like.Var(1:2, 1:2) ...
			+ V_mu_th_th * mu_diff + 0.5 * sh.V_mu_mu_th_th * mu_diff^2;

		c = 1/(theta.N-2);
		X01_inv = (-2/(omega*(1-phi^2)*(1-phi^2+c))) * [1-phi^2, -phi; phi*(1-phi^2), -0.5*(1+phi^2+c)];
		sh.like2.Sigma = X01_inv * sh.like2.Var * X01_inv';
		[V, d] = eig(sh.like2.Sigma, 'vector');
		sh.d1 = max(abs(d));
		sh.d2 = min(abs(d));
		sh.eps = 0.25*pi - acos(max(abs(V(:,1))));

		% Following quantities are used only for testing derivatives
		sh.L_mu_shape.v = L_mu;
		sh.L_mu_shape.g = [L_mu_th; L_mu_mu];
		sh.L_mu_shape.H = [L_mu_th_th, L_mu_mu_th; L_mu_mu_th', L_mu_mu_mu];

		sh.L_mu_mu_shape.v = L_mu_mu;
		sh.L_mu_mu_shape.g = L_mu_mu_th;
		sh.L_mu_mu_shape.H = L_mu_mu_th_th;

		sh.V_mu_mu_shape.v = sh.V_mu_mu;
		sh.V_mu_mu_shape.g = sh.V_mu_mu_th;
		sh.V_mu_mu_shape.H = sh.V_mu_mu_th_th;

		sh.H_mu_mu_shape.v = H33;
		sh.H_mu_mu_shape.g = H_mu_mu_th;
		sh.H_mu_mu_shape.H = H_mu_mu_th_th;

		sh.L11.v = sh.like.H(1,1);
		sh.L11.g = L_mu_th_th(1,1);
		sh.L11.H = L_mu_mu_th_th(1,1);

		sh.L12.v = sh.like.H(1,2);
		sh.L12.g = L_mu_th_th(1,2);
		sh.L12.H = L_mu_mu_th_th(1,2);

		sh.L22.v = sh.like.H(2,2);
		sh.L22.g = L_mu_th_th(2,2);
		sh.L22.H = L_mu_mu_th_th(2,2);

		%{
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
		%}

		%{
		sh.int2.v = L_mu_mu;
		sh.int2.g = L_mu_mu_th;
		sh.int2.H = L_mu_mu_th_th;

		sh.int3.v = L_mu;
		sh.int3.g = L_mu_th;
		sh.int3.H = L_mu_th_th;
		%}

		sh.prior2.v = sh.prior.v;
		sh.prior2.g = sh.prior.g(1:2);
		sh.prior2.H = sh.prior.H(1:2, 1:2);

		sh.post2.v = sh.like2.v + sh.prior.v;% + I;
		sh.post2.g = sh.like2.g + sh.prior.g(1:2);% + I_th;
		sh.post2.H = sh.like2.H + sh.prior.H(1:2, 1:2);% + I_th_th;

		sh.L_mu_mu_mu = L_mu_mu_mu;
	else
		sh.th2 = sh.th;
		sh.prior2 = sh.prior;
		sh.like2 = sh.like;
		sh.post2 = sh.post;
	end

	% New stuff related to third derivatives
	%sh.L12_norm = (sh.like2.H(1,2) + sh.I.H(1,2)) / hp;
	%sh.L11_const = (sh.like2.H(1,1) + sh.I.H(1,1)) - sh.L12_norm * h;
	%sh.L22_norm = (sh.like2.H(2,2) + sh.I.H(2,2)) / hpp;
	%sh.L22_const = (sh.like2.H(2,2) + sh.I.H(2,2)) - sh.L12_norm * hpp;
	sh.L12_norm = sh.like2.H(1,2) / hp;
	sh.L11_const = sh.like2.H(1,1) - sh.L12_norm * h;
	sh.L22_norm = sh.like2.H(2,2) / hpp;
	sh.L22_const = sh.like2.H(2,2) - sh.L12_norm * hpp;

	sh.omqiota = h;
end
