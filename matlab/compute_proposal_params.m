function sh_prime = compute_proposal_params(model, prior, y, mode_sh, theta, hmout)

	has_mu = prior.has_mu;
	if has_mu
		th_length_string = 'Long';
	else
		th_length_string = 'Short';
	end

	% Table of chi2(1) and chi2(2) quantiles to help with code.
	% chi2inv is very expensive in Matlab and users may not have it in toolbox.
	% p      q, nu=1  q, nu=2
	% -      -------  -------
    % 0.5    0.4549   1.3863
    % 0.8    1.6424   3.2189
    % 0.9    2.7055   4.6052
    % 0.95   3.8415   5.9915
    % 0.98   5.4119   7.8240 
    % 0.99   6.6349   9.2103
    % 0.995  7.8794   10.5966

	% Assess trust of quadratic model at mode, set parameters accordingly
	Hbar = mode_sh.post2.H;
	th2 = theta.th(1:2);
	global_chi2 = -(th2 - mode_sh.th2)' * Hbar * (th2 - mode_sh.th2);
	if global_chi2 < 5.9915
		trust = 3;
	elseif global_chi2 < 9.2103
		trust = 2;
	else
		trust = 1;
	end

	% Set trust-dependent parameter values.
	step_decision_chi2_q = [3.8415, 5.4119, 7.8794];
	step_Hhat_chi2_q = [3.8415, 5.4119, 7.8794];
	step_Hhat_coeff = [0.8, 0.6, 0.5];
	step_Hbar_coeff = [0.1, 0.3, 0.5];

	max_iters = 5;
	sh = compute_shape(prior, hmout, theta);
 	theta_prime = theta;
 	sh_prime = sh;
 	I = (rand < 1e-4);
	for iter = 1:max_iters

		% Eigenvector decomposition, local chi2 values
		H = sh_prime.post2.H;
		g = sh_prime.post2.g;
		
		[V, d] = eig(H, 'vector');
		[d, ind] = sort(d);
		V = V(:, ind);

		v2 = (V' * g).^2;
		local_chi2 = -v2 ./ d;
		if (iter == 1)
			old_local_chi2 = local_chi2;
			old_raw_th2 = th2 - H\g;
		end

		% Decide if we can rule out new evaluation based on negative definite H
		% and chi2 values low enough.
		if max(d) < 0 && (max(local_chi2) < step_decision_chi2_q(trust))
			break;
		end
		if max(d) < -1 && H(2,2) < 2 * H(1,1)
			break;
		end

		% No, compute new value of theta prime.

		% First regularize H to Hhat by manipulating eigenvalues.
        q = step_Hhat_chi2_q(trust);
		d(1) = min(d(1), -v2(1)/q);
		d(2) = min(d(2), -v2(2)/q);
		Hhat = V*diag(d)*V';

		gbar = Hbar * (sh_prime.th2 - mode_sh.th2);

		% Next blend in a bit of global information, depending on trust levels.
		H = step_Hhat_coeff(trust) * Hhat + step_Hbar_coeff(trust) * Hbar;
		g = step_Hhat_coeff(trust) * sh_prime.post2.g + step_Hbar_coeff(trust) * gbar;

		% Use this information to generate another value of theta prime.
		th2_prime = sh_prime.th2 - H\g;
		if has_mu
			mu_prime = theta_prime.mu - mode_sh.post.g(3) / mode_sh.post.H(3, 3);
			th_prime = [th2_prime; mu_prime];
		else
			th_prime = th2_prime;
		end
		theta_prime = fill_theta_from_th(theta_prime, th_prime, has_mu);

		% Compute shape at new value at try again
		hmout = hessianMethod(model, y, theta_prime, 'GradHess', th_length_string);
		sh_prime = compute_shape(prior, hmout, theta_prime);
	end

	th2_int = theta_prime.th(1:2);
	[omqiota, omqpiota, omqppiota] = compute_omqiota(theta.N, th2_int);
	H_int = sh_prime.like2.H;
	%fprintf("Input H_int: (%f, %f, %f)\n", H_int(1,1), H_int(1,2), H_int(2,2))

	omqi_0 = omqiota;

	w1 = [omqiota; omqpiota] / sqrt(omqiota^2 + omqpiota^2);
	w2 = [-omqpiota; omqiota] / sqrt(omqiota^2 + omqpiota^2);

	n_steps = 5;
	h = 1/n_steps;
	Hp = sh_prime.prior2.H;
	gp = sh_prime.prior2.g;

	H = sh_prime.like2.Hess;
	Hess_12 = H(1,2);
	Hess_22 = H(2,2);
	V = sh_prime.like2.Var;
	L = H + V;

	Delta = (mode_sh.th2 - sh_prime.th2);
	lm = V(2,2) / Hess_22;
	for step = 1:n_steps
		delta = h * Delta; % -((H_int + Hp)\g);
		th2_int = th2_int + delta;
		[omqiota, omqpiota, omqppiota] = compute_omqiota(theta.N, th2_int);

		% First idea, based on incorrect view that L12 \approx H12, using omqi
		%{
		H_int_12 = sh_prime.L12_norm * omqpiota;
		H_int_22_1 = sh_prime.L22_const + sh_prime.L12_norm * omqppiota;
		H_int_22 = min(H_int_22_1, sh_prime.L22_norm * omqppiota);
		H_int_11 = sh_prime.L11_const + sh_prime.L12_norm * omqiota;
		H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
		%}

		% Second idea, based on stretching in omqi direction, noting in
		% perpendicular direction
		%{
		omqi2 = omqiota^2; omqpi2 = omqpiota^2;
		omqix = omqiota * omqpiota;
		sqrtr = sqrt(omqiota/omqi_0);
		W = (1/(omqi2 + omqpi2)) * ...
			[omqi2*sqrtr + omqpi2, omqix*(sqrtr-1); ...
		     omqix*(sqrtr-1), omqpi2*sqrtr + omqi2];
		H_int = W * sh_prime.like2.H * W;
		%}

		% Third idea, based on empirical regularity of derivatives of
		% log(-L11) and log(-L22)
		%{
		rho20 = H_int(1,2)^2/(H_int(1,1) * H_int(2,2));
		lambda = 0.2;
		c11 = -0.0; c12 = -0.8; c21 = 1.5; c22 = -5.4;

		H_int_11 = H_int(1,1) * exp(c11*delta(1) + c12*delta(2));
		H_int_22 = H_int(2,2) * exp(c21*delta(1) + c22*delta(2));
		H_int_12 = H_int(1,2) * ...
			exp(-(H_int(1,1)/H_int(1,2))*delta(1) + (H_int(2,2)/H_int(1,2))*delta(2));
		H_int_12 = min(H_int_12, sqrt((1-lambda + lambda*rho20)*H_int_11*H_int_22));
		H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
		%}

		% Fourth idea, based on analytical derivation of derivatives of H22
		phi = tanh(th2_int(2));
		% These five lines confirmed Jan 11
		Cov_2_22 = (2*(1-phi^2)) * V(1,2) - 4*phi * V(2,2);
		H221 = H_int(2,2) + (2*(1-phi^2)) * V(1,1) - 4*phi * V(1,2);
		H222 = -2*(3*phi^2+1) * Hess_12 - 6*phi * Hess_22 + Cov_2_22;
		Hess_22 = Hess_22 + H221 * delta(1) + H222 * delta(2);
		Hess_12 = Hess_12 + H_int(1,2) * delta(1) + H_int(2,2) * delta(2);

		S = 1-sqrt(V(1,1)/((theta.N-1)/2));
		V(1,1) = V(1,1) + 2 * V(1,1) * S * delta(1) - V(1,1) * S * delta(2);
		%V(1,2) = -V(1,1) * S * delta(1) - V(1,2) * delta(2);
		V(2,2) = lm * Hess_22;
		L111 = -0.0 * H_int(1,1);
		L112 = -0.5 * H_int(1,1);
		L111 = H_int(1,1) + 2*V(1,1) * S;
		%L112 = -V(1,1) * S;
		L221 = H221 * (1+lm);
		L222 = H222;
		H_int_11 = H_int(1,1) + L111 * delta(1) + L112 * delta(2);
		H_int_22 = Hess_22 + V(2,2);
		H_int_12 = H_int(1,2) + L112 * delta(1) + L221 * delta(2);
		H_int_11 = min(H_int_11, H_int_12^2/(0.95*H_int_22));
		H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
	end
	%fprintf("New H_int: (%f, %f, %f)\n", H_int(1,1), H_int(1,2), H_int(2,2))

	% Test of third idea, based on knowledge of L at mode.
	%{
	Delta = (mode_sh.th2 - sh_prime.th2);
	lambda = 0.2;
	c11 = -0.0; c12 = -0.8; c21 = 1.5; c22 = -5.4;
	H_int = sh_prime.like2.H;
	for step = 1:n_steps
		delta = h * Delta;
		rho2 = H_int(1,2)^2/(H_int(1,1) * H_int(2,2));
		H_int_11 = H_int(1,1) * exp(c11*delta(1) + c12*delta(2));
		H_int_22 = H_int(2,2) * exp(c21*delta(1) + c22*delta(2));
		%H_int_12 = H_int(1,2) + c12*H_int(1,1)*delta(1) + c21*H_int(2,2)*delta(2);
		H_int_12 = H_int(1,2) * exp(c12*(H_int(1,1)/H_int(1,2))*delta(1) + c21*(H_int(2,2)/H_int(1,2))*delta(2));
		H_int_12 = min(H_int_12, sqrt((1-lambda + lambda*rho2)*H_int_11*H_int_22));
		H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
	end
	%}
	%fprintf("Mode based H_int: (%f, %f, %f)\n\n", H_int(1,1), H_int(1,2), H_int(2,2))

	omqi_hat = omqiota;
	H_hat = H_int;      % likelihood only
	th_hat = th2_int;   % posterior

	gbar = Hbar * (sh_prime.th2 - mode_sh.th2);

	% Next blend in a bit of global information, depending on trust levels.
	sh_prime.params.th2 = theta_prime.th(1:2);
	sh_prime.params.n_iters = iter;
	sh_prime.params.th_hat = th_hat;
	d = th_hat - mode_sh.th2;
	sh_prime.params.th_hat_distance = -0.5 * d' * mode_sh.post2.H * d;

	sh_prime.params.Hhat = [H_hat(1,1), H_hat(1,2), H_hat(2,2)];
	[V, D] = eig(H_hat);
	sh_prime.params.Hhat_eigs = sort(diag(D))';
	sh_prime.params.Hhat_Vmax = max(abs(V(:,1)));
	sh_prime.params.Hhat_wHw1 = w1'*H_hat*w1;
	sh_prime.params.Hhat_wHw2 = w2'*H_hat*w2;
	sh_prime.params.Hhat_omqi = omqi_hat;
	H0 = sh_prime.like2.H;
	[V0, D0] = eig(H0);
	sh_prime.params.H0 = [H0(1,1), H0(1,2), H0(2,2)];
	sh_prime.params.V0 = [hmout.q_theta.Var(1,1), hmout.q_theta.Var(1,2), hmout.q_theta.Var(2,2)];
	sh_prime.params.H0_eigs = sort(diag(D0))';
	sh_prime.params.H0_Vmax = max(abs(V0(:,1)));
	sh_prime.params.H0_wHw1 = w1'*H0*w1;
	sh_prime.params.H0_wHw2 = w2'*H0*w2;
	sh_prime.params.H0_omqi = omqi_0;
	sigma = 1./sqrt(diag(-D));
	lambda = 2.5;
	L_plus_1 = directional_3rd(theta.N, H_hat, th_hat, lambda*sigma(1)*V(:,1));
	L_plus_2 = directional_3rd(theta.N, H_hat, th_hat, lambda*sigma(2)*V(:,2));
	x1 = min(1, abs(L_plus_1)/lambda^2);
	x2 = min(1, abs(L_plus_2)/lambda^2);
	fact1 = max(1 - x1, 0.5) * 0.95;
	fact2 = max(1 - x2, 0.5) * 0.95;
	D_new = diag(diag(D) .* [fact1; fact2]);
	sh_prime.params.H2_3rd = V*D_new*V' + Hp;

	% For mu|theta,y draw
	sh_prime.params.like_H33 = sh.like.H(3,3);
	sh_prime.params.like_g3 = sh.like.g(3);
	sh_prime.params.prior_H33 = sh.prior.H(3,3);
	sh_prime.params.prior_g3 = sh.prior.g(3);
end
