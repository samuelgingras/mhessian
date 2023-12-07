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
	mode_Hhat_chi2_q = [2.7055, 3.8415, 5.4119];

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
	w1 = [omqiota; omqpiota] / sqrt(omqiota^2 + omqpiota^2);
	w2 = [-omqpiota; omqiota] / sqrt(omqiota^2 + omqpiota^2);
	H_int = sh_prime.like2.H;

	n_steps = 4;
	h = 1/n_steps;
	Hp = sh_prime.prior2.H;
	for step = 1:n_steps
		th2_int = th2_int - h * ((H_int + Hp)\g);
		[omqiota, omqpiota, omqppiota] = compute_omqiota(theta.N, th2_int);
		H_int_12 = sh_prime.L12_norm * omqpiota;
		H_int_22_1 = sh_prime.L22_const + sh_prime.L12_norm * omqppiota;
		H_int_22 = min(H_int_22_1, sh_prime.L22_norm * omqppiota);
		H_int_11 = sh_prime.L11_const + sh_prime.L12_norm * omqiota;
		H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
	end
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
	sh_prime.params.Hhat_eigs = diag(D)';
	sh_prime.params.Hhat_Vmax = max(abs(V(:,1)));
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
