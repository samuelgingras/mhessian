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
	%mode_Hhat_coeff = [0.9, 0.7, 0.5];
	%mode_Hbar_coeff = [0.1, 0.3, 0.5];
	mode_Hhat_coeff = [0.0, 0.0, 0.0];
	mode_Hbar_coeff = [1.0, 1.0, 1.0];

	max_iters = 5;
	sh = compute_shape(prior, hmout, theta);
 	theta_prime = theta;
 	sh_prime = sh;
 	I = (rand < 1e-3);
	for iter = 1:max_iters

		% Eigenvector decomposition, local chi2 values
		H = sh_prime.post2.H;
		g = sh_prime.post2.g;
		[V, D] = eig(H);
		v2 = (V' * g).^2;
		d = diag(D);
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

		% {
		if I
			fprintf("Iteration %i, trust %i\n", iter, trust)
			fprintf("Old H, g, th2:\n")
			display(sh.post2.H)
			display(sh.post2.g)
			display(theta.th(1:2))
			display(old_local_chi2)
			display(old_raw_th2)
			fprintf("Current H, g, th2:\n")
			display(sh_prime.post2.H)
			display(sh_prime.post2.g)
			display(sh_prime.th2)
			display(local_chi2)
		end
		% }

		% First regularize H to Hhat by manipulating eigenvalues.
        q = step_Hhat_chi2_q(trust);
		D(1,1) = min(D(1,1), -v2(1)/q);
		D(2,2) = min(D(2,2), -v2(2)/q);
		Hhat = V*D*V';

		gbar = Hbar * (sh_prime.th2 - mode_sh.th2);

		% Next blend in a bit of global information, depending on trust levels.
		H = step_Hhat_coeff(trust) * Hhat + step_Hbar_coeff(trust) * Hbar;
		g = step_Hhat_coeff(trust) * sh_prime.post2.g + step_Hbar_coeff(trust) * gbar;

		% Use this information to generate another value of theta prime.
		if max(d) < 0
			th2_prime = sh_prime.th2 - H\g;
			%th2_prime = sh_prime.th2 + g/(g'*g);
		else
			th2_prime = sh_prime.th2 - H\g;
			%th2_prime = sh_prime.th2 + g/(g'*g);
		end
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

	% Next option, regularize H to Hhat.
	%{
    q = mode_Hhat_chi2_q(trust);
	D(1,1) = min(D(1,1), -v2(1)/q);
	D(2,2) = min(D(2,2), -v2(2)/q);
	Hhat = V*D*V';
	th_hat = theta_prime.th(1:2) - V*inv(D)*V'*g;
	%}

	th_mid = theta_prime.th(1:2) - 0.5 * V*inv(D)*V'*g;
	[omqiota, omqpiota, omqppiota] = compute_omqiota(theta.N, th_mid);
	H_mid_12 = sh_prime.L12_norm * omqpiota;
	H_mid_22_1 = sh_prime.L22_const + sh_prime.L12_norm * omqppiota;
	H_mid_22 = min(H_mid_22_1, sh_prime.L22_norm * omqppiota);
	H_mid_11 = sh_prime.L11_const + sh.L12_norm * omqiota;
	H_mid = [H_mid_11, H_mid_12; H_mid_12, H_mid_22];
	if max(eig(H_mid)) > -0.01
		fprintf("Eigenvalue greater than -0.01\n")
		display(th_mid)
		display(H_mid)
	end
	th_hat = th_mid - 0.5 * (H_mid \ g);
	[omqiota, omqpiota, omqppiota] = compute_omqiota(theta.N, th_hat);
	H_hat_12 = sh_prime.L12_norm * omqpiota;
	H_hat_22_1 = sh_prime.L22_const + sh_prime.L12_norm * omqppiota;
	H_hat_22 = min(H_hat_22_1, sh_prime.L12_norm * omqiota);
	H_hat_11 = sh_prime.L11_const + sh_prime.L12_norm * omqiota;
	Hhat = [H_hat_11, H_hat_12; H_hat_12, H_hat_22];

	gbar = Hbar * (sh_prime.th2 - mode_sh.th2);

	% Next blend in a bit of global information, depending on trust levels.
	sh_prime.params.th2 = theta_prime.th(1:2);
	sh_prime.params.g2 = mode_Hhat_coeff(trust) * g + mode_Hbar_coeff(trust) * gbar;
	sh_prime.params.H2 = mode_Hhat_coeff(trust) * Hhat + mode_Hbar_coeff(trust) * Hbar;
	sh_prime.params.n_iters = iter;
	sh_prime.params.mean2 = theta_prime.th(1:2) - sh_prime.params.H2\sh_prime.params.g2;
	d = th_hat - mode_sh.th2;
	sh_prime.params.th_hat_distance = -0.5 * d' * mode_sh.post2.H * d;

	[Lp, sh_prime.params.H2_3rd] = directional_3rd(theta.N, sh_prime.params.H2, th_hat);

	% For mu|theta,y draw
	sh_prime.params.like_H33 = sh.like.H(3,3);
	sh_prime.params.like_g3 = sh.like.g(3);
	sh_prime.params.prior_H33 = sh.prior.H(3,3);
	sh_prime.params.prior_g3 = sh.prior.g(3);
end
