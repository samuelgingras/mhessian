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
	V = sh_prime.like2.Var;
	L = H + V;

	if max(eig(L)) > -1
		fprintf("Bad eig going in\n")
		display(H);
		display(V);
		display(L);
	end

	Delta = (mode_sh.th2 - sh_prime.th2);
	lm = V(2,2) / H(2,2);
	d1 = sh_prime.d1;
	d2 = sh_prime.d2;
	epsilon = sh_prime.eps;
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
		% Update of H
		H111 = L(1,1);
		H112 = L(1,2);
		H121 = H112;
		H122 = L(2,2);
		Cov_1_22 = 2*(1-phi^2) * V(1,1) - 4*phi * V(1,2);
		Cov_2_22 = 2*(1-phi^2) * V(1,2) - 4*phi * V(2,2);
		H221 = H(2,2) + Cov_1_22;
		mean_222 = -2*(3*phi^2+1) * H(1,2) - 6*phi * H(2,2);
		H222 = mean_222 + Cov_2_22;
		H(1,1) = H(1,1) + H111 * delta(1) + H112 * delta(2);
		H(1,2) = H(1,2) + H121 * delta(1) + H122 * delta(2);
		H(2,1) = H(1,2);
		H(2,2) = H(2,2) + H221 * delta(1) + H222 * delta(2);

		% Update of V;
		omega = exp(th2_int(1));
		beta = -2*sqrt(V(1,1)/(0.5*(theta.N-1)));
		d1 = d1;% * exp(0.2 * (delta(1) - 2*phi*delta(2)));
		d2 = d2 * exp(beta * delta(1) + 0.0*delta(2));
		epsilon = epsilon * exp(1.0 * delta(1));
		vmax = epsilon + sqrt(0.5);
		vmin = sqrt(1-vmax^2);
		eigV = [vmax, vmin; vmin, -vmax];
		Sigma = eigV * diag([d1, d2]) * eigV';
		X01 = -0.5*omega*[(1+phi^2), -2*phi; 2*phi*(1-phi^2), -2*(1-phi^2)];
		%V = X01 * Sigma * X01';

		S = 1-sqrt(V(1,1)/((theta.N-1)/2));
		S111 = -2*V(1,1)*sqrt(V(1,1)/((theta.N-1)/2));
		%S112 = -3*V(1,2);
		V111 = 2*V(1,1) + S111;
		V112 = -L(1,1) - V(1,2); % 2*V(1,2) + S112;
		V121 = V112;
		S122 = 0; 6*V(1,2);
		S222 = 0; %-20*V(1,2);
		V122 = Cov_1_22 + V(2,2) + S122;
		V(1,1) = V(1,1) + V111 * delta(1) + V112 * delta(2);
		V(1,2) = V(1,2) + V121 * delta(1) + V122 * delta(2);
		V(2,1) = V(1,2);
		V221 = 2*Cov_1_22 + S122;
		V222 = 2*Cov_2_22 + S222;
		V(2,2) = lm * H(2,2);

		% Update of L;
		L = H + V;
		L(1,1) = min(L(1,1), L(1,2)^2/(0.95*L(2,2)));

		%{
		if max(eig(L)) > -1
			fprintf("L eigenvalue above -1 in compute_proposal_params\n")
			display(H);
			display(V);
			display(L);
			display(lm);
			display(Delta);
			display(step);
		end
		%}

		%{
		L111 = H_int(1,1) + 2*V(1,1)*S;
		L112 = -H_int(1,1);
		%L112 = H_int(1,2) - V(1,1)*S;
		L221 = H221 * (1+lm);
		L222 = H222;
		%}
		%H_int = H + V;
		%H_int_11 = H_int(1,1) + L111 * delta(1) + L112 * delta(2);
		%H_int_22 = Hess_22 + V(2,2);
		%H_int_12 = H_int(1,2) + L112 * delta(1) + L221 * delta(2);
		%H_int_11 = min(H_int_11, H_int_12^2/(0.95*H_int_22));
		%H_int = [H_int_11, H_int_12; H_int_12, H_int_22];
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
	H_hat = L;          % likelihood only, no prior
	V_hat = V;
	th_hat = th2_int;   % posterior

	gbar = Hbar * (sh_prime.th2 - mode_sh.th2);

	% Next blend in a bit of global information, depending on trust levels.
	sh_prime.params.th2 = theta_prime.th(1:2);
	sh_prime.params.n_iters = iter;
	sh_prime.params.th_hat = th_hat;
	d = th_hat - mode_sh.th2;
	sh_prime.params.th_hat_distance = -0.5 * d' * mode_sh.post2.H * d;

	d = theta_prime.th(1:2) - mode_sh.th2;
	sh_prime.params.mode_distance = -0.5 * d' * mode_sh.like2.H * d;

	% Prediction of Vhat based on Eigendecomposition of Sigma
	phi_prime = tanh(sh_prime.th2(2));
	d1 = sh_prime.d1;% * exp(0.2 * (Delta(1) - 2*phi*Delta(2)));
	b1 = -2*sqrt(sh_prime.like2.Var(1,1)/(0.5*(theta.N-1)));
	b2 = 2*(1-phi_prime)*b1;
	d2 = sh_prime.d2 * exp(b1 * Delta(1) + b2 * Delta(2));
	ep = sh_prime.eps * exp(-0.0 * Delta(1));
	eigV = [cos(0.25*pi-ep), sin(0.25*pi-ep); sin(0.25*pi-ep), -cos(0.25*pi-ep)];
	Sigma = eigV * diag([d1, d2]) * eigV';

	omega_hat = exp(mode_sh.th2(1));
	phi_hat = tanh(mode_sh.th2(2));
	c = 1/(theta.N-2);
	X01 = 0.5*omega_hat*[(1+phi_hat^2+c), -2*phi_hat; 2*phi_hat*(1-phi_hat^2), -2*(1-phi_hat^2)];
	Vhat = X01 * Sigma * X01';
	%sh_prime.params.Vhat = [Vhat(1,1), Vhat(1,2), Vhat(2,2)];
	sh_prime.params.Vhat = [V_hat(1,1), V_hat(1,2), V_hat(2,2)];

	% Prediction of Vhat based on Psi30, Psi21, Psi12, Psi03
	H11 = sh_prime.like2.Hess(1,1);
	H12 = sh_prime.like2.Hess(1,2);
	H22 = sh_prime.like2.Hess(2,2);
	V11 = sh_prime.like2.Var(1,1);
	V12 = sh_prime.like2.Var(1,2);
	V22 = sh_prime.like2.Var(2,2);
	phi = tanh(sh_prime.th2(2));
	H111 = H11 + V11;
	H112 = H12 + V12;
	H121 = H112;
	H122 = H22 + V22;
	Cov_1_22 = 2*(1-phi^2) * V11 - 4*phi * V12;
	Cov_2_22 = 2*(1-phi^2) * V12 - 4*phi * V22;
	H221 = H22 + Cov_1_22;
	mean_222 = -2*(3*phi^2+1) * H12 - 6*phi * H22;
	H222 = mean_222 + Cov_2_22;
	H11hat = H11 + H111 * Delta(1) + H112 * Delta(2);
	H12hat = H12 + H121 * Delta(1) + H122 * Delta(2);
	H22hat = H22 + H221 * Delta(1) + H222 * Delta(2);

	S111 = -2*V11*sqrt(V11/((theta.N-1)/2));
	S112 = -3*V12 - H11 - V11;
	S122 = V22;
	S222 = -20*V12;

	V111 = 2*V11 + S111;
	V112 = 2*V12 + S112;
	V11hat = V11 + V111 * Delta(1) + V112 * Delta(2);

	V121 = V112;
	V122 = Cov_1_22 + V22 + S122;
	V12hat = V12 + V121 * Delta(1) + V122 * Delta(2);

	V221 = 2*Cov_1_22;% + S122;
	V222 = 2*Cov_2_22;% + S222;
	V22hat = V22 + V221 * Delta(1) + V222 * Delta(2); %lm * H22hat;

	%sh_prime.params.Hhat = [H_hat(1,1), H_hat(1,2), H_hat(2,2)];
	%sh_prime.params.Hhat = [H(1,1) + Vhat(1,1), H(1,2) + Vhat(1,2), H(2,2) + Vhat(2,2)];
	sh_prime.params.Hhat = [H11hat, H12hat, H22hat];
	sh_prime.params.Vhat = [V11hat, V12hat, V22hat];
	sh_prime.params.Lhat = sh_prime.params.Hhat + sh_prime.params.Vhat;
	sh_prime.params.Delta = Delta;

	[V, D] = eig(H_hat);
	sh_prime.params.Hhat_eigs = sort(diag(D))';
	sh_prime.params.Hhat_Vmax = max(abs(V(:,1)));
	sh_prime.params.Hhat_wHw1 = w1'*H_hat*w1;
	sh_prime.params.Hhat_wHw2 = w2'*H_hat*w2;
	sh_prime.params.Hhat_omqi = omqi_hat;
	H0 = sh_prime.like2.Hess;
	[V0, D0] = eig(H0);
	sh_prime.params.H0 = [H0(1,1), H0(1,2), H0(2,2)];
	sh_prime.params.V0 = [sh_prime.like2.Var(1,1), sh_prime.like2.Var(1,2), sh_prime.like2.Var(2,2)];
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
