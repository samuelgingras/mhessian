function [theta_mode, hmout, mode_sh] = init_theta(prior, theta, model, y)

	max_uphill_steps = 20;
	max_steps_per_uphill_step = 10;
	chi2_p = 0.99;

	[lnp, g, H, hmout] = compute_lnpgH(theta);
	if prior.has_mu
		theta.th(3) = mean(hmout.xC);
		theta.mu = theta.th(3);
		[lnp, g, H, hmout] = compute_lnpgH(theta);
	end
	best_lnp = lnp;
	first_lnp = lnp;

	for i = 1 : max_uphill_steps

		% Regularize step
		[V, D] = eig(H);
		v = V' * g;
		% 0.01 v^2 limits gain_guess to 50 per dimension
		D_inv = -diag(1./max(abs(diag(D)), 0.01 * v.^2)); 
		th_step = -V*D_inv*v;
		gain_guess = -0.5 * v'*D_inv*v;
		if gain_guess < 0.02
			break;
		end

		for j = 1 : max_steps_per_uphill_step
			theta_prime = fill_theta_from_th(theta, theta.th + th_step, prior.has_mu);
			[lnp, g, H, hmout] = compute_lnpgH(theta_prime);
			if lnp > best_lnp || isnan(best_lnp)
				theta = theta_prime;
				current_increase = lnp - best_lnp;
				best_lnp = lnp;
				break;
			end
			th_step = 0.5 * th_step;
		end
	end

	% Compute value, gradient and Hessian of log target at current value th
	function [lnp, g, H, hmout] = compute_lnpgH(theta)
		if prior.has_mu
			th_length_string = 'Long';
		else
			th_length_string = 'Short';
		end
		[lnp, g, H] = prior.log_eval(prior, theta.th);
		hmout = hessianMethod(model, y, theta, 'GradHess', th_length_string);
		lnp = lnp + hmout.lnp_y__x + hmout.lnp_x - hmout.lnq_x__y;
		g = g + hmout.q_theta.grad;
		H = H + hmout.q_theta.Hess + hmout.q_theta.Var;
	end

	total_increase = lnp - first_lnp;
	theta_mode = theta;
	mode_sh = compute_shape(prior, hmout, theta_mode);
end
