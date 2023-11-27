function theta = fill_theta_from_th(theta_in, th, has_mu)
	theta = theta_in;
	theta.th = th;
	theta.omega = exp(th(1));
	theta.phi = tanh(th(2));
	if has_mu
		theta.mu = th(3);
	end
end
