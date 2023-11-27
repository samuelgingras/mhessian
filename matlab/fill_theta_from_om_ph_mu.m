function theta = fill_theta_from_om_ph_mu(theta_in, has_mu)
	theta = theta_in;
	if has_mu
		theta.th = [log(theta_in.omega); atanh(theta_in.phi); theta_in.mu];
	else
		theta.th = [log(theta_in.omega); atanh(theta_in.phi)];
	end
end
