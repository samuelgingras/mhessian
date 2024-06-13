function L_plus = directional_3rd(n, L_opt_th_th, th2, delta)

	[omq0, omqp0, omqpp0] = compute_omqiota(n, th2);

	L12_0 = L_opt_th_th(1,2) / omqp0;
	g = L12_0 * [omq0; omqp0];
	H = L12_0 * [omq0, omqp0; omqp0, omqpp0];
	L = omq0 * L12_0;
	L_plus = compute_omqiota(n, th2 + delta) * L12_0;
	L_minus = compute_omqiota(n, th2 - delta) * L12_0;
	L_plus = L_plus - L - delta'*g - 0.5 * delta'*H*delta;
	L_minus = L_minus - L + delta'*g - 0.5 * delta'*H*delta;
	L_plus = 0.5 * (L_plus - L_minus);
end
