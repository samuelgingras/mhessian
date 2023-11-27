function [L_plus_res, L_stretch] = directional_3rd(n, L_opt_th_th, th2, varargin)

	[omq0, omqp0, omqpp0] = compute_omqiota(n, th2);

	if nargin == 4
		delta = varargin{1};
	else
		alpha = 2.5;
		[V, D] = eig(L_opt_th_th);
		v = V(:,1);
		%v = [1; omqp0/omq0];
    	sigma = 1/sqrt(-v'*L_opt_th_th*v);
    	delta = alpha * sigma * v;
    end

	L12_0 = L_opt_th_th(1,2) / omqp0;
	g = L12_0 * [omq0; omqp0];
	H = L12_0 * [omq0, omqp0; omqp0, omqpp0];
	L = omq0 * L12_0;
	L_plus = compute_omqiota(n, th2 + delta) * L12_0;
	L_minus = compute_omqiota(n, th2 - delta) * L12_0;
	L_plus_res = L_plus - L - delta'*g - 0.5 * delta'*H*delta;
	L_minus_res = L_minus - L + delta'*g - 0.5 * delta'*H*delta;
	L_plus_res = 0.5 * L_plus_res - L_minus_res;

	beta = 0.2 + abs(L_plus_res) / ((delta'*delta)^2);
	beta = min(beta, 0.9 * delta'*L_opt_th_th*delta / ((delta'*delta)^2));

	delta_perp = [-omqp0/omq0; 1];
	beta_perp = 0.1 * delta_perp'*L_opt_th_th*delta_perp / ((delta_perp'*delta_perp));
	L_stretch = L_opt_th_th + beta * delta * delta' + beta_perp * delta_perp * delta_perp';
end
