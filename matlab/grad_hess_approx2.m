% New approximations of Egrad, EHess and Vgrad
% Old way of setting H(1, 1)
% Old way of setting H(2, 3)
function [g, H, V] = grad_hess_approx2(theta, state, long_th)
	[g, H, V] = new_grad_hess_approx(theta, state, long_th);
    if( isfield(theta, 'x') )
        n = theta.x.N;
    else
        n = theta.N;
    end
	H(1, 1) = g(1) - n/2;
	H(1, 2) = 0; H(2, 1) = 0;
	if length(g) > 2
		H(1, 3) = 0; H(3, 1) = 0;
		H(2, 3) = 0; H(3, 2) = 0;
	end
end