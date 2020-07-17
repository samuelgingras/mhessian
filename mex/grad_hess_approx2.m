% New approximations of Egrad, EHess and Vgrad
% Old way of setting H(1, 1)
% Old way of setting H(2, 3)
function [g, H, V] = grad_hess_approx2(theta, state, u)
	[g, H, V] = new_grad_hess_approx(theta, state);
	n = theta.N;
	H(1, 1) = g(1) - n/2;
	H(1, 2) = 0; H(2, 1) = 0;
	H(1, 3) = 0; H(3, 1) = 0;
	H(2, 3) = 0; H(3, 2) = 0;
end