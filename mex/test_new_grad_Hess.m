model = 'gaussian_SV';
y = load('SP500_return');
n = length(y);
u = randn(n,1);
u = u-mean(u);
u = n*u/(u'*u);

mu = -10.00;
phi = 0.9840;
omega = 40;

theta.N     = n;
theta.mu    = mu;
theta.phi   = phi;
theta.omega = omega;

[hmout, state] = hessianMethod(model, y, theta);

%[g, H, V] = grad_hess_approx(theta, state, u);
for m=1:10000
  [gnew, Hnew, Vnew] = new_grad_hess_approx(theta, state, 1);
end
gnew
Vnew
Hnew + Vnew