% Set seeds of random number generators
rng(1)
drawState(12)
drawObs(123)
hessianMethod(1234)

% Set model, values of true parameters, simulate data
model  = 'gaussian_SV';     % Observation model
theta_true.mu = -9.5;
theta_true.phi = 0.97;
theta_true.omega = 25.0;
theta_true = fill_theta_from_om_ph_mu(theta_true, true);
theta_true.N = 5000;
x = drawState(theta_true);
y = drawObs(x, model, theta_true);

% Set first point of evaluation
theta.mu = -9.8;
theta.phi = 0.98;
theta.omega = 30.0;
theta = fill_theta_from_om_ph_mu(theta_true, true);

% Set prior distribution (mean and variance are arguments)
%prior = set_MVN_prior(true, [3.6; 2.5; -10.5], ...
%                      [1.25, 0.5, 0.0; 0.5, 0.25, 0.0; 0.0, 0.0, 0.25]);
%prior = set_GaBeN_prior(true, 1, 50, 19, 1, -9, 1);
prior = set_LNBeN_prior(true, -1.8, 1, 19, 1, -10, 4);

th_prime = theta.th + [0.1; 0.0; 0.0];
theta_prime = fill_theta_from_th(theta, th_prime, true);

hmout = hessianMethod(model, y, theta, 'GradHess', 'Long');
sh = compute_shape(prior, hmout, theta);
hmout_prime = hessianMethod(model, y, theta_prime, 'GradHess', 'Long');
sh_pr = compute_shape(prior, hmout_prime, theta_prime);

%shape_diagnostics(theta.th, sh.prior, th_prime, sh_pr.prior);
%shape_diagnostics(theta.th, sh.like, th_prime, sh_pr.like);
%shape_diagnostics(theta.th, sh.post, th_prime, sh_pr.post);

%shape_diagnostics(theta.th(1:2), sh.prior2, th_prime(1:2), sh_pr.prior2);
%shape_diagnostics(theta.th(1:2), sh.like2, th_prime(1:2), sh_pr.like2);
shape_diagnostics(theta.th(1:2), sh.post2, th_prime(1:2), sh_pr.post2);

%shape_diagnostics(theta.th(1:2), sh.int1, th_prime(1:2), sh_pr.int1);
%shape_diagnostics(theta.th(1:2), sh.int2, th_prime(1:2), sh_pr.int2);
%shape_diagnostics(theta.th(1:2), sh.int3, th_prime(1:2), sh_pr.int3);
