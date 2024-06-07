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
theta.phi = 0.96;
theta.omega = 23.0;
theta = fill_theta_from_om_ph_mu(theta_true, true);

% Set prior distribution (mean and variance are arguments)
%prior = set_MVN_prior(true, [3.6; 2.5; -10.5], ...
%                      [1.25, 0.5, 0.0; 0.5, 0.25, 0.0; 0.0, 0.0, 0.25]);
prior = set_GaBeN_prior(true, 0.5, 0.5, 20, 1.5, -10, 0.01);
%prior = set_LNBeN_prior(true, -1.8, 1, 19, 1, -10, 4);

th_prime = theta.th + [0.0; 0.0; 0.5];
theta_prime = fill_theta_from_th(theta, th_prime, true);

hmout = hessianMethod(model, y, theta, 'GradHess', 'Long');
sh = compute_shape(prior, hmout, theta);
hmout_prime = hessianMethod(model, y, theta_prime, 'GradHess', 'Long');
sh_pr = compute_shape(prior, hmout_prime, theta_prime);

fprintf("L_mu_shape.H\n")
display(sh.L_mu_shape.H)

shape_diagnostics(theta.th, sh.L_mu_shape, th_prime, sh_pr.L_mu_shape);

fprintf("L_mu_mu_shape\n")
display(sh.L_mu_mu_shape.H)

shape_diagnostics(theta.th(1:2), sh.L_mu_mu_shape, th_prime(1:2), sh_pr.L_mu_mu_shape);

fprintf("V_mu_mu_shape\n")
display(sh.V_mu_mu_shape.H)

shape_diagnostics(theta.th(1:2), sh.V_mu_mu_shape, th_prime(1:2), sh_pr.V_mu_mu_shape);

fprintf("H_mu_mu_shape\n")
display(sh.H_mu_mu_shape.H)

shape_diagnostics(theta.th(1:2), sh.H_mu_mu_shape, th_prime(1:2), sh_pr.H_mu_mu_shape);

shape_diagnostics(theta.th(3), sh.L11, th_prime(3), sh_pr.L11);
shape_diagnostics(theta.th(3), sh.L12, th_prime(3), sh_pr.L12);
shape_diagnostics(theta.th(3), sh.L22, th_prime(3), sh_pr.L22);

%shape_diagnostics(theta.th, sh.prior, th_prime, sh_pr.prior);
%shape_diagnostics(theta.th, sh.like, th_prime, sh_pr.like);
%shape_diagnostics(theta.th, sh.post, th_prime, sh_pr.post);

%sh.prior2.g
%sh.prior2.H

% shape_diagnostics(theta.th(1:2), sh.prior2, th_prime(1:2), sh_pr.prior2);
%shape_diagnostics(theta.th(1:2), sh.like2, th_prime(1:2), sh_pr.like2);
%shape_diagnostics(theta.th(1:2), sh.post2, th_prime(1:2), sh_pr.post2);

%shape_diagnostics(theta.th(1:2), sh.int1, th_prime(1:2), sh_pr.int1);
%shape_diagnostics(theta.th(1:2), sh.int2, th_prime(1:2), sh_pr.int2);
%shape_diagnostics(theta.th(1:2), sh.int3, th_prime(1:2), sh_pr.int3);

d3 = -1:0.01:1;
L11_like = zeros(size(d3));
L11_like2 = zeros(size(d3));
H11_like = zeros(size(d3));
H11_like2 = zeros(size(d3));
V11_like = zeros(size(d3));
V11_like2 = zeros(size(d3));

for m = 1:length(d3);
	th_prime = theta.th + [0.0; 0.0; d3(m)];
	theta_prime = fill_theta_from_th(theta, th_prime, true);
	hmout_prime = hessianMethod(model, y, theta_prime, 'GradHess', 'Long');
	sh_pr = compute_shape(prior, hmout_prime, theta_prime);
	L11_like(m) = sh_pr.like.H(1,1);
	L11_like2(m) = sh_pr.like2.H(1,1);
	H11_like(m) = sh_pr.like.Hess(1,1);
	H11_like2(m) = sh_pr.like2.Hess(1,1);
	V11_like(m) = sh_pr.like.Var(1,1);
	V11_like2(m) = sh_pr.like2.Var(1,1);
end

figure(21)
plot(d3, L11_like)
hold on
plot(d3, L11_like2)
hold off

figure(22)
plot(d3, H11_like)
hold on
plot(d3, H11_like2)
hold off

figure(23)
plot(d3, V11_like)
hold on
plot(d3, V11_like2)
hold off
