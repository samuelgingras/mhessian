% This example tests the correctness of the code for various state space models.

% The idea is to generate a sample from the joint distribution of x and y
% (the values of state and model parameters are fixed) using the following Gibbs
% draws:

%  - x|y,theta using the HESSIAN method approximation as a proposal distribution
%  - y|x,theta using direct simulation from the model

% If the model code (as well as the HESSIAN method code) is working, then 
%  the distribution of each $x_t$ is N(mu, (omega(1-phi^2))^-1), t=1,...,n (marg below) 
%  the distribution of each $x_t - (1-phi)mu - phi x_{t-1}$ is N(0,omega^-1) (cond below)
% The proportion of draws that are less than the correponding Gaussian quantiles are
% computed for the quantiles Q = 0.1,0.2,...,0.9, for t=1,...,n in the marg case and
% t=2,...,n in the cond case.
% These sample proportions are compared with the population proportions, which are,
% by definition, the values Q = 0.1,0.2,...,0.9 themselves.

% The marg and cond structures return information relevant to the comparison.
% The fields p give p-values for a t-test of the hypothesis that the population
% proportions are indeed 0.1,0.2,...,0.9.
% This script displays the minimal (out of 25 observations) of the p-values, for each
% quantile.
% It does so for each of the models.

% Use these simulation parameters for all models
clear sim_parameters
sim_parameters.n_obs = 25;         % Number of observations in artificial data
sim_parameters.Q = 0.1:0.1:0.9;    % Vector of population proportions
% Standard errors are computed using the batch mean method, with the following numbers
% of blocks and their sizes. The simulation sample size is the product of these numbers.
sim_parameters.n_blocks = 1e2;     
sim_parameters.block_size = 1e3;

% Use these state evolution parameters for all models
clear x
x.N = sim_parameters.n_obs;
x.mu = -9;
x.phi = 0.95;
x.omega = 100;

% Do one (student_SV) model using more transparent code.
clear y
y.nu = 12;
theta.x = x;
theta.y = y;
model = 'student_SV';
%[tmarg, tcond] = getting_model_right(model, theta, sim_parameters);

% Now do all models using more concise code

% Initialize parameters for those state space models that have them
burr_SS_y = struct('eta', 1.2, 'kappa', 2.5, 'lambda', 1.1);
gamma_SS_y = struct('kappa', 2);
gammapoisson_SS_y = struct('r', 10);
gengamma_SS_y = struct('eta', 1, 'kappa', 1, 'lambda', 1);
mix_exp_SS_y = struct('p', [0.5; 0.3; 0.2], 'lambda', [1; 2; 4]);
mix_gamma_SS_y = struct('p', [0.5; 0.3; 0.2], 'kappa', [1; 2; 4], 'lambda', [2; 3; 4]);
mix_gaussian_SV_y = struct('p', [0.5; 0.3; 0.2], 'mu', [0; -1; 1], 'sigma', [1; 2; 2]);
student_SV_y = struct('nu', 12);
weibull_SS_y = struct('eta', 2);

model_list = { ...
	struct('name', 'mix_gaussian_SV', 'theta', struct('x', x, 'y', mix_gaussian_SV_y)),
	struct('name', 'burr_SS', 'theta', struct('x', x, 'y', burr_SS_y)),
	struct('name', 'exp_SS', 'theta', struct('x', x)),
	struct('name', 'gamma_SS', 'theta', struct('x', x, 'y', gamma_SS_y)),
	struct('name', 'gammapoisson_SS', 'theta', struct('x', x, 'y', gammapoisson_SS_y)),
	struct('name', 'gaussian_SV', 'theta', struct('x', x)),
	struct('name', 'gengamma_SS', 'theta', struct('x', x, 'y', gengamma_SS_y)),
	struct('name', 'mix_exp_SS', 'theta', struct('x', x, 'y', mix_exp_SS_y)),
	%struct('name', 'mix_gamma_SS', 'theta', struct('x', x, 'y', mix_gamma_SS_y)),
	struct('name', 'poisson_SS', 'theta', struct('x', x)),
	struct('name', 'student_SV', 'theta', struct('x', x, 'y', student_SV_y)),
	%struct('name', 'weibull_SS', 'theta', struct('x', x, 'y', weibull_SS_y))
};

for m = 1:length(model_list)
	model_list{m}.name
	[marg, cond] = getting_model_right(model_list{m}.name, model_list{m}.theta, sim_parameters);
	min(marg.p)
	min(cond.p)
end
