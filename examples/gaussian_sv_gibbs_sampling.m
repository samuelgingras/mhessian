% File: gaussian_sv_gibbs_sampling.m
% 
% Date: 2020-07-29
% Author: Samuel Gingras

% Clean workspace
%clear all
%clc

% Set seeds of random number generators
rng(1)
drawState(12)
drawObs(123)
hessianMethod(1234)

% Set simulation parameters
ndraw  = 20000;             % Number of posterior draws 
burnin = 5000;              % Number of burn-in draws 
ndata  = 5000;              % Number of artificial observations

% Set model, values of true parameters
model  = 'gaussian_SV';     % Observation model
mu0    = -9.5;              % Mean
phi0   = 0.97;              % Autocorrelation
omega0 = 25.0;              % Precision

% Set theta for simulating artificial data
theta.N = ndata;
theta.mu = mu0;
theta.phi = phi0;
theta.omega = omega0;

% Simulate artificial sample
x = drawState(theta)
y = drawObs(x, model, theta);

% Set prior distribution (mean and variance are arguments)
%prior = set_MVN_prior(true, [3.6; 2.5; -10.5], ...
%                      [1.25, 0.5, 0.0; 0.5, 0.25, 0.0; 0.0, 0.0, 0.25]);
%prior = set_GaBeN_prior(true, 1, 50, 19, 1, -9, 1);
prior = set_LNBeN_prior(true, -3, 1, 19, 1, -9, 1);

% Reserve space to store results
postsim.mu     = zeros(ndraw, 1);
postsim.phi    = zeros(ndraw, 1);
postsim.omega  = zeros(ndraw, 1);
postsim.aPr_x  = zeros(ndraw, 1);
postsim.aPr_th = zeros(ndraw, 1);

% Initialize HESSIAN method approximation
hmout = hessianMethod(model, y, theta);

% Initialize state vector
x = hmout.x;
xC = hmout.xC;

% Unpack likelihood evaluations
lnp_x = hmout.lnp_x;
lnp_y__x = hmout.lnp_y__x;
lnq_x__y = hmout.lnq_x__y;


for m = (1-burnin):ndraw

    % -------------------- %
    % Update x|theta,y     %
    % -------------------- %

    % Draw proposal xSt
    hmout = hessianMethod( model, y, theta, 'GuessMode', xC );
    xSt   = hmout.x;
    xCSt  = hmout.xC;

    % Unpack likelihood evaluations
    lnp_xSt = hmout.lnp_x;
    lnp_y__xSt = hmout.lnp_y__x;
    lnq_xSt__y = hmout.lnq_x__y;

    % Compute Hastings ratio
    lnH = lnp_y__xSt + lnp_xSt - lnq_xSt__y;
    lnH = lnH - lnp_y__x - lnp_x + lnq_x__y;

    % Accept or reject proposal xSt
    aPr_x = min(1, exp(lnH));
    if( rand < aPr_x )
        x = xSt;
        x0 = xCSt;
    end

    % -------------------- %
    % Update theta|x,y     %
    % -------------------- %

    % Compute sufficient statistics 
    N  = length(x);
    x1 = x(1);
    xN = x(N);
    S1 = sum(x);
    S2 = sum(x.^2);
    Sx = sum(x(2:N) .* x(1:N-1));
    
    % Transform parameters and evaluate prior density and likelihood
    th = transform_parameters(theta);
    lnp_th = prior.log_eval(prior, th);
    lnp_x__th = log_likelihood(N, x1, xN, S1, S2, Sx, th);

    % Compute covariance matrix for Gaussian Random-Walk
    R = prepare_proposal_gibbs(N, x1, xN, S1, S2, Sx);

    % Draw proposal and evaluate prior density and likelihood 
    thSt = th + R' * randn(3,1);
    lnp_thSt = prior.log_eval(prior, thSt);
    lnp_x__thSt = log_likelihood(N, x1, xN, S1, S2, Sx, thSt);

    % Compute Hastings ratio
    lnH = lnp_x__thSt + lnp_thSt - lnp_x__th - lnp_th;

    % Accept/reject for thSt
    aPr_th = min(1, exp(lnH));
    if( rand < aPr_th )

        % Update theta structure
        theta.mu = thSt(3);
        theta.phi = tanh(thSt(2));
        theta.omega = exp(thSt(1));

        % Update HESSIAN method approximation
        hmout = hessianMethod(model, y, theta, 'GuessMode', xC, 'EvalAtState', x);

        % Unpack likelihood evaluations
        lnp_x = hmout.lnp_x;
        lnp_y__x = hmout.lnp_y__x;
        lnq_x__y = hmout.lnq_x__y;

    end

    % Store simulation results
    if( m > 0 )
        postsim.mu(m)     = theta.mu;
        postsim.phi(m)    = theta.phi;
        postsim.omega(m)  = theta.omega;
        postsim.aPr_x(m)  = aPr_x;
        postsim.aPr_th(m) = aPr_th;
    end
end
