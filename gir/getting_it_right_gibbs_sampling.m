% File: getting_it_right_gibbs_sampling.m
% 
% Date: 2020-07-29
% Author: Samuel Gingras

% Clean workspace
clear all
clc

% Set seed
rng(1)
drawState(12)
drawObs(123)
hessianMethod(1234)

% Set simulation parameters
ndraw  = 1e5;               % Nb of posterior draw 
ndata  = 25;                % Nb of artificial observation

% Set model parameters
model  = 'student_SV';      % Observation model    
mu0    = 0.0;               % Mean 
phi0   = 0.95;              % Autocorrelation
omega0 = 100.0;             % Precision
nu0    = 12.0;              % Student's t degree of freedom

% Set prior distribution
prior = set_MVN_prior(true, [log(omega0); atanh(phi0); mu0], ...
                             diag([0.5; 0.10; 0.10]));

% Reserve space to store results
postsim.mu = zeros(ndraw,1);
postsim.phi = zeros(ndraw,1);
postsim.omega = zeros(ndraw,1);

% Initialize theta at true value
theta.x.N = ndata;
theta.x.mu = mu0;
theta.x.phi = phi0;
theta.x.omega = omega0;
theta.y.nu = nu0;

% Initial draw (y,x)
x = drawState(theta);
y = drawObs(x, model, theta);

% Evaluate initial draw (theta,x,y)
hmout = hessianMethod( model, y, theta, 'EvalAtState', x );

% Unpack likelihood evaluations
lnp_x = hmout.lnp_x;
lnp_y__x = hmout.lnp_y__x;
lnq_x__y = hmout.lnq_x__y;


for m = 1:ndraw
        
    % -------------------- %
    % Update x|theta,y     %
    % -------------------- %

    % Draw proposal xSt
    hmout = hessianMethod( model, y, theta );
    xSt   = hmout.x;

    % Unpack likelihood evaluations
    lnp_xSt = hmout.lnp_x;
    lnp_y__xSt = hmout.lnp_y__x;
    lnq_xSt__y = hmout.lnq_x__y;

    % Compute Hastings ratio
    lnH = lnp_y__xSt + lnp_xSt - lnq_xSt__y;
    lnH = lnH - lnp_y__x - lnp_x + lnq_x__y;

    % Accept/reject for xSt|y
    if( rand < exp(lnH) )
        x = xSt;
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
    
    % Transform parameters and evaluate likelihood and prior density
    th = transform_parameters( theta );
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
    if( rand < exp(lnH) )
        theta.x.mu = thSt(3);
        theta.x.phi = tanh(thSt(2));
        theta.x.omega = exp(thSt(1));
    end

    % -------------------- %
    % Update y|theta,x     %
    % -------------------- %
    
    % Draw y|theta,x
    y = drawObs(x, model, theta);

    % Update hessian approximation for new draw (theta,x,y)
    hmout = hessianMethod(model, y, theta, 'EvalAtState', x);

    % Unpack likelihood evaluations
    lnp_x = hmout.lnp_x;
    lnp_y__x = hmout.lnp_y__x;
    lnq_x__y = hmout.lnq_x__y;

    % -------------------- %
    % Store results        %
    % -------------------- %

    postsim.mu(m) = theta.x.mu;
    postsim.phi(m) = theta.x.phi;
    postsim.omega(m) = theta.x.omega;

end
