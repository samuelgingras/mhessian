% File: getting_it_right_mhessian.m
% 
% Date: 2020-06-24
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
ndraw  = 10^3;              % Nb of simulated block
nblock = 1000;              % Nb of draw by block
ndata  = 25;                % Nb of artificial observation

% Set model parameters
model  = 'student_SV';      % Observation model    
mu0    = 0.0;               % Mean 
phi0   = 0.95;              % Autocorrelation
omega0 = 100.0;             % Precision
nu0    = 12.0;              % Student's t degree of freedom

% Set quantile for check implementation correctness
Q = 0.1:0.1:0.9;

% Reserve space to store results
postsim.count1 = zeros(ndraw,length(Q));
postsim.count2 = zeros(ndraw,length(Q));

% Set theta structure for hessianMethod
theta.x.N = ndata;
theta.x.mu = mu0;
theta.x.phi = phi0;
theta.x.omega = omega0;
theta.y.nu = nu0;

% Initial draw (y,x)
x = drawState( theta );
y = drawObs( x, model, theta );

% Evaluate initial draw (y,x)
hmout = hessianMethod( model, y, theta, 'EvalAtState', x );

% Unpack likelihood evaluations
lnp_x = hmout.lnp_x;
lnp_y__x = hmout.lnp_y__x;
lnq_x__y = hmout.lnq_x__y;

% Simulate counts of fixed block sizes 
for m = 1:ndraw

    % New block: set counts at zeros
    count1 = zeros(ndata,9);
    count2 = zeros(ndata-1,9);

    % Simulate counts for block b
    for b = 1:nblock
        
        % -------------------- %
        % Update x|y           %
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
        % Update y|x           %
        % -------------------- %
        
        % Draw y|x
        y = drawObs( x, model, theta );

        % Update hessian approximation for new draw (y,x)
        hmout = hessianMethod( model, y, theta, 'EvalAtState', x );

        % Unpack likelihood evaluations
        lnp_x = hmout.lnp_x;
        lnp_y__x = hmout.lnp_y__x;
        lnq_x__y = hmout.lnq_x__y;

        % -------------------- %
        % Update indicator     %
        % -------------------- %

        % Normalized state vector
        z1 = (x - mu0) * sqrt(omega0*(1-phi0^2));
        z2 = (x(2:ndata) - mu0 - phi0.*(x(1:ndata-1) - mu0)) * sqrt(omega0);

        % Update counts
        count1 = count1 + (z1 < norminv(Q,0,1));
        count2 = count2 + (z2 < norminv(Q,0,1));

    end

    % Store results for block b
    postsim.count1(m,:) = mean( count1 ./ nblock, 1 );
    postsim.count2(m,:) = mean( count2 ./ nblock, 1 );

end
