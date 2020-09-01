% Clear workspace and set seed
clearvars; clc;
rng(1);
drawObs(12);
drawState(123);
hessianMethod(1234);

% Simulation parameters
ndraw  = 10^3;
nblock = 1000;
nstate = 5;
kmax   = 3;
kdata  = randi([1 kmax], nstate,1);
ndata  = sum(kdata);

% Set model parameters
model = 'flexible_SCD';
mu0 = 0.0;
phi0 = 0.90;
omega0 = 75;
beta0 = [0.15; 0.45; 0.40];

% Set quantile for check implementation correctness
Q = 0.1:0.1:0.9;

% Reserve space to store results
postsim.count1 = zeros(ndraw,length(Q));
postsim.count2 = zeros(ndraw,length(Q));

% Transition matrix for transform parameters 
nmixture = length(beta0);
T_p = transition_matrix_bmix(nmixture);
T_lambda = (1:nmixture)' ./ (1:nmixture) * T_p;

% Set theta structure for hessianMethod 
theta.x.N = nstate;
theta.x.mu = mu0;
theta.x.phi = phi0;
theta.x.omega = omega0;
theta.y.p = T_p * beta0;
theta.y.lambda = T_lambda * beta0;

% Initial draw of x
x = drawState(theta);
xrep = repelem(x, kdata);

% Initialize data structure for hessianMethod with initial draw of y
data.y = drawObs(xrep, 'mix_exp_SS', theta);
data.k = repelem((1:nstate)', kdata);
data.s = ones(ndata,1);

% Evaluate initial draw (y,x)
hmout = hessianMethod( model, data, theta, 'EvalAtState', x );

% Unpack likelihood evaluations
lnp_x = hmout.lnp_x;
lnp_y__x = hmout.lnp_y__x;
lnq_x__y = hmout.lnq_x__y;

% Simulate counts of fixed block sizes 
for m = 1:ndraw

    % New block: set counts at zeros
    count1 = zeros(nstate,9);
    count2 = zeros(nstate-1,9);

    % Simulate counts for block b
    for b = 1:nblock
        
        % -------------------- %
        % Update x|y           %
        % -------------------- %

        % Draw proposal xSt
        hmout = hessianMethod( model, data, theta );
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
            xrep = repelem(x, kdata);
        end

        % -------------------- %
        % Update y|x           %
        % -------------------- %
        
        % Draw y|x
        data.y = drawObs( xrep, 'mix_exp_SS', theta );

        % Update HESSIAN method approximation for new draw (y,x)
        hmout = hessianMethod( model, data, theta, 'EvalAtState', x );

        % Unpack likelihood evaluations
        lnp_x = hmout.lnp_x;
        lnp_y__x = hmout.lnp_y__x;
        lnq_x__y = hmout.lnq_x__y;

        % -------------------- %
        % Update indicator     %
        % -------------------- %

        % Normalized state vector
        z1 = (x - mu0) * sqrt(omega0*(1-phi0^2));
        z2 = (x(2:nstate) - mu0 - phi0.*(x(1:nstate-1) - mu0)) * sqrt(omega0);

        % Update counts
        count1 = count1 + (z1 < norminv(Q,0,1));
        count2 = count2 + (z2 < norminv(Q,0,1));

    end

    % Store results for block b
    postsim.count1(m,:) = mean( count1 ./ nblock, 1 );
    postsim.count2(m,:) = mean( count2 ./ nblock, 1 );

end
