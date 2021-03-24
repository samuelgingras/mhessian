% Clear workspace and set seed
clearvars; clc;
rng(1);
drawObs(12);
drawState(123);
hessianMethod(1234);

% Simulation parameters
ndraw  = 10^2;
nblock = 1000;
ndata  = 10;

% Set observation model
model = 'flexible_SCD_v2';
is_data_augmentation = true;

% Set model parameters
mu0 = 0.0;
phi0 = 0.95;
omega0 = 100;
beta0 = [0.4; 0.3; 0.3];
eta0 = 1.2;
kappa0 = 2.5;
lambda0 = 1.0;

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
theta.y.eta = eta0;
theta.y.kappa = kappa0;
theta.y.lambda = lambda0;
theta.y.beta = beta0;
theta.y.alpha = table_beta_scd(length(beta0)) * beta0;

% Initial draw of (s,x,y)
s = randsample( length(beta0), ndata, true, beta0 );
x = drawState( theta );
y = drawObs_fscd( model, s, x, beta0, eta0, kappa0, lambda0 );

% Initialize data structure for hessianMethod with initial draw of y
data.y = y;
data.s = s;

% Evaluate initial draw (y,x)
hmout = hessianMethod( model, data, theta, 'EvalAtState', x, ...
    'DataAugmentation', is_data_augmentation );

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
        hmout = hessianMethod( model, data, theta, ...
            'DataAugmentation', is_data_augmentation );
        
        % Unpack likelihood evaluations
        lnp_xSt = hmout.lnp_x;
        lnp_y__xSt = hmout.lnp_y__x;
        lnq_xSt__y = hmout.lnq_x__y;

        % Compute Hastings ratio
        lnH = lnp_y__xSt + lnp_xSt - lnq_xSt__y;
        lnH = lnH - lnp_y__x - lnp_x + lnq_x__y;

        % Accept/reject for xSt|y
        aPr_x = min(1, exp(lnH));
        if( rand < aPr_x )
            x = hmout.x;
        end

        % -------------------- %
        % Update y|x           %
        % -------------------- %
        
        % Draw y|x
        if( ~is_data_augmentation )
            data.s = randsample( length(beta0), ndata, true, beta0 );
        end
        data.y = drawObs_fscd( model, data.s, x, beta0, eta0, kappa0, lambda0 );

        % Update HESSIAN method approximation for new draw (y,x)
        hmout = hessianMethod( model, data, theta, 'EvalAtState', x, ...
            'DataAugmentation', is_data_augmentation );

        % Unpack likelihood evaluations
        lnp_x = hmout.lnp_x;
        lnp_y__x = hmout.lnp_y__x;
        lnq_x__y = hmout.lnq_x__y;

        % -------------------- %
        % Update indicator     %
        % -------------------- %

        % Normalized state vector
        z1 = (x - mu0) * sqrt(omega0*(1-phi0^2));
        z2 = (x(2:end) - mu0 - phi0.*(x(1:end-1) - mu0)) * sqrt(omega0);

        % Update counts
        count1 = count1 + (z1 < norminv(Q,0,1));
        count2 = count2 + (z2 < norminv(Q,0,1));

    end

    % Store results for block b
    postsim.count1(m,:) = mean( count1 ./ nblock, 1 );
    postsim.count2(m,:) = mean( count2 ./ nblock, 1 );

end
