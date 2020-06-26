% File: gir_mhessian.m
% 
% Date: 2020-06-24
% Author: Samuel Gingras

% Simulation parameters
M = 10^3;           % Nb of simulated block
B = 1000;           % Nb of draw by block
N = 25;             % Nb of artificial observation

% Model fixed parameters
mu = 0.0;
phi = 0.95;
omega = 100.0;
nu = 12.0;

% Set model
model = 'student_SV';

% Initialize theta structure for hessianMethod
theta.x.N = N;
theta.x.mu = mu;
theta.x.phi = phi;
theta.x.omega = omega;
theta.y.nu = nu;

% Set quantiles
Q = [ 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 ];

% Reserve space to store count results
postsim.count1 = zeros(N,length(Q));
postsim.count2 = zeros(N,length(Q));

% Set seed
rng(1)
drawState(12)
drawObs(123)
hessianMethod(1234)

% Initial draw (y,x)
x = drawState( theta );
y = drawObs( x, model, theta );

% Evaluate initial draw (y,x)
hmout = hessianMethod( model, y, theta, 'EvalAtState', x );

% Unpack hessianMethod output
x0 = hmout.x_mode;
lnp_x = hmout.lnp_x;
lnp_y__x = hmout.lnp_y__x;
lnq_x__y = hmout.lnq_x__y;

% Simulate counts of fixed block sizes 
for m=1:M

    % New block: set counts at zeros
    count1 = zeros(N,9);
    count2 = zeros(N-1,9);

    % Simulate counts for block b
    for b=1:B
        
        % (a) Draw proposal xSt|y
        hmout = hessianMethod( model, y, theta, 'GuessMode', x0 );

        % Unpack hessianMethod output
        xSt = hmout.x;
        x0St = hmout.x_mode;
        lnp_xSt = hmout.lnp_x;
        lnp_y__xSt = hmout.lnp_y__x;
        lnq_xSt__y = hmout.lnq_x__y;

        % Compute Hastings ratio
        lnH = lnp_y__xSt + lnp_xSt - lnq_xSt__y;
        lnH = lnH - lnp_y__x - lnp_x + lnq_x__y;

        % Accept/reject for xSt|y
        if( rand < exp(lnH) )
            x = xSt;
            x0 = x0St;
        end

        % (b) Update y|x
        y = drawObs( x, model, theta );

        % (c) Evaluate new draw (y,x)
        hmout = hessianMethod( model, y, theta, 'GuessMode', x0, 'EvalAtState', x );

        % Unpack hessianMethod output
        x0 = hmout.x_mode;
        lnp_x = hmout.lnp_x;
        lnp_y__x = hmout.lnp_y__x;
        lnq_x__y = hmout.lnq_x__y;

        % Normalized state vector
        z1 = (x - mu) * sqrt(omega*(1-phi^2));
        z2 = (x(2:N) - mu - phi.*(x(1:end-1) - mu)) * sqrt(omega);

        % Update counts
        count1 = count1 + (z1 < norminv(Q,0,1));
        count2 = count2 + (z2 < norminv(Q,0,1));

    end

    % Store results for block b
    postsim.count1(m,:) = mean( count1 ./ B, 1 );
    postsim.count2(m,:) = mean( count2 ./ B, 1 );

end
