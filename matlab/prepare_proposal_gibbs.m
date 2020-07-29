% File: prepare_proposal_gibbs.m
%
% Compute the covariance matrix of the Gaussian jumping distribution.

function R = prepare_proposal_gibbs(N, x1, xN, S1, S2, Sx)

    % Compute OLS estimators for ar(1) linear model
    xbar = S1/N;
    s2   = max( 0.001, S2/N - xbar.^2 );
    rho  = min( 0.999, (Sx - (N+1)*xbar^2 + (x1+xN)*xbar) / (N*s2) );

    % Compute jumping distribution, scaled following Gelman et al. (1996)
    J = (1.37^2)*(1/N) * diag( [2; 1/(1-rho^2); s2*(1+rho)/(1-rho)] );
    R = chol(J);

end