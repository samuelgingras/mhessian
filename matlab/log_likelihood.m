% File: log_likelihood.m
%
% Compute log-likelihood of theta|x using sufficient statistics 
% T, x1, xT, S1, S2 and Sx.

function logf = log_likelihood(N, x1, xN, S1, S2, Sx, theta)

    % Transform back theta in canonical parametrization
    mu = theta(3);
    phi = tanh(theta(2));
    omega = exp(theta(1));

    % Compute log f(theta|x)
    logf = 0.5*N*log(omega) + 0.5*log(1-phi^2);
    logf = logf - 0.5*omega*( -phi^2*(x1^2+xN^2) - 2*mu*(1-phi)*phi*(x1+xN) + 2*mu^2*(1-phi)*phi );
    logf = logf - 0.5*omega*( (1+phi^2)*S2 - 2*phi*Sx - 2*mu*(1-phi)^2*S1 + N*(1-phi)^2*mu^2 );

end