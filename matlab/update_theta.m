% File: update_theta.m
%
% Update theta given x, a vector of latent state variables. Compute
% sufficient statistics once and draw theta|x K times using Random-Walk
% Metropolis before returning the last draw as the updated value.

function theta = update_theta(prior, x, theta, K)

% Compute sufficient statistics 
T = length(x);
x1 = x(1);
xT = x(T);
S1 = sum(x);
S2 = sum(x.^2);
Sx = sum(x(2:T) .* x(1:T-1));

% Compute estimators 
x_bar = S1/T;
s2 = S2/T - x_bar^2;
rho = ( Sx - (T+1)*x_bar^2 + (x1+xT)*x_bar ) / (T*s2);
rho = min(rho, 0.999);
s2 = max(s2, 0.001);

% Compute Sigma_step (covariance of jumping distribution)
r11 = 2;
r22 = 1/(1-rho^2);
r33 = s2*(1+rho)/(1-rho);
Sigma_step = (1.37^2) * (1/T) * diag([r11; r22; r33]);

% Draw K Innovation in one block
R_step = chol(Sigma_step); U = R_step'* randn(3,K);

% Compute log f(theta|x)  
log_f_x = log_likelihood(T, x1, xT, S1, S2, Sx, theta);
log_f = log_prior(prior, theta);
logF = log_f_x + log_f;

for k=1:K
    % Draw from proposal
    theta_St = theta + U(:,k);
  
    % Compute log f(theta_St|x)
    log_f_x = log_likelihood(T, x1, xT, S1, S2, Sx, theta_St);
	log_f = log_prior(prior, theta_St);
    logF_St = log_f_x + log_f;
    
    % Update theta
    if rand < exp(logF_St - logF)
        theta = theta_St;
        logF = logF_St;
    end
end

% Local function:
%
%   Compute log-likelihood of theta|x using sufficient statistics
%   T, x1, xT, S1, S2 and Sx.

function logf = log_likelihood(T, x1, xT, S1, S2, Sx, theta)

% Transform back theta in canonical parametrization
mu = theta(3);
phi = tanh(theta(2));
omega = exp(theta(1));

% Compute log f(theta|x)
logf = 0.5*T*log(omega) + 0.5*log(1-phi^2);
logf = logf - 0.5*omega*( -phi^2*(x1^2+xT^2) - 2*mu*(1-phi)*phi*(x1+xT) + 2*mu^2*(1-phi)*phi );
logf = logf - 0.5*omega*( (1+phi^2)*S2 - 2*phi*Sx - 2*mu*(1-phi)^2*S1 + T*(1-phi)^2*mu^2 );
