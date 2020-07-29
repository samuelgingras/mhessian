function sim = GibbsDraw_MCMC(y, prior, K, M)
		
    N = length(y);
	alC = zeros(N,1);
	model = 'plain_SV';
	
	theta = prior.theta;
	mu = prior.theta(3);
	phi = tanh(prior.theta(2));
	omega = exp(prior.theta(1));
	x = A_prior_draw_copy(mu, phi, omega, N);
	
	for m=1:M
   		% Evaluate log-likelihood: log p(x|theta)
    	[log_p_y, log_p, logQ, alC] = A_eval_copy(model, x, alC, y, mu, phi, omega);
    	logP = log_p_y + log_p;
    
    	% Draw a proposal and evaluate log-likelihood: log p(x_St|theta)
    	[x_St, log_p_y, log_p, logQ_St, alC] = A_draw_eval_copy(model, alC, y, mu, phi, omega);
    	logP_St = log_p_y + log_p;

		HR = exp( logP_St - logQ_St - logP + logQ );
	
    	% Update latent state x
    	if rand < HR
        	x = x_St;
    	end
    
    	% Block 2: Update theta|x 
    	theta = update_theta(x, theta, prior, K);
    
    	% Store results 
    	mu = theta(3);
    	phi = tanh(theta(2));
    	omega = exp(theta(1));
        
    	sim.mu(m) = mu;
    	sim.phi(m) = phi;
    	sim.omega(m) = omega;
		sim.w(m) = HR;
		sim.p(m) = min(1,HR);
	end
end

function theta = update_theta(x, theta, prior, K)

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
	Sigma_step = (1.37^2) * (1/T) * [r11 0 0 ; 0 r22 0 ; 0 0 r33];

	% Draw K Innovation in one block
	R_step = chol(Sigma_step); U = R_step'* randn(3,K);

	% Compute initial loglikelihood 
	logF = posteriorEval(prior, T, x1, xT, S1, S2, Sx, theta);

	for k=1:K
    	theta_St = theta + U(:,k);
    	logF_St = posteriorEval(prior, T, x1, xT, S1, S2, Sx, theta_St);

    	if rand < exp(logF_St - logF)
        	theta = theta_St;
        	logF = logF_St;
    	end
	end
end


function logF = posteriorEval(prior, T, x1, xT, S1, S2, Sx, theta)

	% Transform back theta in canonical parametrization
	mu = theta(3);
	phi = tanh(theta(2));
	omega = exp(theta(1));

	% Compute log f(theta|x)
	logf = 0.5*T*log(omega) + 0.5*log(1-phi^2);
	logf = logf - 0.5*omega*(-phi^2*(x1^2+xT^2) - 2*mu*(1-phi)*phi*(x1+xT) + 2*mu^2*(1-phi)*phi);
	logf = logf - 0.5*omega*((1+phi^2)*S2 - 2*phi*Sx - 2*mu*(1-phi)^2*S1 + T*(1-phi)^2*mu^2);
	
	% Compute log f(theta)
	u = prior.R \ (theta-prior.theta);
	logf_th = -0.5*(u'*u);
	
	% Compute posterior loglikelihood
	logF = logf + logf_th;
end