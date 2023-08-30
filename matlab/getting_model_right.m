% getting_model_right Testing model specifications for correctness
%   [marg, cond] = getting_model_right(model, theta, sim_params)
%   computes outputs
%   * marg, with fields (see below for explanation)
%       mean, an n_obs x n_Q matrix
%       nse, an n_obs x n_Q matrix of numerical standard errors,
%       p, an n_obs x n_Q matrix of p-values
%   * cond, with the same fields, but these are (n_obs - 1) x n_Q matrices.
%   The inputs are
%   * model is a string identifying one of the HESSIAN method models,
%   * theta is a structure with fields
%       - N, the number of observations
%       - mu, the mean of the state vector
%       - phi, the autocorrelation of the state vector
%       - omega, the precision of the innovation of the state vector
%   * sim_params is a structure with the fields
%     - n_obs, the number of observations, and the length of vectors x and y,
%     - n_blocks, the number of blocks used to compute numerical standard errors
%     - block_size, the number of draws of (x, y) per block,
%     - Q a vector of quantiles
%
% The idea is to generate a sample from the joint distribution of x and y
% (the values of state and model parameters are fixed) using the following Gibbs
% draws:
%  - x|y,theta using the HESSIAN method approximation as a proposal distribution
%  - y|x,theta using direct simulation from the model
%
% If the model code (as well as the HESSIAN method code) is working, then 
%  the distribution of each x[t] is N(mu, (omega(1-phi^2))^-1), t=1,...,n (marg below) 
%  the distribution of each x[t] - (1-phi)mu - phi x[t-1] is N(0,omega^-1) (cond below)
% The proportion of draws that are less than the correponding Gaussian quantiles are
% computed for the quantiles Q = 0.1,0.2,...,0.9, for t=1,...,n in the marg case and
% t=2,...,n in the cond case.
% These sample proportions are compared with the population proportions, which are,
% by definition, the values Q = 0.1,0.2,...,0.9 themselves.
%
% The marg and cond structures return information relevant to the comparison.
% The fields mean give the sample proportions, for the values in Q and either
% t=1,...,n (in the marg case) or t=2,...,n (in the cond case)
% The fields p give p-values for a t-test of the hypothesis that the population
% proportions are indeed 0.1,0.2,...,0.9.

function [marg, cond] = getting_model_right(model, theta, sim_params)

    % Set seed
    rng(1)
    drawState(12)
    drawObs(123)
    hessianMethod(1234)

    % Reserve space to store results
    n_Q = length(sim_params.Q);
    marg_Q = zeros(sim_params.n_obs, n_Q);
    marg_Q2 = zeros(sim_params.n_obs, n_Q);
    cond_Q = zeros(sim_params.n_obs-1, n_Q);
    cond_Q2 = zeros(sim_params.n_obs-1, n_Q);

    % Initial draw (y,x)
    x = drawState(theta);
    y = drawObs(x, model, theta);

    % Evaluate initial draw (y,x)
    hmout = hessianMethod(model, y, theta, 'EvalAtState', x);

    % Simulate counts of fixed block sizes 
    for b = 1:sim_params.n_blocks

        % New block: set counts at zeros
        marg_Q_b = zeros(sim_params.n_obs, n_Q);
        cond_Q_b = zeros(sim_params.n_obs-1, n_Q);

        % Simulate counts for block b
        for m = 1:sim_params.block_size
        
            % Update x|y  (draw proposal xSt, accept/reject)
            hmoutSt = hessianMethod(model, y, theta);
            if( rand < hastingsRatio(hmoutSt, hmout) )
                x = hmoutSt.x;
            end

            % Update y|x, compute HESSIAN approximation for new (x,y)
            y = drawObs(x, model, theta);
            hmout = hessianMethod(model, y, theta, 'EvalAtState', x);

            % Construct normalized states and innovations.
            % All elements of z1 and z2 are distributed N(0,1) if code is correct.
            z1 = (x - theta.x.mu) * sqrt(theta.x.omega*(1-theta.x.phi^2));
            z2 = (x(2:end) - theta.x.mu - theta.x.phi.*(x(1:(end-1)) - theta.x.mu)) * sqrt(theta.x.omega);

            % Update counts
            marg_Q_b = marg_Q_b + (z1 < norminv(sim_params.Q, 0, 1));
            cond_Q_b = cond_Q_b + (z2 < norminv(sim_params.Q, 0, 1));
        end

        marg_Q_b = marg_Q_b ./ sim_params.block_size;
        marg_Q = marg_Q + marg_Q_b;
        marg_Q2 = marg_Q2 + marg_Q_b.^2;

        cond_Q_b = cond_Q_b ./ sim_params.block_size;
        cond_Q = cond_Q + cond_Q_b;
        cond_Q2 = cond_Q2 + cond_Q_b.^2;
    end

    marg.mean = marg_Q ./ sim_params.n_blocks;
    qvar = marg_Q2 ./ sim_params.n_blocks - marg.mean.^2;
    marg.nse = sqrt(qvar ./ sim_params.n_blocks);
    marg.p = tcdf((marg.mean - sim_params.Q)./marg.nse, sim_params.n_blocks - 1);

    cond.mean = cond_Q ./ sim_params.n_blocks;
    qvar = cond_Q2 ./ sim_params.n_blocks - cond.mean.^2;
    cond.nse = sqrt(qvar ./ sim_params.n_blocks);
    cond.p = tcdf((cond.mean - sim_params.Q)./cond.nse, sim_params.n_blocks - 1);
end
