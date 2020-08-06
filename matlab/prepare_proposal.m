function [R, v, f] = prepare_proposal(n, g_prior, H_prior, q_theta)

    % Unpack approximation
    g = q_theta.grad;
    H = q_theta.Hess;
    V = q_theta.Var;

    % Replicate grad_hess_approx1
    H(1,1) = -n/2;
    H(1,2) = 0;
    H(2,1) = 0;
    if( length(g) > 2 )
        H(1,3) = 0;
        H(3,1) = 0;
    end

    % Parameters of proposal distribution
    h = 1.0;       % Step size
    theta = 0.0;   % Norton and Fox theta
    alpha = 0.5;   % Multiplier for gradient
    z = 4;         % Parameter of limiter
    
    % Manipulation of g and H
    mH = -H - V - H_prior;
    g = g + g_prior;
    
    % First make sure that matrix parameter is positive definite
    [V,D] = eig(mH);
    D = max(sqrt(abs(D)), eye(size(mH)));
    mH = V*D*D*V';
    R = chol(mH);
    
    % Then apply limiter
    c = norm(R'\g);
    R = R * (1/sqrt(h)) * sqrt(coth((3+z*sqrt(3))/c^2));
    
    % Then apply gradient multiplier
    v = g * alpha;
    
    % Then compute multiplication factor
    f = (1 + 0.5*h*theta);
end