function [R, v, f] = prepare_proposal(g, mH, mV, g_prior, mH_prior)
    % Parameters of proposal distribution
    h = 1.0;       % Step size
    theta = 0.0;   % Norton and Fox theta
    alpha = 0.5;   % Multiplier for gradient
    z = 4;         % Parameter of limiter
    
    % Manipulation of g and H
    mH = mH + mV + mH_prior;
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