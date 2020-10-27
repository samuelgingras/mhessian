function y = drawObs_flexible_scd(s, x, beta, eta, lambda)
    n = length(x);
    y = zeros(n,1);
    for t=1:n
        u = betarnd(s(t), length(beta) - s(t) + 1);
        y(t) = (exp(x(t)) / lambda) * (-log(1-u))^(1/eta);
    end
end
