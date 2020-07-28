function th = transform_parameters(theta)
    mu = theta.mu;
    phi = theta.phi;
    omega = theta.omega;
    th = [ log(omega); atanh(phi); mu ];
end