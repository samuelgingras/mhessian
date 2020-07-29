function th = transform_parameters(theta)

    % Unpack theta structure
    if( isfield(theta, 'x') )
        mu = theta.x.mu;
        phi = theta.x.phi;
        omega = theta.x.omega;
    else
        mu = theta.mu;
        phi = theta.phi;
        omega = theta.omega;
    end

    % Transform parameters
    th = [ log(omega); atanh(phi); mu ];

end