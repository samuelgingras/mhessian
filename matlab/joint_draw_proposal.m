function [theta, x] = joint_draw_proposal(model, data, theta, prior, g, H, V)

    % Transform state parameters
    th = transform_parameters( theta );

    % Prepare proposal
    [ lnp_th, g_prior, H_prior ] = log_prior( prior, th );
    [ R , v, f ] = prepare_proposal( g, -H, -V, g_prior, -H_prior );

    % Draw thSt
    uSt = randn(3,1);
    thSt = th + (1/f) * R\(R'\v+uSt);

    % Update structure for HESSIAN method
    theta.mu = thSt(3);
    theta.phi = tanh(thSt(2));
    theta.omega = exp(thSt(1));
    
    % Draw xSt|thSt 
    [ hmout, state ] = hessianMethod( model, data, theta );

    % Compute gradient and Hessian approximation
    [ gSt, HSt, VSt ] = grad_hess( theta, state );

    % Prepare backward proposal
    [ lnp_thSt, gSt_prior, HSt_prior ] = log_prior( prior, thSt );
    [ RSt, vSt, fSt ] = prepare_proposal( gSt, -HSt, -VSt, gSt_prior, -HSt_prior );

end
