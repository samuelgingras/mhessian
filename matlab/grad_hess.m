function [g, H, V] = grad_hess(theta, state, long_th)
    
    [ g, H, V ] = grad_hess_approx1( theta, state, long_th );
    % [ g, H, V ] = grad_hess_approx2( theta, state, long_th );
    % [ g, H, V ] = grad_hess_approx3( theta, state, long_th );
    
end