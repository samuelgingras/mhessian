function y = simulate_sample(model,theta)
    x = drawState( theta );
    y = drawObs( x, model, theta );
end