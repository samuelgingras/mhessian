% Test code for prior specifications

th = [3.8; 2.6; -10];
delta = [0.0; 0.1; 0.1];

priors = cell(3,1);
priors{1} = set_MVN_prior(true, [3.6; 2.5; -10.5], ...
                         [1.25, 0.5, 0.0; 0.5, 0.25, 0.0; 0.0, 0.0, 0.25]);
priors{2} = set_GaBeN_prior(true, 1, 50, 19, 1, -9, 1);
priors{3} = set_LNBeN_prior(true, -3, 1, 19, 1, -9, 1);

for i = 1:3
    prior = priors{i,1};
    [v, g, H] = prior.log_eval(prior, th);
    [vp, gp, Hp] = prior.log_eval(prior, th + delta);
    v
    vp
    vp_approx_1 = v + g'*delta
    vp_approx_2 = v + g'*delta + 0.5*delta'*H*delta
end
