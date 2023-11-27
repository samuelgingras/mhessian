th1 = [3.1; 2.2; -10];
th2 = [3.0; 2.3; -11];
prior = set_GaBeN_prior(true, 1, 50, 19, 1, -9, 1);
[sh1.v, sh1.g, sh1.H] = prior.log_eval(prior, th1);
[sh2.v, sh2.g, sh2.H] = prior.log_eval(prior, th2);

shape_diagnostics(th1, sh1, th2, sh2)