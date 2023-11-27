function bivariate_aPr(sim)
	H = sim.sh.post2.H;
	[V, D] = eig(H);
	th = sim.sh.th2;
	m1 = V(2,1)/V(1,1);
	m2 = V(2,2)/V(1,2);
	b1 = th(2) - th(1)*m1;
	b2 = th(2) - th(1)*m2;

	th1_lo = min(sim.th(:,1));
	th1_hi = max(sim.th(:,1));
	th1_grid = th1_lo:0.01:th1_hi;
	c1 = exp(sim.sh.th2(1)) * (1-tanh(sim.sh.th2(2)))^2;
	y1 = atanh(1-sqrt(c1./exp(th1_grid)));
	c2 = exp(sim.sh.th2(1)) * (1-tanh(sim.sh.th2(2))^2);
	y2 = atanh(sqrt(1-c2./exp(th1_grid)));


	figure(1);
	subplot(1,2,1)
	scatter(sim.th(:,1), sim.th(:,2), (1-10*log(sim.aPr)))
	axis image
	hold on;
		refline(m1, b1);
		refline(m2, b2);
		plot(th1_grid, y1, 'red');
		plot(th1_grid, y2, 'green');
	hold off

	subplot(1,2,2)
	scatter(sim.th(:,1), sim.th(:,2), (1-10*log(sim.aPr)))
	axis image
	hold on;
		scatter(sim.thSt(:,1), sim.thSt(:,2), 2, 'red')
		refline(m1, b1);
		refline(m2, b2);
	hold off;

	figure(2);
	subplot(1,2,1)
	scatter(sim.th(:,2), sim.th(:,3), (1-10*log(sim.aPr)))

	subplot(1,2,2)
	scatter(sim.th(:,2), sim.th(:,3), (1-10*log(sim.aPr)))
	hold on	
		scatter(sim.thSt(:,2), sim.thSt(:,3), 2, 'red')
	hold off
end
