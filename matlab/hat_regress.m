function hat_regress(sim)

	phi_mode = tanh(sim.sh.th2(2));
	He_mode = sim.sh.like2.Hess
	Va_mode = sim.sh.like2.Var

	vv1 = [-2*(3*phi_mode^2 + 1), -6*phi_mode]
	Q3_mean_analytic = vv1 * He_mode(:,2)

	vv2 = [2*(1-phi_mode^2), -4*phi_mode];
	Q02_analytic = vv2 * Va_mode(:,1)
	Q12_analytic = vv2 * Va_mode(:,2)

	th = sim.th;
	f = sim.niter == 1;

	L0 = sim.H0;
	V0 = sim.V0;
	H0 = sim.H0 - sim.V0;

	ywHhatw2 = log(-sim.Hhat_wHw2(f));
	ywH0w2 = log(-sim.H0_wHw2(f));

	%H11
	yl11 = log(abs(L0(f,1)));
	yv11 = log(abs(V0(f,1)));
	yH11 = H0(f,1);
	yV11 = V0(f,1);
	yL11 = L0(f,1);
	yl11new = log(abs(L0(f,1) - sim.sh.L11_const));

	%H12
	yl12 = log(abs(L0(f,2)));
	yv12 = log(abs(V0(f,2)));
	yH12 = H0(f,2);
	yV12 = V0(f,2);

	%H22
	yl22 = log(abs(L0(f,3)));
	yh22 = log(abs(H0(f,3)));
	yH22 = H0(f,3);
	yv22 = log(abs(V0(f,3)));
	yV22 = V0(f,3);

	%X = [sim.niter(f), th(f,1), th(f,2)];
	X = [sim.niter(f), th(f,1), th(f,2)];

	L0_mean = mean(L0);
	H0_mean = mean(H0);
	V0_mean = mean(V0);
	fprintf("L mean = (%f, %f, %f)\n", L0_mean(1), L0_mean(2), L0_mean(3));
	fprintf("H mean = (%f, %f, %f)\n", H0_mean(1), H0_mean(2), H0_mean(3));
	fprintf("V mean = (%f, %f, %f)\n\n", V0_mean(1), V0_mean(2), V0_mean(3));

	XX = [sim.niter(f), log(V0(f,1)), log(V0(f,3)), th(f,1), th(f,2)];
	[B,BINT,R,RINT,STATS] = regress(log(V0(f,2)), XX);
	R2 = STATS(1);
	fprintf("log V12 on (1, log V11, log V22): b = (%f, %f, %f, %f, %f), R2 = %f\n\n", B(1), B(2), B(3), B(4), B(5), R2);

	[B,BINT,R,RINT,STATS] = regress(yl11, X);
	R2 = STATS(1);
	fprintf("log -L11 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yl11new, X);
	R2 = STATS(1);
	fprintf("log abs(L11 - L11_const) on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yv11, X);
	R2 = STATS(1);
	fprintf("log V11 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yl12, X);
	R2 = STATS(1);
	fprintf("log L12 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yv12, X);
	R2 = STATS(1);
	fprintf("log V12 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);
	fprintf("beta2 over beta1 = %f\n\n", B(3)/B(2));

	[B,BINT,R,RINT,STATS] = regress(yl22, X);
	R2 = STATS(1);
	fprintf("log -L22 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yh22, X);
	R2 = STATS(1);
	fprintf("log -H22 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yv22, X);
	R2 = STATS(1);
	fprintf("log V22 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yH11, X);
	R2 = STATS(1);
	fprintf("H11 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yH12, X);
	R2 = STATS(1);
	fprintf("H12 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yH22, X);
	R2 = STATS(1);
	fprintf("H22 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yV11, X);
	R2 = STATS(1);
	fprintf("V11 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yV12, X);
	R2 = STATS(1);
	fprintf("V12 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);
	fprintf("beta2 over beta1 = %f\n", B(3)/B(2));

	[B,BINT,R,RINT,STATS] = regress(yV22, X);
	R2 = STATS(1);
	fprintf("V22 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(yL11, X);
	R2 = STATS(1);
	fprintf("L11 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(ywH0w2, X);
	R2 = STATS(1);
	fprintf("log w2H0w2 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);

	[B,BINT,R,RINT,STATS] = regress(ywHhatw2, X);
	R2 = STATS(1);
	fprintf("log w2Hhatw2 on (1, th_1, th_2): b = (%f, %f, %f), R2 = %f\n", B(1), B(2), B(3), R2);
end