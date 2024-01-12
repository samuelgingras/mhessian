function [BINT, R2] = hat(sim)
	th = sim.th;

	th_hat = sim.th_hat;
	H_hat = sim.Hhat;
	e = sim.Hhat_eigs;
	Vmax = sim.Hhat_Vmax;
	wHw1 = sim.Hhat_wHw1;
	wHw2 = sim.Hhat_wHw2;
	omqi = sim.Hhat_omqi;

	H0 = sim.H0;
	V0 = sim.V0;
	He0 = sim.H0 - sim.V0;
	e0 = sim.H0_eigs;
	Vmax0 = sim.H0_Vmax;
	wHw01 = sim.H0_wHw1;
	wHw02 = sim.H0_wHw2;
	omqi0 = sim.H0_omqi;

	true_H_hat = sim.sh.like2.H;
	[V, d] = eig(true_H_hat, 'vector');
	true_Vmax = max(abs(V(:,1)));
	true_phi_hat = tanh(sim.sh.th2(2));
	[true_omqi, true_omqpi, true_omqppi] = compute_omqiota(3139, sim.sh.th2);

	true_w2 = (true_omqi^2 + true_omqpi^2)^(-0.5) * [-true_omqpi; true_omqi];
	true_wHw2 = true_w2' * true_H_hat * true_w2;

	f = (sim.niter == 1);

	siz = exp(th(f,1) - mean(th(f,1)) + 2);
	col = e0(f,2);
	col = sim.Hhat_omqi(f) ./ sim.H0_omqi(f);
	col = omqi0(f);

	%{
	figure(1);
	scatter(th(:,1), e0(:,2), [], th(:,2));
	title('th1, e02')
	figure(2);
	scatter(th(:,1), H0(:,2).^2 ./ (H0(:,1) .* H0(:,3)), [], th(:,2));
	title('th1, rho2 L')
	figure(3);
	scatter(th(:,1), V0(:,2).^2 ./ (V0(:,1) .* V0(:,3)), [], th(:,2));
	title('th1, rho2 V')
	figure(4);
	scatter(th(:,1), He0(:,2).^2 ./ (He0(:,1) .* He0(:,3)), [], th(:,2));
	title('th1, rho2 H')
	figure(5);
	hist(V0(:,1), 50)
	%}

	%{
	% Predicting wHw2
	figure(1);
	scatter(sim.sigma_u(f), log(-sim.Hhat_wHw2(f)))
	title('sigma_u and w2H2w')
	yline(log(-true_wHw2));

	figure(2);
	scatter(sim.sigma(f), log(-sim.Hhat_wHw2(f)))
	title('sigma and w2H2w')
	yline(log(-true_wHw2));

	% Predicting L11
	figure(3);
	scatter(th(f,1), log(-H0(f,1)), [], col);
	axis equal;
	title('th 1, log(L11)')

	figure(4);
	scatter(th(f,2), log(-H0(f,1))-min(log(-H0(f,1))), [], col);
	axis equal;
	title('th 2, log(L11)')

	figure(3);
	scatter(log(omqi0(f)), log(-H0(f,1)), [], col);
	axis equal;
	title('log(omqi0), log(L11)')

	figure(4);
	scatter(th(f,1), H0(f,1), [], col);
	title('th 1, L11')

	figure(5);
	scatter(th(f,2), H0(f,1), [], col);
	title('th 2, L11')

	figure(6);
	scatter(log(omqi0(f)), H0(f,1), [], col);
	title('log(omqi0), L11')
	%}

	%{
	figure(1);
	scatter(th(f,1), th(f,2))
	hold on;
	scatter(th_hat(f,1), th_hat(f,2));
	hold off;
	title('th1, th2, values and mode guesses');

	figure(2);
	scatter(log(omqi0(f)), log(abs(V0(f,2))), siz);
	axis equal;
	title('log omqi, log|V12|')

	figure(3);
	scatter(log(omqi0(f)), log(abs(V0(f,3))), siz);
	axis equal;
	title('log omqi, log|V22|')

	figure(4)
	scatter(th(f,1), log(-H0(f,1)), [], col)
	axis equal;
	title('th1, log(L11)')

	figure(7)
	scatter(th(f,1), log(abs(V0(f,2))), [], col)
	axis equal;
	title('th1, log(V12)')

	figure(8);
	scatter(He0(f,1), V0(f,1), siz(f), col)
	title('Hess11 and Var11')
	axis equal

	figure(9);
	scatter(He0(f,2), V0(f,2), siz(f), col)
	title('Hess12 and Var12')
	axis equal
	
	figure(10);
	scatter(He0(f,3), V0(f,3), siz(f), col)
	title('Hess22 and Var22')
	axis equal

	figure(11);
	scatter(log(abs(V0(f,2))), log(V0(f,3)), siz, col)
	title('log Var12 and log Var22')
	axis equal

	figure(12);
	scatter(log(abs(V0(f,1))), log(V0(f,2)), siz, col)
	title('log Var11 and log Var12')
	axis equal

	figure(13);
	scatter(log(abs(He0(f,3))), log(V0(f,3)), siz, col)
	title('Hess22 and Var22')
	axis equal
	%}

	% {
	figure(1);
	scatter(th(f,1), th(f,2));
	title('theta and theta hat')
	hold on;
	scatter(th_hat(f,1), th_hat(f,2), [], col);
	xline(sim.sh.th(1));
	yline(sim.sh.th(2));
	hold off;

	figure(2);
	scatter(e(f,1), e(f,2), [], col);
	title('Two eigenvalues of H hat')
	xline(d(1));
	yline(d(2));

	figure(3);
	scatter(H_hat(f,1), H_hat(f,2), [], col);
	title('Hhat 11 and Hhat 12')
	xline(true_H_hat(1,1));
	yline(true_H_hat(1,2));

	figure(4);
	scatter(H_hat(f,2), H_hat(f,3), [], col);
	title('Hhat 12 and Hhat 22')
	xline(true_H_hat(1,2));
	yline(true_H_hat(2,2));

	figure(5);
	scatter(H_hat(f,1), H_hat(f,3), [], col);
	title('Hhat 11 and Hhat 22')
	xline(true_H_hat(1,1));
	yline(true_H_hat(2,2));

	figure(6);
	scatter(H_hat(f,3), Vmax(f), [], col);
	title('Hhat 22 and Vmax')
	xline(true_H_hat(2,2));
	yline(true_Vmax);
	yline(2*true_phi_hat/sqrt(4*true_phi_hat^2 + 1), 'r');
	yline((2+2*true_phi_hat)/sqrt(4*(1+true_phi_hat)^2 + 1), 'r');

	colour = th(f,2);
	figure(7);
	scatter(H0(f,1), H_hat(f,1), [], colour);
	title('Transformation of H11')
	xline(true_H_hat(1,1));
	yline(true_H_hat(1,1));
	axis image;

	figure(8);
	scatter(H0(f,2), H_hat(f,2), [], colour);
	title('Transformation of H12')
	xline(true_H_hat(1,2));
	yline(true_H_hat(1,2));
	axis image;

	figure(9)
	scatter(H0(f,3), H_hat(f,3), [], colour);
	title('Transformation of H22')
	xline(true_H_hat(2,2));
	yline(true_H_hat(2,2));
	axis image;

	figure(10)
	scatter(e0(f,1), e(f,1), [], col);
	title('Transformation of eig 1')
	xline(d(1));
	yline(d(1));
	axis image;

	figure(11)
	scatter(e0(f,2), e(f,2), [], col);
	title('Transformation of eig 2')
	xline(d(2));
	yline(d(2));
	axis image;

	figure(12)
	scatter(Vmax0(f), Vmax(f), [], col);
	title('Transformation of Vmax')
	xline(true_Vmax);
	yline(true_Vmax);
	axis image;

	figure(13)
	scatter(wHw01(f), wHw1(f), [], col);
	title('Transformation of wHw1')
	axis image

	figure(14)
	scatter(wHw02(f), wHw2(f), [], col);
	title('Transformation of wHw2')
	xline(true_wHw2);
	yline(true_wHw2);
	axis image

	figure(15)
	scatter(omqi0(f), omqi(f), [], col);
	title('Transformation of omqi')
	axis image
	%}
end
