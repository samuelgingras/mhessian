function hat(sim)
	th = sim.th;
	th_hat = sim.th_hat;
	H_hat = sim.Hhat;
	e = sim.Hhat_eigs;
	cos_th = sim.Hhat_Vmax;
	true_H_hat = sim.sh.like2.H;
	[V, d] = eig(true_H_hat, 'vector');
	true_Vmax = max(abs(V(:,1)));

	figure(1);
	scatter(th(:,1), th(:,2));
	title('theta and theta hat')
	hold on;
	scatter(th_hat(:,1), th_hat(:,2))
	xline(sim.sh.th(1));
	yline(sim.sh.th(2));
	hold off;

	figure(2);
	scatter(e(:,1), e(:,2));
	title('Two eigenvalues of H hat')
	hold on;
	xline(d(1));
	yline(d(2));
	hold off;

	figure(3);
	scatter(H_hat(:,1), H_hat(:,2));
	title('Hhat 11 and Hhat 12')
	hold on;
	xline(true_H_hat(1,1));
	yline(true_H_hat(1,2));
	hold off;

	figure(4);
	scatter(H_hat(:,2), H_hat(:,3));
	title('Hhat 12 and Hhat 22')
	hold on;
	xline(true_H_hat(1,2));
	yline(true_H_hat(2,2));
	hold off;

	figure(5);
	scatter(H_hat(:,1), H_hat(:,3));
	title('Hhat 11 and Hhat 22')
	hold on;
	xline(true_H_hat(1,1));
	yline(true_H_hat(2,2));
	hold off;

	figure(6);
	scatter(H_hat(:,3), cos_th);
	title('Hhat 22 and Vmax')
	hold on;
	xline(true_H_hat(2,2));
	yline(true_Vmax);
	yline(2/sqrt(5), 'r');
	yline(4/sqrt(17), 'r');
	hold off;
end
