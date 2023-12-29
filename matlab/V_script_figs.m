f = (s.niter == 1);
phi = s.phi;
phi_c = 1-phi;
phi_c2 = phi_c.^2;
phi2_c = (1-phi.^2);
V11 = s.V0(:,1);
V11c = 0.5*3139 - phi.^2.*V11;
V12 = s.V0(:,2);
V22 = s.V0(:,3);
V12n = V12./V11;
V22n = V22./V11;
sigma = s.sigma;
omega = s.omega;
H11 = s.H0(:,1);
H12 = s.H0(:,2);
H22 = s.H0(:,3);
g1 = 0.5*3139 + H11;
g2 = H12 - phi;

figure(1);
scatter(log(s.sigma(f))+log(2*(1-phi(f))), log(abs(V12(f))), [], s.th(f,1))
title('log(sigma) + log(2(1-phi)), log V12')
axis equal

figure(2);
scatter(s.th(f,1) + 2*log(1-phi(f)), log(V22(f)), [], s.omega(f));
title('log(omega) + 2log(1-phi), log(V22)')
axis equal

figure(3);
scatter(0.5*3139 - phi(f).^2 .* V11(f), V12(f), [], phi(f))
title('n/2 - phi^2 V11, V12, line with slope 0.5')
line(2*V12(f), V12(f))
axis equal
