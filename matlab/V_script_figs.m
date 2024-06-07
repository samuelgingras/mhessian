f = (s.niter == 1);
phi = s.phi(f);
phi_c = 1-phi;
phi_c2 = phi_c.^2;
phi2_c = (1-phi.^2);
V11 = s.V0(f,1);
V11c = 0.5*3139 - phi.^2.*V11;
V12 = s.V0(f,2);
V22 = s.V0(f,3);
V12n = V12./V11;
V22n = V22./V11;
sigma = s.sigma(f);
omega = s.omega(f);
L11 = s.H0(f,1);
L12 = s.H0(f,2);
L22 = s.H0(f,3);
H11 = L11 - V11;
H12 = L12 - V12;
H22 = L22 - V22;
g1 = 0.5*3139 + H11;
g2 = H12 - phi;

th = s.th;
S0 = s.S0;
S0_lm = s.S0_eigs;
ep = 0.25*pi - acos(s.S0_Vmax);

figure(1);
plotmatrix(s.V0(f,:));
title('V11, V12, V22 scatter plots')
axis equal

figure(2);
scatter(th(f,1), S0_lm(f,1));
title('theta 1 vs d1')

figure(3);
scatter(th(f,2), S0_lm(f,1));
title('theta 2 vs d1')

figure(4);
scatter(th(f,1), 2*th(f,1) + log(S0_lm(f,2)));
title('theta 1 vs log(omega^2 *d2)')

figure(5);
scatter(th(f,2), 2*th(f,1) + log(S0_lm(f,2)));
title('theta 2 vs log(omega^2 *d2)')

figure(6);
scatter(th(f,1), th(f,1) + log(ep(f)));
title('theta 1 vs log(omega * epsilon)')

figure(7);
scatter(th(f,2), th(f,1) + log(ep(f)));
title('theta 2 vs log(omega * epsilon)')

figure(8);
scatter(th(f,1) + log(phi_c), log(S0_lm(f,1)))
title('theta 1 + log(1-phi) versus log d1')
axis equal

figure(9);
scatter(log(s.H0_omqi(f,1)), log(S0_lm(f,1)))
title('log(omqi) vs log d1')
axis equal

H111_noS = H11 + 3*V11;
H112_noS = H12 + 3*V12;
H122_noS = H22 + 2*V22 + (2*phi2_c.*V11 - 4*phi.*V12);
H222_noS = -2*(3*phi.^2 + 1).*H12 - 6*phi.*H22 + 3*(2*phi2_c.*V12 - 4*phi.*V22);

fprintf('Regression of coskewness predictions on delta')
iota = f(f);
delta = s.sh.th2' - th(f,1:2);
X = [iota, delta];
[b, bint, r, rint, stats] = regress(L11 - H111_noS.*delta(:,1) - H112_noS.*delta(:,2), X);
display(bint)
display(stats(1))
[b, bint, r, rint, stats] = regress(L12 - H112_noS.*delta(:,1) - H122_noS.*delta(:,2), X);
display(bint)
display(stats(1))
[b, bint, r, rint, stats] = regress(L22 - H122_noS.*delta(:,1) - H222_noS.*delta(:,2), X);
display(bint)
display(stats(1))

X = [iota, th(f,1:2) - s.sh.th2'];

fprintf('Regression of log d2 on th1, th2')
y = log(S0_lm(f,2));
[b, bint, r, rint, stats] = regress(y, X);
display(bint)
display(stats(1))

fprintf('Regression of log d2 plus 2th1 on th1, th2')
y = log(S0_lm(f,2)) + 2*th(f,1);
[b, bint, r, rint, stats] = regress(y, X);
display(bint)
display(stats(1))

fprintf('Regression of log d1 on th1, th2')
y = log(S0_lm(f,1));
[b, bint, r, rint, stats] = regress(y, X);
display(bint)
display(stats(1))

fprintf('Regression of log epsilon on th1, th2')
y = log(ep(f));
[b, bint, r, rint, stats] = regress(y, X);
display(bint)
display(stats(1))




%{
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
%}