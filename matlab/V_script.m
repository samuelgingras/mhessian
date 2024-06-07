t = zeros(23, 9);
V11_mse = zeros(23, 1);
V12_mse = zeros(23, 1);
V22_mse = zeros(23, 1);
V11hat_mse = zeros(23, 1);
V11hat_beta = zeros(23, 3);
V11hat_R2 = zeros(23, 1);
V12hat_mse = zeros(23, 1);
V12hat_beta = zeros(23, 3);
V12hat_R2 = zeros(23, 1);
V22hat_mse = zeros(23, 1);
V22hat_beta = zeros(23, 3);
V22hat_R2 = zeros(23, 1);

th2_mode = zeros(23, 2);
omega_mode = zeros(23, 1);
phi_mode = zeros(23, 1);

V = zeros(23, 3);
H = zeros(23, 3);

for i = 1:23
    s = sim_array{i};
    th2_mode(i,:) = s.sh.th2';
    omega_mode(i) = exp(th2_mode(i,1));
    phi_mode(i) = tanh(th2_mode(i,2));

    f = (s.niter == 1);
    phi = s.phi(f);
    phi_c = 1-phi;
    phi_c2 = phi_c.^2;
    phi2_c = (1-phi.^2);
    V11 = s.V0(f,1);
    V11c = 0.5*3139 - phi.^2.*V11;
    V12 = s.V0(f,2);
    V22 = s.V0(f,3);
    V12n = V12./(phi.^2.*V11);
    V22n = V22./(phi.^2.*V11);
    sigma = s.sigma(f);
    omega = s.omega(f);
    H11 = s.H0(f,1);
    H12 = s.H0(f,2);
    H22 = s.H0(f,3);
    L11 = H11 + V11;
    L12 = H12 + V12;
    L22 = H22 + V22;

    g1 = 0.5*3139 + H11;
    g2 = H12 - phi;

    iota = f(f);
    delta = th2_mode(i,:) - s.th(f,1:2);
    Xth = [iota, delta];
    Xg = [iota, g1, g2];

    H111_noS = H11 + 3*V11;
    H112_noS = H12 + 3*V12;
    H122_noS = H22 + 2*V22 + (2*phi2_c.*V11 - 4*phi.*V12);
    H222_noS = -2*(3*phi.^2 + 1).*H12 - 6*phi.*H22 + 3*(2*phi2_c.*V12 - 4*phi.*V22);

    Lhat11 = s.sh.like2.H(1,1);
    Lhat12 = s.sh.like2.H(1,2);
    Lhat22 = s.sh.like2.H(2,2);
    y11 = Lhat11 - L11 - H111_noS.*delta(:,1) - H112_noS.*delta(:,2);
    y12 = Lhat12 - L12 - H112_noS.*delta(:,1) - H122_noS.*delta(:,2);
    y22 = Lhat22 - L22 - H122_noS.*delta(:,1) - H222_noS.*delta(:,2);

    %y11 = V11;
    %y12 = V12;
    %y22 = V22;

    [b, bint, dum1, dum2, stats] = regress(y11, Xth);
    fprintf("%7.2f, %7.2f, %7.2f, %7.3f, ", b(1), b(2), b(3), stats(1));
    t(i, 1) = b(2);
    t(i, 2) = b(3);
    t(i, 3) = stats(1);

    [b, bint, dum1, dum2, stats] = regress(y12, Xth);
    fprintf("%7.2f, %7.2f, %7.2f, %7.3f, ", b(1), b(2), b(3), stats(1));
    t(i, 4) = b(2);
    t(i, 5) = b(3);
    t(i, 6) = stats(1);

    [b, bint, dum1, dum2, stats] = regress(y22, Xth);
    fprintf("%7.2f, %7.2f, %7.2f, %7.3f\n", b(1), b(2), b(3), stats(1));
    t(i, 7) = b(2);
    t(i, 8) = b(3);
    t(i, 9) = stats(1);

    X = [f(f), s.th(f,1:2) - mean(s.th(f,1:2))];
    %V11_mode = mean(s.V0(f,1)); V(i,1) = V11_mode;
    %V12_mode = mean(s.V0(f,2)); V(i,2) = V12_mode;
    %V22_mode = mean(s.V0(f,3)); V(i,3) = V22_mode;
    V11_mode = s.sh.like2.Var(1,1); V(i,1) = V11_mode;
    V12_mode = s.sh.like2.Var(1,2); V(i,2) = V12_mode;
    V22_mode = s.sh.like2.Var(2,2); V(i,3) = V22_mode;
    H(i,1) = s.sh.like2.Hess(1,1);
    H(i,2) = s.sh.like2.Hess(1,2);
    H(i,3) = s.sh.like2.Hess(2,2);

    Q = 0.5*3139 - phi.^2 .* V11;
    Qhat = Q .* exp(-0.4*delta(:,1) - 0.9*delta(:,2));
    %V11hat = (0.5*3139 - Qhat) ./ (phi_mode(i)^2);
    %V11hat = 0.5*3139 * sqrt(phi_mode(i) ./ phi);

    d = (H11.^2 ./ V11) - 0.5*3139;
    lambda = 4 * (d + sqrt(d.^2 + 0.5*3139*d));
    c = -2*11./(3139+lambda);
    d2 = sqrt(1-V11/(0.5*3139));

    Cov_02 = 2*phi2_c.*V11 - 4*phi.*V12;
    Cov_12 = 2*phi2_c.*V12 - 4*phi.*V22;
    f1 = 0.999; f2 = 1.002;
    f1 = 1; f2 = 1;
    V11hat = V11 + 2*V11.*(1-f1*sqrt(V11/(0.5*3138))).*delta(:,1) - V11.*(1-f2*sqrt(V11/(0.5*3138))).*delta(:,2);
    % (exp version) V11hat = V11 .* exp(2*(1-sqrt(V11/(0.5*3138))).*delta(:,1));
    %V11hat = V11 + (2*V11 - (3139 + 3*lambda)).*delta(:,1);
    %V11hat = V11 + (2*V11 - 3139*(1-d2).^2.*(1+2*d2)).*delta(:,1);

    %V12hat = V12 .* exp(-0.35*delta(:,1) - 0.9*delta(:,2));
    V12hat = V12 - V11.*(1-sqrt(V11/(0.5*3138))).*delta(:,1) + (V22 + Cov_02).*delta(:,2);

    %V22hat = V22 + 2*V22.*delta(:,1) + Cov_12.*delta(:,2);
    H221 = L22 + (2*(1-phi.^2)) .* V11 - 4*phi.*V12;
    H222 = -2*(3*phi.^2+1) .* H12 - 6*phi .* H22 + Cov_12;
    V22hat = V22 + (V22./H22).*(H221.*delta(:,1) + H222.*delta(:,2));

    V11_mse(i) = sqrt(mean((V11 - V11_mode).^2));
    V12_mse(i) = sqrt(mean((V12 - V12_mode).^2));
    V22_mse(i) = sqrt(mean((V22 - V22_mode).^2));
    V11hat_mse(i) = sqrt(mean((V11hat - V11_mode).^2));
    V12hat_mse(i) = sqrt(mean((V12hat - V12_mode).^2));
    V22hat_mse(i) = sqrt(mean((V22hat - V22_mode).^2));
    [b, bint, dum1, dum2, stats] = regress(V11hat - V11_mode, Xth);
    V11hat_beta(i,:) = b;
    V11hat_R2(i) = stats(1);
    [b, bint, dum1, dum2, stats] = regress(V12hat - V12_mode, Xth);
    V12hat_beta(i,:) = b;
    V12hat_R2(i) = stats(1);
    [b, bint, dum1, dum2, stats] = regress(V22hat - V22_mode, Xth);
    V22hat_beta(i,:) = b;
    V22hat_R2(i) = stats(1);
end

newt = zeros(23, 9);
for i=1:23
    s = sim_array{i};
    th2_mode(i,:) = s.sh.th2';
    omega_mode(i) = exp(th2_mode(i,1));
    phi_mode(i) = tanh(th2_mode(i,2));

    f = (s.niter == 1);
    iota = f(f);
    delta = s.th(f,1:2) - th2_mode(i,:);
    Xth = [iota, delta];

    y = log(s.d1(f));%.* sqrt(1-s.phi(f).^2));
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    fprintf("%6.3f, %5.3f, %6.3f, %6.3f, ", b(1), b(2), b(3), stats(1));
    newt(i, 1) = b(1);
    newt(i, 2) = b(2);
    newt(i, 3) = b(3);
   
    y = log(s.omega(f).^2 .* s.d2(f));
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    fprintf("%6.3f, %6.3f, %6.3f, %6.3f, ", b(1), b(2), b(3), stats(1));
    newt(i, 4) = b(1);
    newt(i, 5) = b(2);
    newt(i, 6) = b(3);

    y = log(s.omega(f) .* s.ep(f));
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    fprintf("%6.2f, %7.3f, %6.3f, %6.3f\n", b(1), b(2), b(3), stats(1));
    newt(i, 7) = b(1);
    newt(i, 8) = b(2);
    newt(i, 9) = b(3);

    y = log(s.ep(f)) + log(s.d1(f)) - log(s.d2(f));
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    %fprintf("%6.2f, %7.3f, %6.3f, %6.3f\n", b(1), b(2), b(3), stats(1));

    y = log(s.ep(f)) + log(1-s.phi(f)) + log(s.d1(f)) - log(s.d2(f));
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    %fprintf("%6.2f, %7.3f, %6.3f, %6.3f\n", b(1), b(2), b(3), stats(1));

    y = s.V11_a(f);
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    %fprintf("%6.2f, %7.3f, %6.3f, %6.3f\n", b(1), b(2), b(3), stats(1));

    y = s.V11_b(f);
    [b, bint, dum1, dum2, stats] = regress(y, Xth);
    %fprintf("%6.2f, %7.3f, %6.3f, %6.3f\n", b(1), b(2), b(3), stats(1));
end

col = V(:,3);

figure(11)
scatter(3139-2*V(:,1).*sqrt(V(:,1)/1569), 3139+t(:,1), [], col, 'filled')
refline(1,0)
title("Predicting Psi 30, n-2*V11*sqrt(V11/(n-2)")

figure(12)
scatter(-4*(1-phi_mode).*V(:,1).*sqrt(V(:,1)/1569), t(:,2), [], col, 'filled')
refline(1,0)
title("Predicting Psi 21, -4(1-phi)*V11*sqrt(V11/(n-2))")

figure(13)
scatter(V(:,2), -t(:,2), [], col, 'filled')
refline(3.2,0)
title("Predicting Psi 21, 3*V12")

figure(14)
scatter(V(:,2), t(:,5), [], col, 'filled')
refline(6,0)
title("Predicing Psi 12, 6*V12")

figure(15)
scatter(V(:,2), -t(:,8), [], col, 'filled')
refline(20,0)
title("Predicting Psi 03, 20*V12")

figure(16)
plotmatrix(t(:,[1,2,5,8]))

%{
figure(11)
scatter(2*(1-sqrt(V(:,1)/1569)), newt(:,5))
axis equal

figure(12)
scatter(2*(newt(:,5)-2).*(1-phi_mode), newt(:,6))
axis equal

figure(13)
plotmatrix(newt(:, [2,3,5,6,8,9]))

figure(14)
scatter(2*V(:,1).*(1-phi_mode), V(:,2))
axis equal
%}