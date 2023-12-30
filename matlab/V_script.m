t = zeros(23, 12);
v12 = zeros(23, 3);
for i = 1:23
    y1 = log(V12(f));
    y2 = log(V12) - 0.5*(log(th(f,1) - log(th(f,2))));
    y3 = log(V12 - 2*phi(f).*(1-phi(f)).*V11(f));
    var12(i,1) = var();
    var12(i,2) = var();
    var12(i,3) = var();
    s = sim_array{i};
    f = (s.niter == 1);
    phi = s.phi;
    phi_c = 1-phi;
    phi_c2 = phi_c.^2;
    phi2_c = (1-phi.^2);
    V11 = s.V0(:,1);
    V11c = 0.5*3139 - phi.^2.*V11;
    V11cp = 0.5*3139 - phi.^3.*V11;
    V12 = s.V0(:,2);
    V22 = s.V0(:,3);
    V12n = V12./(phi.^2.*V11);
    V22n = V22./(phi.^2.*V11);
    sigma = s.sigma;
    omega = s.omega;
    H11 = s.H0(:,1);
    H12 = s.H0(:,2);
    H22 = s.H0(:,3);
    g1 = 0.5*3139 + H11;
    g2 = H12 - phi;

    X = [f(f), s.th(f,1), log(1-phi(f))];
    XX = [s.th(f,1), log(1-phi(f))];

    [b, bint, dum1, dum2, stats] = regress(log(V11c(f)), X);
    fprintf("%f, %f, %f, %f, ", b(1), b(2), b(3), stats(1));
    t(i, 1) = b(2);
    t(i, 2) = b(3);
    t(i, 3) = stats(1);
    % log (n/2 - phi^2 V11)

    [b, bint, dum1, dum2, stats] = regress(log(abs(V12n(f))), X);
    fprintf("%f, %f, %f, %f, ", b(1), b(2), b(3), stats(1));
    t(i, 4) = b(2);
    t(i, 5) = b(3);
    t(i, 6) = stats(1);
    % log V12

    [b, bint, dum1, dum2, stats] = regress(log(V22n(f)), X);
    fprintf("%f, %f, %f, %f, ", b(1), b(2), b(3), stats(1));
    t(i, 7) = b(2);
    t(i, 8) = b(3);
    t(i, 9) = stats(1);
    % log V22

    y = phi(f).*V12(f)./((1-phi(f)) .* V11(f));
    [b, bint, dum1, dum2, stats] = regress(log(abs(y)), X);
    fprintf("%f, %f, %f, %f\n", b(1), b(2), b(3), stats(1));
    t(i, 10) = b(2);
    t(i, 11) = b(3);
    t(i, 12) = stats(1);
    % Variation of V12n
 end