% File: efficiency_OBM.m
% 
% Efficiency computations using overlapping batch mean method
function eff = efficiency_OBM(h)
    hmean = mean(h);
    hvar = var(h);
    N = length(h);
    B = round(sqrt(N));
    const = B/((N-B)*(N-B+1));
    h_j = mean(h(1:B));
    nse2 = (h_j - hmean)^2;
    for j = 1:(N-B)
        h_j = h_j + (h(j+B) - h(j))/B;
        nse2 = nse2 + (h_j - hmean)^2;
    end
    nse2 = nse2 * const;
    eff.mean = hmean;           % Sample mean
    eff.std = sqrt(hvar);       % Sample standard deviation
    eff.nse = sqrt(nse2);       % Numerical standard error (simulation noise)
    eff.rne = (hvar/N) / nse2;  % Relative numerical efficiency
end
