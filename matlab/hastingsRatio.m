function HR = hastingsRatio(hmoutSt, hmout)
    lnH = hmoutSt.lnp_y__x + hmoutSt.lnp_x - hmoutSt.lnq_x__y;
    lnH = lnH - (hmout.lnp_y__x + hmout.lnp_x - hmout.lnq_x__y);
    HR = min(1, exp(lnH)) * (~any(isnan(hmoutSt.x)));
end
