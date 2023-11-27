function [new_theta, new_hmout, thetaSt, draw_log] = ...
    joint_draw_update(model, prior, y, mode_sh, theta, hmout, draw_log)

    % Draw thSt
    [lnq_thSt, thetaSt] = theta_draw_eval(model, prior, y, mode_sh, theta, hmout);
    lnp_thSt = prior.log_eval(prior, thetaSt.th);

    % Draw xSt|thSt
    if prior.has_mu
        th_length_string = 'Long';
    else
        th_length_string = 'Short';
    end
    hmoutSt = hessianMethod(model, y, thetaSt, 'GradHess', th_length_string);
    hmoutSt.sh = compute_proposal_params(model, prior, y, mode_sh, thetaSt, hmoutSt);
    
    % Evaluate log q(theta|theta*)
    lnq_th = theta_draw_eval(model, prior, y, mode_sh, thetaSt, hmoutSt, theta);
    lnp_th = prior.log_eval(prior, theta.th);

    % Compute Hastings Ratio
    lnH = WJM_H(hmoutSt) + lnp_thSt - lnq_thSt;
    lnH = lnH - (WJM_H(hmout) + lnp_th - lnq_th);

    % Accept/reject
    aPr = min( 1, exp(lnH) );
    if( rand < aPr )
        new_theta = thetaSt;
        new_hmout = hmoutSt;
        if( draw_log.streak > draw_log.longest_streak )
            draw_log.longest_streak = draw_log.streak;
        end
        draw_log.streak = 0; 
    else
        new_theta = theta;
        new_hmout = hmout;
        draw_log.streak = draw_log.streak + 1;
    end
    draw_log.aPr = aPr;
end
