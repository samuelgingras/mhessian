function [new_theta, new_hmout, draw_log] = ...
    joint_draw_update(model, prior, theta, hmout, y, draw_log)


    % Draw thSt
    [lnq_thSt, thetaSt] = theta_draw_eval(prior, theta, hmout.q_theta);
    lnp_thSt = log_prior_eval(prior, thetaSt);

    % Draw xSt|thSt
    if prior.hyper.has_mu
        th_length_string = 'Long';
    else
        th_length_string = 'Short';
    end
    hmoutSt = hessianMethod(model, y, thetaSt, 'GradHess', th_length_string);
    
    % Skip proposal if fail to draw state
    if( ~any(isnan(hmout.x)) )

        % Evaluate log q(theta|theta*)
        lnq_th = theta_draw_eval(prior, thetaSt, hmoutSt.q_theta, theta);
        lnp_th = log_prior_eval(prior, theta);

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
    else
        fprintf('Oops\n')
        draw_log.aPr = 0;
        draw_log.streak = draw_log.streak + 1;
    end
end
